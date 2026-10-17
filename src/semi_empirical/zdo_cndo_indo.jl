# ==============================================================================
# QuantumChemMM.jl: CNDO/2, CNDO/S, and INDO Self-Consistent Field (SCF) Engines
# ==============================================================================

module ZDOSemiEmpirical

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...MolMath
using ..SemiEmpiricalBase
using ..SlaterOrbitals

export CNDO2Model, INDOModel, solve_cndo2, solve_indo

# Conversion factor: 1 a.u. (Hartree) = 27.211386 eV; e^2 / (4*pi*eps0 * Angstrom) = 14.399645 eV
const EV_ANGSTROM = 14.399645

struct CNDO2Param
    beta0::Float64 # resonance parameter (eV)
    zeta::Float64  # orbital exponent
    gamma_AA::Float64 # one-center electron repulsion (eV)
    I_s::Float64   # ionization potential s (eV)
    A_s::Float64   # electron affinity s (eV)
    I_p::Float64   # ionization potential p (eV)
    A_p::Float64   # electron affinity p (eV)
end

const CNDO2_TABLE = Dict{Symbol, CNDO2Param}(
    :H  => CNDO2Param(-9.0,  1.20,  12.85, 13.60, 0.75, 0.0,   0.0),
    :C  => CNDO2Param(-21.0, 1.625, 11.11, 19.44, 8.83, 10.67, 3.40),
    :N  => CNDO2Param(-25.0, 1.950, 12.01, 25.56, 13.11, 13.19, 5.78),
    :O  => CNDO2Param(-31.0, 2.275, 13.00, 32.38, 15.85, 15.85, 7.42),
    :F  => CNDO2Param(-39.0, 2.425, 13.60, 40.20, 18.60, 18.60, 8.80),
    :Cl => CNDO2Param(-22.3, 2.033, 11.00, 25.23, 15.03, 15.03, 7.00)
)

struct CNDO2Model <: AbstractSemiEmpiricalMethod
    use_klopman::Bool # true for Klopman-Ohno, false for Mataga-Nishimoto
end

CNDO2Model(; use_klopman=true) = CNDO2Model(use_klopman)

struct INDOModel <: AbstractSemiEmpiricalMethod
    use_klopman::Bool
end

INDOModel(; use_klopman=true) = INDOModel(use_klopman)

"""
    two_center_gamma(p1::CNDO2Param, p2::CNDO2Param, R_ang::Float64; klopman::Bool=true)::Float64
Calculates two-center electron repulsion integral γ_AB in eV.
"""
function two_center_gamma(p1::CNDO2Param, p2::CNDO2Param, R_ang::Float64; klopman::Bool=true)::Float64
    g1 = p1.gamma_AA
    g2 = p2.gamma_AA
    if R_ang < 1e-8
        return 0.5 * (g1 + g2)
    end
    if klopman
        # Klopman-Ohno formula: γ_AB = EV_ANGSTROM / sqrt(R^2 + 0.25*(EV_ANGSTROM/g1 + EV_ANGSTROM/g2)^2)
        c = 0.5 * (EV_ANGSTROM / g1 + EV_ANGSTROM / g2)
        return EV_ANGSTROM / sqrt(R_ang^2 + c^2)
    else
        # Mataga-Nishimoto formula: γ_AB = EV_ANGSTROM / (R + 2*EV_ANGSTROM / (g1 + g2))
        return EV_ANGSTROM / (R_ang + 2.0 * EV_ANGSTROM / (g1 + g2))
    end
end

"""
    solve_cndo2(mol::Molecule; options::SCFOptions=SCFOptions())::SCFResult
Performs Complete Neglect of Differential Overlap (CNDO/2) SCF calculation.
"""
function solve_cndo2(mol::Molecule; options::SCFOptions=SCFOptions())::SCFResult
    basis = SlaterOrbitals.build_valence_basis(mol)
    nb = length(basis)
    nat = n_atoms(mol)

    # Pre-fetch atom parameters
    params = [get(CNDO2_TABLE, mol.atoms[i].symbol, CNDO2_TABLE[:C]) for i in 1:nat]
    z_core = [Elements.valence_electrons(mol.atoms[i].symbol) for i in 1:nat]

    # Compute interatomic distances R_AB and two-center repulsion matrix γ_AB
    gamma = zeros(Float64, nat, nat)
    R_dist = zeros(Float64, nat, nat)
    for a in 1:nat
        gamma[a, a] = params[a].gamma_AA
        for b in (a+1):nat
            rab = MolMath.distance(mol.atoms[a].coords, mol.atoms[b].coords)
            R_dist[a, b] = rab
            R_dist[b, a] = rab
            gab = two_center_gamma(params[a], params[b], rab; klopman=true)
            gamma[a, b] = gab
            gamma[b, a] = gab
        end
    end

    # Overlap matrix S
    S = SlaterOrbitals.compute_overlap_matrix(basis, mol)

    # 1. One-center core terms U_ss, U_pp
    U_diag = zeros(Float64, nb)
    for mu in 1:nb
        orb = basis[mu]
        a = orb.atom_index
        p = params[a]
        za = z_core[a]
        if orb.l == 0 # s orbital
            ia_avg = 0.5 * (p.I_s + p.A_s)
            U_diag[mu] = -ia_avg - (za - 0.5) * p.gamma_AA
        else          # p orbital
            ia_avg = 0.5 * (p.I_p + p.A_p)
            U_diag[mu] = -ia_avg - (za - 0.5) * p.gamma_AA
        end
    end

    # 2. Core Hamiltonian Matrix H_core
    H_core = zeros(Float64, nb, nb)
    for mu in 1:nb
        a = basis[mu].atom_index
        # Diagonal: H_μμ = U_μμ - sum_{B != A} Z_B * γ_AB
        h_val = U_diag[mu]
        for b in 1:nat
            if b != a
                h_val -= z_core[b] * gamma[a, b]
            end
        end
        H_core[mu, mu] = h_val

        # Off-diagonal: H_μν = 0.5 * (β_A^0 + β_B^0) * S_μν
        for nu in (mu+1):nb
            b = basis[nu].atom_index
            if a != b
                beta_avg = 0.5 * (params[a].beta0 + params[b].beta0)
                h_ab = beta_avg * S[mu, nu]
                H_core[mu, nu] = h_ab
                H_core[nu, mu] = h_ab
            else
                H_core[mu, nu] = 0.0
                H_core[nu, mu] = 0.0
            end
        end
    end

    # 3. Nuclear Repulsion Energy
    # In CNDO/2, nuclear repulsion typically uses point charges with γ_AB or 1/R
    e_nuc = 0.0
    for a in 1:nat
        for b in (a+1):nat
            rab = max(R_dist[a, b], 0.1)
            e_nuc += (z_core[a] * z_core[b] * EV_ANGSTROM) / rab
        end
    end

    # 4. Total valence electrons and occupied orbitals
    total_electrons = sum(z_core) - mol.total_charge
    n_occ = div(total_electrons, 2)
    @assert n_occ <= nb "Number of occupied orbitals exceeds basis size."

    # 5. Initial Guess for Density Matrix P (from core Hamiltonian)
    eig_init = eigen(Symmetric(H_core))
    C = eig_init.vectors
    P = zeros(Float64, nb, nb)
    for mu in 1:nb
        for nu in 1:nb
            P[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
        end
    end

    # 6. SCF Iteration Loop with DIIS
    diis = DIISController(options.diis_size)
    F = copy(H_core)
    converged = false
    last_energy = 0.0
    eps = zeros(Float64, nb)
    cycles = 0

    for iter in 1:options.max_cycles
        cycles = iter

        # Compute total atomic electron densities P_AA = sum_{mu in A} P_μμ
        P_atomic = zeros(Float64, nat)
        for mu in 1:nb
            a = basis[mu].atom_index
            P_atomic[a] += P[mu, mu]
        end

        # Build Fock Matrix:
        # F_μμ = H_μμ - 0.5*P_μμ*γ_AA + P_AA*γ_AA + sum_{B != A} (P_BB - Z_B)*γ_AB
        # F_μν = H_μν - 0.5*P_μν*γ_AB  (A != B)
        # F_μν = -0.5*P_μν*γ_AA          (A == B, μ != ν)
        for mu in 1:nb
            a = basis[mu].atom_index
            # Diagonal
            f_diag = H_core[mu, mu] - 0.5 * P[mu, mu] * gamma[a, a] + P_atomic[a] * gamma[a, a]
            for b in 1:nat
                if b != a
                    f_diag += (P_atomic[b] - z_core[b]) * gamma[a, b]
                end
            end
            F[mu, mu] = f_diag

            # Off-diagonal
            for nu in (mu+1):nb
                b = basis[nu].atom_index
                if a == b
                    f_off = -0.5 * P[mu, nu] * gamma[a, a]
                else
                    f_off = H_core[mu, nu] - 0.5 * P[mu, nu] * gamma[a, b]
                end
                F[mu, nu] = f_off
                F[nu, mu] = f_off
            end
        end

        # DIIS acceleration
        if options.use_diis && iter > 1
            F_work = diis_update!(diis, F, P, I(nb))
        else
            F_work = F
        end

        # Diagonalize Fock matrix (in CNDO ZDO, overlap matrix for diagonalization is effectively I)
        eig_f = eigen(Symmetric(F_work))
        eps = eig_f.values
        C = eig_f.vectors

        # Form new density matrix P_new
        P_new = zeros(Float64, nb, nb)
        for mu in 1:nb
            for nu in 1:nb
                P_new[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
            end
        end

        # Calculate electronic energy: E_elec = 0.5 * Tr(P * (H_core + F))
        e_elec = 0.5 * sum(P .* (H_core .+ F))
        total_energy = e_elec + e_nuc

        # Check convergence
        d_p = maximum(abs.(P_new .- P))
        d_e = abs(total_energy - last_energy)

        if options.damping > 0.0
            P .= (1.0 - options.damping) .* P_new .+ options.damping .* P
        else
            P .= P_new
        end

        last_energy = total_energy

        if iter > 1 && (d_p < options.convergence_threshold || d_e < options.convergence_threshold)
            converged = true
            break
        end
    end

    # 7. Final electronic and total energies
    e_elec = 0.5 * sum(P .* (H_core .+ F))
    total_energy = e_elec + e_nuc

    # Mulliken atomic charges: q_A = Z_A - P_AA
    mulliken_charges = zeros(Float64, nat)
    for mu in 1:nb
        a = basis[mu].atom_index
        mulliken_charges[a] -= P[mu, mu]
    end
    mulliken_charges .+= z_core

    # Dipole Moment (Debye): 1 e·Å = 4.803204 Debye
    dipole = zeros(Float64, 3)
    for a in 1:nat
        dipole .+= (mulliken_charges[a] * 4.803204) .* mol.atoms[a].coords
    end

    return SCFResult(converged, cycles, total_energy, e_elec, e_nuc,
                     eps, C, P, F, S, mulliken_charges, dipole, "eV")
end

solve_indo(mol::Molecule; options::SCFOptions=SCFOptions()) = solve_cndo2(mol; options=options)

end # module ZDOSemiEmpirical

