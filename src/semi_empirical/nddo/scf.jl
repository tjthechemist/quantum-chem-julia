# ==============================================================================
# QuantumChemMM.jl: NDDO (MNDO, AM1, PM3, RM1, PM6) Roothaan-Hall SCF Solver
# ==============================================================================

module NDDOSCF

using LinearAlgebra
using Printf
using ....Types
using ....Elements
using ....MolMath
using ...SemiEmpiricalBase
using ...SlaterOrbitals
using ..NDDOParameters
using ..NDDOBase

export NDDOMethod, solve_nddo, NDDOResult

# 1 eV = 23.060548 kcal/mol
const EV_TO_KCAL = 23.060548

struct NDDOMethod <: AbstractSemiEmpiricalMethod
    model::Symbol # :mndo, :am1, :pm3, :rm1, :pm6
end

NDDOMethod(model::Symbol=:am1) = NDDOMethod(model)

struct NDDOResult
    model::Symbol
    converged::Bool
    cycles::Int
    total_energy_ev::Float64
    electronic_energy_ev::Float64
    nuclear_repulsion_ev::Float64
    heat_of_formation_kcal::Float64
    mo_energies_ev::Vector{Float64}
    mo_coefficients::Matrix{Float64}
    density_matrix::Matrix{Float64}
    fock_matrix::Matrix{Float64}
    mulliken_charges::Vector{Float64}
    dipole_moment::Vector{Float64}
end

"""
    solve_nddo(mol::Molecule, method::NDDOMethod=NDDOMethod(:am1); options::SCFOptions=SCFOptions())::NDDOResult
Runs Self-Consistent Field (SCF) calculation using the NDDO family methods (MNDO, AM1, PM3, RM1, PM6).
"""
function solve_nddo(mol::Molecule, method::NDDOMethod=NDDOMethod(:am1);
                    options::SCFOptions=SCFOptions())::NDDOResult
    basis = SlaterOrbitals.build_valence_basis(mol)
    nb = length(basis)
    nat = n_atoms(mol)

    # 1. Fetch atom parameters and valence counts
    params = [NDDOParameters.get_nddo_param(method.model, mol.atoms[i].symbol) for i in 1:nat]
    z_core = [Elements.valence_electrons(mol.atoms[i].symbol) for i in 1:nat]

    # 2. Compute overlap matrix S
    S = SlaterOrbitals.compute_overlap_matrix(basis, mol)

    # 3. Nuclear Repulsion Energy
    e_nuc = 0.0
    for a in 1:nat
        for b in (a+1):nat
            rab = MolMath.distance(mol.atoms[a].coords, mol.atoms[b].coords)
            g_ss_ab = NDDOBase.calc_nddo_two_electron_integral(:s, :s, :s, :s, params[a], params[b], mol.atoms[a].coords .- mol.atoms[b].coords)
            e_nuc += NDDOBase.calc_core_repulsion(params[a], params[b], z_core[a], z_core[b], rab, g_ss_ab)
        end
    end

    # 4. Core Hamiltonian H_core
    H_core = zeros(Float64, nb, nb)
    for mu in 1:nb
        orb_mu = basis[mu]
        a = orb_mu.atom_index
        pA = params[a]

        # One-center diagonal core energy U_ss or U_pp
        u_val = (orb_mu.l == 0) ? pA.U_ss : pA.U_pp

        # Attraction to external cores B: V_{A, B} = -Z_B * (mu mu | s_B s_B)
        v_ext = 0.0
        for b in 1:nat
            if b != a
                rAB_vec = mol.atoms[a].coords .- mol.atoms[b].coords
                g_mu_sB = NDDOBase.calc_nddo_two_electron_integral(orb_mu.m_type, orb_mu.m_type, :s, :s, pA, params[b], rAB_vec)
                v_ext -= z_core[b] * g_mu_sB
            end
        end
        H_core[mu, mu] = u_val + v_ext

        # Off-diagonal elements
        for nu in (mu+1):nb
            orb_nu = basis[nu]
            b = orb_nu.atom_index
            if a != b
                # Resonance integral beta: 0.5 * (beta_mu + beta_nu) * S_munu
                h_res = NDDOBase.calc_nddo_resonance_integral(pA, params[b], orb_mu.l, orb_nu.l, S[mu, nu])
                H_core[mu, nu] = h_res
                H_core[nu, mu] = h_res
            else
                H_core[mu, nu] = 0.0
                H_core[nu, mu] = 0.0
            end
        end
    end

    # 5. Occupied Orbitals
    total_electrons = sum(z_core) - mol.total_charge
    n_occ = div(total_electrons, 2)

    # 6. Initial Guess Density Matrix from Core Hamiltonian
    eig_init = eigen(Symmetric(H_core))
    C = eig_init.vectors
    P = zeros(Float64, nb, nb)
    for mu in 1:nb
        for nu in 1:nb
            P[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
        end
    end

    # 7. SCF Roothaan-Hall Iteration Loop with DIIS
    diis = DIISController(options.diis_size)
    F = copy(H_core)
    eps = zeros(Float64, nb)
    converged = false
    last_energy = 0.0
    cycles = 0

    for iter in 1:options.max_cycles
        cycles = iter

        # Form Fock Matrix:
        # F_μν = H_μν + sum_{λ, σ} P_{λσ} [ (μν|λσ) - 0.5*(μλ|νσ) ]
        F .= H_core

        # Coulomb & Exchange contributions
        for mu in 1:nb
            a = basis[mu].atom_index
            for nu in mu:nb
                b = basis[nu].atom_index
                val_coulomb = 0.0
                val_exchange = 0.0

                for lam in 1:nb
                    c_atom = basis[lam].atom_index
                    for sig in 1:nb
                        d_atom = basis[sig].atom_index
                        p_val = P[lam, sig]
                        if abs(p_val) < 1e-12
                            continue
                        end

                        # NDDO selection rule: (μν|λσ) non-zero only if atom(μ)==atom(ν) and atom(λ)==atom(σ)
                        if a == b && c_atom == d_atom
                            r_ac = mol.atoms[a].coords .- mol.atoms[c_atom].coords
                            g_coul = NDDOBase.calc_nddo_two_electron_integral(basis[mu].m_type, basis[nu].m_type,
                                                                             basis[lam].m_type, basis[sig].m_type,
                                                                             params[a], params[c_atom], r_ac)
                            val_coulomb += p_val * g_coul
                        end

                        # Exchange selection rule: (μλ|νσ) non-zero only if atom(μ)==atom(λ) and atom(ν)==atom(σ)
                        if a == c_atom && b == d_atom
                            r_ab = mol.atoms[a].coords .- mol.atoms[b].coords
                            g_exch = NDDOBase.calc_nddo_two_electron_integral(basis[mu].m_type, basis[lam].m_type,
                                                                             basis[nu].m_type, basis[sig].m_type,
                                                                             params[a], params[b], r_ab)
                            val_exchange += 0.5 * p_val * g_exch
                        end
                    end
                end

                f_tot = H_core[mu, nu] + val_coulomb - val_exchange
                F[mu, nu] = f_tot
                F[nu, mu] = f_tot
            end
        end

        # DIIS Acceleration
        F_work = options.use_diis && iter > 1 ? diis_update!(diis, F, P, I(nb)) : F

        # Diagonalize Fock matrix
        eig_f = eigen(Symmetric(F_work))
        eps = eig_f.values
        C = eig_f.vectors

        # Form new density matrix
        P_new = zeros(Float64, nb, nb)
        for mu in 1:nb
            for nu in 1:nb
                P_new[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
            end
        end

        # Energy calculation
        e_elec = 0.5 * sum(P .* (H_core .+ F))
        tot_energy = e_elec + e_nuc

        d_p = maximum(abs.(P_new .- P))
        d_e = abs(tot_energy - last_energy)

        if options.damping > 0.0
            P .= (1.0 - options.damping) .* P_new .+ options.damping .* P
        else
            P .= P_new
        end

        last_energy = tot_energy

        if iter > 1 && (d_p < options.convergence_threshold || d_e < options.convergence_threshold)
            converged = true
            break
        end
    end

    # 8. Heat of Formation (ΔH_f)
    e_elec = 0.5 * sum(P .* (H_core .+ F))
    total_energy_ev = e_elec + e_nuc

    # Electronic binding energy = total_energy - sum(isolated atom electronic energies)
    e_bind_ev = total_energy_ev - sum(params[i].isolated_e for i in 1:nat)
    e_bind_kcal = e_bind_ev * EV_TO_KCAL

    # Delta H_f = E_binding + sum(atomic heat of formation)
    heat_of_formation = e_bind_kcal + sum(params[i].heat_atom for i in 1:nat)

    # 9. Net Atomic Charges
    mulliken_charges = zeros(Float64, nat)
    for mu in 1:nb
        a = basis[mu].atom_index
        mulliken_charges[a] -= P[mu, mu]
    end
    mulliken_charges .+= z_core

    # Dipole Moment
    dipole = zeros(Float64, 3)
    for a in 1:nat
        dipole .+= (mulliken_charges[a] * 4.803204) .* mol.atoms[a].coords
    end

    return NDDOResult(method.model, converged, cycles, total_energy_ev,
                      e_elec, e_nuc, heat_of_formation, eps, C, P, F,
                      mulliken_charges, dipole)
end

function print_nddo_summary(res::NDDOResult)
    println("==================================================")
    @printf("       NDDO Method: %-6s Calculation Results      \n", uppercase(string(res.model)))
    println("==================================================")
    @printf("  Convergence Status:       %s (%d cycles)\n", res.converged ? "CONVERGED" : "FAILED", res.cycles)
    @printf("  Electronic Energy:        %16.6f eV\n", res.electronic_energy_ev)
    @printf("  Nuclear Repulsion Energy: %16.6f eV\n", res.nuclear_repulsion_ev)
    @printf("  TOTAL ENERGY:             %16.6f eV\n", res.total_energy_ev)
    @printf("  Heat of Formation (ΔHf):  %16.4f kcal/mol\n", res.heat_of_formation_kcal)
    dip_mag = norm(res.dipole_moment)
    @printf("  Dipole Moment:            %16.4f Debye\n", dip_mag)
    println("--------------------------------------------------")
    println("  Frontier Molecular Orbital Energies (eV):")
    for i in 1:min(8, length(res.mo_energies_ev))
        @printf("    MO #%2d:  %12.4f eV\n", i, res.mo_energies_ev[i])
    end
    println("==================================================")
end

end # module NDDOSCF

