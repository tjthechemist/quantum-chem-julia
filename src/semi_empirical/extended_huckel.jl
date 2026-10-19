# ==============================================================================
# QuantumChemMM.jl: Extended Hückel Theory (EHT)
# ==============================================================================

module ExtendedHuckel

using LinearAlgebra
using Printf
using ..Types
using ..Elements
using ..SemiEmpiricalBase
using ..SlaterOrbitals

export ExtendedHuckelModel, solve_extended_huckel, EHTResult

struct VSIPData
    vsip_s::Float64 # in eV (positive ionization potential, H_ii = -VSIP)
    vsip_p::Float64 # in eV
end

const VSIP_TABLE = Dict{Symbol, VSIPData}(
    :H  => VSIPData(13.60, 0.0),
    :He => VSIPData(24.59, 0.0),
    :Li => VSIPData(5.39, 3.54),
    :Be => VSIPData(9.32, 6.00),
    :B  => VSIPData(14.05, 8.30),
    :C  => VSIPData(21.40, 11.40),
    :N  => VSIPData(26.00, 13.40),
    :O  => VSIPData(32.30, 14.80),
    :F  => VSIPData(40.00, 18.10),
    :Na => VSIPData(5.14, 3.00),
    :Mg => VSIPData(7.65, 4.00),
    :Al => VSIPData(10.70, 6.00),
    :Si => VSIPData(17.30, 9.20),
    :P  => VSIPData(18.60, 14.00),
    :S  => VSIPData(20.00, 13.30),
    :Cl => VSIPData(30.00, 15.00),
    :Br => VSIPData(24.00, 12.50),
    :I  => VSIPData(20.00, 10.50)
)

struct ExtendedHuckelModel <: AbstractSemiEmpiricalMethod
    K::Float64 # Wolfsberg-Helmholz constant (standard: 1.75)
end

ExtendedHuckelModel(; K=1.75) = ExtendedHuckelModel(K)

struct EHTResult
    converged::Bool
    total_energy_ev::Float64
    mo_energies_ev::Vector{Float64}
    mo_coefficients::Matrix{Float64}
    density_matrix::Matrix{Float64}
    overlap_matrix::Matrix{Float64}
    hamiltonian_matrix::Matrix{Float64}
    mulliken_charges::Vector{Float64}
    overlap_populations::Dict{Tuple{Int, Int}, Float64}
end

"""
    solve_extended_huckel(mol::Molecule, model::ExtendedHuckelModel=ExtendedHuckelModel())::EHTResult
Performs all-valence Extended Hückel Theory calculation for any 3D molecular structure.
"""
function solve_extended_huckel(mol::Molecule, model::ExtendedHuckelModel=ExtendedHuckelModel())::EHTResult
    basis = SlaterOrbitals.build_valence_basis(mol)
    nb = length(basis)

    # 1. Compute analytical overlap matrix S
    S = SlaterOrbitals.compute_overlap_matrix(basis, mol)

    # 2. Build Hamiltonian matrix H
    H = zeros(Float64, nb, nb)
    for mu in 1:nb
        orb = basis[mu]
        sym = mol.atoms[orb.atom_index].symbol
        vsip = get(VSIP_TABLE, sym, VSIPData(15.0, 10.0))
        H_diag = -(orb.l == 0 ? vsip.vsip_s : vsip.vsip_p)
        H[mu, mu] = H_diag
    end

    # Off-diagonal elements: H_μν = 0.5 * K * S_μν * (H_μμ + H_νν)
    for mu in 1:nb
        for nu in (mu+1):nb
            s_val = S[mu, nu]
            h_off = 0.5 * model.K * s_val * (H[mu, mu] + H[nu, nu])
            H[mu, nu] = h_off
            H[nu, nu] = h_off # symmetric
            H[nu, mu] = h_off
        end
    end

    # 3. Solve Generalized Eigenvalue Problem: H * C = S * C * eps
    # S^(-1/2) symmetric orthogonalization
    eig_s = eigen(Symmetric(S))
    # Threshold tiny eigenvalues for numerical stability
    s_inv_sqrt_vals = [val > 1e-10 ? 1.0 / sqrt(val) : 0.0 for val in eig_s.values]
    X = eig_s.vectors * Diagonal(s_inv_sqrt_vals) * eig_s.vectors'

    H_prime = X' * H * X
    eig_h = eigen(Symmetric(H_prime))

    eps_ev = eig_h.values
    C = X * eig_h.vectors

    # 4. Count total valence electrons
    n_valence = sum(Elements.valence_electrons(at.symbol) for at in mol.atoms) - mol.total_charge
    n_occ_orbitals = div(n_valence, 2)
    has_unpaired = (n_valence % 2 != 0)

    # 5. Density Matrix: P_μν = 2 * sum_{i=1}^{n_occ} C_μi * C_νi
    P = zeros(Float64, nb, nb)
    for mu in 1:nb
        for nu in 1:nb
            sum_occ = 0.0
            for i in 1:n_occ_orbitals
                sum_occ += 2.0 * C[mu, i] * C[nu, i]
            end
            if has_unpaired
                sum_occ += 1.0 * C[mu, n_occ_orbitals + 1] * C[nu, n_occ_orbitals + 1]
            end
            P[mu, nu] = sum_occ
        end
    end

    # 6. Total Energy: sum of occupied orbital energies
    total_energy_ev = 0.0
    for i in 1:n_occ_orbitals
        total_energy_ev += 2.0 * eps_ev[i]
    end
    if has_unpaired
        total_energy_ev += 1.0 * eps_ev[n_occ_orbitals + 1]
    end

    # 7. Mulliken Population Analysis
    # PS product matrix
    PS = P * S
    nat = n_atoms(mol)
    mulliken_charges = zeros(Float64, nat)

    for i in 1:nat
        z_val = Elements.valence_electrons(mol.atoms[i].symbol)
        gross_pop = 0.0
        for mu in 1:nb
            if basis[mu].atom_index == i
                gross_pop += PS[mu, mu]
            end
        end
        mulliken_charges[i] = z_val - gross_pop
    end

    # Mulliken Overlap Populations between atoms A and B:
    # OP_AB = 2 * sum_{mu in A} sum_{nu in B} P_mu,nu * S_mu,nu
    overlap_pops = Dict{Tuple{Int, Int}, Float64}()
    for a in 1:nat
        for b in (a+1):nat
            op_val = 0.0
            for mu in 1:nb
                basis[mu].atom_index == a || continue
                for nu in 1:nb
                    basis[nu].atom_index == b || continue
                    op_val += 2.0 * P[mu, nu] * S[mu, nu]
                end
            end
            overlap_pops[(a, b)] = op_val
        end
    end

    return EHTResult(true, total_energy_ev, eps_ev, C, P, S, H,
                     mulliken_charges, overlap_pops)
end

function print_eht_summary(res::EHTResult)
    println("==================================================")
    println("           Extended Hückel Theory (EHT)           ")
    println("==================================================")
    @printf("  Total Electronic Energy:  %16.6f eV\n", res.total_energy_ev)
    println("--------------------------------------------------")
    println("  Frontier Molecular Orbitals (eV):")
    n_mo = length(res.mo_energies_ev)
    for i in 1:min(8, n_mo)
        @printf("    MO #%2d:  %12.4f eV\n", i, res.mo_energies_ev[i])
    end
    println("--------------------------------------------------")
    println("  Mulliken Net Atomic Charges (e):")
    for (i, q) in enumerate(res.mulliken_charges)
        @printf("    Atom %2d:  %+8.4f\n", i, q)
    end
    println("==================================================")
end

end # module ExtendedHuckel

