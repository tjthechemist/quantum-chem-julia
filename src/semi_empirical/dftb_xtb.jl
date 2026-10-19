# ==============================================================================
# QuantumChemMM.jl: Tight-Binding (DFTB & GFN-xTB) Engines
# ==============================================================================

module TightBinding

using LinearAlgebra
using Printf
using ..Types
using ..Elements
using ..MolMath
using ..SemiEmpiricalBase
using ..SlaterOrbitals
using ..DispersionD3

export TightBindingModel, solve_tight_binding, DFTBResult

struct TBAtomParam
    eps_s::Float64    # atomic level s (eV)
    eps_p::Float64    # atomic level p (eV)
    hubbard_U::Float64# chemical hardness / Hubbard U (eV)
    alpha_rep::Float64# repulsion decay (Å⁻¹)
    k_rep::Float64    # repulsion prefactor (eV)
end

const TB_PARAMS = Dict{Symbol, TBAtomParam}(
    :H  => TBAtomParam(-13.6,  0.0, 11.5, 3.5,  25.0),
    :C  => TBAtomParam(-19.5, -10.5, 9.8, 3.2, 120.0),
    :N  => TBAtomParam(-24.5, -13.2, 11.0, 3.4, 160.0),
    :O  => TBAtomParam(-30.0, -15.0, 12.5, 3.6, 210.0),
    :F  => TBAtomParam(-38.0, -18.0, 14.0, 3.8, 280.0),
    :P  => TBAtomParam(-18.0, -10.0, 8.5, 2.8, 140.0),
    :S  => TBAtomParam(-20.0, -11.5, 9.2, 2.9, 180.0),
    :Cl => TBAtomParam(-25.0, -13.0, 10.5, 3.1, 230.0)
)

struct TightBindingModel <: AbstractSemiEmpiricalMethod
    variant::Symbol   # :dftb2 (SCC-DFTB), :gfn0_xtb, :gfn1_xtb, :gfn2_xtb
    include_dispersion::Bool
end

TightBindingModel(variant::Symbol=:gfn2_xtb; include_dispersion=true) =
    TightBindingModel(variant, include_dispersion)

struct DFTBResult
    variant::Symbol
    converged::Bool
    cycles::Int
    total_energy_ev::Float64
    band_energy_ev::Float64
    charge_energy_ev::Float64
    repulsion_energy_ev::Float64
    dispersion_energy_ev::Float64
    mo_energies_ev::Vector{Float64}
    mo_coefficients::Matrix{Float64}
    mulliken_charges::Vector{Float64}
end

"""
    hubbard_gamma(UA::Float64, UB::Float64, RAB::Float64)::Float64
DFTB Hubbard charge interaction function γ_AB in eV.
"""
function hubbard_gamma(UA::Float64, UB::Float64, RAB::Float64)::Float64
    if RAB < 1e-8
        return 0.5 * (UA + UB)
    end
    # Klopman-Ohno type interpolation
    c = 0.5 * (14.399645 / UA + 14.399645 / UB)
    return 14.399645 / sqrt(RAB^2 + c^2)
end

"""
    solve_tight_binding(mol::Molecule, model::TightBindingModel=TightBindingModel();
                        options::SCFOptions=SCFOptions())::DFTBResult
Runs Self-Consistent Charge DFTB / GFN-xTB calculation.
"""
function solve_tight_binding(mol::Molecule, model::TightBindingModel=TightBindingModel();
                             options::SCFOptions=SCFOptions())::DFTBResult
    basis = SlaterOrbitals.build_valence_basis(mol)
    nb = length(basis)
    nat = n_atoms(mol)

    params = [get(TB_PARAMS, mol.atoms[i].symbol, TB_PARAMS[:C]) for i in 1:nat]
    z_val = [Elements.valence_electrons(mol.atoms[i].symbol) for i in 1:nat]

    # 1. Overlap Matrix S
    S = SlaterOrbitals.compute_overlap_matrix(basis, mol)

    # 2. Hubbard Gamma Matrix γ_AB
    gamma = zeros(Float64, nat, nat)
    for a in 1:nat
        for b in a:nat
            rab = MolMath.distance(mol.atoms[a].coords, mol.atoms[b].coords)
            gab = hubbard_gamma(params[a].hubbard_U, params[b].hubbard_U, rab)
            gamma[a, b] = gab
            gamma[b, a] = gab
        end
    end

    # 3. Pairwise Core Repulsion Energy: E_rep = sum_{A < B} k_rep * exp(-alpha * R)
    e_rep = 0.0
    for a in 1:nat
        for b in (a+1):nat
            rab = MolMath.distance(mol.atoms[a].coords, mol.atoms[b].coords)
            alpha_ab = 0.5 * (params[a].alpha_rep + params[b].alpha_rep)
            k_ab = sqrt(params[a].k_rep * params[b].k_rep)
            e_rep += k_ab * exp(-alpha_ab * rab)
        end
    end

    # 4. Dispersion Energy (D3)
    e_disp_ev = 0.0
    if model.include_dispersion
        e_disp_kcal, _ = DispersionD3.calc_d3_dispersion(mol)
        e_disp_ev = e_disp_kcal / 23.060548 # kcal/mol to eV
    end

    # 5. Non-SCC Zero-order Hamiltonian H0
    H0 = zeros(Float64, nb, nb)
    for mu in 1:nb
        orb_mu = basis[mu]
        a = orb_mu.atom_index
        pA = params[a]
        H0[mu, mu] = (orb_mu.l == 0) ? pA.eps_s : pA.eps_p

        for nu in (mu+1):nb
            orb_nu = basis[nu]
            b = orb_nu.atom_index
            if a != b
                pB = params[b]
                rab = MolMath.distance(mol.atoms[a].coords, mol.atoms[b].coords)
                s_val = S[mu, nu]
                # Distance-dependent hopping
                e_avg = sqrt(abs(H0[mu, mu] * ((orb_nu.l == 0) ? pB.eps_s : pB.eps_p)))
                hop = -1.75 * e_avg * s_val * exp(-0.15 * rab)
                H0[mu, nu] = hop
                H0[nu, mu] = hop
            end
        end
    end

    # 6. Generalized Eigenvalue Setup
    eig_s = eigen(Symmetric(S))
    s_inv_sqrt = [val > 1e-10 ? 1.0 / sqrt(val) : 0.0 for val in eig_s.values]
    X = eig_s.vectors * Diagonal(s_inv_sqrt) * eig_s.vectors'

    total_electrons = sum(z_val) - mol.total_charge
    n_occ = div(total_electrons, 2)

    # If non-self-consistent (GFN0-xTB / DFTB1), solve once
    delta_q = zeros(Float64, nat)
    converged = false
    cycles = 0
    eps = zeros(Float64, nb)
    C = zeros(Float64, nb, nb)
    P = zeros(Float64, nb, nb)
    H = copy(H0)

    max_iter = (model.variant == :gfn0_xtb) ? 1 : options.max_cycles

    for iter in 1:max_iter
        cycles = iter

        # Form shifted Hamiltonian H_μν = H0_μν + 0.5 * S_μν * sum_C (γ_AC + γ_BC) * Δq_C
        H .= H0
        if model.variant != :gfn0_xtb
            for mu in 1:nb
                a = basis[mu].atom_index
                shift_a = sum(gamma[a, c] * delta_q[c] for c in 1:nat)
                for nu in mu:nb
                    b = basis[nu].atom_index
                    shift_b = sum(gamma[b, c] * delta_q[c] for c in 1:nat)
                    h_shift = 0.5 * S[mu, nu] * (shift_a + shift_b)
                    H[mu, nu] += h_shift
                    H[nu, mu] = H[mu, nu]
                end
            end
        end

        # Solve H * C = S * C * eps
        H_prime = X' * H * X
        eig_h = eigen(Symmetric(H_prime))
        eps = eig_h.values
        C = X * eig_h.vectors

        # Density Matrix P
        P .= 0.0
        for mu in 1:nb
            for nu in 1:nb
                P[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
            end
        end

        # Mulliken Gross Populations and new Δq
        PS = P * S
        delta_q_new = zeros(Float64, nat)
        for i in 1:nat
            gross_pop = sum(PS[mu, mu] for mu in 1:nb if basis[mu].atom_index == i)
            # Δq = q_net = Z_val - gross_pop
            delta_q_new[i] = z_val[i] - gross_pop
        end

        # Check convergence on charges
        err_q = maximum(abs.(delta_q_new .- delta_q))
        delta_q .= delta_q_new

        if err_q < 1e-6 || model.variant == :gfn0_xtb
            converged = true
            break
        end
    end

    # 7. Energy Decomposition
    # Band energy: E_band = sum_{i=1}^{n_occ} 2 * eps_i
    e_band = 2.0 * sum(eps[1:n_occ])
    # Second-order charge fluctuation: E_2nd = 0.5 * sum_{A, B} Δq_A * Δq_B * γ_AB
    e_charge = 0.5 * sum(delta_q[a] * delta_q[b] * gamma[a, b] for a in 1:nat for b in 1:nat)

    # In DFTB2 / GFN-xTB: E_tot = E_band - 0.5 * sum(Δq*γ*Δq) + E_rep + E_disp
    total_energy_ev = e_band - e_charge + e_rep + e_disp_ev

    return DFTBResult(model.variant, converged, cycles, total_energy_ev,
                      e_band, e_charge, e_rep, e_disp_ev, eps, C, delta_q)
end

function print_tb_summary(res::DFTBResult)
    println("==================================================")
    @printf("     Tight-Binding Method: %-8s Results          \n", uppercase(string(res.variant)))
    println("==================================================")
    @printf("  Convergence Status:       %s (%d iterations)\n", res.converged ? "CONVERGED" : "FAILED", res.cycles)
    @printf("  Band Energy (E_band):     %16.6f eV\n", res.band_energy_ev)
    @printf("  Charge Fluctuation (E_q): %16.6f eV\n", res.charge_energy_ev)
    @printf("  Repulsive Energy (E_rep): %16.6f eV\n", res.repulsion_energy_ev)
    @printf("  Dispersion (E_disp):      %16.6f eV\n", res.dispersion_energy_ev)
    @printf("  TOTAL ENERGY:             %16.6f eV\n", res.total_energy_ev)
    println("--------------------------------------------------")
    println("  Frontier Molecular Orbital Energies (eV):")
    for i in 1:min(8, length(res.mo_energies_ev))
        @printf("    MO #%2d:  %12.4f eV\n", i, res.mo_energies_ev[i])
    end
    println("--------------------------------------------------")
    println("  Mulliken Net Charges (e):")
    for (i, q) in enumerate(res.mulliken_charges)
        @printf("    Atom %2d:  %+8.4f\n", i, q)
    end
    println("==================================================")
end

end # module TightBinding

