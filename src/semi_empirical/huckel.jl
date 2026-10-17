# ==============================================================================
# QuantumChemMM.jl: Simple Hückel Molecular Orbital Theory (HMO)
# ==============================================================================

module HuckelTheory

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...Topology
using ..SemiEmpiricalBase

export HuckelModel, solve_huckel, HuckelResult

struct HuckelHeteroParam
    h::Float64   # alpha_X = alpha_0 + h * beta_0
    k::Float64   # beta_XY = k * beta_0
    n_pi::Int    # number of pi electrons contributed
end

const HUCKEL_PARAMS = Dict{Symbol, HuckelHeteroParam}(
    :C => HuckelHeteroParam(0.0, 1.0, 1),
    :N => HuckelHeteroParam(0.5, 1.0, 1), # pyridine-type
    :O => HuckelHeteroParam(1.0, 1.0, 1), # carbonyl-type
    :F => HuckelHeteroParam(3.0, 0.7, 2),
    :Cl=> HuckelHeteroParam(2.0, 0.4, 2)
)

struct HuckelResult
    pi_atom_indices::Vector{Int}
    orbital_energies_m::Vector{Float64} # in units of beta (relative to alpha_0)
    orbital_energies_ev::Vector{Float64}# in eV
    coefficients::Matrix{Float64}
    occupations::Vector{Float64}
    total_pi_energy_beta::Float64
    total_pi_energy_ev::Float64
    delocalization_energy_beta::Float64
    delocalization_energy_ev::Float64
    pi_charges::Vector{Float64}
    bond_orders::Dict{Tuple{Int, Int}, Float64}
end

struct HuckelModel <: AbstractSemiEmpiricalMethod
    alpha0_ev::Float64  # default: -11.26 eV
    beta0_ev::Float64   # default: -2.70 eV
end

HuckelModel(; alpha0_ev=-11.26, beta0_ev=-2.70) = HuckelModel(alpha0_ev, beta0_ev)

"""
    solve_huckel(mol::Molecule, model::HuckelModel=HuckelModel())::HuckelResult
Performs Simple Hückel Theory calculation for conjugated systems.
"""
function solve_huckel(mol::Molecule, model::HuckelModel=HuckelModel())::HuckelResult
    perceive_topology!(mol)

    # 1. Identify conjugated pi atoms (sp2 / aromatic carbons and heteroatoms)
    pi_atoms = Int[]
    for at in mol.atoms
        if at.symbol in (:C, :N, :O, :F, :Cl)
            # Atoms with at least one double bond or in small rings
            push!(pi_atoms, at.index)
        end
    end

    n_pi = length(pi_atoms)
    if n_pi == 0
        error("No conjugated π atoms found in molecule $(mol.name).")
    end

    # Mapping from original atom index to pi system index
    pi_map = Dict{Int, Int}(pi_atoms[i] => i for i in 1:n_pi)

    # 2. Build Hückel Hamiltonian matrix H in units of beta:
    # H_ii = h_i
    # H_ij = k_ij if bonded, 0 otherwise
    H = zeros(Float64, n_pi, n_pi)
    total_pi_electrons = 0

    for i in 1:n_pi
        orig_i = pi_atoms[i]
        sym = mol.atoms[orig_i].symbol
        param = get(HUCKEL_PARAMS, sym, HuckelHeteroParam(0.0, 1.0, 1))
        H[i, i] = param.h
        total_pi_electrons += param.n_pi
    end

    for b in mol.bonds
        if haskey(pi_map, b.i) && haskey(pi_map, b.j)
            pi_i = pi_map[b.i]
            pi_j = pi_map[b.j]
            s1 = mol.atoms[b.i].symbol
            s2 = mol.atoms[b.j].symbol
            k1 = get(HUCKEL_PARAMS, s1, HuckelHeteroParam(0.0, 1.0, 1)).k
            k2 = get(HUCKEL_PARAMS, s2, HuckelHeteroParam(0.0, 1.0, 1)).k
            k_ij = sqrt(k1 * k2)
            H[pi_i, pi_j] = k_ij
            H[pi_j, pi_i] = k_ij
        end
    end

    # 3. Diagonalize H (in Hückel convention, positive beta is bonding,
    # so we sort from most bonding to least bonding: largest positive eigenvalue is lowest energy)
    eig = eigen(Symmetric(H))
    perm = sortperm(eig.values, rev=true) # descending order of m (most bonding first)
    m_values = eig.values[perm]           # energies = alpha + m * beta
    C = eig.vectors[:, perm]

    # Convert to physical eV: E_i = alpha0 + m_i * beta0
    # Since beta0 is negative (-2.7 eV), larger m corresponds to lower (more negative) physical energy
    e_ev = model.alpha0_ev .+ m_values .* model.beta0_ev

    # 4. Occupations (Aufbau principle: 2 electrons per orbital)
    occ = zeros(Float64, n_pi)
    electrons_left = total_pi_electrons
    for i in 1:n_pi
        if electrons_left >= 2
            occ[i] = 2.0
            electrons_left -= 2
        elseif electrons_left == 1
            occ[i] = 1.0
            electrons_left -= 1
        else
            occ[i] = 0.0
        end
    end

    # 5. Total Pi Energy
    total_m = sum(occ .* m_values)
    total_ev = sum(occ .* e_ev)

    # Delocalization energy: compared to isolated localized double bonds (each has 2 * 1.0 beta)
    n_double_bonds = div(total_pi_electrons, 2)
    deloc_m = total_m - (2.0 * n_double_bonds)
    deloc_ev = deloc_m * model.beta0_ev

    # 6. Pi Charges & Bond Orders
    pi_charges = zeros(Float64, n_pi)
    for i in 1:n_pi
        density_i = sum(occ[k] * (C[i, k]^2) for k in 1:n_pi)
        orig_i = pi_atoms[i]
        sym = mol.atoms[orig_i].symbol
        n_core = get(HUCKEL_PARAMS, sym, HuckelHeteroParam(0.0, 1.0, 1)).n_pi
        pi_charges[i] = n_core - density_i
    end

    bond_orders = Dict{Tuple{Int, Int}, Float64}()
    for b in mol.bonds
        if haskey(pi_map, b.i) && haskey(pi_map, b.j)
            pi_i = pi_map[b.i]
            pi_j = pi_map[b.j]
            p_ij = sum(occ[k] * C[pi_i, k] * C[pi_j, k] for k in 1:n_pi)
            bond_orders[(b.i, b.j)] = p_ij
        end
    end

    return HuckelResult(pi_atoms, m_values, e_ev, C, occ,
                        total_m, total_ev, deloc_m, deloc_ev,
                        pi_charges, bond_orders)
end

function print_huckel_summary(res::HuckelResult)
    println("==================================================")
    println("         Hückel Molecular Orbital (HMO)           ")
    println("==================================================")
    @printf("  Total Pi Energy:         %12.4f β (%10.3f eV)\n", res.total_pi_energy_beta, res.total_pi_energy_ev)
    @printf("  Delocalization Energy:   %12.4f β (%10.3f eV)\n", res.delocalization_energy_beta, res.delocalization_energy_ev)
    println("--------------------------------------------------")
    println("  Orbital Energies & Occupancies:")
    for (i, (m, ev, oc)) in enumerate(zip(res.orbital_energies_m, res.orbital_energies_ev, res.occupations))
        @printf("    MO #%2d:  m = %7.4f β (%9.3f eV)  Occ: %.1f\n", i, m, ev, oc)
    end
    println("--------------------------------------------------")
    println("  Pi Bond Orders (Coulson):")
    for ((i, j), bo) in res.bond_orders
        @printf("    Bond (%2d - %2d):  p = %6.4f  (Total order ≈ %6.4f)\n", i, j, bo, 1.0 + bo)
    end
    println("==================================================")
end

end # module HuckelTheory

