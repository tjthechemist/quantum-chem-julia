# ==============================================================================
# QuantumChemMM.jl: Population Analysis (Mulliken & Löwdin)
# ==============================================================================

module PopulationAnalysis

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...SlaterOrbitals

export mulliken_analysis, lowdin_analysis, PopulationResult

struct PopulationResult
    atomic_charges::Vector{Float64}
    orbital_populations::Vector{Float64}
    overlap_populations::Dict{Tuple{Int, Int}, Float64} # Bond orders
    method::Symbol
end

"""
    mulliken_analysis(P::Matrix{Float64}, S::Matrix{Float64},
                      basis::Vector{OrbitalInfo}, mol::Molecule)::PopulationResult
Performs standard Mulliken population analysis.
"""
function mulliken_analysis(P::Matrix{Float64}, S::Matrix{Float64},
                           basis::Vector{OrbitalInfo}, mol::Molecule)::PopulationResult
    nb = length(basis)
    nat = n_atoms(mol)

    # Net population matrix PS
    PS = P * S
    orb_pop = [PS[mu, mu] for mu in 1:nb]

    atomic_charges = zeros(Float64, nat)
    for i in 1:nat
        z_val = Elements.valence_electrons(mol.atoms[i].symbol)
        gross_pop = sum(orb_pop[mu] for mu in 1:nb if basis[mu].atom_index == i)
        atomic_charges[i] = z_val - gross_pop
    end

    overlap_pops = Dict{Tuple{Int, Int}, Float64}()
    for a in 1:nat
        for b in (a+1):nat
            op = 0.0
            for mu in 1:nb
                basis[mu].atom_index == a || continue
                for nu in 1:nb
                    basis[nu].atom_index == b || continue
                    op += 2.0 * P[mu, nu] * S[mu, nu]
                end
            end
            overlap_pops[(a, b)] = op
        end
    end

    return PopulationResult(atomic_charges, orb_pop, overlap_pops, :mulliken)
end

"""
    lowdin_analysis(P::Matrix{Float64}, S::Matrix{Float64},
                    basis::Vector{OrbitalInfo}, mol::Molecule)::PopulationResult
Performs symmetric Löwdin population analysis using S^(1/2) * P * S^(1/2).
"""
function lowdin_analysis(P::Matrix{Float64}, S::Matrix{Float64},
                         basis::Vector{OrbitalInfo}, mol::Molecule)::PopulationResult
    nb = length(basis)
    nat = n_atoms(mol)

    # Compute S^(1/2)
    eig_s = eigen(Symmetric(S))
    s_half = eig_s.vectors * Diagonal(sqrt.(max.(eig_s.values, 1e-12))) * eig_s.vectors'

    # Lowdin density matrix P_low = S^(1/2) * P * S^(1/2)
    P_low = s_half * P * s_half
    orb_pop = [P_low[mu, mu] for mu in 1:nb]

    atomic_charges = zeros(Float64, nat)
    for i in 1:nat
        z_val = Elements.valence_electrons(mol.atoms[i].symbol)
        gross_pop = sum(orb_pop[mu] for mu in 1:nb if basis[mu].atom_index == i)
        atomic_charges[i] = z_val - gross_pop
    end

    return PopulationResult(atomic_charges, orb_pop, Dict{Tuple{Int,Int}, Float64}(), :lowdin)
end

end # module PopulationAnalysis

