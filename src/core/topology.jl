# ==============================================================================
# QuantumChemMM.jl: Molecular Topology, Graph Perception, and Interaction Lists
# ==============================================================================

module Topology

using ..Types
using ..Elements
using ..MolMath
using LinearAlgebra

export build_bonds!, perceive_topology!, topological_distance_matrix,
       get_nonbonded_pairs

"""
    build_bonds!(mol::Molecule; tolerance::Float64=0.45)
Detects covalent bonds based on interatomic distances and covalent radii:
r_ij <= (r_cov(i) + r_cov(j) + tolerance).
"""
function build_bonds!(mol::Molecule; tolerance::Float64=0.45)
    empty!(mol.bonds)
    nat = n_atoms(mol)
    for i in 1:nat
        for j in (i+1):nat
            at_i = mol.atoms[i]
            at_j = mol.atoms[j]
            r_cov_sum = Elements.covalent_radius(at_i.symbol) + Elements.covalent_radius(at_j.symbol)
            dist = MolMath.distance(at_i.coords, at_j.coords)
            if dist <= (r_cov_sum + tolerance) && dist >= 0.4
                # Estimate bond order from distance
                order = 1.0
                if at_i.symbol in (:C, :N, :O) && at_j.symbol in (:C, :N, :O)
                    if dist < (r_cov_sum - 0.18)
                        order = 3.0
                    elseif dist < (r_cov_sum - 0.08)
                        order = 2.0
                    end
                end
                push!(mol.bonds, Bond(i, j, order))
            end
        end
    end
    return mol
end

"""
    perceive_topology!(mol::Molecule)
Automatically perceives bonds (if not already defined), angles, proper dihedrals,
and improper torsions from the molecular graph.
"""
function perceive_topology!(mol::Molecule)
    if isempty(mol.bonds)
        build_bonds!(mol)
    end

    nat = n_atoms(mol)
    adj = [Int[] for _ in 1:nat]
    for b in mol.bonds
        push!(adj[b.i], b.j)
        push!(adj[b.j], b.i)
    end

    # 1. Angles (i - j - k) where j is vertex
    empty!(mol.angles)
    for j in 1:nat
        nbrs = adj[j]
        nn = length(nbrs)
        for idx1 in 1:nn
            for idx2 in (idx1+1):nn
                i = nbrs[idx1]
                k = nbrs[idx2]
                push!(mol.angles, Angle(min(i, k), j, max(i, k)))
            end
        end
    end

    # 2. Proper Dihedrals (i - j - k - l)
    empty!(mol.dihedrals)
    seen_dihedrals = Set{Tuple{Int, Int, Int, Int}}()
    for b in mol.bonds
        j = b.i
        k = b.j
        for i in adj[j]
            i == k && continue
            for l in adj[k]
                l == j && continue
                l == i && continue
                key1 = (i, j, k, l)
                key2 = (l, k, j, i)
                if !(key1 in seen_dihedrals || key2 in seen_dihedrals)
                    push!(seen_dihedrals, key1)
                    push!(mol.dihedrals, Dihedral(i, j, k, l))
                end
            end
        end
    end

    # 3. Improper Dihedrals (for trivalent planar or near-planar atoms)
    empty!(mol.impropers)
    for center in 1:nat
        nbrs = adj[center]
        if length(nbrs) == 3
            push!(mol.impropers, Improper(center, nbrs[1], nbrs[2], nbrs[3]))
        end
    end

    return mol
end

"""
    topological_distance_matrix(mol::Molecule)
Computes shortest path graph distance between all atom pairs using Floyd-Warshall.
"""
function topological_distance_matrix(mol::Molecule)::Matrix{Int}
    nat = n_atoms(mol)
    dist = fill(9999, nat, nat)
    for i in 1:nat
        dist[i, i] = 0
    end
    for b in mol.bonds
        dist[b.i, b.j] = 1
        dist[b.j, b.i] = 1
    end
    for k in 1:nat
        for i in 1:nat
            for j in 1:nat
                if dist[i, k] + dist[k, j] < dist[i, j]
                    dist[i, j] = dist[i, k] + dist[k, j]
                end
            end
        end
    end
    return dist
end

"""
    get_nonbonded_pairs(mol::Molecule)
Returns (pairs_14, pairs_non14) where pairs_14 have 1-4 scaling applied,
and pairs_non14 are separated by 4 or more bonds. 1-2 and 1-3 pairs are excluded.
"""
function get_nonbonded_pairs(mol::Molecule)
    top_dist = topological_distance_matrix(mol)
    nat = n_atoms(mol)
    pairs_14 = Tuple{Int, Int}[]
    pairs_non14 = Tuple{Int, Int}[]

    for i in 1:nat
        for j in (i+1):nat
            d = top_dist[i, j]
            if d == 3
                push!(pairs_14, (i, j))
            elseif d >= 4 || d >= 9999 # unbonded disconnected
                push!(pairs_non14, (i, j))
            end
        end
    end
    return pairs_14, pairs_non14
end

end # module Topology

