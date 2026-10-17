# ==============================================================================
# QuantumChemMM.jl: Conformational Search & Boltzmann Population Analysis
# Systematic and Monte Carlo Torsional Sampling with Force Field Minimization
# ==============================================================================

module ConformationalSearch

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...MolMath
using ...Topology
using ...MMBase
using ...MMEnergyGradient
using ...OptimizationAlgorithms

export Conformer, ConformerEnsemble, search_conformers,
       boltzmann_populations, print_conformer_summary

struct Conformer
    index::Int
    energy_kcal::Float64
    relative_energy_kcal::Float64
    boltzmann_population::Float64
    molecule::Molecule
end

struct ConformerEnsemble
    conformers::Vector{Conformer}
    temperature_k::Float64
    lowest_energy_kcal::Float64
end

"""
    find_rotatable_bonds(mol::Molecule)::Vector{Tuple{Int, Int, Int, Int}}
Detects flexible rotatable single bonds (excluding terminal bonds like C-H or C-CH3).
Returns list of (i, j, k, l) dihedrals controlling each rotatable bond j-k.
"""
function find_rotatable_bonds(mol::Molecule)::Vector{Tuple{Int, Int, Int, Int}}
    perceive_topology!(mol)
    nat = n_atoms(mol)

    # Adjacency list
    adj = [Int[] for _ in 1:nat]
    for b in mol.bonds
        push!(adj[b.i], b.j)
        push!(adj[b.j], b.i)
    end

    rotatables = Tuple{Int, Int, Int, Int}[]

    for b in mol.bonds
        # Must be single bond
        b.order > 1.2 && continue
        j = b.i
        k = b.j

        # Skip terminal atoms (coordination <= 1)
        length(adj[j]) <= 1 && continue
        length(adj[k]) <= 1 && continue

        # Find non-hydrogen neighbors for proper dihedral definition
        nbrs_j = filter(x -> x != k, adj[j])
        nbrs_k = filter(x -> x != j, adj[k])
        isempty(nbrs_j) && continue
        isempty(nbrs_k) && continue

        i = nbrs_j[1]
        l = nbrs_k[1]
        push!(rotatables, (i, j, k, l))
    end

    return rotatables
end

"""
    rotate_dihedral!(mol::Molecule, i::Int, j::Int, k::Int, l::Int, target_deg::Float64)
Rotates all atoms on the 'l' side of the j-k bond to achieve the target dihedral angle.
"""
function rotate_dihedral!(mol::Molecule, i::Int, j::Int, k::Int, l::Int, target_deg::Float64)
    curr_rad = MolMath.dihedral_angle(mol.atoms[i].coords, mol.atoms[j].coords,
                                      mol.atoms[k].coords, mol.atoms[l].coords)
    curr_deg = curr_rad * 180.0 / π
    delta_deg = target_deg - curr_deg
    delta_rad = delta_deg * π / 180.0

    # Rotation axis: unit vector from j to k
    axis = mol.atoms[k].coords .- mol.atoms[j].coords
    axis_hat = axis ./ max(norm(axis), 1e-12)

    # Find all atoms connected to k (excluding j) via BFS
    moving_atoms = Int[]
    visited = fill(false, n_atoms(mol))
    visited[j] = true
    queue = [k]
    visited[k] = true

    # Adjacency
    adj = [Int[] for _ in 1:n_atoms(mol)]
    for b in mol.bonds
        push!(adj[b.i], b.j)
        push!(adj[b.j], b.i)
    end

    while !isempty(queue)
        curr = popfirst!(queue)
        push!(moving_atoms, curr)
        for nbr in adj[curr]
            if !visited[nbr]
                visited[nbr] = true
                push!(queue, nbr)
            end
        end
    end

    # Rodrigues' rotation formula: v_rot = v*cos(θ) + (u × v)*sin(θ) + u*(u · v)*(1 - cos(θ))
    cos_t = cos(delta_rad)
    sin_t = sin(delta_rad)
    origin = mol.atoms[j].coords

    for idx in moving_atoms
        v = mol.atoms[idx].coords .- origin
        v_rot = v .* cos_t .+ cross(axis_hat, v) .* sin_t .+ axis_hat .* (dot(axis_hat, v) * (1.0 - cos_t))
        mol.atoms[idx].coords .= origin .+ v_rot
    end

    return mol
end

"""
    search_conformers(mol::Molecule, ff_generator::Function;
                      n_conformations::Int=12, temp_k::Float64=298.15,
                      rms_threshold::Float64=0.25)::ConformerEnsemble
Performs systematic/stochastic torsional sampling and energy minimization to find
all low-energy conformers and their equilibrium Boltzmann population percentages.
"""
function search_conformers(mol::Molecule, ff_generator::Function;
                          n_conformations::Int=12, temp_k::Float64=298.15,
                          rms_threshold::Float64=0.25)::ConformerEnsemble
    rot_bonds = find_rotatable_bonds(mol)
    candidates = Molecule[]

    # Include original structure
    push!(candidates, deepcopy(mol))

    if !isempty(rot_bonds)
        angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        for _ in 1:n_conformations
            cand = deepcopy(mol)
            for (i, j, k, l) in rot_bonds
                rand_angle = rand(angles)
                rotate_dihedral!(cand, i, j, k, l, rand_angle)
            end
            push!(candidates, cand)
        end
    end

    # Minimize each candidate and collect unique conformers
    minimized_list = Tuple{Float64, Molecule}[]

    for cand in candidates
        ff = ff_generator(cand)
        optimize_geometry!(cand, ff; method=:lbfgs, max_steps=100, grad_tol=1e-3)
        e = total_energy(evaluate_mm(ff, cand; calc_grad=false)[1])
        push!(minimized_list, (e, cand))
    end

    # Sort by energy
    sort!(minimized_list, by=x -> x[1])
    lowest_e = isempty(minimized_list) ? 0.0 : minimized_list[1][1]

    # Cluster unique conformers by RMS coordinate deviation
    unique_conformers = Tuple{Float64, Molecule}[]
    for (e, cand) in minimized_list
        is_dup = false
        c_mat = coordinates_matrix(cand)
        for (_, u_cand) in unique_conformers
            u_mat = coordinates_matrix(u_cand)
            rms = sqrt(sum((c_mat .- u_mat).^2) / n_atoms(cand))
            if rms < rms_threshold
                is_dup = true
                break
            end
        end
        if !is_dup
            push!(unique_conformers, (e, cand))
        end
    end

    # Compute Boltzmann populations: p_i = exp(-dE / (kB * T)) / Z
    KB = 0.0019872041 # kcal/(mol·K)
    rel_energies = [e - lowest_e for (e, _) in unique_conformers]
    weights = [exp(-dE / (KB * temp_k)) for dE in rel_energies]
    z_part = sum(weights)
    populations = weights ./ z_part

    confs = Conformer[]
    for (idx, ((e, cand), dE, pop)) in enumerate(zip(unique_conformers, rel_energies, populations))
        push!(confs, Conformer(idx, e, dE, pop, cand))
    end

    return ConformerEnsemble(confs, temp_k, lowest_e)
end

function print_conformer_summary(ens::ConformerEnsemble)
    println("==================================================")
    println("            Conformer Ensemble Analysis           ")
    println("==================================================")
    @printf("  Temperature:          %8.2f K\n", ens.temperature_k)
    @printf("  Lowest Energy:        %14.4f kcal/mol\n", ens.lowest_energy_kcal)
    @printf("  Unique Conformers:    %d\n", length(ens.conformers))
    println("--------------------------------------------------")
    println("  Conf #   Energy (kcal/mol)   ΔE (kcal/mol)   Boltzmann %")
    for c in ens.conformers
        @printf("    %2d       %14.4f        %8.4f        %7.2f %%\n",
                c.index, c.energy_kcal, c.relative_energy_kcal, c.boltzmann_population * 100.0)
    end
    println("==================================================")
end

end # module ConformationalSearch

