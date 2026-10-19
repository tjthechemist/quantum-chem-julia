# ==============================================================================
# QuantumChemMM.jl: Intrinsic Reaction Coordinate (IRC) Path Tracing
# Traces Minimum Energy Reaction Paths from Transition State to Reactant and Product
# ==============================================================================

module IRCMethod

using LinearAlgebra
using Printf
using ..Types
using ..MMBase
using ..MMEnergyGradient

export trace_irc, IRCResult, IRCPoint

struct IRCPoint
    arc_length::Float64 # in Å·sqrt(amu) or Å
    energy_kcal::Float64
    coords::Matrix{Float64}
end

struct IRCResult
    forward_path::Vector{IRCPoint}
    reverse_path::Vector{IRCPoint}
    ts_energy_kcal::Float64
    reactant_energy_kcal::Float64
    product_energy_kcal::Float64
    barrier_forward_kcal::Float64
    barrier_reverse_kcal::Float64
end

"""
    trace_irc(ts_mol::Molecule, ff_generator::Function;
              transition_vector::Vector{Float64}=Float64[],
              step_size::Float64=0.05, max_steps::Int=50,
              verbose::Bool=false)::IRCResult
Traces the Intrinsic Reaction Coordinate (IRC) forward and reverse from a transition state.
"""
function trace_irc(ts_mol::Molecule, ff_generator::Function;
                  transition_vector::Vector{Float64}=Float64[],
                  step_size::Float64=0.05, max_steps::Int=50,
                  verbose::Bool=false)::IRCResult
    nat = n_atoms(ts_mol)
    ndim = 3 * nat

    ff_ts = ff_generator(ts_mol)
    e_ts = total_energy(evaluate_mm(ff_ts, ts_mol; calc_grad=false)[1])
    ts_coords = vec(coordinates_matrix(ts_mol))

    # If transition vector is not provided, estimate it from Hessian mode
    t_vec = if isempty(transition_vector)
        # Compute numerical Hessian to get negative mode
        delta = 1e-3
        H = zeros(Float64, ndim, ndim)
        for at_j in 1:nat
            for coord_j in 1:3
                col = 3 * (at_j - 1) + coord_j
                ts_mol.atoms[at_j].coords[coord_j] += delta
                ff_p = ff_generator(ts_mol)
                _, gp = evaluate_mm(ff_p, ts_mol; calc_grad=true)

                ts_mol.atoms[at_j].coords[coord_j] -= 2.0 * delta
                ff_m = ff_generator(ts_mol)
                _, gm = evaluate_mm(ff_m, ts_mol; calc_grad=true)

                ts_mol.atoms[at_j].coords[coord_j] += delta
                H[:, col] = (vec(gp) .- vec(gm)) ./ (2.0 * delta)
            end
        end
        H .= 0.5 .* (H .+ H')
        eig = eigen(Symmetric(H))
        eig.vectors[:, 1]
    else
        transition_vector ./ norm(transition_vector)
    end

    # Helper function to trace in one direction (+1 for forward, -1 for reverse)
    function trace_direction(direction::Float64)
        path = IRCPoint[]
        curr_mol = deepcopy(ts_mol)
        curr_coords = ts_coords .+ (direction * 0.05) .* t_vec
        set_coordinates!(curr_mol, reshape(curr_coords, 3, nat))

        total_arc = 0.0

        for step in 1:max_steps
            ff = ff_generator(curr_mol)
            ec, gmat = evaluate_mm(ff, curr_mol; calc_grad=true)
            e_curr = total_energy(ec)
            g = vec(gmat)
            norm_g = norm(g)

            push!(path, IRCPoint(total_arc, e_curr, coordinates_matrix(curr_mol)))

            if norm_g < 0.05
                break # Reached local minimum (reactant or product)
            end

            # Step downhill along gradient: dx = -step_size * (g / ||g||)
            step_dir = -g ./ norm_g
            curr_coords .+= step_size .* step_dir
            total_arc += step_size
            set_coordinates!(curr_mol, reshape(curr_coords, 3, nat))
        end
        return path
    end

    verbose && println("Tracing IRC in forward direction...")
    forward = trace_direction(+1.0)
    verbose && println("Tracing IRC in reverse direction...")
    reverse = trace_direction(-1.0)

    e_prod = isempty(forward) ? e_ts : forward[end].energy_kcal
    e_react = isempty(reverse) ? e_ts : reverse[end].energy_kcal

    b_forward = e_ts - e_react
    b_reverse = e_ts - e_prod

    return IRCResult(forward, reverse, e_ts, e_react, e_prod, b_forward, b_reverse)
end

end # module IRCMethod

