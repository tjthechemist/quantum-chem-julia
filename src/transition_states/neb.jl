# ==============================================================================
# QuantumChemMM.jl: Nudged Elastic Band (NEB) and Climbing Image NEB (CI-NEB)
# Finds Reaction Paths and Transition State Geometries between Reactant and Product
# ==============================================================================

module NEBMethod

using LinearAlgebra
using Printf
using ..Types
using ..MMBase
using ..MMEnergyGradient

export NEBPath, run_neb, NEBResult

struct NEBPath
    images::Vector{Molecule} # array of molecular replicas along reaction path
    n_images::Int
    spring_k::Float64        # spring constant in kcal/(mol·Å²)
    climbing_image::Bool     # CI-NEB enabled
end

function NEBPath(reactant::Molecule, product::Molecule;
                 n_intermediate::Int=7, spring_k::Float64=20.0,
                 climbing_image::Bool=true)
    nat = n_atoms(reactant)
    @assert nat == n_atoms(product) "Reactant and Product must have same number of atoms."

    images = Molecule[]
    push!(images, deepcopy(reactant))

    # Linear interpolation of coordinates between reactant and product
    r_coords = coordinates_matrix(reactant)
    p_coords = coordinates_matrix(product)

    for step in 1:n_intermediate
        frac = step / (n_intermediate + 1)
        inter_mol = deepcopy(reactant)
        inter_coords = (1.0 - frac) .* r_coords .+ frac .* p_coords
        set_coordinates!(inter_mol, inter_coords)
        push!(images, inter_mol)
    end

    push!(images, deepcopy(product))
    return NEBPath(images, length(images), spring_k, climbing_image)
end

struct NEBResult
    converged::Bool
    iterations::Int
    energies_kcal::Vector{Float64}
    transition_state_image::Molecule
    activation_energy_kcal::Float64
    reaction_energy_kcal::Float64
    ts_index::Int
end

"""
    run_neb(neb::NEBPath, ff_generator::Function;
            max_steps::Int=200, force_tol::Float64=0.05,
            step_size::Float64=0.02, verbose::Bool=false)::NEBResult
Executes Nudged Elastic Band (or Climbing Image NEB) optimization.
ff_generator: mol -> AbstractForceField.
"""
function run_neb(neb::NEBPath, ff_generator::Function;
                 max_steps::Int=200, force_tol::Float64=0.05,
                 step_size::Float64=0.02, verbose::Bool=false)::NEBResult
    n_img = neb.n_images
    nat = n_atoms(neb.images[1])
    n_dim = 3 * nat

    energies = zeros(Float64, n_img)
    grad_list = [zeros(Float64, 3, nat) for _ in 1:n_img]

    converged = false
    iter = 0

    for step in 1:max_steps
        iter = step

        # 1. Evaluate energy and true potential forces on all images
        for k in 1:n_img
            ff = ff_generator(neb.images[k])
            ec, g = evaluate_mm(ff, neb.images[k]; calc_grad=true)
            energies[k] = total_energy(ec)
            grad_list[k] .= g
        end

        # Find highest energy image (candidate for climbing image)
        max_e, ts_cand = findmax(energies[2:n_img-1])
        ts_idx = ts_cand + 1 # offset by 1 because reactant is index 1

        max_neb_force = 0.0

        # 2. Compute tangents and NEB nudged forces on interior images
        for k in 2:(n_img-1)
            r_prev = vec(coordinates_matrix(neb.images[k-1]))
            r_curr = vec(coordinates_matrix(neb.images[k]))
            r_next = vec(coordinates_matrix(neb.images[k+1]))

            # Henkelman-Jónsson improved tangent
            v_prev = r_curr .- r_prev
            v_next = r_next .- r_curr

            e_prev = energies[k-1]
            e_curr = energies[k]
            e_next = energies[k+1]

            # Tangent vector tau
            tau = if e_next > e_curr && e_curr > e_prev
                v_next
            elseif e_next < e_curr && e_curr < e_prev
                v_prev
            else
                dE_max = max(abs(e_next - e_curr), abs(e_prev - e_curr))
                dE_min = min(abs(e_next - e_curr), abs(e_prev - e_curr))
                if e_next > e_prev
                    v_next .* dE_max .+ v_prev .* dE_min
                else
                    v_next .* dE_min .+ v_prev .* dE_max
                end
            end

            norm_tau = norm(tau)
            tau_hat = norm_tau > 1e-12 ? (tau ./ norm_tau) : zeros(n_dim)

            # Potential force F_pot = -grad
            f_pot = -vec(grad_list[k])

            # Check if this is the climbing image
            f_neb = zeros(n_dim)
            if neb.climbing_image && k == ts_idx && step > 10
                # Invert potential force along reaction path: F_climb = F_pot - 2*(F_pot · tau_hat)*tau_hat
                f_neb .= f_pot .- 2.0 * dot(f_pot, tau_hat) .* tau_hat
            else
                # Perpendicular potential force: F_pot^perp = F_pot - (F_pot · tau_hat)*tau_hat
                f_perp = f_pot .- dot(f_pot, tau_hat) .* tau_hat

                # Parallel spring force: F_spring^parallel = k * (|r_next - r_curr| - |r_curr - r_prev|) * tau_hat
                dist_next = norm(v_next)
                dist_prev = norm(v_prev)
                f_spring_par = (neb.spring_k * (dist_next - dist_prev)) .* tau_hat

                f_neb .= f_perp .+ f_spring_par
            end

            f_norm = norm(f_neb) / sqrt(nat)
            max_neb_force = max(max_neb_force, f_norm)

            # Update coordinates of image k (Steepest descent step)
            new_coords = r_curr .+ step_size .* f_neb
            set_coordinates!(neb.images[k], reshape(new_coords, 3, nat))
        end

        if verbose && (step % 20 == 1 || step == max_steps)
            act_e = energies[ts_idx] - energies[1]
            @printf("  NEB Step %4d | Barrier: %10.3f kcal/mol | Max Force: %8.4f kcal/(mol·Å)\n",
                    step, act_e, max_neb_force)
        end

        if max_neb_force < force_tol
            converged = true
            break
        end
    end

    # Final energy evaluation
    for k in 1:n_img
        ff = ff_generator(neb.images[k])
        energies[k] = total_energy(evaluate_mm(ff, neb.images[k]; calc_grad=false)[1])
    end

    max_e, ts_cand = findmax(energies[2:n_img-1])
    ts_idx = ts_cand + 1
    activation_e = energies[ts_idx] - energies[1]
    reaction_e = energies[end] - energies[1]

    return NEBResult(converged, iter, energies, deepcopy(neb.images[ts_idx]),
                     activation_e, reaction_e, ts_idx)
end

end # module NEBMethod

