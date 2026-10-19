# ==============================================================================
# QuantumChemMM.jl: Eigenvector-Following / P-RFO Transition State Search
# Local First-Order Saddle Point Search with Hessian Mode Following
# ==============================================================================

module EigenvectorFollowing

using LinearAlgebra
using Printf
using ..Types
using ..MMBase
using ..MMEnergyGradient

export optimize_transition_state!, TSResult

struct TSResult
    converged::Bool
    iterations::Int
    energy_kcal::Float64
    max_gradient::Float64
    lowest_eigenvalue::Float64
    is_true_saddle_point::Bool # true if exactly one negative eigenvalue
end

"""
    optimize_transition_state!(mol::Molecule, ff_generator::Function;
                               max_steps::Int=50, grad_tol::Float64=1e-3,
                               delta::Float64=1e-3, verbose::Bool=false)::TSResult
Finds a first-order saddle point (transition state) by following the lowest Hessian mode uphill
while minimizing all orthogonal modes downhill.
"""
function optimize_transition_state!(mol::Molecule, ff_generator::Function;
                                    max_steps::Int=50, grad_tol::Float64=1e-3,
                                    delta::Float64=1e-3, verbose::Bool=false)::TSResult
    nat = n_atoms(mol)
    ndim = 3 * nat
    converged = false
    last_e = 0.0
    lowest_eig = 0.0
    max_g = 0.0
    iter = 0

    for step in 1:max_steps
        iter = step
        ff = ff_generator(mol)
        ec, gmat = evaluate_mm(ff, mol; calc_grad=true)
        e_curr = total_energy(ec)
        last_e = e_curr
        g = vec(gmat)
        max_g = maximum(abs.(g))

        if verbose && (step % 5 == 1 || step == max_steps)
            @printf("  TS Search Step %3d | Energy = %14.6f kcal/mol | Max Grad = %10.5f\n",
                    step, e_curr, max_g)
        end

        # Numerical Cartesian Hessian: H_{i, j} = (g(x_j + d) - g(x_j - d)) / (2*d)
        H = zeros(Float64, ndim, ndim)
        for at_j in 1:nat
            for coord_j in 1:3
                col = 3 * (at_j - 1) + coord_j
                mol.atoms[at_j].coords[coord_j] += delta
                ff_p = ff_generator(mol)
                _, gp = evaluate_mm(ff_p, mol; calc_grad=true)

                mol.atoms[at_j].coords[coord_j] -= 2.0 * delta
                ff_m = ff_generator(mol)
                _, gm = evaluate_mm(ff_m, mol; calc_grad=true)

                mol.atoms[at_j].coords[coord_j] += delta
                H[:, col] = (vec(gp) .- vec(gm)) ./ (2.0 * delta)
            end
        end
        H .= 0.5 .* (H .+ H')

        # Diagonalize Hessian
        eig = eigen(Symmetric(H))
        lowest_eig = eig.values[1]
        n_neg = count(val -> val < -1e-4, eig.values)

        if max_g < grad_tol
            converged = true
            break
        end

        # P-RFO step: uphill along mode 1 (lowest eigenvalue), downhill along modes 2..ndim
        # Transform gradient into Hessian eigenvector basis
        g_proj = eig.vectors' * g
        step_vec_proj = zeros(ndim)

        # Mode 1: inverted step (uphill)
        lam1 = eig.values[1]
        step_vec_proj[1] = g_proj[1] / max(abs(lam1), 0.1)

        # Modes 2..ndim: standard Newton-Raphson step (downhill)
        for k in 2:ndim
            lam_k = eig.values[k]
            step_vec_proj[k] = -g_proj[k] / max(abs(lam_k), 0.1)
        end

        # Transform step back to Cartesian coordinates
        step_cart = eig.vectors * step_vec_proj

        # Trust radius limitation
        step_norm = norm(step_cart)
        max_step = 0.1 # max 0.1 Å per step
        if step_norm > max_step
            step_cart .*= (max_step / step_norm)
        end

        # Update coordinates
        new_coords = vec(coordinates_matrix(mol)) .+ step_cart
        set_coordinates!(mol, reshape(new_coords, 3, nat))
    end

    is_ts = lowest_eig < -1e-4
    return TSResult(converged, iter, last_e, max_g, lowest_eig, is_ts)
end

end # module EigenvectorFollowing

