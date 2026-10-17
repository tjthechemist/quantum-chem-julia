# ==============================================================================
# QuantumChemMM.jl: Geometry Optimization (Steepest Descent, CG, L-BFGS)
# ==============================================================================

module OptimizationAlgorithms

using LinearAlgebra
using Printf
using ...Types
using ...MMBase
using ...MMEnergyGradient
using ..LineSearch

export OptimizationResult, optimize_geometry!

struct OptimizationResult
    converged::Bool
    iterations::Int
    initial_energy::Float64
    final_energy::Float64
    max_gradient::Float64
    rms_gradient::Float64
end

"""
    optimize_geometry!(mol::Molecule, ff::AbstractForceField;
                       method::Symbol=:lbfgs, max_steps::Int=500,
                       grad_tol::Float64=1e-3, verbose::Bool=false)::OptimizationResult
Performs energy minimization of the molecular structure using Steepest Descent (:sd),
Conjugate Gradient (:cg), or L-BFGS (:lbfgs).
"""
function optimize_geometry!(mol::Molecule, ff::AbstractForceField;
                            method::Symbol=:lbfgs, max_steps::Int=500,
                            grad_tol::Float64=1e-3, verbose::Bool=false)::OptimizationResult
    nat = n_atoms(mol)
    n_dim = 3 * nat

    # Helper to pack and unpack coordinates into a 1D vector
    function pack_coords()::Vector{Float64}
        mat = coordinates_matrix(mol)
        return vec(mat)
    end

    function unpack_coords!(x::Vector{Float64})
        mat = reshape(x, 3, nat)
        set_coordinates!(mol, mat)
    end

    # Objective function and gradient wrapper
    function eval_fg(x::Vector{Float64})
        unpack_coords!(x)
        ec, gmat = evaluate_mm(ff, mol; calc_grad=true)
        return total_energy(ec), vec(gmat)
    end

    function eval_f(x::Vector{Float64})
        unpack_coords!(x)
        return energy(ff, mol)
    end

    x = pack_coords()
    f_val, g = eval_fg(x)
    init_energy = f_val

    # Memory vectors for L-BFGS (m = 6 history)
    m_hist = 6
    s_list = Vector{Float64}[]
    y_list = Vector{Float64}[]
    rho_list = Float64[]

    # Direction vector for CG
    p_cg = -g
    last_g = copy(g)

    converged = false
    max_g = 0.0
    rms_g = 0.0
    iter = 0

    for step in 1:max_steps
        iter = step

        # Gradient norms
        max_g = maximum(abs.(g))
        rms_g = sqrt(sum(g.^2) / n_dim)

        if verbose && (step % 20 == 1 || step == max_steps)
            @printf("  Step %4d | Energy = %14.6f kcal/mol | Max Grad = %10.5f\n", step, f_val, max_g)
        end

        if max_g < grad_tol || rms_g < 0.5 * grad_tol
            converged = true
            break
        end

        # Determine search direction p
        p = zeros(n_dim)

        if method == :sd
            # Steepest Descent: p = -g
            p = -g
        elseif method == :cg
            # Polak-Ribiere Conjugate Gradient
            if step == 1
                p = -g
            else
                beta_pr = dot(g, g .- last_g) / max(dot(last_g, last_g), 1e-12)
                beta_pr = max(0.0, beta_pr) # Powell restart
                p = -g .+ beta_pr .* p_cg
            end
            p_cg = copy(p)
            last_g = copy(g)
        elseif method == :lbfgs
            # Two-loop recursion L-BFGS
            q = copy(g)
            k = length(s_list)
            alphas = zeros(k)

            for i in k:-1:1
                alphas[i] = rho_list[i] * dot(s_list[i], q)
                q .-= alphas[i] .* y_list[i]
            end

            # Initial Hessian approximation scaling
            gamma_k = (k > 0) ? dot(s_list[end], y_list[end]) / max(dot(y_list[end], y_list[end]), 1e-12) : 0.01
            r = gamma_k .* q

            for i in 1:k
                beta = rho_list[i] * dot(y_list[i], r)
                r .+= s_list[i] .* (alphas[i] - beta)
            end
            p = -r
        end

        # Ensure search direction is descent
        if dot(p, g) > 0.0
            p = -g
        end

        # Line search for step length
        step_len = LineSearch.backtracking_line_search(eval_f, g, x, p; alpha_init=1.0, c1=1e-4)

        # Update position
        s = step_len .* p
        x_new = x .+ s
        f_new, g_new = eval_fg(x_new)
        y = g_new .- g

        # Store history for L-BFGS
        if method == :lbfgs
            sy = dot(s, y)
            if sy > 1e-12
                push!(s_list, s)
                push!(y_list, y)
                push!(rho_list, 1.0 / sy)
                if length(s_list) > m_hist
                    popfirst!(s_list)
                    popfirst!(y_list)
                    popfirst!(rho_list)
                end
            end
        end

        x = x_new
        f_val = f_new
        g = g_new
    end

    unpack_coords!(x)
    return OptimizationResult(converged, iter, init_energy, f_val, max_g, rms_g)
end

end # module OptimizationAlgorithms

