# ==============================================================================
# QuantumChemMM.jl: Geometry Optimization (Steepest Descent, CG, L-BFGS)
# ==============================================================================

module OptimizationAlgorithms

using LinearAlgebra
using Printf
using ..Types
using ..MMBase
using ..MMEnergyGradient
using ..LineSearch

export OptimizationResult, optimize_geometry!

const OPTIMIZATION_METHODS = (:sd, :cg, :lbfgs)

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
    eval_fg = function ()
        ec, gmat = evaluate_mm(ff, mol; calc_grad=true)
        return total_energy(ec), vec(gmat)
    end
    eval_f = () -> energy(ff, mol)
    return _optimize_geometry!(mol, eval_fg, eval_f; method=method,
                               max_steps=max_steps, grad_tol=grad_tol,
                               verbose=verbose, energy_unit="kcal/mol")
end

"""
    optimize_geometry!(mol, energy_function, gradient_function; kwargs...)

Optimize a molecule using callbacks that accept the molecule and return an energy
or a 3×N gradient, respectively. This supports finite-difference gradients for
electronic-structure methods without duplicating the optimization algorithms.
"""
function optimize_geometry!(mol::Molecule, energy_function::Function,
                            gradient_function::Function;
                            method::Symbol=:lbfgs, max_steps::Int=500,
                            grad_tol::Float64=1e-4, verbose::Bool=false)::OptimizationResult
    eval_fg = function ()
        energy_value = Float64(energy_function(mol))
        gradient_matrix = gradient_function(mol)
        size(gradient_matrix) == (3, n_atoms(mol)) ||
            throw(DimensionMismatch("Geometry gradient must have size (3, n_atoms(mol))."))
        return energy_value, vec(Float64.(gradient_matrix))
    end
    eval_f = () -> Float64(energy_function(mol))
    return _optimize_geometry!(mol, eval_fg, eval_f; method=method,
                               max_steps=max_steps, grad_tol=grad_tol,
                               verbose=verbose, energy_unit="Hartree")
end

function _optimize_geometry!(mol::Molecule, eval_fg::Function, eval_f::Function;
                             method::Symbol, max_steps::Int, grad_tol::Float64,
                             verbose::Bool, energy_unit::String)::OptimizationResult
    method in OPTIMIZATION_METHODS ||
        throw(ArgumentError("Unsupported optimization method '$method'. Use :sd, :cg, or :lbfgs."))
    max_steps > 0 || throw(ArgumentError("max_steps must be positive."))
    grad_tol > 0.0 || throw(ArgumentError("grad_tol must be positive."))
    nat = n_atoms(mol)
    nat > 0 || throw(ArgumentError("Cannot optimize a molecule with no atoms."))
    n_dim = 3 * nat

    function pack_coords()::Vector{Float64}
        mat = coordinates_matrix(mol)
        return vec(mat)
    end

    function unpack_coords!(x::Vector{Float64})
        mat = reshape(x, 3, nat)
        set_coordinates!(mol, mat)
    end

    function evaluate_fg(x::Vector{Float64})
        unpack_coords!(x)
        return eval_fg()
    end

    function evaluate_f(x::Vector{Float64})
        unpack_coords!(x)
        return eval_f()
    end

    x = pack_coords()
    f_val, g = evaluate_fg(x)
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
            @printf("  Step %4d | Energy = %14.8f %s | Max Grad = %10.5g\n",
                    step, f_val, energy_unit, max_g)
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
        step_len = LineSearch.backtracking_line_search(evaluate_f, g, x, p; alpha_init=1.0, c1=1e-4)

        # Update position
        s = step_len .* p
        x_new = x .+ s
        f_new, g_new = evaluate_fg(x_new)
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

    max_g = maximum(abs.(g))
    rms_g = sqrt(sum(g.^2) / n_dim)
    converged |= max_g < grad_tol || rms_g < 0.5 * grad_tol
    unpack_coords!(x)
    return OptimizationResult(converged, iter, init_energy, f_val, max_g, rms_g)
end

end # module OptimizationAlgorithms
