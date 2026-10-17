# ==============================================================================
# QuantumChemMM.jl: Backtracking Line Search with Armijo Conditions
# ==============================================================================

module LineSearch

using LinearAlgebra

export backtracking_line_search

"""
    backtracking_line_search(f, grad_f, x, p; alpha_init=1.0, c1=1e-4, rho=0.5, max_iter=20)
Finds step size alpha satisfying the Armijo sufficient decrease condition:
f(x + alpha * p) <= f(x) + c1 * alpha * dot(grad_f, p).
"""
function backtracking_line_search(f, grad_f_val::AbstractVector{<:Real},
                                  x::AbstractVector{<:Real},
                                  p::AbstractVector{<:Real};
                                  alpha_init::Float64=1.0, c1::Float64=1e-4,
                                  rho::Float64=0.5, max_iter::Int=20)::Float64
    alpha = alpha_init
    f0 = f(x)
    directional_deriv = dot(grad_f_val, p)

    # If search direction is not a descent direction, flip or return small step
    if directional_deriv > 0.0
        directional_deriv = -directional_deriv
        p = -p
    end

    for _ in 1:max_iter
        x_trial = x .+ alpha .* p
        f_trial = f(x_trial)
        if f_trial <= f0 + c1 * alpha * directional_deriv
            return alpha
        end
        alpha *= rho
    end

    return alpha
end

end # module LineSearch

