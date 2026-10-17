# ==============================================================================
# QuantumChemMM.jl: Non-bonded Interactions (van der Waals & Electrostatics)
# ==============================================================================

module NonBonded

using LinearAlgebra
using ...MolMath

export LJParam, BuckinghamParam, CoulombParam,
       calc_lj_energy_and_grad, calc_coulomb_energy_and_grad,
       calc_buckingham_energy_and_grad,
       lorentz_berthelot, geometric_combination

# Electrostatic conversion constant: e^2 / (4 * π * ε0 * Å) in kcal/mol
const COULOMB_CONSTANT = 332.063711

struct LJParam
    sigma::Float64 # in Å
    epsilon::Float64 # in kcal/mol
end

struct BuckinghamParam
    A::Float64   # in kcal/mol
    B::Float64   # in Å⁻¹
    C::Float64   # in kcal/(mol·Å⁶)
end

function lorentz_berthelot(p1::LJParam, p2::LJParam)::LJParam
    sigma_ij = 0.5 * (p1.sigma + p2.sigma)
    eps_ij = sqrt(p1.epsilon * p2.epsilon)
    return LJParam(sigma_ij, eps_ij)
end

function geometric_combination(p1::LJParam, p2::LJParam)::LJParam
    sigma_ij = sqrt(p1.sigma * p2.sigma)
    eps_ij = sqrt(p1.epsilon * p2.epsilon)
    return LJParam(sigma_ij, eps_ij)
end

function calc_lj_energy_and_grad(param::LJParam, ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real};
                                 scale::Float64=1.0)
    r = MolMath.distance(ri, rj)
    if r < 1e-6
        return 0.0, zeros(3), zeros(3)
    end
    sr = param.sigma / r
    sr6 = sr^6
    sr12 = sr6^2

    energy = scale * 4.0 * param.epsilon * (sr12 - sr6)
    dE_dr = scale * (24.0 * param.epsilon / r) * (-2.0 * sr12 + sr6)

    gi, gj = MolMath.distance_gradient(ri, rj)
    return energy, dE_dr .* gi, dE_dr .* gj
end

function calc_coulomb_energy_and_grad(qi::Float64, qj::Float64,
                                      ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real};
                                      epsilon_r::Float64=1.0, scale::Float64=1.0)
    r = MolMath.distance(ri, rj)
    if r < 1e-6
        return 0.0, zeros(3), zeros(3)
    end
    pref = (COULOMB_CONSTANT * qi * qj) / epsilon_r
    energy = scale * pref / r
    dE_dr = -scale * pref / (r^2)

    gi, gj = MolMath.distance_gradient(ri, rj)
    return energy, dE_dr .* gi, dE_dr .* gj
end

function calc_buckingham_energy_and_grad(param::BuckinghamParam, ri::AbstractVector{<:Real},
                                        rj::AbstractVector{<:Real}; scale::Float64=1.0)
    r = MolMath.distance(ri, rj)
    if r < 1e-6
        return 0.0, zeros(3), zeros(3)
    end
    exp_term = param.A * exp(-param.B * r)
    disp_term = param.C / (r^6)

    energy = scale * (exp_term - disp_term)
    dE_dr = scale * (-param.B * exp_term + 6.0 * param.C / (r^7))

    gi, gj = MolMath.distance_gradient(ri, rj)
    return energy, dE_dr .* gi, dE_dr .* gj
end

end # module NonBonded

