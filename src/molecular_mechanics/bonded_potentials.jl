# ==============================================================================
# QuantumChemMM.jl: Bond and Angle Potentials (Class I and Class II)
# ==============================================================================

module BondedPotentials

using LinearAlgebra
using ..MolMath

export HarmonicBond, MorseBond, Class2Bond,
       HarmonicAngle, FourierAngle, UreyBradleyAngle, Class2Angle,
       calc_bond_energy_and_grad, calc_angle_energy_and_grad

# --- Bond Types ---

struct HarmonicBond
    i::Int
    j::Int
    r0::Float64   # equilibrium distance in Å
    kb::Float64   # force constant in kcal/(mol·Å²)
end

struct MorseBond
    i::Int
    j::Int
    r0::Float64
    De::Float64   # well depth in kcal/mol
    alpha::Float64 # in Å⁻¹
end

struct Class2Bond
    i::Int
    j::Int
    r0::Float64
    k2::Float64   # quadratic constant
    k3::Float64   # cubic constant (-2.55 * k2 in MM3)
    k4::Float64   # quartic constant
end

# --- Angle Types ---

struct HarmonicAngle
    i::Int
    j::Int        # vertex
    k::Int
    theta0::Float64 # equilibrium angle in radians
    ktheta::Float64 # in kcal/(mol·rad²)
end

struct FourierAngle
    i::Int
    j::Int        # vertex
    k::Int
    theta0::Float64
    ktheta::Float64
    c0::Float64
    c1::Float64
    c2::Float64
end

struct UreyBradleyAngle
    i::Int
    j::Int
    k::Int
    theta0::Float64
    ktheta::Float64
    rub0::Float64   # 1-3 equilibrium distance
    kub::Float64    # 1-3 force constant
end

struct Class2Angle
    i::Int
    j::Int
    k::Int
    theta0::Float64
    k2::Float64
    k3::Float64
    k4::Float64
end

# --- Energy & Gradient Evaluations ---

function calc_bond_energy_and_grad(param::HarmonicBond, ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real})
    r = MolMath.distance(ri, rj)
    dr = r - param.r0
    energy = 0.5 * param.kb * dr^2
    dE_dr = param.kb * dr
    gi, gj = MolMath.distance_gradient(ri, rj)
    return energy, dE_dr .* gi, dE_dr .* gj
end

function calc_bond_energy_and_grad(param::MorseBond, ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real})
    r = MolMath.distance(ri, rj)
    dr = r - param.r0
    exp_term = exp(-param.alpha * dr)
    energy = param.De * (1.0 - exp_term)^2
    dE_dr = 2.0 * param.De * param.alpha * (1.0 - exp_term) * exp_term
    gi, gj = MolMath.distance_gradient(ri, rj)
    return energy, dE_dr .* gi, dE_dr .* gj
end

function calc_bond_energy_and_grad(param::Class2Bond, ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real})
    r = MolMath.distance(ri, rj)
    dr = r - param.r0
    energy = param.k2 * dr^2 + param.k3 * dr^3 + param.k4 * dr^4
    dE_dr = 2.0 * param.k2 * dr + 3.0 * param.k3 * dr^2 + 4.0 * param.k4 * dr^3
    gi, gj = MolMath.distance_gradient(ri, rj)
    return energy, dE_dr .* gi, dE_dr .* gj
end

function calc_angle_energy_and_grad(param::HarmonicAngle, ri::AbstractVector{<:Real},
                                    rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})
    theta = MolMath.angle_between(ri, rj, rk)
    dtheta = theta - param.theta0
    energy = 0.5 * param.ktheta * dtheta^2
    dE_dtheta = param.ktheta * dtheta
    gi, gj, gk = MolMath.angle_gradient(ri, rj, rk)
    return energy, dE_dtheta .* gi, dE_dtheta .* gj, dE_dtheta .* gk
end

function calc_angle_energy_and_grad(param::FourierAngle, ri::AbstractVector{<:Real},
                                    rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})
    theta = MolMath.angle_between(ri, rj, rk)
    cos_t = cos(theta)
    cos_2t = cos(2.0 * theta)
    energy = param.ktheta * (param.c0 + param.c1 * cos_t + param.c2 * cos_2t)
    dE_dtheta = param.ktheta * (-param.c1 * sin(theta) - 2.0 * param.c2 * sin(2.0 * theta))
    gi, gj, gk = MolMath.angle_gradient(ri, rj, rk)
    return energy, dE_dtheta .* gi, dE_dtheta .* gj, dE_dtheta .* gk
end

function calc_angle_energy_and_grad(param::UreyBradleyAngle, ri::AbstractVector{<:Real},
                                    rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})
    theta = MolMath.angle_between(ri, rj, rk)
    dtheta = theta - param.theta0
    e_ang = 0.5 * param.ktheta * dtheta^2
    dE_dtheta = param.ktheta * dtheta
    gi, gj, gk = MolMath.angle_gradient(ri, rj, rk)

    r_ik = MolMath.distance(ri, rk)
    dr_ik = r_ik - param.rub0
    e_ub = 0.5 * param.kub * dr_ik^2
    dE_dr = param.kub * dr_ik
    g_ub_i, g_ub_k = MolMath.distance_gradient(ri, rk)

    energy = e_ang + e_ub
    return energy, (dE_dtheta .* gi) .+ (dE_dr .* g_ub_i), (dE_dtheta .* gj), (dE_dtheta .* gk) .+ (dE_dr .* g_ub_k)
end

function calc_angle_energy_and_grad(param::Class2Angle, ri::AbstractVector{<:Real},
                                    rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})
    theta = MolMath.angle_between(ri, rj, rk)
    dtheta = theta - param.theta0
    energy = param.k2 * dtheta^2 + param.k3 * dtheta^3 + param.k4 * dtheta^4
    dE_dtheta = 2.0 * param.k2 * dtheta + 3.0 * param.k3 * dtheta^2 + 4.0 * param.k4 * dtheta^3
    gi, gj, gk = MolMath.angle_gradient(ri, rj, rk)
    return energy, dE_dtheta .* gi, dE_dtheta .* gj, dE_dtheta .* gk
end

end # module BondedPotentials

