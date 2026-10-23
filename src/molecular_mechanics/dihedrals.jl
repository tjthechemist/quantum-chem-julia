# ==============================================================================
# QuantumChemMM.jl: Proper Dihedral and Improper Torsion Potentials
# ==============================================================================

module Dihedrals

using LinearAlgebra
using ..MolMath

export PeriodicTorsion, FourierTerm, RyckaertBellemansTorsion,
       HarmonicImproper, CosineImproper,
       calc_dihedral_energy_and_grad, calc_improper_energy_and_grad

struct FourierTerm
    n::Int           # periodicity
    Vn::Float64      # barrier height in kcal/mol
    gamma::Float64   # phase angle in radians
end

struct PeriodicTorsion
    i::Int
    j::Int
    k::Int
    l::Int
    terms::Vector{FourierTerm}
end

struct RyckaertBellemansTorsion
    i::Int
    j::Int
    k::Int
    l::Int
    c::Vector{Float64} # C0, C1, C2, C3, C4, C5 in kcal/mol
end

struct HarmonicImproper
    central::Int
    i::Int
    j::Int
    k::Int
    omega0::Float64   # equilibrium angle in radians
    k_imp::Float64    # in kcal/(mol·rad²)
end

struct CosineImproper
    central::Int
    i::Int
    j::Int
    k::Int
    k_inv::Float64
    c0::Float64
    c1::Float64
    c2::Float64
end

function calc_dihedral_energy_and_grad(param::PeriodicTorsion, ri::AbstractVector{<:Real},
                                       rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real},
                                       rl::AbstractVector{<:Real})
    phi = MolMath.dihedral_angle(ri, rj, rk, rl)
    energy = 0.0
    dE_dphi = 0.0

    for t in param.terms
        arg = t.n * phi - t.gamma
        energy += 0.5 * t.Vn * (1.0 + cos(arg))
        dE_dphi += -0.5 * t.Vn * t.n * sin(arg)
    end

    gi, gj, gk, gl = MolMath.dihedral_gradient(ri, rj, rk, rl)
    return energy, dE_dphi .* gi, dE_dphi .* gj, dE_dphi .* gk, dE_dphi .* gl
end

function calc_dihedral_energy_and_grad(param::RyckaertBellemansTorsion, ri::AbstractVector{<:Real},
                                       rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real},
                                       rl::AbstractVector{<:Real})
    phi = MolMath.dihedral_angle(ri, rj, rk, rl)
    psi = phi - π
    cos_psi = cos(psi)

    energy = 0.0
    dE_dcos = 0.0
    pow_cos = 1.0

    for (idx, c_n) in enumerate(param.c)
        n = idx - 1
        energy += c_n * (cos_psi^n)
        if n >= 1
            dE_dcos += n * c_n * (cos_psi^(n - 1))
        end
    end

    # dE/dphi = dE/dcos * (-sin(psi))
    dE_dphi = -dE_dcos * sin(psi)

    gi, gj, gk, gl = MolMath.dihedral_gradient(ri, rj, rk, rl)
    return energy, dE_dphi .* gi, dE_dphi .* gj, dE_dphi .* gk, dE_dphi .* gl
end

function calc_improper_energy_and_grad(param::HarmonicImproper, rc::AbstractVector{<:Real},
                                       ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real},
                                       rk::AbstractVector{<:Real})
    omega = MolMath.out_of_plane_angle(rc, ri, rj, rk)
    domega = omega - param.omega0
    energy = 0.5 * param.k_imp * domega^2
    dE_domega = param.k_imp * domega

    gradients = _improper_angle_gradients(rc, ri, rj, rk, dE_domega)
    return energy, gradients...
end

function calc_improper_energy_and_grad(param::CosineImproper, rc::AbstractVector{<:Real},
                                       ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real},
                                       rk::AbstractVector{<:Real})
    omega = MolMath.out_of_plane_angle(rc, ri, rj, rk)
    energy = param.k_inv * (param.c0 + param.c1 * cos(omega) +
                            param.c2 * cos(2.0 * omega))
    dE_domega = param.k_inv * (-param.c1 * sin(omega) -
                               2.0 * param.c2 * sin(2.0 * omega))

    gradients = _improper_angle_gradients(rc, ri, rj, rk, dE_domega)
    return energy, gradients...
end

function _improper_angle_gradients(rc::AbstractVector{<:Real},
                                   ri::AbstractVector{<:Real},
                                   rj::AbstractVector{<:Real},
                                   rk::AbstractVector{<:Real},
                                   dE_domega::Real)
    delta = 1e-6
    coords = [Float64.(rc), Float64.(ri), Float64.(rj), Float64.(rk)]
    gradients = [zeros(3) for _ in 1:4]

    for atom in eachindex(coords), coord in eachindex(coords[atom])
        coords[atom][coord] += delta
        w_plus = MolMath.out_of_plane_angle(coords...)
        coords[atom][coord] -= 2 * delta
        w_minus = MolMath.out_of_plane_angle(coords...)
        coords[atom][coord] += delta
        gradients[atom][coord] = dE_domega * (w_plus - w_minus) / (2 * delta)
    end

    return gradients
end

end # module Dihedrals
