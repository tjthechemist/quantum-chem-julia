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

    # Numerical finite difference for out-of-plane gradients to guarantee robust geometric stability
    delta = 1e-6
    g_c = zeros(3)
    g_i = zeros(3)
    g_j = zeros(3)
    g_k = zeros(3)

    for coord in 1:3
        # central atom
        rc[coord] += delta
        w_plus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        rc[coord] -= 2*delta
        w_minus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        rc[coord] += delta
        g_c[coord] = dE_domega * (w_plus - w_minus) / (2*delta)

        # atom i
        ri[coord] += delta
        w_plus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        ri[coord] -= 2*delta
        w_minus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        ri[coord] += delta
        g_i[coord] = dE_domega * (w_plus - w_minus) / (2*delta)

        # atom j
        rj[coord] += delta
        w_plus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        rj[coord] -= 2*delta
        w_minus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        rj[coord] += delta
        g_j[coord] = dE_domega * (w_plus - w_minus) / (2*delta)

        # atom k
        rk[coord] += delta
        w_plus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        rk[coord] -= 2*delta
        w_minus = MolMath.out_of_plane_angle(rc, ri, rj, rk)
        rk[coord] += delta
        g_k[coord] = dE_domega * (w_plus - w_minus) / (2*delta)
    end

    return energy, g_c, g_i, g_j, g_k
end

end # module Dihedrals

