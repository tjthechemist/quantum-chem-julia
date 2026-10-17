# ==============================================================================
# QuantumChemMM.jl: Geometric and Mathematical Utilities
# ==============================================================================

module MolMath

using LinearAlgebra

export distance, angle_between, dihedral_angle, out_of_plane_angle,
       distance_gradient, angle_gradient, dihedral_gradient

"""
    distance(r1, r2)
Calculates Euclidean distance between two 3D points.
"""
function distance(r1::AbstractVector{<:Real}, r2::AbstractVector{<:Real})::Float64
    return norm(r1 .- r2)
end

"""
    angle_between(ri, rj, rk)
Calculates angle at vertex rj formed by ri - rj and rk - rj in radians.
"""
function angle_between(ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})::Float64
    vij = ri .- rj
    vkj = rk .- rj
    nij = norm(vij)
    nkj = norm(vkj)
    if nij < 1e-12 || nkj < 1e-12
        return 0.0
    end
    cos_theta = clamp(dot(vij, vkj) / (nij * nkj), -1.0, 1.0)
    return acos(cos_theta)
end

"""
    dihedral_angle(ri, rj, rk, rl)
Calculates proper dihedral (torsion) angle in radians between [-π, π].
"""
function dihedral_angle(ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real},
                        rk::AbstractVector{<:Real}, rl::AbstractVector{<:Real})::Float64
    b1 = rj .- ri
    b2 = rk .- rj
    b3 = rl .- rk

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    norm_b2 = norm(b2)
    if norm_b2 < 1e-12 || norm(n1) < 1e-12 || norm(n2) < 1e-12
        return 0.0
    end

    b2_hat = b2 ./ norm_b2
    m1 = cross(n1, b2_hat)

    x = dot(n1, n2)
    y = dot(m1, n2)
    return atan(y, x)
end

"""
    out_of_plane_angle(rc, ri, rj, rk)
Calculates Wilson out-of-plane angle (inversion) where rc is the central atom
and ri, rj, rk are bonded neighbors.
"""
function out_of_plane_angle(rc::AbstractVector{<:Real}, ri::AbstractVector{<:Real},
                            rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})::Float64
    v_ci = ri .- rc
    v_cj = rj .- rc
    v_ck = rk .- rc

    n_plane = cross(v_ci, v_cj)
    norm_n = norm(n_plane)
    norm_ck = norm(v_ck)

    if norm_n < 1e-12 || norm_ck < 1e-12
        return 0.0
    end

    sin_omega = clamp(dot(n_plane, v_ck) / (norm_n * norm_ck), -1.0, 1.0)
    return asin(sin_omega)
end

# Analytical Gradients

"""
    distance_gradient(ri, rj)
Returns (grad_ri, grad_rj) for distance r = ||ri - rj||.
"""
function distance_gradient(ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real})
    diff = ri .- rj
    r = norm(diff)
    if r < 1e-12
        return zeros(3), zeros(3)
    end
    u = diff ./ r
    return u, -u
end

"""
    angle_gradient(ri, rj, rk)
Returns (grad_ri, grad_rj, grad_rk) for angle θ at vertex rj.
"""
function angle_gradient(ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real}, rk::AbstractVector{<:Real})
    vij = ri .- rj
    vkj = rk .- rj
    nij = norm(vij)
    nkj = norm(vkj)

    if nij < 1e-12 || nkj < 1e-12
        return zeros(3), zeros(3), zeros(3)
    end

    uij = vij ./ nij
    ukj = vkj ./ nkj
    cos_theta = clamp(dot(uij, ukj), -0.999999999, 0.999999999)
    sin_theta = sqrt(1.0 - cos_theta^2)

    grad_i = (cos_theta .* uij .- ukj) ./ (nij * sin_theta)
    grad_k = (cos_theta .* ukj .- uij) ./ (nkj * sin_theta)
    grad_j = -(grad_i .+ grad_k)

    return grad_i, grad_j, grad_k
end

"""
    dihedral_gradient(ri, rj, rk, rl)
Analytical gradient of dihedral angle w.r.t. ri, rj, rk, rl.
"""
function dihedral_gradient(ri::AbstractVector{<:Real}, rj::AbstractVector{<:Real},
                           rk::AbstractVector{<:Real}, rl::AbstractVector{<:Real})
    b1 = rj .- ri
    b2 = rk .- rj
    b3 = rl .- rk

    n1 = cross(b1, b2)
    n2 = cross(b2, b3)

    norm_n1_sq = dot(n1, n1)
    norm_n2_sq = dot(n2, n2)
    norm_b2 = norm(b2)

    if norm_n1_sq < 1e-14 || norm_n2_sq < 1e-14 || norm_b2 < 1e-12
        return zeros(3), zeros(3), zeros(3), zeros(3)
    end

    g_i = -norm_b2 ./ norm_n1_sq .* n1
    g_l =  norm_b2 ./ norm_n2_sq .* n2

    dot_b1_b2 = dot(b1, b2)
    dot_b3_b2 = dot(b3, b2)
    b2_sq = norm_b2^2

    g_j = ((dot_b1_b2 / b2_sq) - 1.0) .* g_i .- (dot_b3_b2 / b2_sq) .* g_l
    g_k = -(g_i .+ g_j .+ g_l)

    return g_i, g_j, g_k, g_l
end

end # module MolMath

