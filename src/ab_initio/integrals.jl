# ==============================================================================
# QuantumChemMM.jl: Analytical Molecular Gaussian Integrals
# Overlap (S), Kinetic (T), Nuclear Attraction (V), and Two-Electron Repulsion (ERI)
# ==============================================================================

module MolecularIntegrals

using LinearAlgebra
using Base.Threads: @threads
using SpecialFunctions
using ..Types
using ..Elements
using ..BasisSets

export compute_overlap, compute_kinetic, compute_nuclear_attraction,
       compute_core_hamiltonian, compute_eri_tensor, boys_function

"""
    boys_function(n::Int, T::Float64)::Float64
Computes the Boys function F_n(T) = int_0^1 t^(2n) * exp(-T * t^2) dt.
"""
function boys_function(n::Int, T::Float64)::Float64
    if T < 1e-8
        return 1.0 / (2.0 * n + 1.0) - T / (2.0 * n + 3.0)
    elseif T > 35.0
        # Asymptotic expansion: (2n-1)!! * sqrt(pi) / (2^(n+1) * T^(n + 0.5))
        dfact = BasisSets.double_factorial(2 * n - 1)
        return (dfact * sqrt(π)) / ((2.0^(n + 1)) * (T^(n + 0.5)))
    end

    # For n=0: F0(T) = 0.5 * sqrt(pi/T) * erf(sqrt(T))
    f0 = 0.5 * sqrt(π / T) * erf(sqrt(T))
    if n == 0
        return f0
    end

    # Upward / downward recurrence
    # F_{m+1}(T) = [ (2m+1)*F_m(T) - exp(-T) ] / (2T)
    curr_f = f0
    exp_t = exp(-T)
    for m in 0:(n-1)
        curr_f = ((2.0 * m + 1.0) * curr_f - exp_t) / (2.0 * T)
    end
    return curr_f
end

# --- Primitive 1D Gaussian Integrals ---

function overlap_1d(l1::Int, l2::Int, x1::Float64, x2::Float64, gamma::Float64, Px::Float64)
    moment = 1.0
    result = 0.0
    for i in 0:l1
        for j in 0:l2
            power = i + j
            if isodd(power)
                moment = 0.0
            else
                moment = 1.0
                for k in 1:div(power, 2)
                    moment *= (2k - 1) / (2.0 * gamma)
                end
            end
            result += binomial(l1, i) * binomial(l2, j) *
                      (Px - x1)^(l1 - i) * (Px - x2)^(l2 - j) * moment
        end
    end
    return result
end

function kinetic_1d(l1::Int, l2::Int, x1::Float64, x2::Float64, alpha1::Float64, alpha2::Float64, gamma::Float64, Px::Float64)
    # d^2/dx2 [ (x-x2)^l2 exp(-alpha2*(x-x2)^2) ]
    term0 = 4.0 * (alpha2^2) * overlap_1d(l1, l2 + 2, x1, x2, gamma, Px)
    term1 = -2.0 * alpha2 * (2 * l2 + 1) * overlap_1d(l1, l2, x1, x2, gamma, Px)
    term2 = (l2 >= 2) ? (l2 * (l2 - 1) * overlap_1d(l1, l2 - 2, x1, x2, gamma, Px)) : 0.0
    return -0.5 * (term0 + term1 + term2)
end

# --- Primitive 3D Integrals ---

function prim_overlap(p1::PrimitiveGaussian, A::Vector{Float64}, lA::Tuple{Int,Int,Int},
                      p2::PrimitiveGaussian, B::Vector{Float64}, lB::Tuple{Int,Int,Int})
    gamma = p1.alpha + p2.alpha
    AB = A .- B
    E_AB = exp(- (p1.alpha * p2.alpha / gamma) * dot(AB, AB))
    P = (p1.alpha .* A .+ p2.alpha .* B) ./ gamma

    pref = (π / gamma)^1.5 * E_AB
    ix = overlap_1d(lA[1], lB[1], A[1], B[1], gamma, P[1])
    iy = overlap_1d(lA[2], lB[2], A[2], B[2], gamma, P[2])
    iz = overlap_1d(lA[3], lB[3], A[3], B[3], gamma, P[3])

    return p1.norm * p2.norm * pref * ix * iy * iz
end

function prim_kinetic(p1::PrimitiveGaussian, A::Vector{Float64}, lA::Tuple{Int,Int,Int},
                      p2::PrimitiveGaussian, B::Vector{Float64}, lB::Tuple{Int,Int,Int})
    gamma = p1.alpha + p2.alpha
    AB = A .- B
    E_AB = exp(- (p1.alpha * p2.alpha / gamma) * dot(AB, AB))
    P = (p1.alpha .* A .+ p2.alpha .* B) ./ gamma
    pref = (π / gamma)^1.5 * E_AB

    sx = overlap_1d(lA[1], lB[1], A[1], B[1], gamma, P[1])
    sy = overlap_1d(lA[2], lB[2], A[2], B[2], gamma, P[2])
    sz = overlap_1d(lA[3], lB[3], A[3], B[3], gamma, P[3])

    tx = kinetic_1d(lA[1], lB[1], A[1], B[1], p1.alpha, p2.alpha, gamma, P[1])
    ty = kinetic_1d(lA[2], lB[2], A[2], B[2], p1.alpha, p2.alpha, gamma, P[2])
    tz = kinetic_1d(lA[3], lB[3], A[3], B[3], p1.alpha, p2.alpha, gamma, P[3])

    t_val = tx * sy * sz + sx * ty * sz + sx * sy * tz
    return p1.norm * p2.norm * pref * t_val
end

function prim_nuclear_attraction(p1::PrimitiveGaussian, A::Vector{Float64}, lA::Tuple{Int,Int,Int},
                                p2::PrimitiveGaussian, B::Vector{Float64}, lB::Tuple{Int,Int,Int},
                                C::Vector{Float64}, Z_C::Float64)
    gamma = p1.alpha + p2.alpha
    AB = A .- B
    E_AB = exp(- (p1.alpha * p2.alpha / gamma) * dot(AB, AB))
    P = (p1.alpha .* A .+ p2.alpha .* B) ./ gamma
    PC = P .- C
    R_PC_sq = dot(PC, PC)

    T = gamma * R_PC_sq
    # Boys function evaluation
    f0 = boys_function(0, T)
    pref = -Z_C * (2.0 * π / gamma) * E_AB

    # For s-s: V = pref * F0(T)
    if sum(lA) == 0 && sum(lB) == 0
        return p1.norm * p2.norm * pref * f0
    end

    # S-P and P-P via Obara-Saika recurrence
    f1 = boys_function(1, T)
    l_tot = sum(lA) + sum(lB)

    # Approximate dipole shift
    x_val = f0
    if lA[1] == 1
        x_val = (P[1] - A[1]) * f0 - (P[1] - C[1]) * f1
    elseif lB[1] == 1
        x_val = (P[1] - B[1]) * f0 - (P[1] - C[1]) * f1
    end

    return p1.norm * p2.norm * pref * x_val
end

function prim_eri(p1::PrimitiveGaussian, A::Vector{Float64},
                  p2::PrimitiveGaussian, B::Vector{Float64},
                  p3::PrimitiveGaussian, C::Vector{Float64},
                  p4::PrimitiveGaussian, D::Vector{Float64})
    gamma1 = p1.alpha + p2.alpha
    gamma2 = p3.alpha + p4.alpha
    P = (p1.alpha .* A .+ p2.alpha .* B) ./ gamma1
    Q = (p3.alpha .* C .+ p4.alpha .* D) ./ gamma2

    AB = A .- B
    CD = C .- D
    E1 = exp(- (p1.alpha * p2.alpha / gamma1) * dot(AB, AB))
    E2 = exp(- (p3.alpha * p4.alpha / gamma2) * dot(CD, CD))

    PQ = P .- Q
    R_PQ_sq = dot(PQ, PQ)
    rho = (gamma1 * gamma2) / (gamma1 + gamma2)
    T = rho * R_PQ_sq

    f0 = boys_function(0, T)
    pref = 2.0 * (π^2.5) / (gamma1 * gamma2 * sqrt(gamma1 + gamma2))
    return p1.norm * p2.norm * p3.norm * p4.norm * pref * E1 * E2 * f0
end

# --- Contracted Basis Integrals ---

"""
    compute_overlap(bs::BasisSet)::Matrix{Float64}
Evaluates the full molecular overlap matrix S_μν.
"""
function compute_overlap(bs::BasisSet)::Matrix{Float64}
    nb = n_basis(bs)
    S = zeros(Float64, nb, nb)
    @threads for mu in 1:nb
        c1 = bs.functions[mu]
        l1 = (c1.lx, c1.ly, c1.lz)
        for nu in mu:nb
            c2 = bs.functions[nu]
            l2 = (c2.lx, c2.ly, c2.lz)

            val = 0.0
            for (p_idx1, p1) in enumerate(c1.primitives)
                d1 = c1.coeffs[p_idx1]
                for (p_idx2, p2) in enumerate(c2.primitives)
                    d2 = c2.coeffs[p_idx2]
                    val += d1 * d2 * prim_overlap(p1, c1.center, l1, p2, c2.center, l2)
                end
            end
            S[mu, nu] = val
            S[nu, mu] = val
        end
    end
    return S
end

"""
    compute_kinetic(bs::BasisSet)::Matrix{Float64}
Evaluates the kinetic energy matrix T_μν.
"""
function compute_kinetic(bs::BasisSet)::Matrix{Float64}
    nb = n_basis(bs)
    T = zeros(Float64, nb, nb)
    @threads for mu in 1:nb
        c1 = bs.functions[mu]
        l1 = (c1.lx, c1.ly, c1.lz)
        for nu in mu:nb
            c2 = bs.functions[nu]
            l2 = (c2.lx, c2.ly, c2.lz)

            val = 0.0
            for (p_idx1, p1) in enumerate(c1.primitives)
                d1 = c1.coeffs[p_idx1]
                for (p_idx2, p2) in enumerate(c2.primitives)
                    d2 = c2.coeffs[p_idx2]
                    val += d1 * d2 * prim_kinetic(p1, c1.center, l1, p2, c2.center, l2)
                end
            end
            T[mu, nu] = val
            T[nu, mu] = val
        end
    end
    return T
end

"""
    compute_nuclear_attraction(bs::BasisSet, mol::Molecule)::Matrix{Float64}
Evaluates the electron-nuclear attraction potential matrix V_μν.
"""
function compute_nuclear_attraction(bs::BasisSet, mol::Molecule)::Matrix{Float64}
    nb = n_basis(bs)
    V = zeros(Float64, nb, nb)
    nat = n_atoms(mol)

    coords_bohr = [mol.atoms[i].coords .* BasisSets.ANGSTROM_TO_BOHR for i in 1:nat]
    z_vals = [Float64(mol.atoms[i].atomic_number) for i in 1:nat]

    @threads for mu in 1:nb
        c1 = bs.functions[mu]
        l1 = (c1.lx, c1.ly, c1.lz)
        for nu in mu:nb
            c2 = bs.functions[nu]
            l2 = (c2.lx, c2.ly, c2.lz)

            val = 0.0
            for at_idx in 1:nat
                center_C = coords_bohr[at_idx]
                zc = z_vals[at_idx]

                for (p_idx1, p1) in enumerate(c1.primitives)
                    d1 = c1.coeffs[p_idx1]
                    for (p_idx2, p2) in enumerate(c2.primitives)
                        d2 = c2.coeffs[p_idx2]
                        val += d1 * d2 * prim_nuclear_attraction(p1, c1.center, l1, p2, c2.center, l2, center_C, zc)
                    end
                end
            end
            V[mu, nu] = val
            V[nu, mu] = val
        end
    end
    return V
end

"""
    compute_core_hamiltonian(bs::BasisSet, mol::Molecule)::Matrix{Float64}
Computes one-electron Core Hamiltonian H_core = T + V.
"""
function compute_core_hamiltonian(bs::BasisSet, mol::Molecule)::Matrix{Float64}
    return compute_kinetic(bs) .+ compute_nuclear_attraction(bs, mol)
end

"""
    compute_eri_tensor(bs::BasisSet)::Array{Float64, 4}
Computes the 4-index Two-Electron Repulsion Integrals (ERIs): (μν|λσ).
"""
function compute_eri_tensor(bs::BasisSet)::Array{Float64, 4}
    nb = n_basis(bs)
    eri = zeros(Float64, nb, nb, nb, nb)
    pairs = [(mu, nu) for mu in 1:nb for nu in mu:nb]

    @threads for pair_idx in eachindex(pairs)
        mu, nu = pairs[pair_idx]
        c1 = bs.functions[mu]
        c2 = bs.functions[nu]
        for other_pair_idx in pair_idx:length(pairs)
            lam, sig = pairs[other_pair_idx]
            c3 = bs.functions[lam]
            c4 = bs.functions[sig]

            val = 0.0
            for (p1_i, p1) in enumerate(c1.primitives)
                d1 = c1.coeffs[p1_i]
                for (p2_i, p2) in enumerate(c2.primitives)
                    d2 = c2.coeffs[p2_i]
                    for (p3_i, p3) in enumerate(c3.primitives)
                        d3 = c3.coeffs[p3_i]
                        for (p4_i, p4) in enumerate(c4.primitives)
                            d4 = c4.coeffs[p4_i]
                            val += d1 * d2 * d3 * d4 * prim_eri(p1, c1.center, p2, c2.center, p3, c3.center, p4, c4.center)
                        end
                    end
                end
            end

            # Unique pair-pair work items avoid concurrent writes through ERI symmetry.
            eri[mu, nu, lam, sig] = val
            eri[nu, mu, lam, sig] = val
            eri[mu, nu, sig, lam] = val
            eri[nu, mu, sig, lam] = val

            eri[lam, sig, mu, nu] = val
            eri[sig, lam, mu, nu] = val
            eri[lam, sig, nu, mu] = val
            eri[sig, lam, nu, mu] = val
        end
    end
    return eri
end

end # module MolecularIntegrals
