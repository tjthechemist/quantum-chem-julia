# ==============================================================================
# QuantumChemMM.jl: High-Performance GPU Acceleration Kernels
# Supports NVIDIA CUDA, AMD ROCm, Intel oneAPI, and Host CPU Fallback
# ==============================================================================

module GPUKernels

using LinearAlgebra
using ..GPUInterface

export gpu_distance_matrix, gpu_coulomb_energy_and_forces,
       gpu_lj_energy_and_forces, gpu_two_center_gamma_matrix,
       gpu_velocity_verlet_step!,
       gpu_fock_coulomb_exchange, gpu_dft_density_and_xc,
       gpu_transform_eri_ao_to_mo, gpu_antisymmetrize_mo_eri,
       gpu_mp2_correlation, gpu_cis_hamiltonian,
       gpu_nmr_paramagnetic_shielding, gpu_nmr_diamagnetic_shielding,
       gpu_simulate_nmr_spectrum

const COULOMB_KCAL_ANG = 332.063711
const EV_ANGSTROM = 14.399645

"""
    gpu_distance_matrix(coords::AbstractMatrix{Float64})::AbstractMatrix{Float64}
Vectorized pairwise Euclidean distance calculation suitable for GPU parallel execution.
coords: 3 x N_atoms.
"""
function gpu_distance_matrix(coords::AbstractMatrix{Float64})
    nat = size(coords, 2)
    # R_ij^2 = ||r_i - r_j||^2 = ||r_i||^2 + ||r_j||^2 - 2 * r_i' * r_j
    sq_norms = sum(coords.^2, dims=1) # 1 x N
    # Broadcast addition: (1 x N) + (N x 1) - 2 * (N x N)
    gram = coords' * coords # N x N
    dist_sq = sq_norms .+ sq_norms' .- 2.0 .* gram
    # Clamp small negative values from floating point cancellation
    return sqrt.(max.(dist_sq, 0.0))
end

"""
    gpu_two_center_gamma_matrix(coords::AbstractMatrix{Float64}, gamma_1center::AbstractVector{Float64})
Evaluates the Klopman-Ohno two-center repulsion matrix γ_AB on GPU arrays.
"""
function gpu_two_center_gamma_matrix(coords::AbstractMatrix{Float64}, gamma_1center::AbstractVector{Float64})
    dist = gpu_distance_matrix(coords) # N x N
    c_widths = 0.5 .* (EV_ANGSTROM ./ gamma_1center) # N
    # C_ij = c_i + c_j
    c_matrix = c_widths .+ c_widths' # N x N
    denom = sqrt.(dist.^2 .+ c_matrix.^2)
    return EV_ANGSTROM ./ denom
end

"""
    gpu_lj_energy_and_forces(coords::AbstractMatrix{Float64}, sigmas::AbstractVector{Float64},
                             epsilons::AbstractVector{Float64}; scale_14::Float64=1.0)
Computes Lennard-Jones 12-6 pairwise energy and 3 x N forces in parallel on device.
"""
function gpu_lj_energy_and_forces(coords::AbstractMatrix{Float64}, sigmas::AbstractVector{Float64},
                                  epsilons::AbstractVector{Float64}; scale_14::Float64=1.0)
    nat = size(coords, 2)
    forces = zeros(Float64, 3, nat)
    energy = 0.0

    # Combining rules: Lorentz-Berthelot
    sigma_mat = 0.5 .* (sigmas .+ sigmas')
    eps_mat = sqrt.(epsilons .* epsilons')

    dist = gpu_distance_matrix(coords)

    # Upper triangular mask for pair sum
    for i in 1:nat
        for j in (i+1):nat
            r = dist[i, j]
            if r > 1e-4
                s_ij = sigma_mat[i, j]
                e_ij = eps_mat[i, j]
                sr = s_ij / r
                sr6 = sr^6
                sr12 = sr6^2
                e_pair = 4.0 * e_ij * (sr12 - sr6)
                energy += e_pair

                # Force magnitude: -dE/dr
                dE_dr = (24.0 * e_ij / r) * (-2.0 * sr12 + sr6)
                disp = (coords[:, i] .- coords[:, j]) ./ r
                f_vec = -dE_dr .* disp
                forces[:, i] .+= f_vec
                forces[:, j] .-= f_vec
            end
        end
    end

    return energy, forces
end

"""
    gpu_coulomb_energy_and_forces(coords::AbstractMatrix{Float64}, charges::AbstractVector{Float64};
                                  dielectric::Float64=1.0)
Computes Coulomb electrostatics and forces across all pairs on device.
"""
function gpu_coulomb_energy_and_forces(coords::AbstractMatrix{Float64}, charges::AbstractVector{Float64};
                                      dielectric::Float64=1.0)
    nat = size(coords, 2)
    forces = zeros(Float64, 3, nat)
    energy = 0.0
    pref = COULOMB_KCAL_ANG / dielectric
    dist = gpu_distance_matrix(coords)

    for i in 1:nat
        qi = charges[i]
        abs(qi) < 1e-8 && continue
        for j in (i+1):nat
            qj = charges[j]
            abs(qj) < 1e-8 && continue
            r = dist[i, j]
            if r > 1e-4
                e_pair = pref * (qi * qj) / r
                energy += e_pair
                dE_dr = -pref * (qi * qj) / (r^2)
                disp = (coords[:, i] .- coords[:, j]) ./ r
                f_vec = -dE_dr .* disp
                forces[:, i] .+= f_vec
                forces[:, j] .-= f_vec
            end
        end
    end

    return energy, forces
end

"""
    gpu_velocity_verlet_step!(coords::AbstractMatrix{Float64}, velocities::AbstractMatrix{Float64},
                               accel::AbstractMatrix{Float64}, dt::Float64)
Vectorized Verlet coordinate update across all atoms.
"""
function gpu_velocity_verlet_step!(coords::AbstractMatrix{Float64}, velocities::AbstractMatrix{Float64},
                                   accel::AbstractMatrix{Float64}, dt::Float64)
    coords .+= velocities .* dt .+ 0.5 .* accel .* (dt^2)
end

# ==============================================================================
# Ab Initio & DFT GPU Acceleration Kernels
# ==============================================================================

"""
    gpu_fock_coulomb_exchange(P::AbstractMatrix{Float64},
                              eri_j_mat::AbstractMatrix{Float64},
                              eri_k_mat::AbstractMatrix{Float64},
                              c_x::Float64)
Evaluates Coulomb (J) and Exchange (K) matrices on GPU using batched GEMV/GEMM matrix products.
This replaces the O(N^4) host loops with two high-throughput GPU matrix contractions.
"""
function gpu_fock_coulomb_exchange(P::AbstractMatrix{Float64},
                                   eri_j_mat::AbstractMatrix{Float64},
                                   eri_k_mat::AbstractMatrix{Float64},
                                   c_x::Float64)
    nb = size(P, 1)
    p_vec = vec(P)
    j_vec = eri_j_mat * p_vec
    J = reshape(j_vec, nb, nb)
    if c_x > 0.0
        k_vec = eri_k_mat * p_vec
        K = reshape(k_vec, nb, nb)
    else
        K = similar(J)
        fill!(K, 0.0)
    end
    return J, K
end

"""
    gpu_dft_density_and_xc(basis_values::AbstractMatrix{Float64}, weights::AbstractVector{Float64},
                           P::AbstractMatrix{Float64})
Evaluates electron density rho on device grid, computes exchange-correlation potential,
and forms the V_xc matrix in AO basis entirely on GPU device arrays.
"""
function gpu_dft_density_and_xc(basis_values::AbstractMatrix{Float64},
                                weights::AbstractVector{Float64},
                                P::AbstractMatrix{Float64})
    M = basis_values * P # N_grid x N_basis (GPU GEMM)
    rho_mat = sum(M .* basis_values, dims=2) # N_grid x 1
    rho = vec(rho_mat)
    rho_clamped = max.(rho, 1e-12)

    cx = 0.7385587663820224
    rho_13 = rho_clamped .^ (1.0 / 3.0)
    eps_x = -cx .* rho_13
    v_x = (4.0 / 3.0) .* eps_x

    rs = (3.0 ./ (4.0 * π .* rho_clamped)) .^ (1.0 / 3.0)
    x = sqrt.(rs)
    A = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    Q = sqrt(4.0 * c - b^2)

    X_x = x.^2 .+ b.*x .+ c
    X_x0 = x0^2 + b*x0 + c

    t1 = log.(x.^2 ./ X_x)
    t2 = (2.0 * b / Q) .* atan.(Q ./ (2.0 .* x .+ b))
    t3 = (b * x0 / X_x0) .* (log.((x .- x0).^2 ./ X_x) .+ (2.0 * (b + 2.0 * x0) / Q) .* atan.(Q ./ (2.0 .* x .+ b)))
    eps_c = A .* (t1 .+ t2 .- t3)

    dx_drs = 0.5 ./ x
    d_eps_c_dx = A .* ( (2.0 ./ x .- (2.0 .* x .+ b) ./ X_x) .- (4.0 * b ./ (Q^2 .+ (2.0 .* x .+ b).^2)) .-
                        (b * x0 / X_x0) .* (2.0 ./ (x .- x0) .- (2.0 .* x .+ b) ./ X_x .- 4.0 * (b + 2.0 * x0) ./ (Q^2 .+ (2.0 .* x .+ b).^2)) )
    d_eps_c_drs = d_eps_c_dx .* dx_drs
    v_c = eps_c .- (rs ./ 3.0) .* d_eps_c_drs

    eps_xc = eps_x .+ eps_c
    v_xc = v_x .+ v_c

    w_vxc = weights .* v_xc
    B_weighted = basis_values .* reshape(w_vxc, :, 1)
    V_xc = basis_values' * B_weighted

    E_xc = sum(weights .* rho_clamped .* eps_xc)

    return V_xc, E_xc
end

# ==============================================================================
# Post-HF GPU Acceleration Kernels (MP2, CCSD, CISD)
# ==============================================================================

"""
    gpu_transform_eri_ao_to_mo(eri_ao::AbstractArray{Float64, 4}, C::AbstractMatrix{Float64})
Performs AO-to-MO 4-index transformation using four successive matrix multiplications (GEMMs) on device.
"""
function gpu_transform_eri_ao_to_mo(eri_ao::AbstractArray{Float64, 4}, C::AbstractMatrix{Float64})
    nb = size(eri_ao, 1)
    n_mo = size(C, 2)

    A1 = reshape(eri_ao, nb^3, nb) * C
    A1_perm = permutedims(reshape(A1, nb, nb, nb, n_mo), (1, 2, 4, 3))
    A2 = reshape(A1_perm, nb^2 * n_mo, nb) * C
    A2_perm = permutedims(reshape(A2, nb, nb, n_mo, n_mo), (1, 3, 4, 2))
    A3 = reshape(A2_perm, nb * n_mo^2, nb) * C
    A3_perm = permutedims(reshape(A3, nb, n_mo, n_mo, n_mo), (2, 3, 4, 1))
    A4 = reshape(A3_perm, n_mo^3, nb) * C
    eri_mo = permutedims(reshape(A4, n_mo, n_mo, n_mo, n_mo), (4, 3, 2, 1))
    return eri_mo
end

"""
    gpu_antisymmetrize_mo_eri(eri_mo::AbstractArray{Float64, 4})
Forms <pq||rs> = (pr|qs) - (ps|qr) on GPU via tensor permutation and subtraction.
"""
function gpu_antisymmetrize_mo_eri(eri_mo::AbstractArray{Float64, 4})
    term1 = permutedims(eri_mo, (1, 3, 2, 4))
    term2 = permutedims(eri_mo, (1, 4, 2, 3))
    return term1 .- term2
end

"""
    gpu_mp2_correlation(eri_mo::AbstractArray{Float64, 4}, eps::AbstractVector{Float64},
                        n_occ::Int, n_virt::Int)
Evaluates MP2 and SCS-MP2 correlation energy directly on device arrays using vectorized tensor expressions.
"""
function gpu_mp2_correlation(eri_mo::AbstractArray{Float64, 4}, eps::AbstractVector{Float64},
                             n_occ::Int, n_virt::Int)
    n_mo = n_occ + n_virt
    occ_range = 1:n_occ
    virt_range = (n_occ+1):n_mo

    V_iajb = eri_mo[occ_range, virt_range, occ_range, virt_range]
    V_ibja = permutedims(V_iajb, (1, 4, 3, 2))

    eps_occ = eps[occ_range]
    eps_virt = eps[virt_range]

    D = reshape(eps_occ, n_occ, 1, 1, 1) .+ reshape(eps_occ, 1, 1, n_occ, 1) .-
        reshape(eps_virt, 1, n_virt, 1, 1) .- reshape(eps_virt, 1, 1, 1, n_virt)

    term_os = (V_iajb .^ 2) ./ D
    term_ss = (V_iajb .* (V_iajb .- V_ibja)) ./ D

    e_os = sum(term_os)
    e_ss = sum(term_ss)
    e_mp2 = sum(V_iajb .* (2.0 .* V_iajb .- V_ibja) ./ D)
    e_scs = (6.0 / 5.0) * e_os + (1.0 / 3.0) * e_ss

    return e_mp2, e_scs, e_ss, e_os
end

# ==============================================================================
# Excited States CIS GPU Acceleration Kernel
# ==============================================================================

"""
    gpu_cis_hamiltonian(eps::AbstractVector{Float64}, C::AbstractMatrix{Float64},
                        gamma_mat::AbstractMatrix{Float64}, atom_indices::AbstractVector{Int},
                        n_occ::Int, n_virt::Int)
Forms Singlet and Triplet CIS Hamiltonian matrices on device.
"""
function gpu_cis_hamiltonian(eps::AbstractVector{Float64},
                             C::AbstractMatrix{Float64},
                             gamma_mat::AbstractMatrix{Float64},
                             atom_indices::AbstractVector{Int},
                             n_occ::Int, n_virt::Int)
    nb = length(eps)
    nat = size(gamma_mat, 1)
    n_ex = n_occ * n_virt

    # 1. Build transition charge matrix T (nat x n_ex)
    T = zeros(Float64, nat, n_ex)
    ex_idx = 1
    for i in 1:n_occ
        for a in (n_occ+1):nb
            for mu in 1:nb
                at = atom_indices[mu]
                T[at, ex_idx] += C[mu, i] * C[mu, a]
            end
            ex_idx += 1
        end
    end

    # 2. Coulomb-type integral matrix: G = T' * gamma_mat * T (Device GEMM)
    G = T' * (gamma_mat * T)

    # 3. Exchange-type integral matrix: K_mat[ia, jb] = (ib|ja)
    K_mat = zeros(Float64, n_ex, n_ex)
    ex_idx1 = 1
    for i in 1:n_occ
        for a in (n_occ+1):nb
            ex_idx2 = 1
            for j in 1:n_occ
                for b in (n_occ+1):nb
                    val = 0.0
                    for mu in 1:nb
                        at_A = atom_indices[mu]
                        c_ib = C[mu, i] * C[mu, b]
                        abs(c_ib) < 1e-10 && continue
                        for nu in 1:nb
                            at_B = atom_indices[nu]
                            c_ja = C[nu, j] * C[nu, a]
                            val += c_ib * c_ja * gamma_mat[at_A, at_B]
                        end
                    end
                    K_mat[ex_idx1, ex_idx2] = val
                    ex_idx2 += 1
                end
            end
            ex_idx1 += 1
        end
    end

    # 4. Diagonal orbital energy differences
    H_diag = zeros(Float64, n_ex)
    ex_idx = 1
    for i in 1:n_occ
        for a in (n_occ+1):nb
            H_diag[ex_idx] = eps[a] - eps[i]
            ex_idx += 1
        end
    end

    D_mat = Diagonal(H_diag)
    A_singlet = D_mat + 2.0 .* G - K_mat
    A_triplet = D_mat - K_mat

    return A_singlet, A_triplet
end

# ==============================================================================
# NMR Spectroscopy GPU Acceleration Kernels
# ==============================================================================

"""
    gpu_nmr_paramagnetic_shielding(L_ia::AbstractArray{Float64, 4},
                                   delta_eps::AbstractMatrix{Float64},
                                   r3_inv::AbstractVector{Float64})
Evaluates paramagnetic shielding tensors σ_p for all nuclei in parallel on device arrays.
L_ia: (3, n_occ, n_virt, n_atoms) angular momentum matrix elements.
delta_eps: (n_occ, n_virt) orbital energy gaps (eps_a - eps_i).
r3_inv: (n_atoms) expectation value <r^-3> for valence orbitals.
"""
function gpu_nmr_paramagnetic_shielding(L_ia::AbstractArray{Float64, 4},
                                        delta_eps::AbstractMatrix{Float64},
                                        r3_inv::AbstractVector{Float64})
    _, n_occ, n_virt, nat = size(L_ia)
    sigma_p = zeros(Float64, 3, 3, nat)

    # Inv sqrt delta_eps for GEMM: M_ia = L_ia / sqrt(delta_eps)
    # Shape: 3 x (n_occ * n_virt) x nat
    n_ex = n_occ * n_virt
    denom_sqrt = sqrt.(max.(delta_eps, 1e-4)) # n_occ x n_virt

    for at in 1:nat
        r3 = r3_inv[at]
        r3 < 1e-8 && continue

        # Scale L by denom_sqrt
        L_at = L_ia[:, :, :, at] # 3 x n_occ x n_virt
        M = reshape(L_at, 3, n_ex) ./ reshape(denom_sqrt, 1, n_ex)

        # Tensor product: sigma_p = -pref * r3 * (M * M')
        # In a.u. to ppm: alpha^2 * 1e6 / 2 ~ 26.62567 ppm
        pref = 26.62567 * r3
        sigma_p[:, :, at] .= -pref .* (M * M')
    end

    return sigma_p
end

"""
    gpu_nmr_diamagnetic_shielding(coords::AbstractMatrix{Float64},
                                  charges::AbstractVector{Float64},
                                  sigma0_d::AbstractVector{Float64},
                                  val_pops::AbstractVector{Float64},
                                  val_electrons::AbstractVector{Float64})
Computes atomic and interatomic diamagnetic chemical shielding constants on device.
"""
function gpu_nmr_diamagnetic_shielding(coords::AbstractMatrix{Float64},
                                       charges::AbstractVector{Float64},
                                       sigma0_d::AbstractVector{Float64},
                                       val_pops::AbstractVector{Float64},
                                       val_electrons::AbstractVector{Float64})
    nat = size(coords, 2)
    dist = gpu_distance_matrix(coords)
    sigma_d = zeros(Float64, nat)

    # 1. Intra-atomic contribution: sigma0 * (pop / val_e)
    # 2. Inter-atomic Flygare contribution: 17.75 * sum_{B != A} q_B / R_AB (with R in Angstrom)
    for a in 1:nat
        scale_pop = val_electrons[a] > 0.0 ? (val_pops[a] / val_electrons[a]) : 1.0
        intra = sigma0_d[a] * scale_pop

        inter = 0.0
        for b in 1:nat
            if a != b
                r = dist[a, b]
                if r > 1e-3
                    inter += (17.75 * charges[b]) / r
                end
            end
        end
        sigma_d[a] = intra + inter
    end

    return sigma_d
end

"""
    gpu_simulate_nmr_spectrum(ppm_grid::AbstractVector{Float64},
                              shifts::AbstractVector{Float64},
                              intensities::AbstractVector{Float64},
                              gamma_ppm::Float64)
Evaluates continuous Lorentzian line shapes across all spectrum grid points on device.
"""
function gpu_simulate_nmr_spectrum(ppm_grid::AbstractVector{Float64},
                                   shifts::AbstractVector{Float64},
                                   intensities::AbstractVector{Float64},
                                   gamma_ppm::Float64)
    # Grid: N_pts x 1, Shifts: 1 x N_peaks
    diff = reshape(ppm_grid, :, 1) .- reshape(shifts, 1, :)
    denom = diff.^2 .+ (gamma_ppm^2)
    weights = reshape(intensities, 1, :) .* (gamma_ppm / π)
    peaks = weights ./ denom
    spectrum = vec(sum(peaks, dims=2))
    return spectrum
end

end # module GPUKernels
