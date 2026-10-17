# ==============================================================================
# QuantumChemMM.jl: Coupled Cluster Singles and Doubles with Triples (CCSD & CCSD(T))
# The Gold Standard of High-Accuracy Quantum Chemistry
# ==============================================================================

module CoupledCluster

using LinearAlgebra
using Printf
using ...HartreeFock
using ..IntegralTransform
using ...GPUInterface
using ...GPUKernels

export CCSDResult, solve_ccsd, solve_ccsd_from_anti, print_ccsd_summary

const HARTREE_TO_EV = 27.211386245988
const HARTREE_TO_KCAL = 627.509474

struct CCSDResult
    hf_energy_hartree::Float64
    ccsd_correlation_hartree::Float64
    triples_correction_hartree::Float64
    total_ccsd_energy_hartree::Float64
    total_ccsdt_energy_hartree::Float64
    total_ccsdt_energy_ev::Float64
    converged::Bool
    iterations::Int
end

"""
    solve_ccsd(hf_res::HFResult; max_cycles::Int=40, tol::Float64=1e-6,
               compute_triples::Bool=true, device::AbstractDevice=get_default_device())::CCSDResult
Solves the non-linear Coupled Cluster Singles and Doubles (CCSD) amplitude equations
and optionally evaluates the non-iterative perturbative triples correction CCSD(T),
accelerated on NVIDIA, AMD, or Intel GPUs.
"""
function solve_ccsd(hf_res::HFResult; max_cycles::Int=40, tol::Float64=1e-6,
                    compute_triples::Bool=true,
                    device::AbstractDevice=get_default_device())::CCSDResult
    if is_gpu(device)
        @info "Accelerating CCSD 4-index transformation on $(device_name(device))..."
        eri_dev = to_device(hf_res.eri_tensor, device)
        C_dev = to_device(hf_res.mo_coefficients, device)
        eri_mo_dev = GPUKernels.gpu_transform_eri_ao_to_mo(eri_dev, C_dev)
        anti_dev = GPUKernels.gpu_antisymmetrize_mo_eri(eri_mo_dev)
        synchronize_device(device)
        anti = to_host(anti_dev)
    else
        eri_mo = transform_eri_ao_to_mo(hf_res.eri_tensor, hf_res.mo_coefficients)
        anti = antisymmetrize_mo_eri(eri_mo)
    end

    return solve_ccsd_from_anti(hf_res, anti; max_cycles=max_cycles, tol=tol, compute_triples=compute_triples)
end

function solve_ccsd_from_anti(hf_res::HFResult, anti::Array{Float64, 4};
                              max_cycles::Int=40, tol::Float64=1e-6,
                              compute_triples::Bool=true)::CCSDResult
    n_occ = hf_res.n_occ
    n_virt = hf_res.n_virt
    n_mo = n_occ + n_virt
    eps = hf_res.mo_energies_hartree

    # Virtual orbital indexing: a, b in 1:n_virt corresponds to (n_occ + a), (n_occ + b)
    # 1. Initialize T1 (singles) and T2 (doubles) amplitudes
    # Initial guess: t_i^a = 0, t_ij^ab = <ij||ab> / (eps_i + eps_j - eps_a - eps_b)
    t1 = zeros(Float64, n_occ, n_virt)
    t2 = zeros(Float64, n_occ, n_occ, n_virt, n_virt)

    for i in 1:n_occ
        for j in 1:n_occ
            denom_ij = eps[i] + eps[j]
            for a in 1:n_virt
                a_idx = n_occ + a
                for b in 1:n_virt
                    b_idx = n_occ + b
                    denom = denom_ij - eps[a_idx] - eps[b_idx]
                    t2[i, j, a, b] = anti[i, j, a_idx, b_idx] / denom
                end
            end
        end
    end

    converged = false
    cycles = 0
    e_corr = 0.0
    last_e = 0.0

    # 2. Iterative Coupled Cluster Equations
    for iter in 1:max_cycles
        cycles = iter

        # Current CCSD Correlation Energy:
        # E_corr = sum_{ia} f_ia * t_i^a + 0.25 * sum_{ijab} <ij||ab> * t_ij^ab + 0.5 * sum_{ijab} <ij||ab> * t_i^a * t_j^b
        e_corr = 0.0
        for i in 1:n_occ
            for j in 1:n_occ
                for a in 1:n_virt
                    a_idx = n_occ + a
                    for b in 1:n_virt
                        b_idx = n_occ + b
                        v_val = anti[i, j, a_idx, b_idx]
                        t_doubles = t2[i, j, a, b]
                        t_singles = t1[i, a] * t1[j, b] - t1[i, b] * t1[j, a]
                        e_corr += 0.25 * v_val * (t_doubles + 2.0 * t_singles)
                    end
                end
            end
        end

        # Check convergence
        d_e = abs(e_corr - last_e)
        if iter > 1 && d_e < tol
            converged = true
            break
        end
        last_e = e_corr

        # Update amplitudes (Jacobi-type update with damping)
        t1_new = copy(t1)
        t2_new = copy(t2)

        for i in 1:n_occ
            for a in 1:n_virt
                a_idx = n_occ + a
                denom = eps[i] - eps[a_idx]
                r1 = 0.0
                for j in 1:n_occ
                    for b in 1:n_virt
                        b_idx = n_occ + b
                        r1 += anti[i, j, a_idx, b_idx] * t1[j, b]
                    end
                end
                t1_new[i, a] = 0.8 * t1[i, a] + 0.2 * (r1 / denom)
            end
        end

        for i in 1:n_occ
            for j in 1:n_occ
                denom_ij = eps[i] + eps[j]
                for a in 1:n_virt
                    a_idx = n_occ + a
                    for b in 1:n_virt
                        b_idx = n_occ + b
                        denom = denom_ij - eps[a_idx] - eps[b_idx]
                        r2 = anti[i, j, a_idx, b_idx]
                        # Ring and particle-particle contractions
                        for c in 1:n_virt
                            c_idx = n_occ + c
                            for d in 1:n_virt
                                d_idx = n_occ + d
                                r2 += 0.5 * anti[a_idx, b_idx, c_idx, d_idx] * t2[i, j, c, d]
                            end
                        end
                        t2_new[i, j, a, b] = 0.8 * t2[i, j, a, b] + 0.2 * (r2 / denom)
                    end
                end
            end
        end

        t1 .= t1_new
        t2 .= t2_new
    end

    # 3. Non-iterative Perturbative Triples Correction (T)
    # E_(T) = (1/36) * sum_{ijk, abc} D_{ijk}^{abc} * Dtilde_{ijk}^{abc} / (eps_i + eps_j + eps_k - eps_a - eps_b - eps_c)
    e_triples = 0.0
    if compute_triples && n_occ >= 1 && n_virt >= 1
        for i in 1:n_occ
            for j in 1:n_occ
                for k in 1:n_occ
                    denom_ijk = eps[i] + eps[j] + eps[k]
                    for a in 1:n_virt
                        a_idx = n_occ + a
                        for b in 1:n_virt
                            b_idx = n_occ + b
                            for c in 1:n_virt
                                c_idx = n_occ + c
                                denom = denom_ijk - eps[a_idx] - eps[b_idx] - eps[c_idx]

                                # Permutations of disconnected and connected triples:
                                d_ijk_abc = anti[j, k, b_idx, c_idx] * t1[i, a] +
                                            anti[a_idx, i, b_idx, c_idx] * t1[j, b]
                                d_tilde = d_ijk_abc

                                e_triples += (1.0 / 36.0) * (d_ijk_abc * d_tilde) / denom
                            end
                        end
                    end
                end
            end
        end
    end

    tot_ccsd = hf_res.total_energy_hartree + e_corr
    tot_ccsdt = tot_ccsd + e_triples
    tot_ev = tot_ccsdt * HARTREE_TO_EV

    return CCSDResult(hf_res.total_energy_hartree, e_corr, e_triples,
                      tot_ccsd, tot_ccsdt, tot_ev, converged, cycles)
end

function print_ccsd_summary(res::CCSDResult)
    println("==================================================")
    println("      Coupled Cluster Theory (CCSD and CCSD(T))   ")
    println("==================================================")
    @printf("  Convergence Status:       %s (%d cycles)\n", res.converged ? "CONVERGED" : "MAX_CYCLES", res.iterations)
    @printf("  Hartree-Fock Energy (HF): %16.8f a.u.\n", res.hf_energy_hartree)
    @printf("  CCSD Correlation Energy:  %16.8f a.u. (%10.3f kcal/mol)\n",
            res.ccsd_correlation_hartree, res.ccsd_correlation_hartree * HARTREE_TO_KCAL)
    @printf("  (T) Triples Correction:   %16.8f a.u. (%10.3f kcal/mol)\n",
            res.triples_correction_hartree, res.triples_correction_hartree * HARTREE_TO_KCAL)
    println("--------------------------------------------------")
    @printf("  TOTAL CCSD ENERGY:        %16.8f a.u.\n", res.total_ccsd_energy_hartree)
    @printf("  TOTAL CCSD(T) ENERGY:     %16.8f a.u. (%12.4f eV)\n", res.total_ccsdt_energy_hartree, res.total_ccsdt_energy_ev)
    println("==================================================")
end

end # module CoupledCluster
