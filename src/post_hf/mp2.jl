# ==============================================================================
# QuantumChemMM.jl: Second-Order Møller-Plesset Perturbation Theory (MP2 & SCS-MP2)
# ==============================================================================

module MP2Theory

using LinearAlgebra
using Printf
using ..BasisSets
using ...HartreeFock
using ..IntegralTransform
using ...GPUInterface
using ...GPUKernels

export MP2Result, solve_mp2, print_mp2_summary

const HARTREE_TO_EV = 27.211386245988
const HARTREE_TO_KCAL = 627.509474

struct MP2Result
    hf_energy_hartree::Float64
    mp2_correlation_hartree::Float64
    scs_mp2_correlation_hartree::Float64
    total_mp2_energy_hartree::Float64
    total_mp2_energy_ev::Float64
    same_spin_correlation::Float64
    opposite_spin_correlation::Float64
end

"""
    solve_mp2(hf_res::HFResult; device::AbstractDevice=get_default_device())::MP2Result
Computes the second-order Møller-Plesset perturbation correlation energy (MP2 and SCS-MP2)
with GPU acceleration on NVIDIA CUDA, AMD ROCm, and Intel oneAPI.
"""
function solve_mp2(hf_res::HFResult;
                   device::AbstractDevice=get_default_device())::MP2Result
    if is_gpu(device)
        @info "Accelerating MP2 Correlation on $(device_name(device))..."
        eri_dev = to_device(hf_res.eri_tensor, device)
        C_dev = to_device(hf_res.mo_coefficients, device)
        eri_mo_dev = GPUKernels.gpu_transform_eri_ao_to_mo(eri_dev, C_dev)
        eps_dev = to_device(hf_res.mo_energies_hartree, device)
        e_mp2, e_scs, e_ss, e_os = GPUKernels.gpu_mp2_correlation(eri_mo_dev, eps_dev, hf_res.n_occ, hf_res.n_virt)
        synchronize_device(device)
        tot_mp2 = hf_res.total_energy_hartree + e_mp2
        tot_ev = tot_mp2 * HARTREE_TO_EV
        return MP2Result(hf_res.total_energy_hartree, e_mp2, e_scs, tot_mp2, tot_ev, e_ss, e_os)
    end

    n_occ = hf_res.n_occ
    n_virt = hf_res.n_virt
    n_mo = n_occ + n_virt
    eps = hf_res.mo_energies_hartree

    # Transform AO ERIs into MO basis
    eri_mo = transform_eri_ao_to_mo(hf_res.eri_tensor, hf_res.mo_coefficients)

    e_os = 0.0 # opposite-spin
    e_ss = 0.0 # same-spin
    e_mp2 = 0.0

    for i in 1:n_occ
        for j in 1:n_occ
            denom_ij = eps[i] + eps[j]
            for a in (n_occ+1):n_mo
                for b in (n_occ+1):n_mo
                    denom = denom_ij - eps[a] - eps[b]

                    ia_jb = eri_mo[i, a, j, b]
                    ib_ja = eri_mo[i, b, j, a]

                    # Standard closed-shell formula:
                    term_os = (ia_jb^2) / denom
                    term_ss = (ia_jb * (ia_jb - ib_ja)) / denom

                    e_os += term_os
                    e_ss += term_ss
                    e_mp2 += (ia_jb * (2.0 * ia_jb - ib_ja)) / denom
                end
            end
        end
    end

    # Spin-Component-Scaled MP2 (SCS-MP2, Grimme 2003):
    e_scs = (6.0 / 5.0) * e_os + (1.0 / 3.0) * e_ss

    tot_mp2 = hf_res.total_energy_hartree + e_mp2
    tot_ev = tot_mp2 * HARTREE_TO_EV

    return MP2Result(hf_res.total_energy_hartree, e_mp2, e_scs, tot_mp2, tot_ev, e_ss, e_os)
end

function print_mp2_summary(res::MP2Result)
    println("==================================================")
    println("      Second-Order Møller-Plesset Theory (MP2)    ")
    println("==================================================")
    @printf("  Hartree-Fock Energy (HF): %16.8f a.u.\n", res.hf_energy_hartree)
    @printf("  MP2 Correlation Energy:   %16.8f a.u. (%10.3f kcal/mol)\n",
            res.mp2_correlation_hartree, res.mp2_correlation_hartree * HARTREE_TO_KCAL)
    @printf("  SCS-MP2 Correlation:     %16.8f a.u. (%10.3f kcal/mol)\n",
            res.scs_mp2_correlation_hartree, res.scs_mp2_correlation_hartree * HARTREE_TO_KCAL)
    println("--------------------------------------------------")
    @printf("  TOTAL MP2 ENERGY:         %16.8f a.u. (%12.4f eV)\n", res.total_mp2_energy_hartree, res.total_mp2_energy_ev)
    println("==================================================")
end

end # module MP2Theory
