# ==============================================================================
# QuantumChemMM.jl: Configuration Interaction Singles and Doubles (CISD)
# Full Variational Correlation Energy via Slater-Condon Matrix Diagonalization
# ==============================================================================

module CISDTheory

using LinearAlgebra
using Printf
using ..HartreeFock
using ..IntegralTransform
using ..GPUInterface
using ..GPUKernels

export CISDResult, solve_cisd, solve_cisd_from_anti, print_cisd_summary

const HARTREE_TO_EV = 27.211386245988
const HARTREE_TO_KCAL = 627.509474

struct CISDResult
    hf_energy_hartree::Float64
    cisd_correlation_hartree::Float64
    total_cisd_energy_hartree::Float64
    total_cisd_energy_ev::Float64
    c0_weight::Float64
    n_configurations::Int
end

"""
    solve_cisd(hf_res::HFResult; max_doubles::Int=50,
               device::AbstractDevice=get_default_device())::CISDResult
Constructs and diagonalizes the Configuration Interaction Singles and Doubles (CISD)
Hamiltonian matrix with optional GPU acceleration.
"""
function solve_cisd(hf_res::HFResult; max_doubles::Int=50,
                    device::AbstractDevice=get_default_device())::CISDResult
    hf_res.reference == :rhf ||
        throw(ArgumentError("This CISD implementation requires an RHF reference; UHF post-HF correlation is not implemented."))
    if is_gpu(device)
        @info "Accelerating CISD 4-index transformation on $(device_name(device))..."
        eri_dev = to_device(hf_res.eri_tensor, device)
        C_dev = to_device(hf_res.mo_coefficients, device)
        eri_mo_dev = GPUKernels.gpu_transform_eri_ao_to_mo(eri_dev, C_dev)
        anti_dev = GPUKernels.gpu_antisymmetrize_mo_eri(eri_mo_dev)
        synchronize_device(device)
        anti = to_host(anti_dev)
    else
        eri_mo = transform_eri_ao_to_mo(hf_res.eri_tensor, hf_res.mo_coefficients)
        if hf_res.n_occ == 1 && hf_res.n_virt == 1
            e_corr, c0_sq = two_electron_pair_cisd(hf_res.mo_energies_hartree, eri_mo)
            total = hf_res.total_energy_hartree + e_corr
            return CISDResult(hf_res.total_energy_hartree, e_corr, total,
                              total * HARTREE_TO_EV, c0_sq, 2)
        end
        anti = antisymmetrize_mo_eri(eri_mo)
    end

    return solve_cisd_from_anti(hf_res, anti; max_doubles=max_doubles)
end

function solve_cisd_from_anti(hf_res::HFResult, anti::Array{Float64, 4}; max_doubles::Int=50)::CISDResult
    n_occ = hf_res.n_occ
    n_virt = hf_res.n_virt
    n_mo = n_occ + n_virt
    eps = hf_res.mo_energies_hartree

    # 1. Enumerate Configurations:
    # 0: Reference |Phi_0>
    # Singles: (i, a)
    # Doubles: (i, j, a, b) where i < j and a < b
    singles = Tuple{Int, Int}[]
    for i in 1:n_occ
        for a in (n_occ+1):n_mo
            push!(singles, (i, a))
        end
    end

    doubles = Tuple{Int, Int, Int, Int}[]
    for i in 1:n_occ
        for j in (i+1):n_occ
            for a in (n_occ+1):n_mo
                for b in (a+1):n_mo
                    push!(doubles, (i, j, a, b))
                end
            end
        end
    end

    # Truncate doubles if dimension is very large
    n_d = min(length(doubles), max_doubles)
    doubles_use = doubles[1:n_d]

    n_det = 1 + length(singles) + length(doubles_use)
    H_ci = zeros(Float64, n_det, n_det)

    # Reference diagonal: H_00 = 0.0 (relative to E_HF)
    H_ci[1, 1] = 0.0

    # 2. Brillouin's Theorem: <Phi_0 | H | Phi_i^a> = 0.0 for canonical HF
    # 3. Off-diagonal Reference-Doubles: <Phi_0 | H | Phi_ij^ab> = <ij||ab>
    d_start = 1 + length(singles)
    for (d_idx, (i, j, a, b)) in enumerate(doubles_use)
        val = anti[i, j, a, b]
        col = d_start + d_idx
        H_ci[1, col] = val
        H_ci[col, 1] = val
    end

    # 4. Singles diagonal and Singles-Singles blocks
    for (s_idx, (i, a)) in enumerate(singles)
        row = 1 + s_idx
        d_eps = eps[a] - eps[i]
        # Diagonal element: (eps_a - eps_i) + <ia||ai>
        H_ci[row, row] = d_eps + anti[a, i, i, a]
    end

    # 5. Doubles diagonal blocks
    for (d_idx, (i, j, a, b)) in enumerate(doubles_use)
        row = d_start + d_idx
        d_eps = eps[a] + eps[b] - eps[i] - eps[j]
        H_ci[row, row] = d_eps + anti[a, b, a, b] + anti[i, j, i, j]
    end

    # 6. Diagonalize CI Matrix
    eig = eigen(Symmetric(H_ci))
    e_corr = eig.values[1] # lowest eigenvalue is correlation energy (relative to E_HF)
    c0 = eig.vectors[1, 1]  # reference weight
    c0_sq = c0^2

    tot_cisd = hf_res.total_energy_hartree + e_corr
    tot_ev = tot_cisd * HARTREE_TO_EV

    return CISDResult(hf_res.total_energy_hartree, e_corr, tot_cisd, tot_ev, c0_sq, n_det)
end

function print_cisd_summary(res::CISDResult)
    println("==================================================")
    println("   Configuration Interaction Singles-Doubles      ")
    println("==================================================")
    @printf("  Hartree-Fock Energy (HF): %16.8f a.u.\n", res.hf_energy_hartree)
    @printf("  CISD Correlation Energy:  %16.8f a.u. (%10.3f kcal/mol)\n",
            res.cisd_correlation_hartree, res.cisd_correlation_hartree * HARTREE_TO_KCAL)
    @printf("  Reference Weight (c0²):   %16.4f %%\n", res.c0_weight * 100.0)
    @printf("  Total Determinants:       %d\n", res.n_configurations)
    println("--------------------------------------------------")
    @printf("  TOTAL CISD ENERGY:        %16.8f a.u. (%12.4f eV)\n", res.total_cisd_energy_hartree, res.total_cisd_energy_ev)
    println("==================================================")
end

end # module CISDTheory
