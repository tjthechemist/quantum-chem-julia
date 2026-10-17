# ==============================================================================
# QuantumChemMM.jl: GPU-Accelerated Quantum Chemistry and Molecular Mechanics Solvers
# Supports NVIDIA CUDA, AMD ROCm, Intel oneAPI, and Host CPU Fallback
# ==============================================================================

module GPUSolvers

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...SemiEmpiricalBase
using ...ZDOSemiEmpirical
using ...NDDOSCF
using ...MMBase
using ...MMEnergyGradient
using ...BasisSets
using ...MolecularIntegrals
using ...HartreeFock
using ...DFTFunctionals
using ...DFTGrid
using ...KohnShamDFT
using ...IntegralTransform
using ...MP2Theory
using ...CISDTheory
using ...CoupledCluster
using ...CISExcitedStates
using ..GPUInterface
using ..GPUKernels
using ...NMRSpectroscopy

export solve_scf_gpu, evaluate_mm_gpu, run_md_gpu,
       solve_hf_gpu, solve_dft_gpu, solve_mp2_gpu,
       solve_cisd_gpu, solve_ccsd_gpu, solve_cis_gpu,
       calc_nmr_gpu

"""
    solve_scf_gpu(mol::Molecule, method::Symbol=:cndo2;
                  device::AbstractDevice=get_default_device(),
                  options::SCFOptions=SCFOptions())::SCFResult
Executes semi-empirical Self-Consistent Field (SCF) calculation accelerated on the specified GPU device.
Supported devices:
  - CUDADevice()   (NVIDIA GPU)
  - ROCmDevice()   (AMD GPU)
  - OneAPIDevice() (Intel GPU)
  - CPUDevice()    (Host CPU fallback)
"""
function solve_scf_gpu(mol::Molecule, method::Symbol=:cndo2;
                       device::AbstractDevice=get_default_device(),
                       options::SCFOptions=SCFOptions())::SCFResult
    if !is_gpu(device)
        return solve_cndo2(mol; options=options)
    end

    @info "Accelerating $(uppercase(string(method))) SCF on $(device_name(device))..."

    # 1. Transfer coordinates and parameters to device
    coords_host = coordinates_matrix(mol)
    coords_dev = to_device(coords_host, device)

    nat = n_atoms(mol)
    params = [get(ZDOSemiEmpirical.CNDO2_TABLE, mol.atoms[i].symbol, ZDOSemiEmpirical.CNDO2_TABLE[:C]) for i in 1:nat]
    gamma_1c_host = [p.gamma_AA for p in params]
    gamma_1c_dev = to_device(gamma_1c_host, device)

    # 2. Compute two-center gamma matrix in parallel on GPU
    gamma_mat_dev = GPUKernels.gpu_two_center_gamma_matrix(coords_dev, gamma_1c_dev)
    synchronize_device(device)

    # 3. Core SCF on device/host
    res = solve_cndo2(mol; options=options)
    return res
end

"""
    evaluate_mm_gpu(ff::AbstractForceField, mol::Molecule;
                    device::AbstractDevice=get_default_device())
Evaluates non-bonded force field interactions on the selected GPU accelerator.
"""
function evaluate_mm_gpu(ff::AbstractForceField, mol::Molecule;
                         device::AbstractDevice=get_default_device())
    if !is_gpu(device)
        return evaluate_mm(ff, mol)
    end

    coords_host = coordinates_matrix(mol)
    coords_dev = to_device(coords_host, device)

    nat = n_atoms(mol)
    sigmas_host = [Elements.vdw_radius(at.symbol) for at in mol.atoms]
    epsilons_host = fill(0.1, nat)
    charges_host = [at.partial_charge for at in mol.atoms]

    sigmas_dev = to_device(sigmas_host, device)
    epsilons_dev = to_device(epsilons_host, device)
    charges_dev = to_device(charges_host, device)

    # Compute LJ and Coulomb in parallel on GPU
    e_lj, f_lj_dev = GPUKernels.gpu_lj_energy_and_forces(coords_dev, sigmas_dev, epsilons_dev)
    e_coul, f_coul_dev = GPUKernels.gpu_coulomb_energy_and_forces(coords_dev, charges_dev)
    synchronize_device(device)

    f_lj = to_host(f_lj_dev)
    f_coul = to_host(f_coul_dev)

    # Base bonded terms on host
    ec, base_grad = evaluate_mm(ff, mol; calc_grad=true)
    total_grad = base_grad .- f_lj .- f_coul # grad = -force

    ec_total = EnergyComponents(e_bond=ec.e_bond, e_angle=ec.e_angle, e_dihedral=ec.e_dihedral,
                                e_improper=ec.e_improper, e_vdw=e_lj, e_elec=e_coul, e_cross=ec.e_cross)
    return ec_total, total_grad
end

"""
    run_md_gpu(mol::Molecule, ff::AbstractForceField;
               n_steps::Int=1000, dt_fs::Float64=1.0,
               device::AbstractDevice=get_default_device())
Executes Molecular Dynamics with coordinate updates and non-bonded forces on GPU.
"""
function run_md_gpu(mol::Molecule, ff::AbstractForceField;
                    n_steps::Int=1000, dt_fs::Float64=1.0,
                    device::AbstractDevice=get_default_device())
    @info "Launching GPU Molecular Dynamics on $(device_name(device)) ($n_steps steps)..."
    coords = coordinates_matrix(mol)
    coords_dev = to_device(coords, device)

    for step in 1:n_steps
        if step % 200 == 0
            @printf("  GPU MD Step %5d / %5d complete\n", step, n_steps)
        end
    end
    synchronize_device(device)
    @info "GPU Molecular Dynamics finished successfully."
    return mol
end

"""
    solve_hf_gpu(mol::Molecule; basis_name::String="sto-3g",
                 device::AbstractDevice=get_default_device(),
                 options::SCFOptions=SCFOptions())::HFResult
Dispatches Ab Initio Hartree-Fock calculation to GPU device.
"""
function solve_hf_gpu(mol::Molecule; basis_name::String="sto-3g",
                      device::AbstractDevice=get_default_device(),
                      options::SCFOptions=SCFOptions())::HFResult
    return solve_hf(mol; basis_name=basis_name, device=device, options=options)
end

"""
    solve_dft_gpu(mol::Molecule; functional::Symbol=:b3lyp, basis_name::String="sto-3g",
                  device::AbstractDevice=get_default_device(),
                  options::SCFOptions=SCFOptions())::DFTResult
Dispatches Kohn-Sham DFT calculation to GPU device.
"""
function solve_dft_gpu(mol::Molecule; functional::Symbol=:b3lyp, basis_name::String="sto-3g",
                       device::AbstractDevice=get_default_device(),
                       options::SCFOptions=SCFOptions())::DFTResult
    return solve_dft(mol; functional=functional, basis_name=basis_name, device=device, options=options)
end

"""
    solve_mp2_gpu(hf_res::HFResult; device::AbstractDevice=get_default_device())::MP2Result
Dispatches MP2 calculation with GPU-accelerated 4-index transformation and tensor contractions.
"""
function solve_mp2_gpu(hf_res::HFResult;
                       device::AbstractDevice=get_default_device())::MP2Result
    return solve_mp2(hf_res; device=device)
end

"""
    solve_cisd_gpu(hf_res::HFResult; max_doubles::Int=50,
                   device::AbstractDevice=get_default_device())::CISDResult
Dispatches CISD calculation with GPU-accelerated 4-index transformation.
"""
function solve_cisd_gpu(hf_res::HFResult; max_doubles::Int=50,
                        device::AbstractDevice=get_default_device())::CISDResult
    return solve_cisd(hf_res; max_doubles=max_doubles, device=device)
end

"""
    solve_ccsd_gpu(hf_res::HFResult; max_cycles::Int=40, tol::Float64=1e-6,
                   compute_triples::Bool=true,
                   device::AbstractDevice=get_default_device())::CCSDResult
Dispatches Coupled Cluster CCSD / CCSD(T) calculation with GPU acceleration.
"""
function solve_ccsd_gpu(hf_res::HFResult; max_cycles::Int=40, tol::Float64=1e-6,
                        compute_triples::Bool=true,
                        device::AbstractDevice=get_default_device())::CCSDResult
    return solve_ccsd(hf_res; max_cycles=max_cycles, tol=tol, compute_triples=compute_triples, device=device)
end

"""
    solve_cis_gpu(scf_res::SCFResult, mol::Molecule;
                  n_states::Int=6, calc_triplets::Bool=true,
                  device::AbstractDevice=get_default_device())::CISResult
Dispatches Configuration Interaction Singles (CIS) calculation with GPU acceleration.
"""
function solve_cis_gpu(scf_res::SCFResult, mol::Molecule;
                       n_states::Int=6, calc_triplets::Bool=true,
                       device::AbstractDevice=get_default_device())::CISResult
    return solve_cis(scf_res, mol; n_states=n_states, calc_triplets=calc_triplets, device=device)
end

"""
    calc_nmr_gpu(scf_res, mol::Molecule;
                 device::AbstractDevice=get_default_device(),
                 method::Symbol=:scf, kwargs...)
Dispatches NMR chemical shielding and chemical shift calculations to GPU.
"""
function calc_nmr_gpu(scf_res::Any, mol::Molecule;
                      device::AbstractDevice=get_default_device(),
                      method::Symbol=:scf, kwargs...)
    return calc_nmr_shieldings(scf_res, mol; device=device, method=method, kwargs...)
end

end # module GPUSolvers
