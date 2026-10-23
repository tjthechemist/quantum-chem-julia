# ==============================================================================
# Example 08: Multi-Vendor GPU Acceleration (NVIDIA, AMD, Intel)
# Hardware Acceleration Across Molecular Mechanics, Semi-Empirical,
# Ab Initio Hartree-Fock, Kohn-Sham DFT, Post-HF MP2/CCSD, and Excited States CIS
# ==============================================================================

using QuantumChemMM
using LinearAlgebra
using Printf

println("====================================================================")
println("      Multi-Vendor GPU Acceleration in QuantumChemMM.jl             ")
println("      Supported: NVIDIA (CUDA), AMD (ROCm), Intel (oneAPI)          ")
println("====================================================================")

# 1. Device Detection & Status
available_devices = detect_available_devices()
println("\n[1] Detected Available Computational Devices:")
for (idx, dev) in enumerate(available_devices)
    println("  [$idx] $(device_name(dev)) | Vendor: :$(vendor(dev)) | Is GPU: $(is_gpu(dev))")
end

# 2. Molecular System Setup
water_xyz = """3
Water H2O
O   0.000000    0.000000    0.117300
H   0.000000    0.757200   -0.469200
H   0.000000   -0.757200   -0.469200
"""
mol = parse_xyz(water_xyz; name="Water")
println("\nMolecule: $(mol.name) with $(n_atoms(mol)) atoms.")

# 3. Select an actually available device. Optional GPU backends must be loaded
# by the Julia session before detect_available_devices() can report them.
dev_cpu = CPUDevice()
active_dev = length(available_devices) > 1 ? available_devices[2] : dev_cpu
println("\nCurrently executing benchmarks with: $(device_name(active_dev))")
is_gpu(active_dev) ||
    println("No loaded GPU backend detected; GPU solver calls will use CPU fallback.")

# 4. GPU-Accelerated Molecular Mechanics & MD
println("\n--------------------------------------------------------------------")
println(" [A] Molecular Mechanics & Molecular Dynamics on GPU")
println("--------------------------------------------------------------------")
uff = parameterize_uff(mol)
ec_dev, grad_dev = evaluate_mm_gpu(uff, mol; device=active_dev)
@printf("  UFF Non-Bonded Energy on %s: %12.4f kcal/mol\n", vendor(active_dev), total_energy(ec_dev))

# 5. GPU-Accelerated Semi-Empirical SCF (CNDO/2)
println("\n--------------------------------------------------------------------")
println(" [B] Semi-Empirical CNDO/2 SCF on GPU")
println("--------------------------------------------------------------------")
cndo_res = solve_scf_gpu(mol, :cndo2; device=active_dev)
@printf("  CNDO/2 Total Energy on %s: %12.6f eV\n", vendor(active_dev), cndo_res.total_energy)

# 6. GPU-Accelerated Ab Initio Hartree-Fock (RHF)
println("\n--------------------------------------------------------------------")
println(" [C] Ab Initio Hartree-Fock (RHF) with GPU Fock Matrix Construction")
println("--------------------------------------------------------------------")
# High-throughput GEMV tensor contraction builds Coulomb (J) and Exchange (K) matrices
hf_res = solve_hf_gpu(mol; basis_name="sto-3g", device=active_dev)
@printf("  Hartree-Fock Energy: %16.8f a.u. (Converged: %s in %d cycles)\n",
        hf_res.total_energy_hartree, hf_res.converged, hf_res.cycles)

# 7. GPU-Accelerated Kohn-Sham Density Functional Theory (B3LYP & PBE)
println("\n--------------------------------------------------------------------")
println(" [D] Kohn-Sham DFT (B3LYP) with GPU Numerical Grid & GEMV Fock Build")
println("--------------------------------------------------------------------")
# Electron density on quadrature grid and V_xc matrix formed via GPU matrix multiplications
dft_res = solve_dft_gpu(mol; functional=:b3lyp, basis_name="sto-3g", device=active_dev)
@printf("  B3LYP Total Energy:  %16.8f a.u. (XC: %10.4f a.u.)\n",
        dft_res.total_energy_hartree, dft_res.xc_energy_hartree)

# 8. GPU-Accelerated Post-Ab Initio Methods (MP2, CISD, CCSD)
println("\n--------------------------------------------------------------------")
println(" [E] Post-HF Correlation with GPU 4-Index AO-to-MO Transformation")
println("--------------------------------------------------------------------")
# The O(N^5) four quarter-transformations execute as 4 consecutive GPU GEMMs
mp2_res = solve_mp2_gpu(hf_res; device=active_dev)
@printf("  MP2 Total Energy:    %16.8f a.u. (Correlation: %10.6f a.u.)\n",
        mp2_res.total_mp2_energy_hartree, mp2_res.mp2_correlation_hartree)

ccsd_res = solve_ccsd_gpu(hf_res; device=active_dev, compute_triples=true)
@printf("  CCSD Total Energy:   %16.8f a.u.\n", ccsd_res.total_ccsd_energy_hartree)
@printf("  CCSD(T) Energy:      %16.8f a.u. (Triples: %10.6f a.u.)\n",
        ccsd_res.total_ccsdt_energy_hartree, ccsd_res.triples_correction_hartree)

# 9. GPU-Accelerated Excited States (CIS)
println("\n--------------------------------------------------------------------")
println(" [F] Configuration Interaction Singles (CIS) on GPU")
println("--------------------------------------------------------------------")
cis_res = solve_cis_gpu(cndo_res, mol; device=active_dev, n_states=3)
println("  Excited States (Singlets):")
for st in cis_res.singlets
    @printf("    S%d: %6.2f eV (%5.1f nm) | f_osc = %6.4f\n",
            st.state_number, st.excitation_energy_ev, st.wavelength_nm, st.oscillator_strength)
end

# 10. Unified High-Level Dispatcher with device= Parameter
println("\n--------------------------------------------------------------------")
println(" [G] Unified calculate() API with device= Keyword")
println("--------------------------------------------------------------------")
res_unified_hf  = calculate(mol, :hf; device=active_dev)
res_unified_dft = calculate(mol, :b3lyp; device=active_dev)
res_unified_mp2 = calculate(mol, :mp2; device=active_dev)

@printf("  calculate(mol, :hf; device=%s)   -> %12.6f Hartree\n", device_name(active_dev), res_unified_hf.total_energy_hartree)
@printf("  calculate(mol, :b3lyp; device=%s) -> %12.6f Hartree\n", device_name(active_dev), res_unified_dft.total_energy_hartree)
@printf("  calculate(mol, :mp2; device=%s) -> %12.6f Hartree\n", device_name(active_dev), res_unified_mp2.total_energy_hartree)

println("\n====================================================================")
println("  GPU acceleration is used when a supported backend is available.  ")
println("  Otherwise, calculations run on the CPU fallback.                  ")
println("====================================================================")

write_output("example_08_gpu_acceleration.out",
             (available_devices=available_devices, device=active_dev,
              uff=ec_dev, cndo2=cndo_res, hartree_fock=hf_res,
              b3lyp=dft_res, mp2=mp2_res, ccsd=ccsd_res, cis=cis_res,
              unified_hf=res_unified_hf, unified_b3lyp=res_unified_dft,
              unified_mp2=res_unified_mp2);
             title="GPU Acceleration Example Results")
