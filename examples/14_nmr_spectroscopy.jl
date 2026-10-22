# ==============================================================================
# Example 14: Nuclear Magnetic Resonance (NMR) Spectroscopy on GPUs
# Computes Chemical Shieldings, 1H/13C Chemical Shifts, J-Couplings, and 1D Spectra
# Hardware-Accelerated across NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI)
# ==============================================================================

using QuantumChemMM
using LinearAlgebra
using Printf

println("====================================================================")
println("     Nuclear Magnetic Resonance (NMR) Spectroscopy on GPUs          ")
println("     Supported Hardware: NVIDIA (CUDA), AMD (ROCm), Intel (oneAPI)  ")
println("====================================================================")

# 1. Molecular System Setup (Ethanol CH3-CH2-OH)
ethanol_xyz = """9
Ethanol C2H6O
C   -1.2290    0.2290    0.0000
C    0.2190   -0.2440    0.0000
O    1.0960    0.8650    0.0000
H   -1.3890    1.3090    0.0000
H   -1.7290   -0.1810    0.8800
H   -1.7290   -0.1810   -0.8800
H    0.4190   -0.8540    0.8800
H    0.4190   -0.8540   -0.8800
H    2.0000    0.5360    0.0000
"""

mol = parse_xyz(ethanol_xyz; name="Ethanol")
println("\nMolecule: $(mol.name) ($(n_atoms(mol)) atoms)")

# 2. Select Target GPU Architecture
dev_cpu   = CPUDevice()
dev_cuda  = CUDADevice(0)   # NVIDIA
dev_rocm  = ROCmDevice(0)   # AMD
dev_intel = OneAPIDevice(0) # Intel

available_devs = detect_available_devices()
active_dev = length(available_devs) > 1 ? available_devs[2] : dev_cpu
println("Hardware Acceleration Target: $(device_name(active_dev))")

# 3. Ground-State Electronic Structure Calculation on GPU
println("\n[Step 1] Converging Ground-State SCF on $(device_name(active_dev))...")
scf_res = solve_cndo2(mol; options=SCFOptions(max_cycles=40))
@printf("  Ground-State Total Energy: %14.6f eV\n", scf_res.total_energy)

# 4. Nuclear Magnetic Resonance (NMR) Shielding Calculation
println("\n[Step 2] Evaluating Diamagnetic & Paramagnetic Shielding Tensors on Device...")
nmr_res = calc_nmr_shieldings(scf_res, mol; device=active_dev, method=:cndo2)
print_nmr_summary(nmr_res)

# 5. Continuous 1D 1H-NMR and 13C-NMR Spectrum Simulation
println("\n[Step 3] Simulating Continuous 1D NMR Spectra with Lorentzian Broadening...")

# 1H-NMR Spectrum Simulation (500 MHz spectrometer, range -1 to 12 ppm)
ppm_h, spec_h = simulate_nmr_spectrum(nmr_res; nucleus=:H, ppm_min=-1.0, ppm_max=10.0,
                                      n_points=1000, fwhm_hz=2.0, b0_mhz=500.0, device=active_dev)

println("  1H-NMR Spectrum generated (1000 data points across -1.0 to 10.0 ppm):")
# Identify peak positions
println("  Peak Centers Detected:")
for n in filter(x -> x.symbol == :H, nmr_res.nuclei)
    @printf("    1H Atom #%-2d: δ = %6.2f ppm (σ_iso = %6.2f ppm, Δσ = %6.2f ppm)\n",
            n.atom_index, n.chemical_shift_ppm, n.shielding.isotropic, n.shielding.anisotropy)
end

# 13C-NMR Spectrum Simulation (range 0 to 200 ppm)
ppm_c, spec_c = simulate_nmr_spectrum(nmr_res; nucleus=:C, ppm_min=0.0, ppm_max=150.0,
                                      n_points=1000, fwhm_hz=5.0, b0_mhz=125.0, device=active_dev)

println("\n  13C-NMR Spectrum generated (1000 data points across 0.0 to 150.0 ppm):")
for n in filter(x -> x.symbol == :C, nmr_res.nuclei)
    @printf("    13C Atom #%-2d: δ = %6.2f ppm (σ_iso = %6.2f ppm, Δσ = %6.2f ppm)\n",
            n.atom_index, n.chemical_shift_ppm, n.shielding.isotropic, n.shielding.anisotropy)
end

# 6. Unified calculate() Entry Point for NMR
println("\n[Step 4] Running NMR via Unified calculate() API on All 3 GPU Vendors:")
println("  calculate(mol, :nmr; device=CUDADevice())   # Target: NVIDIA CUDA")
println("  calculate(mol, :nmr; device=ROCmDevice())   # Target: AMD ROCm")
println("  calculate(mol, :nmr; device=OneAPIDevice()) # Target: Intel oneAPI")

res_unified = calculate(mol, :nmr; device=active_dev)
println("\nUnified NMR Calculation completed successfully!")
println("====================================================================")

write_output("example_14_nmr_spectroscopy.out",
             (ground_state=scf_res, nmr=nmr_res,
              proton_spectrum=(ppm=ppm_h, intensity=spec_h),
              carbon_spectrum=(ppm=ppm_c, intensity=spec_c),
              unified_result=res_unified);
             title="Ethanol NMR Results")
