# ==============================================================================
# Example 09: Excited States Calculation (CIS) and UV-Vis Spectroscopy
# ==============================================================================

using QuantumChemMM
using Printf

# Formaldehyde H2C=O
formaldehyde_xyz = """4
Formaldehyde CH2O
C    0.000000    0.000000   -0.597000
O    0.000000    0.000000    0.607000
H    0.000000    0.938000   -1.181000
H    0.000000   -0.938000   -1.181000
"""

mol = parse_xyz(formaldehyde_xyz; name="Formaldehyde")
println("1. Converging Ground-State SCF (CNDO/2)...")
scf_res = solve_cndo2(mol; options=SCFOptions(max_cycles=40))
println("Ground State Energy: ", round(scf_res.total_energy, digits=5), " eV")

println("\n2. Solving Configuration Interaction Singles (CIS)...")
cis_res = solve_cis(scf_res, mol; n_states=5, calc_triplets=true)

println("\n=== Singlet Excited States (S1 - S$(length(cis_res.singlets))) ===")
println(" State  Excitation (eV)  Wavelength (nm)  Osc. Strength (f)  Leading Transition")
for st in cis_res.singlets
    lbl = isempty(st.leading_configurations) ? "" : st.leading_configurations[1][4]
    @printf("   S%d       %8.4f          %7.2f            %8.4f       %s\n",
            st.state_number, st.excitation_energy_ev, st.wavelength_nm, st.oscillator_strength, lbl)
end

println("\n=== Triplet Excited States (T1 - T$(length(cis_res.triplets))) ===")
println(" State  Excitation (eV)  Wavelength (nm)  Osc. Strength (f)  Leading Transition")
for st in cis_res.triplets
    lbl = isempty(st.leading_configurations) ? "" : st.leading_configurations[1][4]
    @printf("   T%d       %8.4f          %7.2f            %8.4f       %s\n",
            st.state_number, st.excitation_energy_ev, st.wavelength_nm, st.oscillator_strength, lbl)
end

println("\n3. Generating Simulated Continuous UV-Vis Spectrum Curve...")
singlet_transitions = uv_vis_transitions(cis_res)
wavelengths, absorbance = generate_uv_vis_spectrum(singlet_transitions; lambda_min=50.0, lambda_max=400.0, fwhm_nm=15.0)
peak_idx = argmax(absorbance)
println("Maximum UV-Vis Absorption Peak (λ_max): ", round(wavelengths[peak_idx], digits=2), " nm")
println("Peak Molar Absorptivity ε: ", round(absorbance[peak_idx], digits=1), " L/(mol·cm)")

write_output("example_09_excited_states.out",
             (ground_state=scf_res, cis=cis_res,
              uv_vis_wavelengths_nm=wavelengths,
              uv_vis_absorbance=absorbance,
              maximum_peak_nm=wavelengths[peak_idx],
              peak_absorptivity=absorbance[peak_idx]);
             title="Formaldehyde Excited-State and UV-Vis Results")
