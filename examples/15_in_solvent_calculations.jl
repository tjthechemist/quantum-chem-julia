# ==============================================================================
# Example 15: In-Solvent Calculations Across All Methods
# Demonstrates implicit COSMO/Generalized-Born solvation for every method family
# Solvents supported: :water, :acetonitrile, :dmso, :methanol, :ethanol,
#                     :chloroform, :dichloromethane, :toluene, :hexane, :octanol
# ==============================================================================

include("../src/QuantumChemMM.jl")
using .QuantumChemMM
using Printf

# ── Molecule: Formaldehyde (H2C=O) — small polar molecule ────────────────────
ch2o_xyz = """4
Formaldehyde
C    0.000000    0.000000   -0.597000
O    0.000000    0.000000    0.607000
H    0.000000    0.938000   -1.181000
H    0.000000   -0.938000   -1.181000
"""

mol = parse_xyz(ch2o_xyz; name="Formaldehyde")
println("Molecule: $(mol.name)  ($(n_atoms(mol)) atoms)\n")

# ==============================================================================
# 1. Standalone solvation model: compare all available solvents
# ==============================================================================
println("=" ^ 65)
println("  SOLVATION FREE ENERGY ACROSS SOLVENTS (COSMO/GB model)")
println("=" ^ 65)
@printf("  %-22s  %8s  %10s  %10s\n", "Solvent", "ε", "ΔG_elec", "ΔG_solv (kcal/mol)")
println("-" ^ 65)

solvents = [:water, :dmso, :acetonitrile, :methanol, :ethanol,
            :chloroform, :dichloromethane, :toluene, :hexane]

for sym in solvents
    sv = calc_solvation_energy(mol, sym)
    @printf("  %-22s  %8.2f  %10.4f  %10.4f\n",
            sv.solvent_name, sv.dielectric,
            sv.electrostatic_solv_kcal, sv.total_delta_g_solv_kcal)
end
println("=" ^ 65)
println()

# ==============================================================================
# 2. Molecular Mechanics in solvent
# ==============================================================================
println("── Molecular Mechanics (UFF) ─────────────────────────────────────────")
res_mm_vac = calculate(mol, :uff; solvent=:vacuum)
res_mm_aq  = calculate(mol, :uff; solvent=:water)
@printf("  Gas phase:    %12.4f kcal/mol\n", res_mm_vac.total_energy_kcal)
@printf("  Water:        %12.4f kcal/mol  (ΔG_solv = %+.4f kcal/mol)\n",
        res_mm_aq.total_energy_solv_kcal,
        res_mm_aq.solvation.total_delta_g_solv_kcal)
println()

# ==============================================================================
# 3. Semi-Empirical methods in solvent
# ==============================================================================
println("── Semi-Empirical Methods ────────────────────────────────────────────")
for method in [:cndo2, :am1, :pm3]
    res_vac = calculate(mol, method; solvent=:vacuum)
    res_aq  = calculate(mol, method; solvent=:water)
    @printf("  %-8s  gas: %10.4f eV  |  water: %10.4f eV  (ΔG_solv = %+.4f kcal/mol)\n",
            uppercase(string(method)),
            res_vac.total_energy_ev,
            res_aq.total_energy_solv_ev,
            res_aq.solvation.total_delta_g_solv_kcal)
end
println()

# ==============================================================================
# 4. Hartree-Fock in multiple solvents
# ==============================================================================
println("── Hartree-Fock (STO-3G) ─────────────────────────────────────────────")
res_hf_vac  = calculate(mol, :hf; basis="sto-3g", solvent=:vacuum)
res_hf_aq   = calculate(mol, :hf; basis="sto-3g", solvent=:water)
res_hf_dmso = calculate(mol, :hf; basis="sto-3g", solvent=:dmso)
res_hf_chl  = calculate(mol, :hf; basis="sto-3g", solvent=:chloroform)

@printf("  Gas phase:    %14.6f Hartree\n",    res_hf_vac.total_energy_hartree)
@printf("  Water:        %14.6f Hartree  (ΔG_solv = %+.4f kcal/mol)\n",
        res_hf_aq.total_energy_solv_hartree,
        res_hf_aq.solvation.total_delta_g_solv_kcal)
@printf("  DMSO:         %14.6f Hartree  (ΔG_solv = %+.4f kcal/mol)\n",
        res_hf_dmso.total_energy_solv_hartree,
        res_hf_dmso.solvation.total_delta_g_solv_kcal)
@printf("  Chloroform:   %14.6f Hartree  (ΔG_solv = %+.4f kcal/mol)\n",
        res_hf_chl.total_energy_solv_hartree,
        res_hf_chl.solvation.total_delta_g_solv_kcal)
println()

# ==============================================================================
# 5. DFT in solvent
# ==============================================================================
println("── Density Functional Theory (B3LYP / STO-3G) ───────────────────────")
for func in [:lda, :pbe, :b3lyp]
    res_vac = calculate(mol, func; basis="sto-3g", solvent=:vacuum)
    res_aq  = calculate(mol, func; basis="sto-3g", solvent=:water)
    @printf("  %-8s  gas: %10.6f Ha  |  water: %10.6f Ha  (ΔG_solv = %+.4f kcal/mol)\n",
            uppercase(string(func)),
            res_vac.total_energy_hartree,
            res_aq.total_energy_solv_hartree,
            res_aq.solvation.total_delta_g_solv_kcal)
end
println()

# ==============================================================================
# 6. Post-HF methods in solvent
# ==============================================================================
println("── Post-Ab Initio Methods (STO-3G) ───────────────────────────────────")
for method in [:mp2, :cisd, :ccsd]
    res_vac = calculate(mol, method; basis="sto-3g", solvent=:vacuum)
    res_aq  = calculate(mol, method; basis="sto-3g", solvent=:water)
    @printf("  %-8s  gas: %10.6f Ha  |  water: %10.6f Ha  (ΔG_solv = %+.4f kcal/mol)\n",
            uppercase(string(method)),
            res_vac.total_energy_hartree,
            res_aq.total_energy_solv_hartree,
            res_aq.solvation.total_delta_g_solv_kcal)
end
println()

# ==============================================================================
# 7. NMR in solvent — HF/water reference
# ==============================================================================
println("── NMR Spectroscopy in Water ─────────────────────────────────────────")
res_nmr_vac = calculate(mol, :nmr; basis="sto-3g", solvent=:vacuum)
res_nmr_aq  = calculate(mol, :nmr; basis="sto-3g", solvent=:water)

println("  Gas phase NMR:")
print_nmr_summary(res_nmr_vac.raw_result)

println("  Aqueous NMR (same shieldings, solvated reference energy):")
@printf("  ΔG_solv correction: %+.4f kcal/mol\n",
        res_nmr_aq.solvation.total_delta_g_solv_kcal)
println()

# ==============================================================================
# 8. Detailed solvation breakdown for HF/water
# ==============================================================================
println("── Detailed Solvation Breakdown (HF/water) ───────────────────────────")
print_solvation_summary(res_hf_aq.solvation)

println("\nDone! For all methods the solvated total energy is in:")
println("  result.total_energy_solv_hartree  /  .total_energy_solv_ev  /  .total_energy_solv_kcal")
println("Gas-phase energies are always preserved in result.total_energy_* fields.")

