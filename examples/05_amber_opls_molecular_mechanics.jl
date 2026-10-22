# ==============================================================================
# Example 05: AMBER, CHARMM, and OPLS-AA Force Field Comparisons
# ==============================================================================

using QuantumChemMM

ethanol_xyz = """9
Ethanol C2H5OH
C   -1.2180    0.2810    0.0000
C    0.0000   -0.6270    0.0000
O    1.1870    0.1650    0.0000
H   -1.2840    0.9160    0.8840
H   -1.2840    0.9160   -0.8840
H   -2.0740   -0.3840    0.0000
H    0.0660   -1.2620    0.8840
H    0.0660   -1.2620   -0.8840
H    1.9540   -0.4180    0.0000
"""

mol = parse_xyz(ethanol_xyz; name="Ethanol")

println("=== AMBER / GAFF Evaluation ===")
amber_ff = parameterize_amber(mol)
ec_amber, _ = evaluate_mm(amber_ff, mol)
print_summary(ec_amber)

println("\n=== OPLS-AA Evaluation ===")
opls_ff = parameterize_opls(mol)
ec_opls, _ = evaluate_mm(opls_ff, mol)
print_summary(ec_opls)

println("\n=== CHARMM Evaluation (with Urey-Bradley) ===")
charmm_ff = parameterize_charmm(mol)
ec_charmm, _ = evaluate_mm(charmm_ff, mol)
print_summary(ec_charmm)

println("\n=== Allinger MM3 Class II Evaluation (with Stretch-Bend) ===")
mm3_ff = parameterize_mm3(mol)
ec_mm3, _ = evaluate_mm(mm3_ff, mol)
print_summary(ec_mm3)

write_output("example_05_force_field_comparison.out",
             (amber=ec_amber, opls=ec_opls, charmm=ec_charmm, mm3=ec_mm3);
             title="Ethanol Force-Field Comparison")
