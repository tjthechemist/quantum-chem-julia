# ==============================================================================
# Example 01: Simple Hückel Theory (HMO) Calculation on Benzene
# ==============================================================================

using QuantumChemMM

# Benzene planar geometry in XYZ format
benzene_xyz = """12
Benzene molecule C6H6
C    1.3970    0.0000    0.0000
C    0.6985    1.2100    0.0000
C   -0.6985    1.2100    0.0000
C   -1.3970    0.0000    0.0000
C   -0.6985   -1.2100    0.0000
C    0.6985   -1.2100    0.0000
H    2.4810    0.0000    0.0000
H    1.2405    2.1486    0.0000
H   -1.2405    2.1486    0.0000
H   -2.4810    0.0000    0.0000
H   -0.6985   -2.1486    0.0000
H    1.2405   -2.1486    0.0000
"""

println("Parsing Benzene structure...")
mol = parse_xyz(benzene_xyz; name="Benzene")
println("Atoms: ", n_atoms(mol), " | Detected Bonds: ", n_bonds(mol))

# Run Simple Hückel calculation
model = HuckelModel(alpha0_ev=-11.26, beta0_ev=-2.70)
result = solve_huckel(mol, model)

report_title = "Benzene Hückel Results"
report = format_output(result; title=report_title)
write_output("benzene_huckel.out", result; title=report_title)
println(report)
