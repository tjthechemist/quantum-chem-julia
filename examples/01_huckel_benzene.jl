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

# Save the solver result as a readable plain-text report. The extension is
# caller-selected; the report contents remain plain text.
write_output("benzene_huckel.out", result; title="Benzene Hückel Results")

# Print Summary
println("\n--- Hückel Results ---")
println("Total Pi Energy: ", round(result.total_pi_energy_beta, digits=4), " β (",
        round(result.total_pi_energy_ev, digits=3), " eV)")
println("Delocalization (Aromatic Resonance) Energy: ",
        round(result.delocalization_energy_beta, digits=4), " β (expected: 2.0000 β)")
println("\nOrbital Energies (m where E = α + m*β):")
for (i, m) in enumerate(result.orbital_energies_m)
    println("  MO #$i: m = $(round(m, digits=4)) β  | Occ: $(result.occupations[i])")
end

println("\nCoulson Pi Bond Orders:")
for ((i, j), bo) in result.bond_orders
    println("  Bond ($i - $j): p = $(round(bo, digits=4)) (Total order = $(round(1.0 + bo, digits=4)))")
end
