# ==============================================================================
# Example 02: Extended Hückel Theory (EHT) on Water (H2O)
# ==============================================================================

using QuantumChemMM

water_xyz = """3
Water molecule H2O
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
"""

println("Parsing Water molecule...")
mol = parse_xyz(water_xyz; name="Water")

println("Running Extended Hückel calculation (Wolfsberg-Helmholz K=1.75)...")
eht_res = solve_extended_huckel(mol)

println("\n--- Extended Hückel Results ---")
println("Total Valence Electronic Energy: ", round(eht_res.total_energy_ev, digits=4), " eV")
println("\nMolecular Orbital Energies (eV):")
for (i, eps) in enumerate(eht_res.mo_energies_ev)
    tag = (i == 4) ? " [HOMO]" : (i == 5) ? " [LUMO]" : ""
    println("  MO #$i: ", round(eps, digits=4), " eV", tag)
end

println("\nMulliken Net Atomic Charges (e):")
for (i, q) in enumerate(eht_res.mulliken_charges)
    println("  Atom $i ($(mol.atoms[i].symbol)): ", round(q, digits=4))
end

println("\nMulliken Overlap Populations (Bond Orders):")
for ((a, b), pop) in eht_res.overlap_populations
    println("  Atoms ($a - $b): ", round(pop, digits=4))
end

write_output("example_02_extended_huckel.out", eht_res;
             title="Water Extended Hückel Results")
