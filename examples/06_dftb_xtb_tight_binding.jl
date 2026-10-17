# ==============================================================================
# Example 06: Tight-Binding (SCC-DFTB & GFN-xTB) with D3 Dispersion
# ==============================================================================

using QuantumChemMM

methane_xyz = """5
Methane CH4
C    0.0000    0.0000    0.0000
H    0.6276    0.6276    0.6276
H   -0.6276   -0.6276    0.6276
H   -0.6276    0.6276   -0.6276
H    0.6276   -0.6276   -0.6276
"""

mol = parse_xyz(methane_xyz; name="Methane")

# 1. GFN2-xTB with Grimme D3(BJ) dispersion
println("=== 1. GFN2-xTB Calculation ===")
tb_res = solve_tight_binding(mol, TightBindingModel(:gfn2_xtb; include_dispersion=true))

println("Status:                  ", tb_res.converged ? "Converged" : "Failed", " (", tb_res.cycles, " cycles)")
println("Band Energy:             ", round(tb_res.band_energy_ev, digits=4), " eV")
println("Charge Fluctuation (E_q):", round(tb_res.charge_energy_ev, digits=4), " eV")
println("Repulsive Energy:        ", round(tb_res.repulsion_energy_ev, digits=4), " eV")
println("D3 Dispersion:           ", round(tb_res.dispersion_energy_ev, digits=6), " eV")
println("Total Energy:            ", round(tb_res.total_energy_ev, digits=4), " eV")

println("\nMulliken Net Charges:")
for (i, at) in enumerate(mol.atoms)
    println("  Atom $i ($(at.symbol)): ", round(tb_res.mulliken_charges[i], digits=4))
end

# 2. Standalone Grimme D3 Dispersion Evaluation
println("\n=== 2. Standalone Grimme D3(BJ) Dispersion ===")
d3_kcal, d3_grad = calc_d3_dispersion(mol)
println("Dispersion Energy: ", round(d3_kcal, digits=6), " kcal/mol")

