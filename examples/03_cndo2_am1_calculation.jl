# ==============================================================================
# Example 03: CNDO/2 and AM1 / PM3 SCF Calculations with DIIS
# ==============================================================================

using QuantumChemMM
using LinearAlgebra

# Formaldehyde H2C=O
formaldehyde_xyz = """4
Formaldehyde CH2O
C    0.000000    0.000000   -0.597000
O    0.000000    0.000000    0.607000
H    0.000000    0.938000   -1.181000
H    0.000000   -0.938000   -1.181000
"""

mol = parse_xyz(formaldehyde_xyz; name="Formaldehyde")

# 1. CNDO/2 SCF Calculation
println("=== 1. CNDO/2 SCF Calculation with DIIS ===")
opt = SCFOptions(max_cycles=50, convergence_threshold=1e-6, use_diis=true)
cndo_res = solve_cndo2(mol; options=opt)

println("Converged in: ", cndo_res.cycles, " iterations")
println("Total Energy: ", round(cndo_res.total_energy, digits=5), " eV")
println("Dipole Moment: ", round(norm(cndo_res.dipole_moment), digits=3), " Debye")
println("Mulliken Charges:")
for (i, at) in enumerate(mol.atoms)
    println("  Atom $i ($(at.symbol)): ", round(cndo_res.mulliken_charges[i], digits=4))
end

# 2. AM1 NDDO SCF Calculation
println("\n=== 2. AM1 NDDO SCF Calculation ===")
am1_res = solve_nddo(mol, NDDOMethod(:am1); options=opt)

println("Converged in: ", am1_res.cycles, " iterations")
println("Total Electronic Energy: ", round(am1_res.electronic_energy_ev, digits=5), " eV")
println("Nuclear Repulsion:       ", round(am1_res.nuclear_repulsion_ev, digits=5), " eV")
println("Total Energy:            ", round(am1_res.total_energy_ev, digits=5), " eV")
println("Heat of Formation (ΔHf): ", round(am1_res.heat_of_formation_kcal, digits=3), " kcal/mol")
println("Dipole Moment:           ", round(norm(am1_res.dipole_moment), digits=3), " Debye")

# 3. PM3 NDDO SCF Calculation
println("\n=== 3. PM3 NDDO SCF Calculation ===")
pm3_res = solve_nddo(mol, NDDOMethod(:pm3); options=opt)
println("Total Energy (PM3):      ", round(pm3_res.total_energy_ev, digits=5), " eV")
println("Heat of Formation (PM3): ", round(pm3_res.heat_of_formation_kcal, digits=3), " kcal/mol")

