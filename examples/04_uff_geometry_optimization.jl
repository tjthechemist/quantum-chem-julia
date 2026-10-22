# ==============================================================================
# Example 04: UFF Force Field Setup and L-BFGS Geometry Optimization
# ==============================================================================

using QuantumChemMM
using Printf

# Strained / distorted Ethane molecule
strained_ethane_xyz = """8
Distorted Ethane C2H6
C    0.0000    0.0000    0.0000
C    1.8000    0.0000    0.0000
H   -0.5000    0.9000    0.3000
H   -0.5000   -0.9000    0.3000
H   -0.5000    0.0000   -1.0000
H    2.3000    0.9000    0.3000
H    2.3000   -0.9000    0.3000
H    2.3000    0.0000   -1.0000
"""

mol = parse_xyz(strained_ethane_xyz; name="Ethane")
println("Initial C-C distance: ", round(distance(mol.atoms[1].coords, mol.atoms[2].coords), digits=4), " Å")

# Automatically parameterize molecule using UFF
uff_ff = parameterize_uff(mol)

# Evaluate initial MM energy
ec_init, _ = evaluate_mm(uff_ff, mol; calc_grad=false)
println("\n--- Initial Energy Breakdown ---")
print_summary(ec_init)

# Minimize geometry using L-BFGS optimizer
println("\nStarting L-BFGS Geometry Minimization...")
opt_res = optimize_geometry!(mol, uff_ff; method=:lbfgs, max_steps=200, grad_tol=1e-3, verbose=true)

println("\nMinimization Complete!")
println("Converged:     ", opt_res.converged)
println("Iterations:    ", opt_res.iterations)
println("Final Energy:  ", round(opt_res.final_energy, digits=5), " kcal/mol")
println("Max Residual Gradient: ", round(opt_res.max_gradient, digits=6), " kcal/(mol·Å)")
println("Optimized C-C distance: ", round(distance(mol.atoms[1].coords, mol.atoms[2].coords), digits=4), " Å (expected: ~1.53 Å)")

# Final energy decomposition
ec_final, _ = evaluate_mm(uff_ff, mol; calc_grad=false)
println("\n--- Optimized Energy Breakdown ---")
print_summary(ec_final)

write_output("example_04_uff_optimization.out",
             (initial_energy=ec_init, optimization=opt_res,
              final_energy=ec_final);
             title="Ethane UFF Geometry Optimization")
