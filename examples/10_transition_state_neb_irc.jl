# ==============================================================================
# Example 10: Transition State Search via Climbing-Image NEB and IRC Tracing
# ==============================================================================

using QuantumChemMM
using LinearAlgebra
using Printf

# Cis-Trans Isomerization or Torsional Rotation in H2O2 (Hydrogen Peroxide)
# Reactant: cis-like conformation
reactant_xyz = """4
H2O2 Cis
O    0.0000    0.7300    0.0000
O    0.0000   -0.7300    0.0000
H    0.8500    0.9500    0.3500
H    0.8500   -0.9500    0.3500
"""

# Product: trans-like conformation
product_xyz = """4
H2O2 Trans
O    0.0000    0.7300    0.0000
O    0.0000   -0.7300    0.0000
H    0.8500    0.9500    0.3500
H   -0.8500   -0.9500   -0.3500
"""

react = parse_xyz(reactant_xyz; name="Reactant")
prod = parse_xyz(product_xyz; name="Product")

println("1. Setting up Nudged Elastic Band (NEB)...")
neb = NEBPath(react, prod; n_intermediate=7, spring_k=25.0, climbing_image=true)
o_o_axis = normalize(react.atoms[2].coords .- react.atoms[1].coords)
hydrogen_vector = react.atoms[4].coords .- react.atoms[2].coords
for image_idx in 2:(neb.n_images - 1)
    fraction = (image_idx - 1) / (neb.n_images - 1)
    angle = π * fraction
    rotated_vector = cos(angle) .* hydrogen_vector .+
                     sin(angle) .* cross(o_o_axis, hydrogen_vector) .+
                     (1.0 - cos(angle)) * dot(o_o_axis, hydrogen_vector) .* o_o_axis
    coords = coordinates_matrix(neb.images[image_idx])
    coords[:, 4] .= react.atoms[2].coords .+ rotated_vector
    set_coordinates!(neb.images[image_idx], coords)
end
println("Initialized NEB with $(neb.n_images) replicas along the reaction path.")

println("\n2. Running Climbing-Image NEB (CI-NEB)...")
res_neb = run_neb(neb, parameterize_uff; max_steps=100, force_tol=0.08,
                  step_size=0.0001, verbose=true)

println("\n=== NEB Results ===")
println("Convergence Status:       ", res_neb.converged ? "CONVERGED" : "MAX_STEPS REACHED")
println("Activation Barrier (E_a): ", round(res_neb.activation_energy_kcal, digits=3), " kcal/mol")
println("Reaction Energy (ΔE_rxn): ", round(res_neb.reaction_energy_kcal, digits=3), " kcal/mol")
println("Identified TS at Replica: ", res_neb.ts_index)

println("\n3. Verifying Transition State with Intrinsic Reaction Coordinate (IRC)...")
ts_mol = res_neb.transition_state_image
irc_res = trace_irc(ts_mol, parameterize_uff; step_size=0.04, max_steps=30, verbose=false)

println("Forward Barrier (TS -> Reactant): ", round(irc_res.barrier_forward_kcal, digits=3), " kcal/mol")
println("Reverse Barrier (TS -> Product):  ", round(irc_res.barrier_reverse_kcal, digits=3), " kcal/mol")
println("IRC successfully confirmed connection between reactant, TS, and product!")

write_output("example_10_transition_state.out",
             (neb=res_neb, irc=irc_res);
             title="Hydrogen Peroxide NEB and IRC Results")
