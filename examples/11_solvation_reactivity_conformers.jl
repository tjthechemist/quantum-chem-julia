# ==============================================================================
# Example 11: Solvation Free Energies, Conceptual DFT Reactivity & Conformers
# ==============================================================================

using QuantumChemMM
using Printf

# Acrolein (Propenal) CH2=CH-CHO
acrolein_xyz = """8
Acrolein C3H4O
C   -1.3000    0.4000    0.0000
C   -0.1000   -0.2000    0.0000
C    1.2000    0.5000    0.0000
O    2.2500   -0.1000    0.0000
H   -2.2000   -0.1500    0.0000
H   -1.3500    1.4800    0.0000
H   -0.0500   -1.2800    0.0000
H    1.1500    1.6000    0.0000
"""

mol = parse_xyz(acrolein_xyz; name="Acrolein")

# 1. Ground State Electronic Structure
println("1. Solving AM1 Semi-Empirical Electronic Structure...")
scf_res = solve_nddo(mol, NDDOMethod(:am1))
for (i, at) in enumerate(mol.atoms)
    at.partial_charge = scf_res.mulliken_charges[i]
end

# 2. Conceptual DFT Reactivity Descriptors & Fukui Functions
println("\n2. Conceptual DFT Reactivity Analysis:")
react_indices = calc_reactivity_indices(scf_res, mol)
fukui = calc_condensed_fukui(scf_res, mol)
print_reactivity_summary(react_indices, fukui, mol)

# 3. Solvation Free Energy in Various Solvents
println("\n3. Solvation Free Energy (ΔG_solv) via COSMO/Generalized Born:")
println("  Solvent          Dielectric (ε)   ΔG_solv (kcal/mol)   SASA (Å²)")
for s_name in [:water, :ethanol, :chloroform, :hexane]
    solv_res = calc_solvation_energy(mol, s_name)
    @printf("  %-15s     %6.2f          %8.3f        %7.2f\n",
            solv_res.solvent_name, solv_res.dielectric,
            solv_res.total_delta_g_solv_kcal, solv_res.total_sasa_ang2)
end

# 4. Conformational Search & Boltzmann Populations
println("\n4. Conformational Search around C-C Single Bond:")
ensemble = search_conformers(mol, parameterize_uff; n_conformations=8, temp_k=298.15)
print_conformer_summary(ensemble)

