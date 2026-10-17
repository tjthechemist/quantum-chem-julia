# ==============================================================================
# Example 07: Hybrid QM/MM Calculation with Mechanical & Electrostatic Embedding
# ==============================================================================

using QuantumChemMM

# Acetone in a water cluster (QM: Acetone [atoms 1-10], MM: Water molecule [atoms 11-13])
cluster_xyz = """13
Acetone solvated by Water
C    0.0000    0.1800    0.0000
O    0.0000    1.4000    0.0000
C    1.2800   -0.6300    0.0000
C   -1.2800   -0.6300    0.0000
H    1.2900   -1.2800    0.8800
H    1.2900   -1.2800   -0.8800
H    2.1400    0.0400    0.0000
H   -1.2900   -1.2800    0.8800
H   -1.2900   -1.2800   -0.8800
H   -2.1400    0.0400    0.0000
O    0.0000    3.8000    0.0000
H    0.7600    4.3800    0.0000
H   -0.7600    4.3800    0.0000
"""

mol = parse_xyz(cluster_xyz; name="Acetone_Water")

# Set standard partial charges for MM water (SPC/E model)
mol.atoms[11].partial_charge = -0.8476 # O
mol.atoms[12].partial_charge =  0.4238 # H
mol.atoms[13].partial_charge =  0.4238 # H

# Partition: QM = Acetone (atoms 1 to 10), MM = Water (atoms 11 to 13)
qm_atoms = collect(1:10)

println("=== 1. Mechanical Embedding QM/MM ===")
qmmm_mech = create_qmmm_system(mol, qm_atoms; embedding=:mechanical)
res_mech = evaluate_qmmm(qmmm_mech, NDDOMethod(:am1))
print_qmmm_summary(res_mech)

println("\n=== 2. Electrostatic Embedding QM/MM ===")
qmmm_elec = create_qmmm_system(mol, qm_atoms; embedding=:electrostatic)
res_elec = evaluate_qmmm(qmmm_elec, NDDOMethod(:am1))
print_qmmm_summary(res_elec)

