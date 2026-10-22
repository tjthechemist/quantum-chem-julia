# ==============================================================================
# Example 13: Post-Ab Initio Methods (MP2, CISD, CCSD, and CCSD(T))
# Demonstrates the Quantum Chemical Correlation Ladder from HF to CCSD(T)
# ==============================================================================

using QuantumChemMM
using Printf

# Hydrogen molecule H2
h2_xyz = """2
Hydrogen H2
H    0.000000    0.000000    0.000000
H    0.000000    0.000000    0.741400
"""

mol = parse_xyz(h2_xyz; name="H2")

println("1. Reference State: Ab Initio Hartree-Fock (RHF / STO-3G)...")
hf_res = solve_hf(mol; basis_name="sto-3g")
@printf("  E(HF) = %16.8f a.u.\n", hf_res.total_energy_hartree)

println("\n2. Second-Order Møller-Plesset Perturbation (MP2)...")
mp2_res = solve_mp2(hf_res)
print_mp2_summary(mp2_res)

println("\n3. Configuration Interaction Singles and Doubles (CISD)...")
cisd_res = solve_cisd(hf_res)
print_cisd_summary(cisd_res)

println("\n4. Coupled Cluster with Triples (CCSD and CCSD(T))...")
cc_res = solve_ccsd(hf_res; compute_triples=true)
print_ccsd_summary(cc_res)

println("\n=== Electronic Correlation Energy Ladder Summary ===")
println(" Method       Total Energy (a.u.)  E_corr (kcal/mol)")
@printf(" HF           %16.8f       %8.3f\n", hf_res.total_energy_hartree, 0.0)
@printf(" MP2          %16.8f       %8.3f\n", mp2_res.total_mp2_energy_hartree, mp2_res.mp2_correlation_hartree * 627.509)
@printf(" CISD         %16.8f       %8.3f\n", cisd_res.total_cisd_energy_hartree, cisd_res.cisd_correlation_hartree * 627.509)
@printf(" CCSD         %16.8f       %8.3f\n", cc_res.total_ccsd_energy_hartree, cc_res.ccsd_correlation_hartree * 627.509)
@printf(" CCSD(T)      %16.8f       %8.3f\n", cc_res.total_ccsdt_energy_hartree, (cc_res.ccsd_correlation_hartree + cc_res.triples_correction_hartree) * 627.509)

write_output("example_13_post_hf.out",
             (hartree_fock=hf_res, mp2=mp2_res, cisd=cisd_res,
              coupled_cluster=cc_res);
             title="Hydrogen Post-Hartree-Fock Results")
