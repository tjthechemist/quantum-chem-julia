# ==============================================================================
# Example 12: Ab Initio Hartree-Fock (RHF) and Density Functional Theory (DFT)
# ==============================================================================

using QuantumChemMM
using Printf

# Water molecule H2O
water_xyz = """3
Water molecule H2O
O    0.000000    0.000000    0.117300
H    0.000000    0.757200   -0.469200
H    0.000000   -0.757200   -0.469200
"""

mol = parse_xyz(water_xyz; name="Water")

# 1. Ab Initio Restricted Hartree-Fock (RHF / STO-3G)
println("=== 1. Ab Initio Hartree-Fock (RHF / STO-3G) ===")
hf_res = solve_hf(mol; basis_name="sto-3g")
print_hf_summary(hf_res)

# 2. Kohn-Sham DFT with B3LYP Hybrid Functional
println("\n=== 2. Kohn-Sham DFT (B3LYP / STO-3G) ===")
b3lyp_res = solve_dft(mol; functional=:b3lyp, basis_name="sto-3g")
print_dft_summary(b3lyp_res)

# 3. Kohn-Sham DFT with PBE (GGA Functional)
println("\n=== 3. Kohn-Sham DFT (PBE / STO-3G) ===")
pbe_res = solve_dft(mol; functional=:pbe, basis_name="sto-3g")
print_dft_summary(pbe_res)

# 4. Unified Entry Point calculate()
println("\n=== 4. Testing Unified calculate() Interface ===")
for m in [:hf, :b3lyp, :pbe, :am1, :uff]
    c_res = calculate(mol, m; basis="sto-3g")
    @printf("  Method: %-8s | Energy: %14.6f a.u. (%12.4f eV | %12.2f kcal/mol)\n",
            string(c_res.method), c_res.total_energy_hartree, c_res.total_energy_ev, c_res.total_energy_kcal)
end

