# ==============================================================================
# QuantumChemMM.jl: Comprehensive Unit Test Suite
# Tests MM, Semi-Empirical, Ab Initio, DFT, Post-HF, GPU, Excited & Transition States, and NMR
# ==============================================================================

using Test
using LinearAlgebra

include("../src/QuantumChemMM.jl")
using .QuantumChemMM

@testset "QuantumChemMM Full Comprehensive Test Suite" begin

    @testset "1. Core & Periodic Table Elements" begin
        c_elem = QuantumChemMM.Elements.get_element(:C)
        @test c_elem.atomic_number == 6
        @test c_elem.symbol == :C
        @test isapprox(c_elem.mass, 12.011, atol=0.01)
        @test QuantumChemMM.Elements.valence_electrons(:C) == 4
        @test QuantumChemMM.Elements.valence_electrons(:H) == 1
        @test QuantumChemMM.Elements.valence_electrons(:O) == 6
    end

    @testset "2. Geometry and Mathematical Derivatives" begin
        r1 = [0.0, 0.0, 0.0]
        r2 = [1.0, 0.0, 0.0]
        r3 = [1.0, 1.0, 0.0]
        r4 = [1.0, 1.0, 1.0]

        @test isapprox(distance(r1, r2), 1.0, atol=1e-8)
        @test isapprox(angle_between(r1, r2, r3), π / 2.0, atol=1e-8)
        @test isapprox(abs(dihedral_angle(r1, r2, r3, r4)), π / 2.0, atol=1e-6)

        gi, gj = QuantumChemMM.MolMath.distance_gradient(r1, r2)
        @test isapprox(gi, [-1.0, 0.0, 0.0], atol=1e-6)
        @test isapprox(gj, [ 1.0, 0.0, 0.0], atol=1e-6)
    end

    @testset "3. Chemical File IO (XYZ & PDB)" begin
        h2o_xyz = """3
Water test
O  0.0  0.0  0.1173
H  0.0  0.7572 -0.4692
H  0.0 -0.7572 -0.4692
"""
        mol = parse_xyz(h2o_xyz; name="Water")
        @test n_atoms(mol) == 3
        @test n_bonds(mol) == 2
        @test mol.atoms[1].symbol == :O

        com = center_of_mass(mol)
        @test isapprox(com[1], 0.0, atol=1e-4)
    end

    @testset "4. Molecular Mechanics Force Fields (UFF, AMBER, OPLS, CHARMM, MM3)" begin
        ethane_xyz = """8
Ethane C2H6
C  0.0000  0.0000  0.0000
C  1.5300  0.0000  0.0000
H -0.5000  0.8900  0.3000
H -0.5000 -0.8900  0.3000
H -0.5000  0.0000 -1.0000
H  2.0300  0.8900  0.3000
H  2.0300 -0.8900  0.3000
H  2.0300  0.0000 -1.0000
"""
        mol = parse_xyz(ethane_xyz; name="Ethane")

        uff_ff = parameterize_uff(mol)
        ec_uff, grad_uff = evaluate_mm(uff_ff, mol)
        @test isfinite(total_energy(ec_uff))
        @test size(grad_uff) == (3, 8)

        amber_ff = parameterize_amber(mol)
        @test isfinite(total_energy(evaluate_mm(amber_ff, mol)[1]))

        opls_ff = parameterize_opls(mol)
        @test isfinite(total_energy(evaluate_mm(opls_ff, mol)[1]))

        charmm_ff = parameterize_charmm(mol)
        @test isfinite(total_energy(evaluate_mm(charmm_ff, mol)[1]))

        mm3_ff = parameterize_mm3(mol)
        @test isfinite(total_energy(evaluate_mm(mm3_ff, mol)[1]))
    end

    @testset "5. Geometry Optimization (L-BFGS)" begin
        distorted_xyz = """3
Distorted Water
O  0.0  0.0  0.0
H  0.0  1.3 -0.5
H  0.0 -1.3 -0.5
"""
        mol = parse_xyz(distorted_xyz; name="Water")
        ff = parameterize_uff(mol)
        e_init = total_energy(evaluate_mm(ff, mol; calc_grad=false)[1])

        res = optimize_geometry!(mol, ff; method=:lbfgs, max_steps=50, grad_tol=1e-3)
        @test res.final_energy < e_init
    end

    @testset "6. Simple Hückel Theory (HMO) on Benzene" begin
        benzene_xyz = """12
Benzene
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
        mol = parse_xyz(benzene_xyz; name="Benzene")
        res = solve_huckel(mol)

        @test isapprox(res.orbital_energies_m[1], 2.0, atol=1e-2)
        @test isapprox(res.delocalization_energy_beta, 2.0, atol=1e-2)
        unified_res = calculate(mol, :huckel)
        @test unified_res.orbital_energies_ev ≈ res.orbital_energies_ev
    end

    @testset "7. CNDO/2 and AM1 / PM3 NDDO SCF" begin
        ch2o_xyz = """4
Formaldehyde
C    0.000000    0.000000   -0.597000
O    0.000000    0.000000    0.607000
H    0.000000    0.938000   -1.181000
H    0.000000   -0.938000   -1.181000
"""
        mol = parse_xyz(ch2o_xyz; name="Formaldehyde")

        cndo_res = solve_cndo2(mol; options=SCFOptions(max_cycles=30))
        @test isfinite(cndo_res.total_energy)

        am1_res = solve_nddo(mol, NDDOMethod(:am1); options=SCFOptions(max_cycles=30))
        @test isfinite(am1_res.total_energy_ev)
    end

    @testset "8. Ab Initio Hartree-Fock (RHF)" begin
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2")
        bs = build_basis(mol, "sto-3g")
        @test n_basis(bs) == 2

        S = compute_overlap(bs)
        @test isapprox(S[1, 1], 1.0, atol=1e-6)
        @test isapprox(S[2, 2], 1.0, atol=1e-6)
        @test S[1, 2] > 0.0

        hf_res = solve_hf(mol; basis_name="sto-3g")
        @test hf_res.converged
        @test isapprox(hf_res.total_energy_hartree, -1.117, atol=0.05)
    end

    @testset "9. Density Functional Theory (DFT: B3LYP & PBE)" begin
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2")

        b3lyp_res = solve_dft(mol; functional=:b3lyp, basis_name="sto-3g")
        @test b3lyp_res.converged
        @test isfinite(b3lyp_res.total_energy_hartree)
        @test b3lyp_res.total_energy_hartree < -1.0

        pbe_res = solve_dft(mol; functional=:pbe, basis_name="sto-3g")
        @test pbe_res.converged
        @test isfinite(pbe_res.total_energy_hartree)
    end

    @testset "10. Post-Ab Initio Methods (MP2, CISD, CCSD)" begin
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2")
        hf_res = solve_hf(mol; basis_name="sto-3g")

        # MP2
        mp2_res = solve_mp2(hf_res)
        @test mp2_res.mp2_correlation_hartree < 0.0
        @test mp2_res.total_mp2_energy_hartree < hf_res.total_energy_hartree

        # CISD
        cisd_res = solve_cisd(hf_res)
        @test cisd_res.cisd_correlation_hartree < 0.0
        @test cisd_res.total_cisd_energy_hartree < hf_res.total_energy_hartree

        # CCSD
        cc_res = solve_ccsd(hf_res; compute_triples=true)
        @test cc_res.ccsd_correlation_hartree < 0.0
        @test cc_res.total_ccsd_energy_hartree < hf_res.total_energy_hartree
    end


    @testset "11. Unified calculate() Entry Point (with Solvation)" begin
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2")

        # Gas-phase
        res_hf = calculate(mol, :hf)
        @test res_hf.total_energy_hartree < -1.0
        @test res_hf.orbital_energies_ev ≈ res_hf.raw_result.mo_energies_hartree .* 27.211386245988
        @test res_hf.solvation === nothing
        @test res_hf.total_energy_solv_hartree == res_hf.total_energy_hartree

        res_dft = calculate(mol, :b3lyp)
        @test res_dft.total_energy_hartree < -1.0
        @test res_dft.orbital_energies_ev ≈ res_dft.raw_result.mo_energies_hartree .* 27.211386245988

        res_am1 = calculate(mol, :am1)
        @test res_am1.orbital_energies_ev ≈ res_am1.raw_result.mo_energies_ev

        res_eht = calculate(mol, :eht)
        @test res_eht.orbital_energies_ev ≈ res_eht.raw_result.mo_energies_ev

        res_cndo2 = calculate(mol, :cndo2)
        @test res_cndo2.orbital_energies_ev ≈ res_cndo2.raw_result.mo_energies

        res_mp2 = calculate(mol, :mp2)
        @test res_mp2.total_energy_hartree < res_hf.total_energy_hartree
        @test res_mp2.orbital_energies_ev ≈ res_hf.orbital_energies_ev

        res_uff = mktempdir() do output_dir
            output_path = joinpath(output_dir, "h2-report.custom")
            result = calculate(mol, :uff; output_file=output_path)
            report = format_output(result)
            @test occursin("QuantumChemMM Calculation Report", report)
            @test occursin("total_energy_hartree", report)
            @test isfile(output_path)
            @test read(output_path, String) == report
            @test sprint(show, MIME"text/plain"(), result) == report
            result
        end
        @test isfinite(res_uff.total_energy_kcal)
        @test res_uff.orbital_energies_ev === nothing
        hf_report = format_output(res_hf)
        @test occursin("orbital_energies_ev:", hf_report)
        @test occursin("MO #1:", hf_report)
        @test occursin("eV", hf_report)
        orbital_report = format_output((orbital_energies_ev=collect(1.0:13.0),))
        @test occursin("MO #13: 13 eV", orbital_report)
        @test occursin("e_bond", format_output(res_uff.raw_result))

        # Solvated HF/water: ΔG_solv should shift the total
        res_hf_aq = calculate(mol, :hf; solvent=:water)
        @test res_hf_aq.solvation isa SolvationResult
        @test res_hf_aq.solvation.dielectric ≈ 78.39 atol=0.1
        @test isfinite(res_hf_aq.total_energy_solv_hartree)
        # Solvated total ≠ gas-phase total for polar water
        @test res_hf_aq.total_energy_solv_hartree != res_hf_aq.total_energy_hartree

        # Vacuum keyword should produce identical energies
        res_hf_vac = calculate(mol, :hf; solvent=:vacuum)
        @test res_hf_vac.solvation === nothing
        @test res_hf_vac.total_energy_hartree == res_hf.total_energy_hartree
    end

    @testset "12. Multi-Vendor GPU Layer (NVIDIA, AMD, Intel, CPU)" begin
        devs = detect_available_devices()
        @test length(devs) >= 1
        @test is_gpu(CUDADevice(0))
        @test is_gpu(ROCmDevice(0))
        @test is_gpu(OneAPIDevice(0))
        @test !is_gpu(CPUDevice())

        @test vendor(CUDADevice(0)) == :nvidia
        @test vendor(ROCmDevice(0)) == :amd
        @test vendor(OneAPIDevice(0)) == :intel
        @test vendor(CPUDevice()) == :cpu

        # Test GPU tensor kernels
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2")
        bs = build_basis(mol, "sto-3g")
        eri = compute_eri_tensor(bs)
        nb = n_basis(bs)
        P = fill(0.5, nb, nb)
        eri_j = reshape(eri, nb^2, nb^2)
        eri_k = reshape(permutedims(eri, (1, 3, 2, 4)), nb^2, nb^2)

        J_gpu, K_gpu = gpu_fock_coulomb_exchange(P, eri_j, eri_k, 1.0)
        @test size(J_gpu) == (nb, nb)
        @test size(K_gpu) == (nb, nb)
        @test all(isfinite.(J_gpu))
        @test all(isfinite.(K_gpu))

        # Test 4-index transformation kernel
        C = Matrix(1.0I, nb, nb)
        eri_mo = gpu_transform_eri_ao_to_mo(eri, C)
        @test size(eri_mo) == (nb, nb, nb, nb)
        @test isapprox(eri_mo[1, 1, 1, 1], eri[1, 1, 1, 1], atol=1e-8)

        anti = gpu_antisymmetrize_mo_eri(eri_mo)
        @test size(anti) == (nb, nb, nb, nb)

        # Test GPU Solvers with device parameter
        hf_res = solve_hf(mol; basis_name="sto-3g", device=CPUDevice())
        @test hf_res.converged

        dft_res = solve_dft(mol; functional=:b3lyp, basis_name="sto-3g", device=CPUDevice())
        @test dft_res.converged

        mp2_res = solve_mp2(hf_res; device=CPUDevice())
        @test mp2_res.mp2_correlation_hartree < 0.0

        cc_res = solve_ccsd(hf_res; device=CPUDevice(), compute_triples=true)
        @test cc_res.ccsd_correlation_hartree < 0.0

        # Unified calculate with device=
        res_calc_hf = calculate(mol, :hf; device=CPUDevice())
        @test isfinite(res_calc_hf.total_energy_hartree)

        res_calc_dft = calculate(mol, :b3lyp; device=CPUDevice())
        @test isfinite(res_calc_dft.total_energy_hartree)

        res_calc_mp2 = calculate(mol, :mp2; device=CPUDevice())
        @test isfinite(res_calc_mp2.total_energy_hartree)
    end

    @testset "13. Nuclear Magnetic Resonance (NMR) Spectroscopy on GPU" begin
        water_xyz = """3
Water
O  0.000000    0.000000    0.117300
H  0.000000    0.757200   -0.469200
H  0.000000   -0.757200   -0.469200
"""
        mol = parse_xyz(water_xyz; name="Water")
        scf_res = solve_cndo2(mol)

        # Compute NMR on device
        nmr_res = calc_nmr_shieldings(scf_res, mol; device=CPUDevice())
        @test length(nmr_res.nuclei) == 3
        @test nmr_res.nuclei[1].symbol == :O
        @test nmr_res.nuclei[2].symbol == :H
        @test nmr_res.nuclei[3].symbol == :H

        # Shielding tensor properties
        h1 = nmr_res.nuclei[2]
        @test isfinite(h1.shielding.isotropic)
        @test isfinite(h1.chemical_shift_ppm)
        @test isfinite(h1.shielding.anisotropy)
        @test length(h1.shielding.eigenvalues) == 3
        @test h1.shielding.eigenvalues[1] <= h1.shielding.eigenvalues[2] <= h1.shielding.eigenvalues[3]

        # Spin-spin couplings (geminal 2J_HH)
        @test isfinite(nmr_res.couplings_hz[2, 3])

        # Spectrum simulation on device
        ppm_axis, spec_curve = simulate_nmr_spectrum(nmr_res; nucleus=:H, ppm_min=-2.0, ppm_max=14.0, n_points=500, device=CPUDevice())
        @test length(ppm_axis) == 500
        @test length(spec_curve) == 500
        @test maximum(spec_curve) > 0.0

        # Dispatch via calculate(mol, :nmr)
        res_calc_nmr = calculate(mol, :nmr; device=CPUDevice())
        @test res_calc_nmr.method == :nmr
        @test res_calc_nmr.raw_result isa NMRResult
    end

    @testset "14. Implicit Solvation Across All Method Families" begin
        # Use H2O: polar molecule ensures meaningful ΔG_solv
        water_xyz = """3
Water
O  0.000000    0.000000    0.117300
H  0.000000    0.757200   -0.469200
H  0.000000   -0.757200   -0.469200
"""
        mol = parse_xyz(water_xyz; name="Water")

        # ── Solvation model standalone ──────────────────────────────────────
        solv_water = calc_solvation_energy(mol, :water)
        @test solv_water.dielectric ≈ 78.39 atol=0.1
        @test solv_water.electrostatic_solv_kcal <= 0.0   # always stabilizing
        @test solv_water.total_sasa_ang2 > 0.0

        solv_vacuum = calc_solvation_energy(mol, :vacuum)
        @test solv_vacuum.total_delta_g_solv_kcal == 0.0  # no correction in vacuum

        # ── Molecular Mechanics + solvent ───────────────────────────────────
        res_mm_vac  = calculate(mol, :uff; solvent=:vacuum)
        res_mm_aq   = calculate(mol, :uff; solvent=:water)
        @test res_mm_vac.solvation === nothing
        @test res_mm_aq.solvation isa SolvationResult
        @test isfinite(res_mm_aq.total_energy_solv_kcal)
        # ΔG_solv lowers (stabilizes) total energy in water
        @test res_mm_aq.total_energy_solv_kcal < res_mm_vac.total_energy_kcal

        # ── Semi-Empirical (CNDO/2) + solvent ───────────────────────────────
        res_cndo_vac = calculate(mol, :cndo2; solvent=:vacuum)
        res_cndo_aq  = calculate(mol, :cndo2; solvent=:water)
        @test res_cndo_vac.solvation === nothing
        @test res_cndo_aq.solvation isa SolvationResult
        @test isfinite(res_cndo_aq.total_energy_solv_ev)

        # ── Semi-Empirical (AM1) + solvent ───────────────────────────────────
        res_am1_aq = calculate(mol, :am1; solvent=:water)
        @test res_am1_aq.solvation isa SolvationResult
        @test isfinite(res_am1_aq.total_energy_solv_ev)

        # ── Hartree-Fock + solvent ───────────────────────────────────────────
        res_hf_vac = calculate(mol, :hf;  solvent=:vacuum)
        res_hf_aq  = calculate(mol, :hf;  solvent=:water)
        res_hf_dmso = calculate(mol, :hf; solvent=:dmso)
        @test res_hf_vac.solvation === nothing
        @test res_hf_aq.solvation isa SolvationResult
        @test isfinite(res_hf_aq.total_energy_solv_hartree)
        @test res_hf_aq.total_energy_solv_hartree != res_hf_vac.total_energy_hartree
        @test isfinite(res_hf_dmso.total_energy_solv_hartree)
        # Water (ε=78.4) stabilizes more than DMSO (ε=46.7) — more negative ΔG_solv
        @test res_hf_aq.solvation.electrostatic_solv_kcal <
              res_hf_dmso.solvation.electrostatic_solv_kcal

        # ── DFT (B3LYP) + solvent ───────────────────────────────────────────
        res_dft_aq = calculate(mol, :b3lyp; solvent=:water)
        @test res_dft_aq.solvation isa SolvationResult
        @test isfinite(res_dft_aq.total_energy_solv_hartree)

        # ── MP2 + solvent ────────────────────────────────────────────────────
        res_mp2_aq = calculate(mol, :mp2; solvent=:water)
        @test res_mp2_aq.solvation isa SolvationResult
        @test isfinite(res_mp2_aq.total_energy_solv_hartree)

        # ── CISD + solvent ───────────────────────────────────────────────────
        res_cisd_aq = calculate(mol, :cisd; solvent=:water)
        @test res_cisd_aq.solvation isa SolvationResult
        @test isfinite(res_cisd_aq.total_energy_solv_hartree)

        # ── CCSD + solvent ───────────────────────────────────────────────────
        res_ccsd_aq = calculate(mol, :ccsd; solvent=:water)
        @test res_ccsd_aq.solvation isa SolvationResult
        @test isfinite(res_ccsd_aq.total_energy_solv_hartree)

        # ── NMR + solvent (underlying HF in water) ───────────────────────────
        res_nmr_aq = calculate(mol, :nmr; solvent=:water)
        @test res_nmr_aq.solvation isa SolvationResult
        @test res_nmr_aq.raw_result isa NMRResult

        # ── Multiple solvents — ΔG_solv ordering check ───────────────────────
        res_hf_chl  = calculate(mol, :hf; solvent=:chloroform)  # ε=4.81
        res_hf_hex  = calculate(mol, :hf; solvent=:hexane)       # ε=1.88
        # More polar solvent → stronger electrostatic stabilization
        @test res_hf_aq.solvation.electrostatic_solv_kcal  <=
              res_hf_chl.solvation.electrostatic_solv_kcal
        @test res_hf_chl.solvation.electrostatic_solv_kcal <=
              res_hf_hex.solvation.electrostatic_solv_kcal

        # ── print_solvation_summary smoke test ───────────────────────────────
        mktemp() do _, buf
            redirect_stdout(buf) do
                print_solvation_summary(res_hf_aq.solvation)
            end
            seekstart(buf)
            out = read(buf, String)
            @test contains(out, "Water")
            @test contains(out, "ΔG_solv")
        end
    end

end
