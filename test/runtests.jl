# ==============================================================================
# QuantumChemMM.jl: Comprehensive Unit Test Suite
# Tests MM, Semi-Empirical, Ab Initio, DFT, Post-HF, GPU, Excited & Transition States, and NMR
# ==============================================================================

using Test
using LinearAlgebra

include("../src/QuantumChemMM.jl")
using .QuantumChemMM

struct TestAccelerator <: QuantumChemMM.AbstractDevice end
QuantumChemMM.GPUInterface.is_gpu(::TestAccelerator) = true
QuantumChemMM.GPUInterface.device_name(::TestAccelerator) = "Test accelerator"
QuantumChemMM.GPUInterface.to_device(x::AbstractArray, ::TestAccelerator) = x
QuantumChemMM.GPUInterface.synchronize_device(::TestAccelerator) = nothing

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

    @testset "5b. Quantum Geometry Optimization" begin
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.95
"""
        for method in (:am1, :hf, :pbe)
            mol = parse_xyz(h2_xyz; name="H2 optimization")
            initial = calculate(mol, method; basis="sto-3g",
                                device=CPUDevice())
            result = optimize_geometry!(mol, method; basis="sto-3g",
                                        device=CPUDevice(), max_steps=12,
                                        grad_tol=1e-3)
            final = calculate(mol, method; basis="sto-3g",
                              device=CPUDevice())

            @test result.initial_energy ≈ initial.total_energy_hartree atol=1e-8
            @test result.final_energy < result.initial_energy
            @test result.final_energy ≈ final.total_energy_hartree atol=1e-8
            @test all(isfinite, coordinates_matrix(mol))
        end

        mol = parse_xyz(h2_xyz; name="H2 optimization")
        @test_throws ArgumentError optimize_geometry!(mol, :mp2)
        @test_throws ArgumentError optimize_geometry!(mol, :hf;
                                                       finite_difference_step=0.0)
    end

    @testset "5c. IR and Raman Spectroscopy" begin
        water_xyz = """3
Water
O  0.0000  0.0000  0.0000
H  0.9572  0.0000  0.0000
H -0.2390  0.9270  0.0000
"""
        mol = parse_xyz(water_xyz; name="Water spectroscopy")
        build_bonds!(mol)
        ff = parameterize_uff(mol)

        ir_modes = calc_vibrational_frequencies(mol, ff)
        @test length(ir_modes) == 3 * n_atoms(mol)
        ir_shift, ir_intensity = generate_ir_spectrum(ir_modes)
        @test length(ir_shift) == length(ir_intensity)
        @test all(isfinite, ir_intensity)

        raman_modes = calc_raman_modes(mol, ff)
        @test length(raman_modes) == length(ir_modes)
        @test all(mode -> isfinite(mode.raman_activity) &&
                          isfinite(mode.depolarization_ratio), raman_modes)
        @test all(mode -> 0.0 <= mode.depolarization_ratio <= 0.75,
                  raman_modes)
        raman_shift, raman_intensity = generate_raman_spectrum(raman_modes)
        @test length(raman_shift) == length(raman_intensity)
        @test all(isfinite, raman_intensity)
        @test maximum(raman_intensity) > 0.0
        @test_throws ArgumentError generate_raman_spectrum(raman_modes;
                                                           laser_wavelength_nm=0.0)

        h2_xyz = """2
H2 QM spectroscopy
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        h2 = parse_xyz(h2_xyz; name="H2 QM spectroscopy")
        qm_modes = calc_vibrational_frequencies(h2, :am1; delta=0.01,
                                                ir_delta=0.01,
                                                device=CPUDevice())
        @test length(qm_modes) == 3 * n_atoms(h2)
        @test all(mode -> isfinite(mode.frequency_cm1) &&
                          isfinite(mode.ir_intensity_km_mol), qm_modes)
        for method in (:hf, :pbe)
            electronic_modes = calc_vibrational_frequencies(
                h2, method; delta=0.01, ir_delta=0.01, device=CPUDevice())
            @test length(electronic_modes) == length(qm_modes)
            @test all(mode -> isfinite(mode.frequency_cm1) &&
                              isfinite(mode.ir_intensity_km_mol),
                      electronic_modes)
        end
        qm_raman_modes = calc_raman_modes(h2, :am1; delta=0.01,
                                          ir_delta=0.01,
                                          device=CPUDevice())
        @test length(qm_raman_modes) == length(qm_modes)
        @test all(mode -> isfinite(mode.raman_activity), qm_raman_modes)
        @test maximum(mode.raman_activity for mode in qm_raman_modes) > 0.0
        dft_raman_modes = calc_raman_modes(h2, :pbe; delta=0.01,
                                           ir_delta=0.01, device=CPUDevice())
        @test length(dft_raman_modes) == length(qm_modes)
        @test all(mode -> isfinite(mode.raman_activity), dft_raman_modes)
    end

    @testset "5d. UV-Visible Spectroscopy" begin
        h2_xyz = """2
H2 UV-Vis
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2 UV-Vis")
        for method in (:cndo2, :hf, :pbe)
            transitions = calc_uv_vis(mol, method; n_states=2,
                                      device=CPUDevice())
            @test !isempty(transitions)
            @test all(tr -> tr.energy_ev > 0.0 && tr.wavelength_nm > 0.0 &&
                            isfinite(tr.oscillator_strength), transitions)
            wavelength, intensity =
                generate_uv_vis_spectrum(transitions; n_points=80)
            @test length(wavelength) == 80
            @test all(isfinite, intensity)
        end
        @test_throws ArgumentError calc_uv_vis(mol, :am1)
        @test_throws ArgumentError calc_uv_vis(mol, :hf; n_states=0)
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

    @testset "7b. Periodic tight-binding bands and DOS" begin
        graphene = CrystalStructure(
            [2.46 1.23 0.0;
             0.0  2.13 0.0;
             0.0  0.0 20.0],
            [0.0 1/3;
             0.0 1/3;
             0.0 0.0],
            [:C, :C]; name="Graphene prototype",
            periodic_dimensions=[1, 2])

        result = solve_periodic_tb(graphene; electrons_per_cell=2,
                                   kmesh=(6, 6, 1), hopping_ev=-2.7,
                                   cutoff_angstrom=1.6, dos_points=120)
        @test size(result.bands_ev) == (36, 2)
        @test result.electrons_per_cell == 2
        @test isfinite(result.fermi_level_ev)
        @test isfinite(result.total_energy_ev_per_cell)
        @test result.band_gap_ev < 0.1
        @test length(result.dos_energies_ev) == 120
        @test all(isfinite, result.dos_states_per_ev_per_cell)

        band_path = periodic_band_structure(
            graphene, [[0.0, 0.0, 0.0], [1/3, 1/3, 0.0], [0.5, 0.0, 0.0]];
            labels=["Γ", "K", "M"], points_per_segment=8,
            hopping_ev=-2.7, cutoff_angstrom=1.6)
        @test size(band_path.bands_ev) == (17, 2)
        @test band_path.high_symmetry_indices == [1, 9, 17]
        @test issorted(band_path.path_distance_per_angstrom)
        @test band_path.labels == ["Γ", "K", "M"]

        @test_throws ArgumentError solve_periodic_tb(graphene;
            electrons_per_cell=5, kmesh=(2, 2, 1), cutoff_angstrom=1.6)
        @test_throws ArgumentError solve_periodic_tb(graphene;
            electrons_per_cell=2, kmesh=(2, 2, 2), cutoff_angstrom=1.6)
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
        @test hf_res.effective_nuclear_charges == [1.0, 1.0]

        iodine = Molecule("Iodine")
        push!(iodine.atoms, Atom(1, :I, [0.0, 0.0, 0.0]))
        iodine_ecps = build_ecp(iodine; name="qcm-prototype")
        @test ecp_core_electrons(iodine_ecps) == 46
        @test effective_nuclear_charges(iodine, iodine_ecps) == [7.0]

        ecp_basis = BasisSet("test", [
            ContractedGaussian(1, [0.0, 0.0, 0.0], 0, 0, 0,
                               [PrimitiveGaussian(1.0, 1.0)], [1.0])
        ])
        ecp_matrix = apply_ecp_matrix(iodine, ecp_basis; name="qcm-prototype", device=CPUDevice())
        @test size(ecp_matrix) == (1, 1)
        @test isfinite(ecp_matrix[1, 1])

        iodine_dimer = Molecule("Iodine dimer")
        push!(iodine_dimer.atoms, Atom(1, :I, [0.0, 0.0, 0.0]))
        push!(iodine_dimer.atoms, Atom(2, :I, [0.0, 0.0, 2.67]))
        basis_functions = ContractedGaussian[]
        for atom in iodine_dimer.atoms
            center = atom.coords .* QuantumChemMM.BasisSets.ANGSTROM_TO_BOHR
            for (lx, ly, lz) in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
                alpha = 0.5
                norm = QuantumChemMM.BasisSets.gaussian_norm(alpha, lx, ly, lz)
                primitive = PrimitiveGaussian(alpha, norm)
                push!(basis_functions, ContractedGaussian(atom.index, center, lx, ly, lz,
                                                          [primitive], [1.0]))
            end
        end
        iodine_basis = BasisSet("test-valence-basis", basis_functions)
        hf_ecp_cpu = solve_hf(iodine_dimer; basis_set=iodine_basis, ecp_name="qcm-prototype",
                              device=CPUDevice(), options=SCFOptions(max_cycles=2))
        hf_ecp_accel = solve_hf(iodine_dimer; basis_set=iodine_basis, ecp_name="qcm-prototype",
                                device=TestAccelerator(), options=SCFOptions(max_cycles=2))
        @test hf_ecp_cpu.effective_nuclear_charges == [7.0, 7.0]
        @test hf_ecp_cpu.n_occ == 7
        @test hf_ecp_accel.core_hamiltonian ≈ hf_ecp_cpu.core_hamiltonian
        unified_ecp = calculate(iodine_dimer, :hf; basis_set=iodine_basis,
                                ecp_name="qcm-prototype", device=CPUDevice(),
                                options=SCFOptions(max_cycles=2))
        @test unified_ecp.raw_result.effective_nuclear_charges == [7.0, 7.0]

        dft_ecp_cpu = solve_dft(iodine_dimer; functional=:lda, basis_set=iodine_basis,
                                ecp_name="qcm-prototype", device=CPUDevice(),
                                options=SCFOptions(max_cycles=2))
        dft_ecp_accel = solve_dft(iodine_dimer; functional=:lda, basis_set=iodine_basis,
                                  ecp_name="qcm-prototype", device=TestAccelerator(),
                                  options=SCFOptions(max_cycles=2))
        @test dft_ecp_cpu.effective_nuclear_charges == [7.0, 7.0]
        @test dft_ecp_accel.effective_nuclear_charges == dft_ecp_cpu.effective_nuclear_charges
        @test dft_ecp_accel.nuclear_repulsion_hartree == dft_ecp_cpu.nuclear_repulsion_hartree
    end

    @testset "8a. RHF/UHF charge and spin occupations" begin
        h2plus = parse_xyz("""2
H2+
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""; name="H2+")
        h2plus.total_charge = 1
        h2plus.multiplicity = 2
        doublet = solve_uhf(h2plus; device=CPUDevice())
        @test doublet.converged
        @test doublet.reference == :uhf
        @test (doublet.n_alpha, doublet.n_beta) == (1, 0)
        @test tr(doublet.alpha_density_matrix * doublet.overlap_matrix) ≈ 1.0 atol=1e-8
        @test tr(doublet.beta_density_matrix * doublet.overlap_matrix) ≈ 0.0 atol=1e-8
        @test_throws ArgumentError solve_hf(h2plus; device=CPUDevice())
        @test_throws ArgumentError solve_dft(h2plus; functional=:lda, device=CPUDevice())
        @test_throws ArgumentError solve_mp2(doublet; device=CPUDevice())

        h2triplet = parse_xyz("""2
H2 triplet
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""; name="H2 triplet")
        h2triplet.multiplicity = 3
        triplet = solve_uhf(h2triplet; device=CPUDevice())
        @test triplet.converged
        @test (triplet.n_alpha, triplet.n_beta) == (2, 0)
        @test tr(triplet.density_matrix * triplet.overlap_matrix) ≈ 2.0 atol=1e-8
        @test_throws ArgumentError solve_dft(h2triplet; functional=:lda, device=CPUDevice())

        oxygen = Molecule("O triplet")
        push!(oxygen.atoms, Atom(1, :O, [0.0, 0.0, 0.0]))
        oxygen.multiplicity = 3
        oxygen_triplet = solve_uhf(oxygen; device=CPUDevice())
        @test oxygen_triplet.converged
        @test (oxygen_triplet.n_alpha, oxygen_triplet.n_beta) == (5, 3)
        @test tr(oxygen_triplet.alpha_density_matrix * oxygen_triplet.overlap_matrix) ≈ 5.0 atol=1e-8
        @test tr(oxygen_triplet.beta_density_matrix * oxygen_triplet.overlap_matrix) ≈ 3.0 atol=1e-8

        h3plus = parse_xyz("""3
H3+
H 0.0 0.0 0.0
H 1.0 0.0 0.0
H 0.5 0.8660254 0.0
"""; name="H3+")
        h3plus.total_charge = 1
        closed_shell_ion = solve_hf(h3plus; device=CPUDevice())
        @test closed_shell_ion.reference == :rhf
        @test closed_shell_ion.n_alpha == closed_shell_ion.n_beta == 1

        unified_doublet = calculate(h2plus, :uhf; device=CPUDevice())
        @test unified_doublet.raw_result.n_alpha == 1
        @test unified_doublet.raw_result.n_beta == 0
        accelerator_doublet = solve_uhf(h2plus; device=TestAccelerator())
        @test accelerator_doublet.total_energy_hartree ≈ doublet.total_energy_hartree atol=1e-8
        h2plus.multiplicity = 1
        @test_throws ArgumentError solve_uhf(h2plus; device=CPUDevice())
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

    @testset "9b. TDA-TDHF and adiabatic TDA-TDDFT" begin
        h2_xyz = """2
H2
H  0.0  0.0  0.0
H  0.0  0.0  0.7414
"""
        mol = parse_xyz(h2_xyz; name="H2")

        tdhf = calculate(mol, :tdhf; basis="sto-3g", device=CPUDevice())
        @test tdhf.method == :tdhf
        @test tdhf.raw_result isa TDResult
        @test !isempty(tdhf.raw_result.singlets)
        @test !isempty(tdhf.raw_result.triplets)
        @test all(state -> state.excitation_energy_ev > 0.0,
                  tdhf.raw_result.singlets)
        @test all(state -> isfinite(state.oscillator_strength),
                  tdhf.raw_result.singlets)
        tdhf_accelerator = calculate(mol, :tdhf; basis="sto-3g",
                                     calc_triplets=false, n_states=1,
                                     device=TestAccelerator())
        @test tdhf_accelerator.raw_result.singlets[1].excitation_energy_ev ≈
              tdhf.raw_result.singlets[1].excitation_energy_ev atol=1e-8

        tddft = calculate(mol, :tddft; functional=:pbe, n_states=1,
                          device=CPUDevice())
        @test tddft.method == :tddft
        @test tddft.raw_result.functional == "pbe"
        @test length(tddft.raw_result.singlets) == 1
        @test isempty(tddft.raw_result.triplets)
        @test tddft.raw_result.singlets[1].excitation_energy_ev > 0.0
        tddft_accelerator = calculate(mol, :tddft; functional=:pbe,
                                      n_states=1, device=TestAccelerator())
        @test tddft_accelerator.raw_result.singlets[1].excitation_energy_ev ≈
              tddft.raw_result.singlets[1].excitation_energy_ev atol=1e-8

        tda = calculate(mol, :tda; functional=:lda, n_states=1,
                        device=CPUDevice())
        @test tda.method == :tddft
        @test tda.raw_result.functional == "lda"
        @test length(tda.raw_result.singlets) == 1

        @test_throws ArgumentError calculate(mol, :tddft; functional=:pbe,
                                             calc_triplets=true)
        @test_throws ArgumentError calculate(mol, :tda; functional=:invalid)
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
