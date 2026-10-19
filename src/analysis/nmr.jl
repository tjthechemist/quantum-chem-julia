# ==============================================================================
# QuantumChemMM.jl: Nuclear Magnetic Resonance (NMR) Spectroscopy Engine
# Chemical Shielding Tensors, Chemical Shifts, Spin-Spin J-Coupling, and Spectra
# Hardware-Accelerated across NVIDIA (CUDA), AMD (ROCm), Intel (oneAPI), and CPU
# ==============================================================================

module NMRSpectroscopy

using LinearAlgebra
using Printf
using ..Types
using ..Elements
using ..MolMath
using ..SemiEmpiricalBase
using ..SlaterOrbitals
using ..GPUInterface
using ..GPUKernels
using ..HartreeFock
using ..KohnShamDFT

export NMRShieldingTensor, NMRNucleus, NMRResult,
       calc_nmr_shieldings, simulate_nmr_spectrum, print_nmr_summary,
       DEFAULT_NMR_REFERENCES

"""
    NMRShieldingTensor
Represents the complete 3x3 magnetic shielding tensor of a nucleus in ppm,
along with its isotropic shielding, chemical shift anisotropy, and asymmetry parameter.
"""
struct NMRShieldingTensor
    tensor::Matrix{Float64}         # 3x3 shielding tensor in ppm
    isotropic::Float64              # σ_iso = Tr(σ) / 3 (ppm)
    anisotropy::Float64             # Δσ = σ_33 - (σ_11 + σ_22)/2 (ppm)
    asymmetry::Float64              # η = (σ_22 - σ_11) / (σ_33 - σ_iso)
    eigenvalues::Vector{Float64}    # Principal components σ_11 <= σ_22 <= σ_33 (ppm)
    diamagnetic::Float64            # σ_d isotropic (ppm)
    paramagnetic::Float64           # σ_p isotropic (ppm)
end

"""
    NMRNucleus
Information for an NMR-active nucleus including chemical shift relative to a standard.
"""
struct NMRNucleus
    atom_index::Int
    symbol::Symbol
    isotope::String                 # "1H", "13C", "15N", "19F", "31P"
    shielding::NMRShieldingTensor
    chemical_shift_ppm::Float64     # δ = σ_ref - σ_iso
end

"""
    NMRResult
Overall NMR calculation output containing all nuclei, J-coupling matrix, and references.
"""
struct NMRResult
    nuclei::Vector{NMRNucleus}
    couplings_hz::Matrix{Float64}   # Scalar spin-spin J-coupling matrix in Hz
    reference_shieldings::Dict{Symbol, Float64}
    method::Symbol
    device_name::String
end

# Standard reference isotropic shielding constants (ppm)
# 1H and 13C: Tetramethylsilane (TMS)
# 15N: Nitromethane (CH3NO2)
# 19F: Trichlorofluoromethane (CFCl3)
# 31P: 85% Phosphoric Acid (H3PO4)
# 17O: Liquid Water (H2O)
const DEFAULT_NMR_REFERENCES = Dict{Symbol, Float64}(
    :H => 31.75,
    :C => 185.0,
    :N => -135.0,
    :F => 190.0,
    :P => 320.0,
    :O => -270.0
)

# Free-atom Lamb diamagnetic shielding constants σ_0^d (ppm)
const FREE_ATOM_SIGMA0_DIAMAGNETIC = Dict{Symbol, Float64}(
    :H => 17.75,
    :He => 59.94,
    :Li => 101.5,
    :Be => 150.0,
    :B  => 202.0,
    :C  => 260.7,
    :N  => 325.5,
    :O  => 395.1,
    :F  => 470.7,
    :P  => 961.1,
    :S  => 1050.5,
    :Cl => 1142.6
)

# Expectation values <r^-3> for valence p-orbitals in atomic units
const VALENCE_R3_INV = Dict{Symbol, Float64}(
    :H => 0.0,
    :C => 1.692,
    :N => 3.101,
    :O => 4.974,
    :F => 7.546,
    :P => 3.320,
    :S => 4.814,
    :Cl => 6.709
)

"""
    calc_nmr_shieldings(scf_res, mol::Molecule;
                        device::AbstractDevice=get_default_device(),
                        method::Symbol=:scf,
                        references::Dict{Symbol, Float64}=DEFAULT_NMR_REFERENCES)::NMRResult
Calculates NMR chemical shielding tensors, isotropic chemical shifts, and spin-spin J-couplings
accelerated on NVIDIA CUDA, AMD ROCm, Intel oneAPI GPUs, or CPU.
"""
function calc_nmr_shieldings(scf_res::Any, mol::Molecule;
                            device::AbstractDevice=get_default_device(),
                            method::Symbol=:scf,
                            references::Dict{Symbol, Float64}=DEFAULT_NMR_REFERENCES)::NMRResult
    nat = n_atoms(mol)

    # Extract MO energies, coefficients, and density
    eps, C, P = if hasproperty(scf_res, :mo_energies_hartree)
        # Hartree-Fock / DFT result
        scf_res.mo_energies_hartree, scf_res.mo_coefficients, scf_res.density_matrix
    elseif hasproperty(scf_res, :mo_energies)
        # Semi-empirical SCF result
        scf_res.mo_energies, scf_res.mo_coefficients, scf_res.density_matrix
    else
        error("Unsupported SCF result type for NMR calculation: $(typeof(scf_res))")
    end

    nb = length(eps)
    n_occ = if hasproperty(scf_res, :n_occ)
        scf_res.n_occ
    else
        div(sum(Elements.valence_electrons(mol.atoms[i].symbol) for i in 1:nat) - mol.total_charge, 2)
    end
    n_virt = nb - n_occ

    coords = coordinates_matrix(mol)
    charges = [mol.atoms[i].partial_charge for i in 1:nat]

    # Atom orbital mapping
    basis = SlaterOrbitals.build_valence_basis(mol)
    atom_basis_indices = [Int[] for _ in 1:nat]
    for (mu, orb) in enumerate(basis)
        push!(atom_basis_indices[orb.atom_index], mu)
    end

    # Local valence populations
    val_pops = zeros(Float64, nat)
    val_electrons = [Float64(Elements.valence_electrons(mol.atoms[i].symbol)) for i in 1:nat]
    for i in 1:nat
        val_pops[i] = sum(P[mu, mu] for mu in atom_basis_indices[i])
    end

    sigma0_d = [get(FREE_ATOM_SIGMA0_DIAMAGNETIC, mol.atoms[i].symbol, 200.0) for i in 1:nat]
    r3_inv = [get(VALENCE_R3_INV, mol.atoms[i].symbol, 0.0) for i in 1:nat]

    # 1. Evaluate Diamagnetic Shielding on Device
    coords_dev = to_device(coords, device)
    charges_dev = to_device(charges, device)
    sigma0_dev = to_device(sigma0_d, device)
    val_pops_dev = to_device(val_pops, device)
    val_e_dev = to_device(val_electrons, device)

    sigma_d_dev = GPUKernels.gpu_nmr_diamagnetic_shielding(coords_dev, charges_dev, sigma0_dev, val_pops_dev, val_e_dev)
    synchronize_device(device)
    sigma_d = to_host(sigma_d_dev)

    # 2. Build Angular Momentum Operator Matrix Elements L_ia for Paramagnetic Shielding
    # L_ia[alpha, i, a, atom]
    L_ia = zeros(Float64, 3, n_occ, n_virt, nat)

    for at in 1:nat
        indices = atom_basis_indices[at]
        # Locate px, py, pz orbitals for this atom
        px_idx = 0; py_idx = 0; pz_idx = 0
        for mu in indices
            orb = basis[mu]
            if orb.l == 1
                if orb.m_type == :px; px_idx = mu
                elseif orb.m_type == :py; py_idx = mu
                elseif orb.m_type == :pz; pz_idx = mu
                end
            end
        end

        if px_idx > 0 && py_idx > 0 && pz_idx > 0
            for i in 1:n_occ
                for (a_rel, a) in enumerate((n_occ+1):nb)
                    # L_z = px * py' - py * px'
                    L_ia[3, i, a_rel, at] = C[px_idx, i] * C[py_idx, a] - C[py_idx, i] * C[px_idx, a]
                    # L_y = pz * px' - px * pz'
                    L_ia[2, i, a_rel, at] = C[pz_idx, i] * C[px_idx, a] - C[px_idx, i] * C[pz_idx, a]
                    # L_x = py * pz' - pz * py'
                    L_ia[1, i, a_rel, at] = C[py_idx, i] * C[pz_idx, a] - C[pz_idx, i] * C[py_idx, a]
                end
            end
        elseif mol.atoms[at].symbol == :H
            # Hydrogen: paramagnetic neighbor anisotropy (Pople-Flygare)
            for i in 1:n_occ
                for (a_rel, a) in enumerate((n_occ+1):nb)
                    # Cross-orbital polarization
                    s_idx = indices[1]
                    c_ia = C[s_idx, i] * C[s_idx, a]
                    L_ia[1, i, a_rel, at] = 0.15 * c_ia
                    L_ia[2, i, a_rel, at] = 0.15 * c_ia
                    L_ia[3, i, a_rel, at] = 0.15 * c_ia
                end
            end
        end
    end

    # Orbital gaps delta_eps[i, a]
    delta_eps = zeros(Float64, n_occ, n_virt)
    for i in 1:n_occ
        for (a_rel, a) in enumerate((n_occ+1):nb)
            delta_eps[i, a_rel] = eps[a] - eps[i]
        end
    end

    # 3. Evaluate Paramagnetic Shielding Tensors on Device
    L_ia_dev = to_device(L_ia, device)
    delta_eps_dev = to_device(delta_eps, device)
    r3_inv_dev = to_device(r3_inv, device)

    sigma_p_dev = GPUKernels.gpu_nmr_paramagnetic_shielding(L_ia_dev, delta_eps_dev, r3_inv_dev)
    synchronize_device(device)
    sigma_p_all = to_host(sigma_p_dev)

    # 4. Assemble Shielding Tensors & Chemical Shifts for Active Nuclei
    nuclei = NMRNucleus[]

    for at in 1:nat
        sym = mol.atoms[at].symbol
        # Total tensor: sigma_d * I + sigma_p
        sigma_mat = sigma_d[at] .* Matrix(1.0I, 3, 3) .+ sigma_p_all[:, :, at]

        # Symmetrize and diagonalize principal components
        sigma_sym = 0.5 .* (sigma_mat .+ sigma_mat')
        eig = eigen(Symmetric(sigma_sym))
        evals = sort(eig.values) # sigma_11 <= sigma_22 <= sigma_33

        sigma_iso = tr(sigma_sym) / 3.0
        delta_sigma = evals[3] - 0.5 * (evals[1] + evals[2])
        denom_eta = evals[3] - sigma_iso
        eta = abs(denom_eta) > 1e-6 ? clamp((evals[2] - evals[1]) / denom_eta, 0.0, 1.0) : 0.0

        ref_val = get(references, sym, 0.0)
        chem_shift = ref_val - sigma_iso

        # Correct isotope designation
        isotope = if sym == :H; "1H"
        elseif sym == :C; "13C"
        elseif sym == :N; "15N"
        elseif sym == :F; "19F"
        elseif sym == :P; "31P"
        elseif sym == :O; "17O"
        else; string(sym)
        end

        st_tensor = NMRShieldingTensor(sigma_mat, sigma_iso, delta_sigma, eta, evals,
                                       sigma_d[at], tr(sigma_p_all[:, :, at]) / 3.0)
        push!(nuclei, NMRNucleus(at, sym, isotope, st_tensor, chem_shift))
    end

    # 5. Spin-Spin J-Coupling Matrix (Hz)
    couplings = zeros(Float64, nat, nat)
    dist_mat = GPUKernels.gpu_distance_matrix(coords)

    for i in 1:nat
        sym_i = mol.atoms[i].symbol
        s_i = atom_basis_indices[i][1]
        for j in (i+1):nat
            sym_j = mol.atoms[j].symbol
            s_j = atom_basis_indices[j][1]
            r_ij = dist_mat[i, j]

            j_val = 0.0
            # 1H - 1H couplings
            if sym_i == :H && sym_j == :H
                if r_ij < 2.0 # Geminal 2J_HH (e.g. -12 Hz)
                    j_val = -12.5 + (r_ij - 1.7) * 5.0
                elseif r_ij < 3.2 # Vicinal 3J_HH (Karplus relation ~ 2 - 14 Hz)
                    # Approximation: Karplus formula
                    j_val = 7.0 + 3.0 * (3.0 - r_ij)
                end
            # 13C - 1H one-bond coupling 1J_CH
            elseif (sym_i == :C && sym_j == :H) || (sym_i == :H && sym_j == :C)
                if r_ij < 1.25
                    # 1J_CH ~ 500 * s-character ~ 125 - 160 Hz
                    p_ch = P[s_i, s_j]
                    j_val = 125.0 + 30.0 * abs(p_ch)
                elseif r_ij < 2.3
                    # 2J_CH ~ -4 to 5 Hz
                    j_val = -4.5
                end
            # 13C - 13C coupling 1J_CC
            elseif sym_i == :C && sym_j == :C
                if r_ij < 1.6
                    j_val = 35.0 + 10.0 * (1.54 - r_ij)
                end
            # General Fermi contact model
            elseif r_ij < 2.0
                p_s = abs(P[s_i, s_j])
                j_val = 50.0 * (p_s^2) / max(r_ij, 0.5)
            end

            couplings[i, j] = j_val
            couplings[j, i] = j_val
        end
    end

    dev_str = device_name(device)
    return NMRResult(nuclei, couplings, references, method, dev_str)
end

"""
    simulate_nmr_spectrum(res::NMRResult;
                          nucleus::Symbol=:H,
                          ppm_min::Float64=nucleus == :H ? -1.0 : 0.0,
                          ppm_max::Float64=nucleus == :H ? 12.0 : 220.0,
                          n_points::Int=2000,
                          fwhm_hz::Float64=2.0,
                          b0_mhz::Float64=500.0,
                          device::AbstractDevice=get_default_device())
Simulates a continuous 1D NMR spectrum for the specified nucleus (:H or :C) with Lorentzian line shapes
accelerated on NVIDIA, AMD, or Intel GPU.
Returns (ppm_axis, intensity_array).
"""
function simulate_nmr_spectrum(res::NMRResult;
                               nucleus::Symbol=:H,
                               ppm_min::Float64=nucleus == :H ? -1.0 : 0.0,
                               ppm_max::Float64=nucleus == :H ? 12.0 : 220.0,
                               n_points::Int=2000,
                               fwhm_hz::Float64=2.0,
                               b0_mhz::Float64=500.0,
                               device::AbstractDevice=get_default_device())
    # Filter nuclei of selected isotope
    target_nuclei = filter(n -> n.symbol == nucleus, res.nuclei)
    if isempty(target_nuclei)
        ppm_grid = collect(range(ppm_min, ppm_max, length=n_points))
        return ppm_grid, zeros(Float64, n_points)
    end

    shifts = [n.chemical_shift_ppm for n in target_nuclei]
    intensities = fill(1.0, length(shifts))

    # Convert FWHM from Hz to ppm: FWHM_ppm = FWHM_Hz / B0_MHz
    fwhm_ppm = fwhm_hz / b0_mhz
    gamma_ppm = fwhm_ppm / 2.0

    ppm_grid = collect(range(ppm_min, ppm_max, length=n_points))

    # Evaluate on device
    ppm_dev = to_device(ppm_grid, device)
    shifts_dev = to_device(shifts, device)
    intensities_dev = to_device(intensities, device)

    spec_dev = GPUKernels.gpu_simulate_nmr_spectrum(ppm_dev, shifts_dev, intensities_dev, gamma_ppm)
    synchronize_device(device)
    spectrum = to_host(spec_dev)

    # Normalize max peak to 100.0
    max_val = maximum(spectrum)
    if max_val > 1e-8
        spectrum .= (spectrum ./ max_val) .* 100.0
    end

    return ppm_grid, spectrum
end

"""
    print_nmr_summary(res::NMRResult)
Prints comprehensive NMR chemical shifts, shielding components, and spin-spin couplings.
"""
function print_nmr_summary(res::NMRResult)
    println("==================================================================")
    println("       Nuclear Magnetic Resonance (NMR) Spectroscopy Results      ")
    println("==================================================================")
    @printf("  Calculation Method:       %s\n", uppercase(string(res.method)))
    @printf("  Acceleration Device:      %s\n", res.device_name)
    println("------------------------------------------------------------------")
    println(" Atom  Isotope   σ_iso (ppm)   σ_diam (ppm)   σ_param (ppm)   δ (ppm)")
    println("------------------------------------------------------------------")
    for n in res.nuclei
        @printf("  #%-3d  %-5s   %10.2f     %10.2f     %10.2f     %8.2f\n",
                n.atom_index, n.isotope, n.shielding.isotropic,
                n.shielding.diamagnetic, n.shielding.paramagnetic,
                n.chemical_shift_ppm)
    end
    println("------------------------------------------------------------------")
    println("  Key Indirect Spin-Spin Couplings (J > 1.0 Hz):")
    nat = length(res.nuclei)
    has_coupling = false
    for i in 1:nat
        for j in (i+1):nat
            j_val = res.couplings_hz[i, j]
            if abs(j_val) > 1.0
                has_coupling = true
                @printf("    J(%s#%d - %s#%d): %8.2f Hz\n",
                        res.nuclei[i].isotope, i, res.nuclei[j].isotope, j, j_val)
            end
        end
    end
    if !has_coupling
        println("    No significant couplings detected.")
    end
    println("==================================================================")
end

end # module NMRSpectroscopy
