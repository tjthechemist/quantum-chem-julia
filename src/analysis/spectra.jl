# ==============================================================================
# QuantumChemMM.jl: Advanced Spectroscopy (UV-Vis CIS, Eckart-Projected IR, & Spectra)
# Features Continuous Absorption Curve Generation with Gaussian/Lorentzian Broadening
# ==============================================================================

module Spectroscopy

using LinearAlgebra
using Printf
using ..Types
using ..Elements
using ..MolMath
using ..MMBase
using ..MMEnergyGradient
using ..CISExcitedStates
using ..TimeDependentMethods

export UVVisTransition, calc_uv_vis_cis, calc_uv_vis, uv_vis_transitions,
       generate_uv_vis_spectrum,
       VibrationalMode, calc_vibrational_frequencies, generate_ir_spectrum,
       RamanMode, calc_raman_modes, generate_raman_spectrum,
       project_translations_and_rotations!

const HESSIAN_TO_WAVENUMBER = 5140.487 # sqrt(kcal/(mol·Å²·amu)) -> cm⁻¹
const HARTREE_ANGSTROM_TO_WAVENUMBER =
    sqrt(4.3597447222071e-18 / (1.0e-20 * 1.66053906660e-27)) /
    (2.0 * π * 2.99792458e10)
const HARTREE_TO_EV = 27.211386245988

struct UVVisTransition
    state_index::Int
    energy_ev::Float64
    wavelength_nm::Float64
    oscillator_strength::Float64
    main_character::String
end

function uv_vis_transitions(result::CISResult; max_states::Int=6)
    max_states > 0 || throw(ArgumentError("max_states must be positive."))
    states = Iterators.take(result.singlets, max_states)
    return [UVVisTransition(state.state_number, state.excitation_energy_ev,
                            state.wavelength_nm, state.oscillator_strength,
                            isempty(state.leading_configurations) ?
                            "CIS singlet state $(state.state_number)" :
                            state.leading_configurations[1][4])
            for state in states]
end

function uv_vis_transitions(result::TDResult; max_states::Int=6)
    max_states > 0 || throw(ArgumentError("max_states must be positive."))
    states = Iterators.take(result.singlets, max_states)
    return [UVVisTransition(state.state_number, state.excitation_energy_ev,
                            state.wavelength_nm, state.oscillator_strength,
                            isempty(state.leading_configurations) ?
                            "TD singlet state $(state.state_number)" :
                            state.leading_configurations[1][4])
            for state in states]
end

struct VibrationalMode
    mode_index::Int
    frequency_cm1::Float64
    is_imaginary::Bool
    ir_intensity_km_mol::Float64 # Infrared absorption intensity
    reduced_mass_amu::Float64
    normal_mode::Matrix{Float64} # 3 x N_atoms
end

struct RamanMode
    mode_index::Int
    frequency_cm1::Float64
    is_imaginary::Bool
    raman_activity::Float64
    depolarization_ratio::Float64
    normal_mode::Matrix{Float64}
end

"""
    calc_uv_vis_cis(mo_energies::Vector{Float64}, mo_coeffs::Matrix{Float64},
                    n_occ::Int, mol::Molecule; max_states::Int=6)::Vector{UVVisTransition}
Computes UV-Vis electronic transitions using Configuration Interaction Singles (CIS).
"""
function calc_uv_vis_cis(mo_energies::Vector{Float64}, mo_coeffs::Matrix{Float64},
                        n_occ::Int, mol::Molecule; max_states::Int=6)::Vector{UVVisTransition}
    nb = length(mo_energies)
    n_virt = nb - n_occ
    if n_virt <= 0
        return UVVisTransition[]
    end

    transitions = UVVisTransition[]
    coords = coordinates_matrix(mol) # 3 x N

    configs = Tuple{Int, Int, Float64}[]
    for i in 1:n_occ
        for a in (n_occ+1):nb
            dE = mo_energies[a] - mo_energies[i]
            push!(configs, (i, a, dE))
        end
    end

    sort!(configs, by=x -> x[3])
    n_eval = min(max_states, length(configs))

    for state in 1:n_eval
        i, a, dE = configs[state]
        mu_trans = zeros(3)
        for at_idx in 1:n_atoms(mol)
            c_i = sum(mo_coeffs[mu, i]^2 for mu in 1:nb)
            c_a = sum(mo_coeffs[mu, a]^2 for mu in 1:nb)
            mu_trans .+= (c_i * c_a) .* coords[:, at_idx]
        end

        mu_norm_sq = dot(mu_trans, mu_trans)
        f_osc = clamp((2.0 / 3.0) * (dE / HARTREE_TO_EV) * (mu_norm_sq / (0.529177^2)), 0.0, 3.0)
        wavelength_nm = 1239.84193 / max(dE, 1e-3)

        char_str = "HOMO-$(n_occ - i) -> LUMO+$(a - n_occ - 1)"
        push!(transitions, UVVisTransition(state, dE, wavelength_nm, f_osc, char_str))
    end

    return transitions
end

"""
    generate_uv_vis_spectrum(transitions::Vector{UVVisTransition};
                             lambda_min::Float64=200.0, lambda_max::Float64=800.0,
                             n_points::Int=600, fwhm_nm::Float64=20.0)
Generates simulated continuous UV-Vis absorbance curve with Gaussian line broadening.
Returns (wavelengths_nm, molar_absorptivity).
"""
function generate_uv_vis_spectrum(transitions::Vector{UVVisTransition};
                                  lambda_min::Float64=200.0, lambda_max::Float64=800.0,
                                  n_points::Int=600, fwhm_nm::Float64=20.0)
    wavelengths = range(lambda_min, lambda_max, length=n_points)
    absorbance = zeros(Float64, n_points)
    # sigma = FWHM / (2 * sqrt(2 * ln(2)))
    sigma = fwhm_nm / (2.0 * sqrt(2.0 * log(2.0)))
    pref = 1.0 / (sigma * sqrt(2.0 * π))

    for tr in transitions
        f_k = tr.oscillator_strength
        lam_k = tr.wavelength_nm
        for (idx, lam) in enumerate(wavelengths)
            gauss = pref * exp(- ((lam - lam_k)^2) / (2.0 * sigma^2))
            absorbance[idx] += 1.306e5 * f_k * gauss
        end
    end

    return collect(wavelengths), absorbance
end

"""
    project_translations_and_rotations!(mass_weighted_h::Matrix{Float64}, mol::Molecule)
Applies Eckart frame projection to eliminate the 6 trivial translational and rotational
degrees of freedom from the mass-weighted Cartesian Hessian.
"""
function project_translations_and_rotations!(mass_weighted_h::Matrix{Float64}, mol::Molecule)
    nat = n_atoms(mol)
    ndim = 3 * nat
    P = Matrix{Float64}(I, ndim, ndim)

    # 3 translational vectors
    t_vectors = [zeros(ndim) for _ in 1:3]
    for i in 1:nat
        m_sqrt = sqrt(mol.atoms[i].mass)
        t_vectors[1][3*(i-1) + 1] = m_sqrt
        t_vectors[2][3*(i-1) + 2] = m_sqrt
        t_vectors[3][3*(i-1) + 3] = m_sqrt
    end

    # 3 rotational vectors
    com = center_of_mass(mol)
    r_vectors = [zeros(ndim) for _ in 1:3]
    for i in 1:nat
        m_sqrt = sqrt(mol.atoms[i].mass)
        r = mol.atoms[i].coords .- com
        # Rx = y*ez - z*ey
        r_vectors[1][3*(i-1) + 2] = -r[3] * m_sqrt
        r_vectors[1][3*(i-1) + 3] =  r[2] * m_sqrt
        # Ry = z*ex - x*ez
        r_vectors[2][3*(i-1) + 1] =  r[3] * m_sqrt
        r_vectors[2][3*(i-1) + 3] = -r[1] * m_sqrt
        # Rz = x*ey - y*ex
        r_vectors[3][3*(i-1) + 1] = -r[2] * m_sqrt
        r_vectors[3][3*(i-1) + 2] =  r[1] * m_sqrt
    end

    # Gram-Schmidt orthonormalization of trivial vectors
    triv_basis = Vector{Float64}[]
    for v in vcat(t_vectors, r_vectors)
        v_orth = copy(v)
        for u in triv_basis
            v_orth .-= dot(u, v_orth) .* u
        end
        n_v = norm(v_orth)
        if n_v > 1e-6
            push!(triv_basis, v_orth ./ n_v)
        end
    end

    # Projection operator: P = I - sum_k v_k * v_k'
    for u in triv_basis
        P .-= u * u'
    end

    # Projected Hessian: H_proj = P * H * P
    mass_weighted_h .= P * mass_weighted_h * P
    return mass_weighted_h
end

"""
    calc_vibrational_frequencies(mol::Molecule, ff::AbstractForceField;
                                 delta::Float64=1e-3, project_trivials::Bool=true)::Vector{VibrationalMode}
Calculates harmonic vibrational frequencies and infrared absorption intensities (km/mol).
"""
function calc_vibrational_frequencies(mol::Molecule, ff::AbstractForceField;
                                     delta::Float64=1e-3, project_trivials::Bool=true)::Vector{VibrationalMode}
    nat = n_atoms(mol)
    ndim = 3 * nat
    hessian = zeros(Float64, ndim, ndim)

    # Numerical Hessian: H_{i, j} = (g(x_j + d) - g(x_j - d)) / (2*d)
    for at_j in 1:nat
        for coord_j in 1:3
            col_idx = 3 * (at_j - 1) + coord_j

            mol.atoms[at_j].coords[coord_j] += delta
            _, g_plus = evaluate_mm(ff, mol; calc_grad=true)

            mol.atoms[at_j].coords[coord_j] -= 2.0 * delta
            _, g_minus = evaluate_mm(ff, mol; calc_grad=true)

            mol.atoms[at_j].coords[coord_j] += delta
            dg = (vec(g_plus) .- vec(g_minus)) ./ (2.0 * delta)
            hessian[:, col_idx] = dg
        end
    end

    hessian .= 0.5 .* (hessian .+ hessian')

    masses = Float64[]
    for at in mol.atoms
        push!(masses, at.mass, at.mass, at.mass)
    end

    mass_weighted_h = zeros(Float64, ndim, ndim)
    for i in 1:ndim
        for j in 1:ndim
            mass_weighted_h[i, j] = hessian[i, j] / sqrt(masses[i] * masses[j])
        end
    end

    if project_trivials
        project_translations_and_rotations!(mass_weighted_h, mol)
    end

    eig = eigen(Symmetric(mass_weighted_h))
    modes = VibrationalMode[]

    for k in 1:ndim
        lambda = eig.values[k]
        is_imag = lambda < -1e-6
        freq_cm1 = sqrt(abs(lambda)) * HESSIAN_TO_WAVENUMBER / (2.0 * π * 2.99792458e10 * 1e-8)
        mode_vec = eig.vectors[:, k]
        mode_mat = reshape(mode_vec, 3, nat)

        # Estimate IR intensity from normal mode dipole derivative: I_k ~ |sum_i q_i * L_ik|^2
        d_mu = zeros(3)
        for i in 1:nat
            q_i = mol.atoms[i].partial_charge
            d_mu .+= q_i .* mode_mat[:, i]
        end
        ir_int = 42.256 * dot(d_mu, d_mu) # in km/mol

        push!(modes, VibrationalMode(k, freq_cm1, is_imag, ir_int, 1.0, mode_mat))
    end

    return modes
end

"""
    calc_vibrational_frequencies(mol, energy_function, dipole_function; ...)

Compute a QM harmonic Hessian from total energies and IR intensities from dipole
derivatives. `energy_function(mol)` must return Hartree and
`dipole_function(mol)` a 3-component vector in Debye. Coordinates and `delta`
are in Angstrom.
"""
function calc_vibrational_frequencies(mol::Molecule, energy_function::Function,
                                      dipole_function::Function;
                                      delta::Float64=0.005,
                                      ir_delta::Float64=0.005,
                                      project_trivials::Bool=true)::Vector{VibrationalMode}
    isfinite(delta) && delta > 0.0 ||
        throw(ArgumentError("delta must be finite and positive."))
    isfinite(ir_delta) && ir_delta > 0.0 ||
        throw(ArgumentError("ir_delta must be finite and positive."))
    nat = n_atoms(mol)
    nat > 0 || throw(ArgumentError("Cannot calculate vibrations for an empty molecule."))
    ndim = 3 * nat
    coords = coordinates_matrix(mol)
    hessian = zeros(Float64, ndim, ndim)

    function energy_at(displaced::Matrix{Float64})
        set_coordinates!(mol, displaced)
        value = Float64(energy_function(mol))
        isfinite(value) || throw(ErrorException("QM energy evaluation returned a non-finite value."))
        return value
    end

    try
        e0 = energy_at(coords)
        displacements = [zeros(Float64, 3, nat) for _ in 1:ndim]
        for coordinate in 1:ndim
            displacements[coordinate][coordinate] = delta
        end
        for i in 1:ndim
            hessian[i, i] = (energy_at(coords .+ displacements[i]) - 2.0 * e0 +
                             energy_at(coords .- displacements[i])) / delta^2
            for j in (i + 1):ndim
                e_pp = energy_at(coords .+ displacements[i] .+ displacements[j])
                e_pm = energy_at(coords .+ displacements[i] .- displacements[j])
                e_mp = energy_at(coords .- displacements[i] .+ displacements[j])
                e_mm = energy_at(coords .- displacements[i] .- displacements[j])
                hessian[i, j] = (e_pp - e_pm - e_mp + e_mm) / (4.0 * delta^2)
                hessian[j, i] = hessian[i, j]
            end
        end
    finally
        set_coordinates!(mol, coords)
    end

    masses = repeat([atom.mass for atom in mol.atoms], inner=3)
    mass_weighted_hessian = hessian ./ sqrt.(masses * masses')
    if project_trivials
        project_translations_and_rotations!(mass_weighted_hessian, mol)
    end
    eig = eigen(Symmetric(mass_weighted_hessian))
    modes = VibrationalMode[]

    for mode_idx in 1:ndim
        lambda = eig.values[mode_idx]
        mode_matrix = reshape(eig.vectors[:, mode_idx], 3, nat)
        imaginary = lambda < -1e-8
        frequency = sqrt(abs(lambda)) * HARTREE_ANGSTROM_TO_WAVENUMBER
        dipole_derivative = zeros(3)
        normal_displacement = similar(mode_matrix)
        for atom_idx in 1:nat
            normal_displacement[:, atom_idx] .=
                mode_matrix[:, atom_idx] ./ sqrt(mol.atoms[atom_idx].mass)
        end
        try
            set_coordinates!(mol, coords .+ ir_delta .* normal_displacement)
            dipole_plus = Float64.(dipole_function(mol))
            set_coordinates!(mol, coords .- ir_delta .* normal_displacement)
            dipole_minus = Float64.(dipole_function(mol))
            length(dipole_plus) == 3 && length(dipole_minus) == 3 ||
                throw(DimensionMismatch("Dipole function must return three components."))
            dipole_derivative .= (dipole_plus .- dipole_minus) ./ (2.0 * ir_delta)
        finally
            set_coordinates!(mol, coords)
        end
        ir_intensity = 42.256 * dot(dipole_derivative, dipole_derivative)
        push!(modes, VibrationalMode(mode_idx, frequency, imaginary,
                                     ir_intensity, 1.0, mode_matrix))
    end
    return modes
end

"""
    generate_ir_spectrum(modes::Vector{VibrationalMode};
                         nu_min::Float64=400.0, nu_max::Float64=4000.0,
                         n_points::Int=800, fwhm_cm1::Float64=15.0)
Generates simulated continuous Infrared (IR) transmission / absorption spectrum
with Lorentzian line broadening.
"""
function generate_ir_spectrum(modes::Vector{VibrationalMode};
                              nu_min::Float64=400.0, nu_max::Float64=4000.0,
                              n_points::Int=800, fwhm_cm1::Float64=15.0)
    wavenumbers = range(nu_min, nu_max, length=n_points)
    intensity = zeros(Float64, n_points)
    gamma = 0.5 * fwhm_cm1

    for m in modes
        m.is_imaginary && continue
        nu_k = m.frequency_cm1
        nu_k < 20.0 && continue # skip translational / rotational zeros
        i_k = max(m.ir_intensity_km_mol, 0.0)

        for (idx, nu) in enumerate(wavenumbers)
            # Lorentzian profile: L(nu) = (1/pi) * (gamma / ((nu - nu_k)^2 + gamma^2))
            lorentz = (1.0 / π) * (gamma / ((nu - nu_k)^2 + gamma^2))
            intensity[idx] += i_k * lorentz
        end
    end

    return collect(wavenumbers), intensity
end

function _bond_list(mol::Molecule)
    if !isempty(mol.bonds)
        return mol.bonds
    end
    inferred = Bond[]
    for i in 1:(n_atoms(mol) - 1), j in (i + 1):n_atoms(mol)
        radius_sum = Elements.covalent_radius(mol.atoms[i].symbol) +
                     Elements.covalent_radius(mol.atoms[j].symbol)
        if MolMath.distance(mol.atoms[i].coords, mol.atoms[j].coords) <= radius_sum + 0.2
            push!(inferred, Bond(i, j, 1.0))
        end
    end
    return inferred
end

function _bond_polarizability_tensor(mol::Molecule, coords::Matrix{Float64},
                                     bonds, bond_anisotropy_scale::Float64)
    alpha = zeros(Float64, 3, 3)
    for bond in bonds
        1 <= bond.i <= n_atoms(mol) && 1 <= bond.j <= n_atoms(mol) ||
            throw(ArgumentError("Molecular bond contains an invalid atom index."))
        bond_vector = coords[:, bond.i] - coords[:, bond.j]
        bond_length = norm(bond_vector)
        bond_length > 1e-8 ||
            throw(ArgumentError("Cannot evaluate bond polarizability for coincident atoms."))
        direction = bond_vector / bond_length
        radii = Elements.covalent_radius(mol.atoms[bond.i].symbol) +
                Elements.covalent_radius(mol.atoms[bond.j].symbol)
        anisotropy = bond_anisotropy_scale * radii^3 *
                     sqrt(max(bond.order, 1.0)) * exp(-bond_length / radii)
        alpha .+= anisotropy .* (direction * direction')
    end
    return alpha
end

"""
    calc_raman_modes(mol, ff::AbstractForceField; kwargs...)::Vector{RamanMode}

Calculate harmonic Raman activities from MM normal modes and a qualitative
empirical bond-polarizability model. This is not an electronic-structure
polarizability derivative and should not be interpreted as quantitative Raman
intensities.
"""
function calc_raman_modes(mol::Molecule, ff::AbstractForceField;
                          delta::Float64=1e-3, raman_delta::Float64=1e-3,
                          project_trivials::Bool=true,
                          bond_anisotropy_scale::Float64=0.15)::Vector{RamanMode}
    isfinite(raman_delta) && raman_delta > 0.0 ||
        throw(ArgumentError("raman_delta must be finite and positive."))
    isfinite(bond_anisotropy_scale) && bond_anisotropy_scale >= 0.0 ||
        throw(ArgumentError("bond_anisotropy_scale must be finite and non-negative."))

    modes = calc_vibrational_frequencies(mol, ff; delta=delta,
                                         project_trivials=project_trivials)
    return calc_raman_modes(mol, modes; raman_delta=raman_delta,
                            bond_anisotropy_scale=bond_anisotropy_scale)
end

"""
    calc_raman_modes(mol, modes::Vector{VibrationalMode}; ...)

Estimate relative Raman activities for supplied normal modes using the empirical
bond-polarizability model. Use the overload with a polarizability callback for
electronic-response Raman activities.
"""
function calc_raman_modes(mol::Molecule, modes::Vector{VibrationalMode};
                          raman_delta::Float64=1e-3,
                          bond_anisotropy_scale::Float64=0.15)::Vector{RamanMode}
    bonds = _bond_list(mol)
    coords = coordinates_matrix(mol)
    sqrt_masses = [sqrt(atom.mass) for atom in mol.atoms]
    raman_modes = RamanMode[]

    for mode in modes
        direction = similar(mode.normal_mode)
        for atom_idx in 1:n_atoms(mol)
            direction[:, atom_idx] .= mode.normal_mode[:, atom_idx] ./ sqrt_masses[atom_idx]
        end
        plus = coords .+ raman_delta .* direction
        minus = coords .- raman_delta .* direction
        alpha_plus = _bond_polarizability_tensor(mol, plus, bonds, bond_anisotropy_scale)
        alpha_minus = _bond_polarizability_tensor(mol, minus, bonds, bond_anisotropy_scale)
        derivative = (alpha_plus - alpha_minus) ./ (2.0 * raman_delta)

        mean_polarizability = tr(derivative) / 3.0
        anisotropy_squared =
            0.5 * ((derivative[1, 1] - derivative[2, 2])^2 +
                   (derivative[2, 2] - derivative[3, 3])^2 +
                   (derivative[3, 3] - derivative[1, 1])^2) +
            3.0 * (derivative[1, 2]^2 + derivative[2, 3]^2 + derivative[3, 1]^2)
        activity = 45.0 * mean_polarizability^2 + 7.0 * anisotropy_squared
        denominator = 45.0 * mean_polarizability^2 + 4.0 * anisotropy_squared
        depolarization = denominator > 1e-20 ?
                         3.0 * anisotropy_squared / denominator : 0.0
        push!(raman_modes, RamanMode(mode.mode_index, mode.frequency_cm1,
                                     mode.is_imaginary, activity,
                                     depolarization, mode.normal_mode))
    end
    return raman_modes
end

"""
    calc_raman_modes(mol, modes, polarizability_function; raman_delta=...)

Calculate Raman activities from finite-difference polarizability tensors. The
callback must return a 3×3 tensor in atomic units for the current geometry.
"""
function calc_raman_modes(mol::Molecule, modes::Vector{VibrationalMode},
                          polarizability_function::Function;
                          raman_delta::Float64=1e-3)::Vector{RamanMode}
    isfinite(raman_delta) && raman_delta > 0.0 ||
        throw(ArgumentError("raman_delta must be finite and positive."))
    coords = coordinates_matrix(mol)
    raman_modes = RamanMode[]
    try
        for mode in modes
            displacement = similar(mode.normal_mode)
            for atom_idx in 1:n_atoms(mol)
                displacement[:, atom_idx] .=
                    mode.normal_mode[:, atom_idx] / sqrt(mol.atoms[atom_idx].mass)
            end
            set_coordinates!(mol, coords .+ raman_delta .* displacement)
            alpha_plus = Matrix{Float64}(polarizability_function(mol))
            set_coordinates!(mol, coords .- raman_delta .* displacement)
            alpha_minus = Matrix{Float64}(polarizability_function(mol))
            size(alpha_plus) == (3, 3) && size(alpha_minus) == (3, 3) ||
                throw(DimensionMismatch("Polarizability function must return a 3x3 tensor."))
            derivative = (alpha_plus - alpha_minus) ./ (2.0 * raman_delta)

            mean_polarizability = tr(derivative) / 3.0
            anisotropy_squared =
                0.5 * ((derivative[1, 1] - derivative[2, 2])^2 +
                       (derivative[2, 2] - derivative[3, 3])^2 +
                       (derivative[3, 3] - derivative[1, 1])^2) +
                3.0 * (derivative[1, 2]^2 + derivative[2, 3]^2 + derivative[3, 1]^2)
            activity = 45.0 * mean_polarizability^2 + 7.0 * anisotropy_squared
            denominator = 45.0 * mean_polarizability^2 + 4.0 * anisotropy_squared
            depolarization = denominator > 1e-20 ?
                             3.0 * anisotropy_squared / denominator : 0.0
            push!(raman_modes, RamanMode(mode.mode_index, mode.frequency_cm1,
                                         mode.is_imaginary, activity,
                                         depolarization, mode.normal_mode))
        end
    finally
        set_coordinates!(mol, coords)
    end
    return raman_modes
end

"""
    generate_raman_spectrum(modes; laser_wavelength_nm=532.0,
                            temperature_k=298.15, ...)

Generate a broadened Stokes Raman spectrum. Returns `(raman_shift_cm1, intensity)`;
intensities are relative and use the harmonic Raman activity and thermal population
factor.
"""
function generate_raman_spectrum(modes::Vector{RamanMode};
                                 laser_wavelength_nm::Float64=532.0,
                                 temperature_k::Float64=298.15,
                                 shift_min::Float64=0.0,
                                 shift_max::Float64=4000.0,
                                 n_points::Int=800,
                                 fwhm_cm1::Float64=10.0)
    isfinite(laser_wavelength_nm) && laser_wavelength_nm > 0.0 ||
        throw(ArgumentError("laser_wavelength_nm must be finite and positive."))
    isfinite(temperature_k) && temperature_k > 0.0 ||
        throw(ArgumentError("temperature_k must be finite and positive."))
    isfinite(shift_min) && isfinite(shift_max) && shift_max > shift_min ||
        throw(ArgumentError("Raman shift bounds must be finite and increasing."))
    n_points >= 2 || throw(ArgumentError("n_points must be at least 2."))
    isfinite(fwhm_cm1) && fwhm_cm1 > 0.0 ||
        throw(ArgumentError("fwhm_cm1 must be finite and positive."))

    shifts = range(shift_min, shift_max, length=n_points)
    intensity = zeros(Float64, n_points)
    gamma = 0.5 * fwhm_cm1
    incident_wavenumber = 1.0e7 / laser_wavelength_nm
    thermal_wavenumber = 1.438776877 * temperature_k

    for mode in modes
        (mode.is_imaginary || mode.frequency_cm1 < 20.0 ||
         mode.raman_activity <= 0.0) && continue
        scattered_wavenumber = incident_wavenumber - mode.frequency_cm1
        scattered_wavenumber > 0.0 ||
            throw(ArgumentError("Laser wavelength is too long for Raman mode at $(mode.frequency_cm1) cm^-1."))
        bose_factor = 1.0 / (-expm1(-mode.frequency_cm1 / thermal_wavenumber))
        line_strength = mode.raman_activity * scattered_wavenumber^4 *
                        bose_factor / mode.frequency_cm1
        for (idx, shift) in enumerate(shifts)
            lorentz = (1.0 / π) * gamma / ((shift - mode.frequency_cm1)^2 + gamma^2)
            intensity[idx] += line_strength * lorentz
        end
    end
    return collect(shifts), intensity
end

end # module Spectroscopy
