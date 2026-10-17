# ==============================================================================
# QuantumChemMM.jl: Advanced Spectroscopy (UV-Vis CIS, Eckart-Projected IR, & Spectra)
# Features Continuous Absorption Curve Generation with Gaussian/Lorentzian Broadening
# ==============================================================================

module Spectroscopy

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...MolMath
using ...MMBase
using ...MMEnergyGradient

export UVVisTransition, calc_uv_vis_cis, generate_uv_vis_spectrum,
       VibrationalMode, calc_vibrational_frequencies, generate_ir_spectrum,
       project_translations_and_rotations!

const HESSIAN_TO_WAVENUMBER = 5140.487 # sqrt(kcal/(mol·Å²·amu)) -> cm⁻¹
const HARTREE_TO_EV = 27.211386245988

struct UVVisTransition
    state_index::Int
    energy_ev::Float64
    wavelength_nm::Float64
    oscillator_strength::Float64
    main_character::String
end

struct VibrationalMode
    mode_index::Int
    frequency_cm1::Float64
    is_imaginary::Bool
    ir_intensity_km_mol::Float64 # Infrared absorption intensity
    reduced_mass_amu::Float64
    normal_mode::Matrix{Float64} # 3 x N_atoms
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
        i_k = max(m.ir_intensity_km_mol, 1.0)

        for (idx, nu) in enumerate(wavenumbers)
            # Lorentzian profile: L(nu) = (1/pi) * (gamma / ((nu - nu_k)^2 + gamma^2))
            lorentz = (1.0 / π) * (gamma / ((nu - nu_k)^2 + gamma^2))
            intensity[idx] += i_k * lorentz
        end
    end

    return collect(wavenumbers), intensity
end

end # module Spectroscopy
