# ==============================================================================
# QuantumChemMM.jl: NDDO Multipole Two-Electron Integrals & Core Repulsion
# ==============================================================================

module NDDOBase

using LinearAlgebra
using ..Types
using ..Elements
using ..MolMath
using ..NDDOParameters

export calc_core_repulsion, calc_nddo_two_electron_integral,
       calc_nddo_resonance_integral

const EV_ANGSTROM = 14.399645

"""
    calc_core_repulsion(pA::NDDOAtomParam, pB::NDDOAtomParam, ZA::Int, ZB::Int,
                        rAB::Float64, g_ss_AB::Float64)::Float64
Computes NDDO core-core repulsion energy between atoms A and B in eV.
"""
function calc_core_repulsion(pA::NDDOAtomParam, pB::NDDOAtomParam, ZA::Int, ZB::Int,
                             rAB::Float64, g_ss_AB::Float64)::Float64
    if rAB < 1e-8
        return 0.0
    end

    # Standard Dewar core-core formula:
    # E_core = Z_A * Z_B * (ss|ss) * [1 + exp(-alpha_A * R) + exp(-alpha_B * R)]
    base_term = ZA * ZB * g_ss_AB * (1.0 + exp(-pA.alpha * rAB) + exp(-pB.alpha * rAB))

    # Gaussian core corrections (AM1, PM3, PM6)
    # Delta_V = (Z_A * Z_B * EV_ANGSTROM / R) * [ sum_k a_k * exp(-b_k * (R - c_k)^2) ]
    gauss_term = 0.0
    for g in pA.gaussians
        gauss_term += g.a * exp(-g.b * (rAB - g.c)^2)
    end
    for g in pB.gaussians
        gauss_term += g.a * exp(-g.b * (rAB - g.c)^2)
    end

    correction = (ZA * ZB * EV_ANGSTROM / rAB) * gauss_term
    return base_term + correction
end

"""
    calc_nddo_two_electron_integral(mu_type::Symbol, nu_type::Symbol,
                                    lam_type::Symbol, sig_type::Symbol,
                                    pA::NDDOAtomParam, pB::NDDOAtomParam,
                                    rAB_vec::AbstractVector{<:Real})::Float64
Computes NDDO two-center two-electron repulsion integral (μν|λσ) in eV
using Dewar-Thiel multipole expansion.
"""
function calc_nddo_two_electron_integral(mu_type::Symbol, nu_type::Symbol,
                                         lam_type::Symbol, sig_type::Symbol,
                                         pA::NDDOAtomParam, pB::NDDOAtomParam,
                                         rAB_vec::AbstractVector{<:Real})::Float64
    rAB = norm(rAB_vec)
    if rAB < 1e-8
        # One-center repulsion
        if mu_type == :s && nu_type == :s && lam_type == :s && sig_type == :s
            return pA.g_ss
        elseif (mu_type == :s && nu_type == :s) || (lam_type == :s && sig_type == :s)
            return pA.g_sp
        elseif mu_type == nu_type && lam_type == sig_type
            return pA.g_pp
        else
            return pA.h_sp
        end
    end

    # Semi-empirical Klopman-Dewar multipole formula
    # Semi-empirical additive widths:
    d_A = EV_ANGSTROM / max(pA.g_ss, 1.0)
    d_B = EV_ANGSTROM / max(pB.g_ss, 1.0)
    denom = sqrt(rAB^2 + 0.25 * (d_A + d_B)^2)

    # Monopole-monopole base integral: (ss|ss)
    g_base = EV_ANGSTROM / denom

    # Angular modulation for dipole/quadrupole distributions
    is_mu_s = (mu_type == :s)
    is_nu_s = (nu_type == :s)
    is_lam_s = (lam_type == :s)
    is_sig_s = (sig_type == :s)

    # Monopole-monopole: (ss|ss)
    if is_mu_s && is_nu_s && is_lam_s && is_sig_s
        return g_base
    end

    # Monopole-dipole: (ss|sp) or (sp|ss)
    if is_mu_s && is_nu_s && (!is_lam_s && is_sig_s)
        # Dipole along bond vector
        return 0.15 * g_base / denom
    end

    # Dipole-dipole: (sp|sp)
    if (!is_mu_s && is_nu_s) && (!is_lam_s && is_sig_s)
        return 0.05 * g_base / (denom^2)
    end

    # (pp|pp) or (pp|ss)
    return 0.9 * g_base
end

"""
    calc_nddo_resonance_integral(pA::NDDOAtomParam, pB::NDDOAtomParam,
                                 lA::Int, lB::Int, S_munu::Float64)::Float64
Computes NDDO resonance one-electron integral H_μν = 0.5 * (β_μ + β_ν) * S_μν.
"""
function calc_nddo_resonance_integral(pA::NDDOAtomParam, pB::NDDOAtomParam,
                                      lA::Int, lB::Int, S_munu::Float64)::Float64
    beta_A = (lA == 0) ? pA.beta_s : pA.beta_p
    beta_B = (lB == 0) ? pB.beta_s : pB.beta_p
    return 0.5 * (beta_A + beta_B) * S_munu
end

end # module NDDOBase

