# ==============================================================================
# QuantumChemMM.jl: RRHO Ideal-Gas Thermochemistry (ZPVE, Enthalpy, Entropy, Gibbs Free Energy)
# ==============================================================================

module Thermochemistry

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ..Spectroscopy

export ThermochemistryResult, calc_thermochemistry, print_thermo_summary

# Universal constants
const R_IDEAL = 1.98720425864083e-3 # kcal/(mol·K)
const KB_CGS = 1.380649e-16         # erg/K
const H_CGS = 6.62607015e-27        # erg·s
const C_LIGHT_CGS = 2.99792458e10   # cm/s
const NA = 6.02214076e23
const PLANCK_PER_C_IN_KCAL = 2.8591436e-3 # (h * c * NA) in (kcal/mol) / cm⁻¹

struct ThermochemistryResult
    temperature_k::Float64
    pressure_atm::Float64
    zpve_kcal::Float64
    u_trans_kcal::Float64
    u_rot_kcal::Float64
    u_vib_kcal::Float64
    u_thermal_kcal::Float64
    s_trans_cal_mol_k::Float64
    s_rot_cal_mol_k::Float64
    s_vib_cal_mol_k::Float64
    total_entropy_cal_mol_k::Float64
    enthalpy_corr_kcal::Float64
    gibbs_corr_kcal::Float64
end

"""
    calc_thermochemistry(mol::Molecule, freqs_cm1::Vector{Float64};
                         temp_k::Float64=298.15, pressure_atm::Float64=1.0)::ThermochemistryResult
Computes standard ideal-gas RRHO thermochemical corrections.
"""
function calc_thermochemistry(mol::Molecule, freqs_cm1::Vector{Float64};
                             temp_k::Float64=298.15, pressure_atm::Float64=1.0)::ThermochemistryResult
    nat = n_atoms(mol)

    # Filter real, physical vibrational frequencies (ignore 6 translational/rotational zeros)
    real_freqs = filter(f -> f > 20.0, freqs_cm1)

    # 1. Zero-Point Vibrational Energy: ZPVE = 0.5 * sum_i (h * c * nu_i)
    zpve_kcal = 0.5 * sum(real_freqs) * PLANCK_PER_C_IN_KCAL

    # 2. Vibrational Thermal Energy & Entropy
    u_vib = 0.0
    s_vib = 0.0

    for nu in real_freqs
        theta_vib = (PLANCK_PER_C_IN_KCAL * nu) / R_IDEAL # vibrational characteristic temp
        x = theta_vib / temp_k
        if x < 100.0
            exp_term = exp(-x)
            # U_vib = R * theta * (1/(exp(x)-1))
            u_vib += R_IDEAL * theta_vib * (exp_term / (1.0 - exp_term))
            # S_vib = R * [ x / (exp(x) - 1) - ln(1 - exp(-x)) ]
            s_vib += (R_IDEAL * 1000.0) * ( (x * exp_term / (1.0 - exp_term)) - log(1.0 - exp_term) )
        end
    end

    # 3. Translational Contributions (Sackur-Tetrode equation)
    u_trans = 1.5 * R_IDEAL * temp_k
    total_m_amu = sum(at.mass for at in mol.atoms)
    # S_trans
    s_trans = (R_IDEAL * 1000.0) * (1.5 * log(total_m_amu) + 2.5 * log(temp_k) - log(pressure_atm) - 1.1648)

    # 4. Rotational Contributions
    is_linear = (nat == 2)
    u_rot = is_linear ? (R_IDEAL * temp_k) : (1.5 * R_IDEAL * temp_k)
    s_rot = (R_IDEAL * 1000.0) * (is_linear ? 15.0 : 25.0) # Standard approximation

    # 5. Total Enthalpy and Gibbs Free Energy corrections
    u_thermal = u_trans + u_rot + u_vib
    total_s = s_trans + s_rot + s_vib
    # H_corr = ZPVE + U_thermal + R*T
    h_corr = zpve_kcal + u_thermal + (R_IDEAL * temp_k)
    # G_corr = H_corr - T * S
    g_corr = h_corr - temp_k * (total_s / 1000.0)

    return ThermochemistryResult(temp_k, pressure_atm, zpve_kcal,
                                 u_trans, u_rot, u_vib, u_thermal,
                                 s_trans, s_rot, s_vib, total_s,
                                 h_corr, g_corr)
end

function print_thermo_summary(res::ThermochemistryResult)
    println("==================================================")
    println("          RRHO Thermochemistry Analysis           ")
    println("==================================================")
    @printf("  Temperature:          %8.2f K\n", res.temperature_k)
    @printf("  Pressure:             %8.2f atm\n", res.pressure_atm)
    @printf("  Zero-Point Energy:    %14.4f kcal/mol\n", res.zpve_kcal)
    @printf("  Thermal Energy (U):   %14.4f kcal/mol\n", res.u_thermal_kcal)
    @printf("  Total Entropy (S):    %14.4f cal/(mol·K)\n", res.total_entropy_cal_mol_k)
    @printf("  Enthalpy Correction:  %14.4f kcal/mol\n", res.enthalpy_corr_kcal)
    @printf("  Gibbs Free Energy:    %14.4f kcal/mol\n", res.gibbs_corr_kcal)
    println("==================================================")
end

end # module Thermochemistry

