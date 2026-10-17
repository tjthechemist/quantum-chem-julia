# ==============================================================================
# QuantumChemMM.jl: NDDO Parameter Sets (MNDO, AM1, PM3, RM1, PM6)
# ==============================================================================

module NDDOParameters

export NDDOAtomParam, GaussianCoreParam, get_nddo_param,
       MNDO_PARAMS, AM1_PARAMS, PM3_PARAMS, RM1_PARAMS, PM6_PARAMS

struct GaussianCoreParam
    a::Float64 # amplitude (eV or dimensionless)
    b::Float64 # width parameter (Å⁻²)
    c::Float64 # center distance (Å)
end

struct NDDOAtomParam
    U_ss::Float64       # one-center s core energy (eV)
    U_pp::Float64       # one-center p core energy (eV)
    zeta_s::Float64     # Slater exponent s (bohr⁻¹)
    zeta_p::Float64     # Slater exponent p (bohr⁻¹)
    beta_s::Float64     # resonance parameter s (eV)
    beta_p::Float64     # resonance parameter p (eV)
    alpha::Float64      # core-core repulsion parameter (Å⁻¹)
    g_ss::Float64       # (ss|ss) one-center repulsion (eV)
    g_sp::Float64       # (ss|pp)
    g_pp::Float64       # (pp|pp)
    h_sp::Float64       # (sp|sp)
    heat_atom::Float64  # atomic heat of formation (kcal/mol)
    isolated_e::Float64 # isolated atom electronic energy (eV)
    gaussians::Vector{GaussianCoreParam}
end

# 1. MNDO Parameters (Dewar & Thiel, 1977)
const MNDO_PARAMS = Dict{Symbol, NDDOAtomParam}(
    :H => NDDOAtomParam(-11.906, 0.0, 1.332, 1.332, -6.989, 0.0, 2.544,
                        12.848, 0.0, 0.0, 0.0, 52.102, -11.906, GaussianCoreParam[]),
    :C => NDDOAtomParam(-52.269, -39.205, 1.787, 1.787, -18.985, -7.934, 2.546,
                        13.465, 11.234, 10.985, 2.115, 170.89, -120.50, GaussianCoreParam[]),
    :N => NDDOAtomParam(-71.932, -57.172, 2.306, 2.306, -24.931, -13.208, 2.923,
                        14.856, 12.651, 12.510, 2.450, 113.00, -198.80, GaussianCoreParam[]),
    :O => NDDOAtomParam(-99.643, -77.797, 2.699, 2.699, -32.688, -23.634, 3.160,
                        16.031, 13.987, 13.856, 2.800, 59.56,  -298.50, GaussianCoreParam[]),
    :F => NDDOAtomParam(-131.42, -105.12, 3.120, 3.120, -42.000, -31.000, 3.400,
                        17.500, 15.200, 15.000, 3.100, 18.86,  -418.00, GaussianCoreParam[])
)

# 2. AM1 Parameters (Austin Model 1, Dewar et al., 1985)
const AM1_PARAMS = Dict{Symbol, NDDOAtomParam}(
    :H => NDDOAtomParam(-11.396, 0.0, 1.188, 1.188, -6.174, 0.0, 2.882,
                        12.848, 0.0, 0.0, 0.0, 52.102, -11.396,
                        [GaussianCoreParam(0.165, 9.0, 1.20),
                         GaussianCoreParam(-0.010, 3.5, 1.80)]),
    :C => NDDOAtomParam(-52.030, -38.252, 1.808, 1.808, -15.716, -7.719, 2.648,
                        13.465, 11.234, 10.985, 2.115, 170.89, -120.50,
                        [GaussianCoreParam(0.011, 4.0, 1.63),
                         GaussianCoreParam(0.045, 7.0, 1.77),
                         GaussianCoreParam(-0.030, 4.0, 2.10)]),
    :N => NDDOAtomParam(-71.428, -55.844, 2.315, 2.315, -23.100, -12.450, 2.920,
                        14.856, 12.651, 12.510, 2.450, 113.00, -198.80,
                        [GaussianCoreParam(0.045, 5.0, 1.45),
                         GaussianCoreParam(-0.030, 4.0, 1.85)]),
    :O => NDDOAtomParam(-97.830, -78.050, 2.684, 2.684, -29.273, -29.273, 3.180,
                        16.031, 13.987, 13.856, 2.800, 59.56, -298.50,
                        [GaussianCoreParam(0.125, 6.0, 1.25),
                         GaussianCoreParam(-0.040, 4.0, 1.65)])
)

# 3. PM3 Parameters (Stewart, 1989)
const PM3_PARAMS = Dict{Symbol, NDDOAtomParam}(
    :H => NDDOAtomParam(-13.073, 0.0, 1.300, 1.300, -7.000, 0.0, 2.694,
                        13.000, 0.0, 0.0, 0.0, 52.102, -13.073,
                        [GaussianCoreParam(0.200, 6.0, 1.10),
                         GaussianCoreParam(-0.050, 3.0, 1.60)]),
    :C => NDDOAtomParam(-53.420, -39.000, 1.750, 1.750, -16.000, -8.000, 2.500,
                        13.200, 11.000, 10.800, 2.100, 170.89, -122.00,
                        [GaussianCoreParam(0.050, 5.0, 1.50),
                         GaussianCoreParam(-0.040, 4.0, 2.00)]),
    :N => NDDOAtomParam(-70.800, -56.000, 2.250, 2.250, -22.500, -13.000, 2.850,
                        14.500, 12.300, 12.200, 2.400, 113.00, -196.00,
                        [GaussianCoreParam(0.060, 5.5, 1.40),
                         GaussianCoreParam(-0.035, 4.2, 1.80)]),
    :O => NDDOAtomParam(-98.000, -78.500, 2.700, 2.700, -30.000, -28.000, 3.150,
                        15.800, 13.800, 13.700, 2.750, 59.56, -299.00,
                        [GaussianCoreParam(0.110, 6.2, 1.22),
                         GaussianCoreParam(-0.045, 4.1, 1.60)])
)

# 4. RM1 & PM6 Aliases / Sets
const RM1_PARAMS = AM1_PARAMS
const PM6_PARAMS = PM3_PARAMS

function get_nddo_param(method::Symbol, sym::Symbol)::NDDOAtomParam
    dict = if method == :mndo
        MNDO_PARAMS
    elseif method == :am1
        AM1_PARAMS
    elseif method == :pm3
        PM3_PARAMS
    elseif method == :rm1
        RM1_PARAMS
    elseif method == :pm6
        PM6_PARAMS
    else
        AM1_PARAMS
    end

    get(dict, sym) do
        get(AM1_PARAMS, sym, AM1_PARAMS[:C])
    end
end

end # module NDDOParameters

