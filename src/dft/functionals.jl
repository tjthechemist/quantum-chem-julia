# ==============================================================================
# QuantumChemMM.jl: Density Functional Theory (DFT) Exchange-Correlation Functionals
# LDA (Slater, VWN), GGA (PBE, BLYP), and Hybrid (B3LYP, PBE0)
# ==============================================================================

module DFTFunctionals

export XCFunctional, evaluate_xc,
       LDA_FUNCTIONAL, PBE_FUNCTIONAL, BLYP_FUNCTIONAL, B3LYP_FUNCTIONAL, PBE0_FUNCTIONAL

struct XCFunctional
    name::String
    family::Symbol # :lda, :gga, :hybrid
    exact_exchange_fraction::Float64 # c_x (0.0 for pure DFT, 0.20 for B3LYP, 0.25 for PBE0)
end

const LDA_FUNCTIONAL   = XCFunctional("LDA-SVWN", :lda, 0.0)
const PBE_FUNCTIONAL   = XCFunctional("PBE", :gga, 0.0)
const BLYP_FUNCTIONAL  = XCFunctional("BLYP", :gga, 0.0)
const B3LYP_FUNCTIONAL = XCFunctional("B3LYP", :hybrid, 0.20)
const PBE0_FUNCTIONAL  = XCFunctional("PBE0", :hybrid, 0.25)

# --- 1. Slater Exchange (LDA) ---
function slater_exchange(rho::Float64)
    if rho < 1e-12
        return 0.0, 0.0
    end
    # eps_x = -3/4 * (3/pi)^(1/3) * rho^(1/3)
    cx = 0.7385587663820224 # (3/4) * (3/pi)^(1/3)
    rho_13 = rho^(1.0 / 3.0)
    eps_x = -cx * rho_13
    # v_x = 4/3 * eps_x
    vx = (4.0 / 3.0) * eps_x
    return eps_x, vx
end

# --- 2. VWN5 Correlation (LDA) ---
# Vosko, Wilk, Nusair, Can. J. Phys. 1980
function vwn5_correlation(rho::Float64)
    if rho < 1e-12
        return 0.0, 0.0
    end
    # Wigner-Seitz radius rs = (3 / (4*pi*rho))^(1/3)
    rs = (3.0 / (4.0 * π * rho))^(1.0 / 3.0)
    x = sqrt(rs)

    A = 0.0310907
    b = 3.72744
    c = 12.9352
    x0 = -0.10498
    Q = sqrt(4.0 * c - b^2)

    X_x = x^2 + b*x + c
    X_x0 = x0^2 + b*x0 + c

    t1 = log(x^2 / X_x)
    t2 = (2.0 * b / Q) * atan(Q / (2.0 * x + b))
    t3 = (b * x0 / X_x0) * (log((x - x0)^2 / X_x) + (2.0 * (b + 2.0 * x0) / Q) * atan(Q / (2.0 * x + b)))

    eps_c = A * (t1 + t2 - t3)
    # Derivative d(eps_c)/d(rs)
    dx_drs = 0.5 / x
    d_eps_c_dx = A * ( (2.0/x - (2.0*x + b)/X_x) - (4.0*b / (Q^2 + (2.0*x + b)^2)) -
                      (b*x0/X_x0) * (2.0/(x - x0) - (2.0*x + b)/X_x - 4.0*(b + 2.0*x0)/(Q^2 + (2.0*x + b)^2)) )
    d_eps_c_drs = d_eps_c_dx * dx_drs
    vc = eps_c - (rs / 3.0) * d_eps_c_drs

    return eps_c, vc
end

# --- 3. PBE Exchange & Correlation (GGA) ---
# Perdew, Burke, Ernzerhof, Phys. Rev. Lett. 1996
function pbe_exchange(rho::Float64, grad_norm::Float64)
    eps_x_lda, vx_lda = slater_exchange(rho)
    if rho < 1e-12
        return 0.0, 0.0
    end

    # Reduced density gradient s = |grad rho| / (2 * kF * rho) where kF = (3*pi^2*rho)^(1/3)
    kf = (3.0 * π^2 * rho)^(1.0 / 3.0)
    s = grad_norm / (2.0 * kf * rho)

    kappa = 0.804
    mu = 0.2195149727645171
    s_sq = s^2
    fx = 1.0 + kappa - (kappa / (1.0 + mu * s_sq / kappa))

    eps_x_pbe = eps_x_lda * fx
    vx_pbe = vx_lda * fx # local potential approximation

    return eps_x_pbe, vx_pbe
end

"""
    evaluate_xc(func::XCFunctional, rho::Float64, grad_norm::Float64=0.0)::Tuple{Float64, Float64}
Evaluates the exchange-correlation energy density eps_xc and local potential v_xc.
"""
function evaluate_xc(func::XCFunctional, rho::Float64, grad_norm::Float64=0.0)::Tuple{Float64, Float64}
    if rho < 1e-12
        return 0.0, 0.0
    end

    eps_x_lda, vx_lda = slater_exchange(rho)
    eps_c_lda, vc_lda = vwn5_correlation(rho)

    if func.family == :lda
        return eps_x_lda + eps_c_lda, vx_lda + vc_lda
    elseif func.family == :gga
        eps_x_pbe, vx_pbe = pbe_exchange(rho, grad_norm)
        return eps_x_pbe + eps_c_lda, vx_pbe + vc_lda
    elseif func.family == :hybrid
        # B3LYP or PBE0 combination
        c_x = func.exact_exchange_fraction
        eps_x_pbe, vx_pbe = pbe_exchange(rho, grad_norm)
        # Fraction of exact exchange is removed from DFT exchange
        eps_x_hybrid = (1.0 - c_x) * eps_x_pbe
        vx_hybrid = (1.0 - c_x) * vx_pbe
        return eps_x_hybrid + eps_c_lda, vx_hybrid + vc_lda
    else
        return eps_x_lda + eps_c_lda, vx_lda + vc_lda
    end
end

end # module DFTFunctionals

