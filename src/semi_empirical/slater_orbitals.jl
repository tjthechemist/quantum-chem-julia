# ==============================================================================
# QuantumChemMM.jl: Slater-Type Orbitals, STO-3G Gaussian Expansion, and Overlaps
# ==============================================================================

module SlaterOrbitals

using LinearAlgebra
using ..Types
using ..Elements

export OrbitalInfo, AtomOrbitals, build_valence_basis,
       compute_overlap_matrix, get_orbital_zeta

struct OrbitalInfo
    atom_index::Int
    n::Int              # Principal quantum number (1 or 2)
    l::Int              # Angular momentum (0=s, 1=p)
    m_type::Symbol      # :s, :px, :py, :pz
    zeta::Float64       # Slater exponent
end

struct AtomOrbitals
    atom_index::Int
    symbol::Symbol
    orbitals::Vector{OrbitalInfo}
end

# Standard Slater orbital exponents (in bohr⁻¹)
# 1 Bohr = 0.529177210903 Å
const BOHR_TO_ANGSTROM = 0.529177210903
const ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

const SLATER_EXPONENTS = Dict{Symbol, Float64}(
    :H  => 1.20,
    :He => 1.6875,
    :Li => 0.650,
    :Be => 0.975,
    :B  => 1.300,
    :C  => 1.625,
    :N  => 1.950,
    :O  => 2.275,
    :F  => 2.425,
    :Na => 0.733,
    :Mg => 0.950,
    :Al => 1.167,
    :Si => 1.383,
    :P  => 1.600,
    :S  => 1.817,
    :Cl => 2.033,
    :Br => 2.050,
    :I  => 1.900
)

get_orbital_zeta(sym::Symbol) = get(SLATER_EXPONENTS, sym, 1.50)

# STO-3G Gaussian expansion fitting parameters (Hehre, Stewart, Pople, JCP 1969)
# 1s expansion:
const STO3G_1S_ALPHA = [0.109818, 0.405771, 2.227660]
const STO3G_1S_COEFF = [0.444635, 0.535328, 0.154329]

# 2sp expansion:
const STO3G_2SP_ALPHA = [0.0751386, 0.231031, 0.994203]
const STO3G_2S_COEFF  = [0.700115, 0.399513, -0.0999672]
const STO3G_2P_COEFF  = [0.391957, 0.607684, 0.155916]

"""
    build_valence_basis(mol::Molecule)::Vector{OrbitalInfo}
Constructs the valence basis set for the molecule (1s for H/He; 2s, 2px, 2py, 2pz for second-row).
"""
function build_valence_basis(mol::Molecule)::Vector{OrbitalInfo}
    basis = OrbitalInfo[]
    for at in mol.atoms
        zeta = get_orbital_zeta(at.symbol)
        if at.symbol in (:H, :He)
            push!(basis, OrbitalInfo(at.index, 1, 0, :s, zeta))
        else
            # Second-row and higher valence s and p orbitals
            push!(basis, OrbitalInfo(at.index, 2, 0, :s, zeta))
            push!(basis, OrbitalInfo(at.index, 2, 1, :px, zeta))
            push!(basis, OrbitalInfo(at.index, 2, 1, :py, zeta))
            push!(basis, OrbitalInfo(at.index, 2, 1, :pz, zeta))
        end
    end
    return basis
end

"""
Analytical Gaussian overlap between two primitives:
g_A(r) = (x - xA)^lx (y - yA)^ly (z - zA)^lz exp(-alphaA * (r - rA)^2)
normalized primitive overlap in 3D.
"""
function primitive_overlap_1d(lA::Int, lB::Int, xA::Float64, xB::Float64, gamma::Float64, Px::Float64)
    # Closed form for l in {0, 1}
    if lA == 0 && lB == 0
        return 1.0
    elseif lA == 1 && lB == 0
        return Px - xA
    elseif lA == 0 && lB == 1
        return Px - xB
    elseif lA == 1 && lB == 1
        return 1.0 / (2.0 * gamma) + (Px - xA) * (Px - xB)
    else
        return 0.0
    end
end

function gaussian_overlap_3d(alphaA::Float64, rA::Vector{Float64}, lA::Tuple{Int, Int, Int},
                             alphaB::Float64, rB::Vector{Float64}, lB::Tuple{Int, Int, Int})
    gamma = alphaA + alphaB
    rAB = rA .- rB
    rAB_sq = dot(rAB, rAB)
    E_AB = exp(- (alphaA * alphaB / gamma) * rAB_sq)
    P = (alphaA .* rA .+ alphaB .* rB) ./ gamma

    # Pre-exponential factor from 3D Gaussian normalization
    # N(alpha, l) = (2*alpha/pi)^(3/4) * sqrt( (4*alpha)^(lx+ly+lz) / ((2lx-1)!! (2ly-1)!! (2lz-1)!!) )
    normA = (2.0 * alphaA / π)^0.75 * sqrt((4.0 * alphaA)^(sum(lA)))
    normB = (2.0 * alphaB / π)^0.75 * sqrt((4.0 * alphaB)^(sum(lB)))

    pref = (π / gamma)^1.5 * E_AB

    Ix = primitive_overlap_1d(lA[1], lB[1], rA[1], rB[1], gamma, P[1])
    Iy = primitive_overlap_1d(lA[2], lB[2], rA[2], rB[2], gamma, P[2])
    Iz = primitive_overlap_1d(lA[3], lB[3], rA[3], rB[3], gamma, P[3])

    return normA * normB * pref * Ix * Iy * Iz
end

function get_orbital_angular(orb::OrbitalInfo)::Tuple{Int, Int, Int}
    if orb.m_type == :s
        return (0, 0, 0)
    elseif orb.m_type == :px
        return (1, 0, 0)
    elseif orb.m_type == :py
        return (0, 1, 0)
    elseif orb.m_type == :pz
        return (0, 0, 1)
    else
        return (0, 0, 0)
    end
end

function get_sto3g_params(orb::OrbitalInfo)
    # Exponents in STO-3G scale with zeta^2
    zeta_sq = orb.zeta^2
    if orb.n == 1
        alphas = STO3G_1S_ALPHA .* zeta_sq
        coeffs = STO3G_1S_COEFF
    else
        alphas = STO3G_2SP_ALPHA .* zeta_sq
        coeffs = (orb.l == 0) ? STO3G_2S_COEFF : STO3G_2P_COEFF
    end
    return alphas, coeffs
end

"""
    compute_overlap_matrix(basis::Vector{OrbitalInfo}, mol::Molecule)::Matrix{Float64}
Evaluates the full valence overlap matrix S_μν = <χ_μ | χ_ν> analytically in atomic units (Bohrs).
"""
function compute_overlap_matrix(basis::Vector{OrbitalInfo}, mol::Molecule)::Matrix{Float64}
    nb = length(basis)
    S = zeros(Float64, nb, nb)
    coords_bohr = [mol.atoms[i].coords .* ANGSTROM_TO_BOHR for i in 1:n_atoms(mol)]

    for mu in 1:nb
        S[mu, mu] = 1.0
        orb_mu = basis[mu]
        r_mu = coords_bohr[orb_mu.atom_index]
        l_mu = get_orbital_angular(orb_mu)
        alphas_mu, coeffs_mu = get_sto3g_params(orb_mu)

        for nu in (mu+1):nb
            orb_nu = basis[nu]
            # Same atom orthogonality
            if orb_mu.atom_index == orb_nu.atom_index
                S[mu, nu] = 0.0
                S[nu, mu] = 0.0
                continue
            end

            r_nu = coords_bohr[orb_nu.atom_index]
            l_nu = get_orbital_angular(orb_nu)
            alphas_nu, coeffs_nu = get_sto3g_params(orb_nu)

            # Contract over the 3 primitive Gaussians of STO-3G
            s_val = 0.0
            for k in 1:3
                for m in 1:3
                    c_km = coeffs_mu[k] * coeffs_nu[m]
                    prim_s = gaussian_overlap_3d(alphas_mu[k], r_mu, l_mu,
                                                 alphas_nu[m], r_nu, l_nu)
                    s_val += c_km * prim_s
                end
            end
            S[mu, nu] = s_val
            S[nu, mu] = s_val
        end
    end
    return S
end

end # module SlaterOrbitals

