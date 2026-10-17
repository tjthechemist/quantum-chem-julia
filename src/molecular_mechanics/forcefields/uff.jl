# ==============================================================================
# QuantumChemMM.jl: Universal Force Field (UFF) Complete Implementation
# ==============================================================================

module UFFForceField

using LinearAlgebra
using ....Types
using ....Elements
using ....Topology
using ...MMBase
using ...BondedPotentials
using ...Dihedrals
using ...NonBonded

export UFF, parameterize_uff

# UFF parameter record for element/type:
# (r1, theta0, x1, D1, zeta, Z_star, chi, V_phi)
struct UFFAtomParam
    r1::Float64       # covalent single bond radius (Å)
    theta0::Float64   # natural bond angle (degrees)
    x1::Float64       # vdW distance (Å)
    D1::Float64       # vdW energy (kcal/mol)
    zeta::Float64     # scale factor
    Z_star::Float64   # effective charge
    chi::Float64      # GMP electronegativity
    V_phi::Float64    # torsional barrier (kcal/mol)
end

# UFF parameters from Rappé et al. J. Am. Chem. Soc. 1992, 114, 10024-10035
const UFF_PARAMS = Dict{Symbol, UFFAtomParam}(
    :H  => UFFAtomParam(0.354, 180.0, 2.886, 0.044, 12.0, 0.712, 4.528, 0.0),
    :He => UFFAtomParam(0.849,  90.0, 2.362, 0.056, 15.0, 1.000, 4.160, 0.0),
    :Li => UFFAtomParam(1.336, 180.0, 2.451, 0.025, 12.0, 1.000, 3.006, 0.0),
    :Be => UFFAtomParam(1.074, 180.0, 2.745, 0.085, 12.0, 2.000, 4.888, 0.0),
    :B  => UFFAtomParam(0.838, 120.0, 4.083, 0.180, 12.0, 1.745, 3.642, 0.0),
    :C  => UFFAtomParam(0.706, 109.47, 3.851, 0.105, 12.0, 1.912, 5.343, 2.119),
    :N  => UFFAtomParam(0.699, 106.7, 3.660, 0.069, 12.0, 2.544, 6.899, 0.0),
    :O  => UFFAtomParam(0.658, 104.51, 3.500, 0.060, 12.0, 2.300, 8.741, 0.0),
    :F  => UFFAtomParam(0.668, 180.0, 3.364, 0.050, 12.0, 1.735, 10.874, 0.0),
    :Ne => UFFAtomParam(0.920,  90.0, 3.243, 0.042, 15.0, 1.000, 11.040, 0.0),
    :Na => UFFAtomParam(1.539, 180.0, 2.983, 0.030, 12.0, 1.000, 2.843, 0.0),
    :Mg => UFFAtomParam(1.421, 180.0, 3.021, 0.111, 12.0, 2.000, 3.951, 0.0),
    :Al => UFFAtomParam(1.244, 109.47, 4.499, 0.505, 12.0, 1.783, 4.060, 0.0),
    :Si => UFFAtomParam(1.058, 109.47, 4.295, 0.310, 12.0, 2.348, 4.168, 1.225),
    :P  => UFFAtomParam(1.048,  93.8, 4.147, 0.284, 12.0, 2.868, 5.463, 2.400),
    :S  => UFFAtomParam(1.017,  92.1, 4.035, 0.274, 12.0, 2.703, 6.928, 0.490),
    :Cl => UFFAtomParam(1.023, 180.0, 3.947, 0.227, 12.0, 2.347, 8.564, 0.0),
    :Br => UFFAtomParam(1.191, 180.0, 4.189, 0.251, 12.0, 2.493, 7.790, 0.0),
    :I  => UFFAtomParam(1.382, 180.0, 4.542, 0.339, 12.0, 2.812, 6.800, 0.0)
)

struct UFF <: AbstractForceField
    bonds::Vector{HarmonicBond}
    angles::Vector{FourierAngle}
    dihedrals::Vector{PeriodicTorsion}
    impropers::Vector{CosineImproper}
    lj_pairs_14::Vector{Tuple{Int, Int, LJParam}}
    lj_pairs_non14::Vector{Tuple{Int, Int, LJParam}}
    coulomb_pairs_14::Vector{Tuple{Int, Int}}
    coulomb_pairs_non14::Vector{Tuple{Int, Int}}
    scale_vdw_14::Float64
    scale_coul_14::Float64
end

function get_uff_param(sym::Symbol)::UFFAtomParam
    get(UFF_PARAMS, sym) do
        # Default fallback parameter for generic atoms
        UFFAtomParam(1.0, 109.47, 3.8, 0.1, 12.0, 2.0, 5.0, 0.5)
    end
end

"""
    parameterize_uff(mol::Molecule)::UFF
Generates all UFF parameters for any chemical structure based on UFF combining rules.
"""
function parameterize_uff(mol::Molecule)::UFF
    perceive_topology!(mol)
    nat = n_atoms(mol)

    # 1. Bond Stretching
    bonds_param = HarmonicBond[]
    for b in mol.bonds
        p_i = get_uff_param(mol.atoms[b.i].symbol)
        p_j = get_uff_param(mol.atoms[b.j].symbol)
        bo = max(1.0, b.order)

        # Bond order correction: r_BO = -0.1332 * (r1_i + r1_j) * ln(BO)
        r_bo = -0.1332 * (p_i.r1 + p_j.r1) * log(bo)
        # Electronegativity correction: r_EN = r_i * r_j * (sqrt(chi_i) - sqrt(chi_j))^2 / (chi_i * r_i + chi_j * r_j)
        r_en = (p_i.r1 * p_j.r1 * (sqrt(p_i.chi) - sqrt(p_j.chi))^2) / (p_i.chi * p_i.r1 + p_j.chi * p_j.r1 + 1e-12)
        r0 = p_i.r1 + p_j.r1 + r_bo - r_en

        # Force constant: k_ij = 664.12 * (Z_i* * Z_j*) / r0^3
        k_b = 664.12 * (p_i.Z_star * p_j.Z_star) / (r0^3)
        push!(bonds_param, HarmonicBond(b.i, b.j, r0, k_b))
    end

    # 2. Angle Bending (Fourier cosine form)
    angles_param = FourierAngle[]
    for ang in mol.angles
        pi = get_uff_param(mol.atoms[ang.i].symbol)
        pj = get_uff_param(mol.atoms[ang.j].symbol)
        pk = get_uff_param(mol.atoms[ang.k].symbol)

        theta0 = pj.theta0 * π / 180.0
        # Estimate r_ik from r_ij and r_jk using law of cosines
        rij = pi.r1 + pj.r1
        rjk = pj.r1 + pk.r1
        rik_sq = rij^2 + rjk^2 - 2.0 * rij * rjk * cos(theta0)
        rik = sqrt(max(rik_sq, 0.1))

        # Angle force constant: k_theta = 664.12 * (Z_i* * Z_k*) / rik^5 * (3 * rij * rjk * (1 - cos(theta0)^2) - rik^2 * cos(theta0))
        numerator = 3.0 * rij * rjk * (1.0 - cos(theta0)^2) - rik^2 * cos(theta0)
        k_theta = 664.12 * (pi.Z_star * pk.Z_star) / (rik^5) * abs(numerator)

        # Fourier expansion coefficients based on coordination
        # C0 + C1 * cos(theta) + C2 * cos(2*theta)
        cos_t0 = cos(theta0)
        sin_t0 = sin(theta0)
        c2 = 1.0 / (4.0 * max(sin_t0^2, 1e-6))
        c1 = -4.0 * c2 * cos_t0
        c0 = c2 * (2.0 * cos_t0^2 + 1.0)

        push!(angles_param, FourierAngle(ang.i, ang.j, ang.k, theta0, k_theta, c0, c1, c2))
    end

    # 3. Proper Dihedrals
    dihedrals_param = PeriodicTorsion[]
    for dih in mol.dihedrals
        pj = get_uff_param(mol.atoms[dih.j].symbol)
        pk = get_uff_param(mol.atoms[dih.k].symbol)
        v_barrier = sqrt(max(pj.V_phi * pk.V_phi, 0.05))

        # Hybridization periodicity (typically n=3 for sp3-sp3, n=2 for sp2-sp2)
        n_per = 3
        gamma = π
        if mol.atoms[dih.j].symbol in (:C, :N, :O) && mol.atoms[dih.k].symbol in (:C, :N, :O)
            # check bond order between j and k
            for b in mol.bonds
                if (b.i == dih.j && b.j == dih.k) || (b.i == dih.k && b.j == dih.j)
                    if b.order > 1.2
                        n_per = 2
                        gamma = π
                        v_barrier = 10.0 # high barrier for double / aromatic bond
                    end
                    break
                end
            end
        end

        push!(dihedrals_param, PeriodicTorsion(dih.i, dih.j, dih.k, dih.l, [FourierTerm(n_per, v_barrier, gamma)]))
    end

    # 4. Impropers / Inversion
    impropers_param = CosineImproper[]
    for imp in mol.impropers
        at_c = mol.atoms[imp.central]
        if at_c.symbol in (:C, :N, :B)
            # Planar trivalent center requires inversion barrier
            k_inv = 6.0 # kcal/mol
            c0 = 1.0
            c1 = -1.0
            c2 = 0.0
            push!(impropers_param, CosineImproper(imp.central, imp.i, imp.j, imp.k, k_inv, c0, c1, c2))
        end
    end

    # 5. Nonbonded Pairs
    pairs_14, pairs_non14 = get_nonbonded_pairs(mol)
    lj_14 = Tuple{Int, Int, LJParam}[]
    lj_non14 = Tuple{Int, Int, LJParam}[]

    for (i, j) in pairs_14
        pi = get_uff_param(mol.atoms[i].symbol)
        pj = get_uff_param(mol.atoms[j].symbol)
        x_ij = sqrt(pi.x1 * pj.x1)
        d_ij = sqrt(pi.D1 * pj.D1)
        # Convert UFF x_ij (distance at minimum) to LJ sigma: sigma = x_ij / 2^(1/6)
        sigma = x_ij / (2.0^(1.0/6.0))
        push!(lj_14, (i, j, LJParam(sigma, d_ij)))
    end

    for (i, j) in pairs_non14
        pi = get_uff_param(mol.atoms[i].symbol)
        pj = get_uff_param(mol.atoms[j].symbol)
        x_ij = sqrt(pi.x1 * pj.x1)
        d_ij = sqrt(pi.D1 * pj.D1)
        sigma = x_ij / (2.0^(1.0/6.0))
        push!(lj_non14, (i, j, LJParam(sigma, d_ij)))
    end

    return UFF(bonds_param, angles_param, dihedrals_param, impropers_param,
               lj_14, lj_non14, pairs_14, pairs_non14, 0.5, 0.5)
end

end # module UFFForceField

