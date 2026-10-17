# ==============================================================================
# QuantumChemMM.jl: MM2 / MM3 Class II Force Field Implementation
# ==============================================================================

module MM3ForceField

using LinearAlgebra
using ....Types
using ....Elements
using ....Topology
using ...MMBase
using ...BondedPotentials
using ...Dihedrals
using ...NonBonded
using ....MolMath

export MM3, parameterize_mm3

struct StretchBendCrossTerm
    i::Int
    j::Int # vertex
    k::Int
    r0_ij::Float64
    r0_kj::Float64
    theta0::Float64
    ksb::Float64 # in kcal/(mol·Å·rad)
end

struct MM3 <: AbstractForceField
    bonds::Vector{Class2Bond}
    angles::Vector{Class2Angle}
    stretch_bends::Vector{StretchBendCrossTerm}
    dihedrals::Vector{PeriodicTorsion}
    impropers::Vector{HarmonicImproper}
    buckingham_pairs::Vector{Tuple{Int, Int, BuckinghamParam}}
    coulomb_pairs::Vector{Tuple{Int, Int}}
    scale_14::Float64
end

const MM3_BUCKINGHAM = Dict{Symbol, BuckinghamParam}(
    :H  => BuckinghamParam(2800.0, 4.3, 15.0),
    :C  => BuckinghamParam(78000.0, 3.5, 400.0),
    :N  => BuckinghamParam(65000.0, 3.6, 350.0),
    :O  => BuckinghamParam(55000.0, 3.8, 300.0),
    :Cl => BuckinghamParam(120000.0, 3.4, 800.0)
)

"""
    parameterize_mm3(mol::Molecule)::MM3
Builds Allinger MM3 Class II parameterization including anharmonicities and stretch-bend cross terms.
"""
function parameterize_mm3(mol::Molecule)::MM3
    perceive_topology!(mol)

    # 1. Class II Anharmonic Bonds
    bonds = Class2Bond[]
    for b in mol.bonds
        s1 = mol.atoms[b.i].symbol
        s2 = mol.atoms[b.j].symbol
        r0 = Elements.covalent_radius(s1) + Elements.covalent_radius(s2)
        k2 = 350.0 # kcal/(mol·Å²)
        if (s1 == :C && s2 == :H) || (s2 == :C && s1 == :H)
            r0 = 1.10
            k2 = 330.0
        elseif (s1 == :C && s2 == :C)
            r0 = b.order > 1.2 ? 1.34 : 1.53
            k2 = b.order > 1.2 ? 600.0 : 316.0
        end
        # MM3 cubic and quartic expansion constants:
        # V = k2 * dr^2 * (1 - 2.55 * dr + (7/12)*(2.55)^2 * dr^2)
        # = k2 * dr^2 - 2.55*k2 * dr^3 + (7/12)*(2.55)^2*k2 * dr^4
        k3 = -2.55 * k2
        k4 = (7.0 / 12.0) * (2.55^2) * k2
        push!(bonds, Class2Bond(b.i, b.j, r0, k2, k3, k4))
    end

    # 2. Class II Anharmonic Angles and Stretch-Bend Cross Terms
    angles = Class2Angle[]
    stretch_bends = StretchBendCrossTerm[]
    for ang in mol.angles
        theta0 = 109.5 * π / 180.0
        k2 = 60.0 # kcal/(mol·rad²)
        k3 = -0.014 * k2
        k4 = 5.6e-5 * k2
        push!(angles, Class2Angle(ang.i, ang.j, ang.k, theta0, k2, k3, k4))

        # Cross term: E_sb = ksb * (dr_ij + dr_kj) * dtheta
        r0_ij = Elements.covalent_radius(mol.atoms[ang.i].symbol) + Elements.covalent_radius(mol.atoms[ang.j].symbol)
        r0_kj = Elements.covalent_radius(mol.atoms[ang.k].symbol) + Elements.covalent_radius(mol.atoms[ang.j].symbol)
        ksb = 15.0 # kcal/(mol·Å·rad)
        push!(stretch_bends, StretchBendCrossTerm(ang.i, ang.j, ang.k, r0_ij, r0_kj, theta0, ksb))
    end

    # 3. Dihedrals
    dihedrals = PeriodicTorsion[]
    for dih in mol.dihedrals
        push!(dihedrals, PeriodicTorsion(dih.i, dih.j, dih.k, dih.l, [
            FourierTerm(1, 0.5, 0.0),
            FourierTerm(2, -0.3, π),
            FourierTerm(3, 1.8, 0.0)
        ]))
    end

    # 4. Impropers
    impropers = HarmonicImproper[]
    for imp in mol.impropers
        push!(impropers, HarmonicImproper(imp.central, imp.i, imp.j, imp.k, 0.0, 25.0))
    end

    # 5. Buckingham exp-6 Nonbonded pairs
    _, pairs_non14 = get_nonbonded_pairs(mol)
    buck_pairs = Tuple{Int, Int, BuckinghamParam}[]
    for (i, j) in pairs_non14
        p_i = get(MM3_BUCKINGHAM, mol.atoms[i].symbol, BuckinghamParam(50000.0, 3.6, 250.0))
        p_j = get(MM3_BUCKINGHAM, mol.atoms[j].symbol, BuckinghamParam(50000.0, 3.6, 250.0))
        A_ij = sqrt(p_i.A * p_j.A)
        B_ij = 0.5 * (p_i.B + p_j.B)
        C_ij = sqrt(p_i.C * p_j.C)
        push!(buck_pairs, (i, j, BuckinghamParam(A_ij, B_ij, C_ij)))
    end

    return MM3(bonds, angles, stretch_bends, dihedrals, impropers,
               buck_pairs, pairs_non14, 0.5)
end

function calc_stretch_bend_energy_and_grad(sb::StretchBendCrossTerm,
                                           ri::AbstractVector{<:Real},
                                           rj::AbstractVector{<:Real},
                                           rk::AbstractVector{<:Real})
    rij = MolMath.distance(ri, rj)
    rkj = MolMath.distance(rk, rj)
    theta = MolMath.angle_between(ri, rj, rk)

    drij = rij - sb.r0_ij
    drkj = rkj - sb.r0_kj
    dtheta = theta - sb.theta0

    energy = sb.ksb * (drij + drkj) * dtheta

    # Gradients w.r.t r_ij, r_kj, theta
    g_drij_i, g_drij_j = MolMath.distance_gradient(ri, rj)
    g_drkj_k, g_drkj_j = MolMath.distance_gradient(rk, rj)
    g_ang_i, g_ang_j, g_ang_k = MolMath.angle_gradient(ri, rj, rk)

    # dE/dri = ksb * dtheta * d(drij)/dri + ksb * (drij + drkj) * d(dtheta)/dri
    gi = (sb.ksb * dtheta) .* g_drij_i .+ (sb.ksb * (drij + drkj)) .* g_ang_i
    gk = (sb.ksb * dtheta) .* g_drkj_k .+ (sb.ksb * (drij + drkj)) .* g_ang_k
    gj = (sb.ksb * dtheta) .* (g_drij_j .+ g_drkj_j) .+ (sb.ksb * (drij + drkj)) .* g_ang_j

    return energy, gi, gj, gk
end

end # module MM3ForceField

