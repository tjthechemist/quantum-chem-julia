# ==============================================================================
# QuantumChemMM.jl: OPLS-AA Force Field Implementation
# ==============================================================================

module OPLSForceField

using LinearAlgebra
using ....Types
using ....Elements
using ....Topology
using ...MMBase
using ...BondedPotentials
using ...Dihedrals
using ...NonBonded

export OPLS, parameterize_opls

struct OPLS <: AbstractForceField
    bonds::Vector{HarmonicBond}
    angles::Vector{HarmonicAngle}
    dihedrals::Vector{PeriodicTorsion}
    impropers::Vector{HarmonicImproper}
    lj_pairs_14::Vector{Tuple{Int, Int, LJParam}}
    lj_pairs_non14::Vector{Tuple{Int, Int, LJParam}}
    coulomb_pairs_14::Vector{Tuple{Int, Int}}
    coulomb_pairs_non14::Vector{Tuple{Int, Int}}
    scale_vdw_14::Float64   # OPLS standard: 0.5
    scale_coul_14::Float64  # OPLS standard: 0.5
end

# Standard OPLS-AA parameters (sigma in Å, epsilon in kcal/mol)
const OPLS_LJ_TABLE = Dict{Symbol, LJParam}(
    :H  => LJParam(2.500, 0.0300),
    :C  => LJParam(3.500, 0.0660),
    :N  => LJParam(3.250, 0.1700),
    :O  => LJParam(3.000, 0.1700),
    :F  => LJParam(2.940, 0.0610),
    :S  => LJParam(3.550, 0.2500),
    :Cl => LJParam(3.400, 0.3000),
    :Br => LJParam(3.700, 0.3900)
)

"""
    parameterize_opls(mol::Molecule)::OPLS
Parameterizes a molecule using the OPLS-AA force field conventions.
"""
function parameterize_opls(mol::Molecule)::OPLS
    perceive_topology!(mol)

    bonds = HarmonicBond[]
    for b in mol.bonds
        s1 = mol.atoms[b.i].symbol
        s2 = mol.atoms[b.j].symbol
        r0 = Elements.covalent_radius(s1) + Elements.covalent_radius(s2)
        kb = 600.0
        if (s1 == :C && s2 == :H) || (s2 == :C && s1 == :H)
            r0 = 1.09
            kb = 680.0
        elseif (s1 == :C && s2 == :C)
            r0 = b.order > 1.2 ? 1.40 : 1.529
            kb = b.order > 1.2 ? 938.0 : 536.0
        end
        push!(bonds, HarmonicBond(b.i, b.j, r0, kb))
    end

    angles = HarmonicAngle[]
    for ang in mol.angles
        theta0 = 109.5 * π / 180.0
        ktheta = 100.0
        push!(angles, HarmonicAngle(ang.i, ang.j, ang.k, theta0, ktheta))
    end

    dihedrals = PeriodicTorsion[]
    for dih in mol.dihedrals
        # OPLS V1, V2, V3 Fourier series
        push!(dihedrals, PeriodicTorsion(dih.i, dih.j, dih.k, dih.l, [
            FourierTerm(1, 1.3, 0.0),
            FourierTerm(2, -0.2, π),
            FourierTerm(3, 0.4, 0.0)
        ]))
    end

    impropers = HarmonicImproper[]
    for imp in mol.impropers
        push!(impropers, HarmonicImproper(imp.central, imp.i, imp.j, imp.k, 0.0, 30.0))
    end

    pairs_14, pairs_non14 = get_nonbonded_pairs(mol)
    lj_14 = Tuple{Int, Int, LJParam}[]
    lj_non14 = Tuple{Int, Int, LJParam}[]

    for (i, j) in pairs_14
        pi = get(OPLS_LJ_TABLE, mol.atoms[i].symbol, LJParam(3.5, 0.1))
        pj = get(OPLS_LJ_TABLE, mol.atoms[j].symbol, LJParam(3.5, 0.1))
        # OPLS uses geometric combination rule: sigma_ij = sqrt(sigma_i * sigma_j)
        push!(lj_14, (i, j, geometric_combination(pi, pj)))
    end

    for (i, j) in pairs_non14
        pi = get(OPLS_LJ_TABLE, mol.atoms[i].symbol, LJParam(3.5, 0.1))
        pj = get(OPLS_LJ_TABLE, mol.atoms[j].symbol, LJParam(3.5, 0.1))
        push!(lj_non14, (i, j, geometric_combination(pi, pj)))
    end

    return OPLS(bonds, angles, dihedrals, impropers,
                lj_14, lj_non14, pairs_14, pairs_non14, 0.5, 0.5)
end

end # module OPLSForceField

