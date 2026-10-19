# ==============================================================================
# QuantumChemMM.jl: CHARMM Force Field Implementation with Urey-Bradley
# ==============================================================================

module CHARMMForceField

using LinearAlgebra
using ..Types
using ..Elements
using ..Topology
using ..MMBase
using ..BondedPotentials
using ..Dihedrals
using ..NonBonded

export CHARMM, parameterize_charmm

struct CHARMM <: AbstractForceField
    bonds::Vector{HarmonicBond}
    angles::Vector{UreyBradleyAngle}
    dihedrals::Vector{PeriodicTorsion}
    impropers::Vector{HarmonicImproper}
    lj_pairs_14::Vector{Tuple{Int, Int, LJParam}}
    lj_pairs_non14::Vector{Tuple{Int, Int, LJParam}}
    coulomb_pairs_14::Vector{Tuple{Int, Int}}
    coulomb_pairs_non14::Vector{Tuple{Int, Int}}
    scale_vdw_14::Float64   # CHARMM standard: 1.0 (with modified Rmin in full param sets)
    scale_coul_14::Float64  # CHARMM standard: 1.0
end

const CHARMM_LJ_TABLE = Dict{Symbol, LJParam}(
    :H  => LJParam(2.352, 0.0220),
    :C  => LJParam(3.564, 0.0700),
    :N  => LJParam(3.296, 0.2000),
    :O  => LJParam(3.029, 0.1200),
    :P  => LJParam(3.742, 0.5850),
    :S  => LJParam(3.564, 0.4500),
    :Cl => LJParam(3.471, 0.2650)
)

"""
    parameterize_charmm(mol::Molecule)::CHARMM
Parameterizes a molecule using CHARMM force field conventions with Urey-Bradley 1-3 terms.
"""
function parameterize_charmm(mol::Molecule)::CHARMM
    perceive_topology!(mol)

    bonds = HarmonicBond[]
    for b in mol.bonds
        s1 = mol.atoms[b.i].symbol
        s2 = mol.atoms[b.j].symbol
        r0 = Elements.covalent_radius(s1) + Elements.covalent_radius(s2)
        kb = 600.0
        if (s1 == :C && s2 == :H) || (s2 == :C && s1 == :H)
            r0 = 1.09
            kb = 644.0
        elseif (s1 == :C && s2 == :C)
            r0 = b.order > 1.2 ? 1.375 : 1.52
            kb = b.order > 1.2 ? 800.0 : 445.0
        end
        push!(bonds, HarmonicBond(b.i, b.j, r0, kb))
    end

    angles = UreyBradleyAngle[]
    for ang in mol.angles
        theta0 = 109.5 * π / 180.0
        ktheta = 80.0
        # Compute default 1-3 distance
        r1 = Elements.covalent_radius(mol.atoms[ang.i].symbol) + Elements.covalent_radius(mol.atoms[ang.j].symbol)
        r2 = Elements.covalent_radius(mol.atoms[ang.j].symbol) + Elements.covalent_radius(mol.atoms[ang.k].symbol)
        rub0 = sqrt(r1^2 + r2^2 - 2.0*r1*r2*cos(theta0))
        kub = 45.0 # kcal/(mol·Å²) Urey-Bradley constant
        push!(angles, UreyBradleyAngle(ang.i, ang.j, ang.k, theta0, ktheta, rub0, kub))
    end

    dihedrals = PeriodicTorsion[]
    for dih in mol.dihedrals
        push!(dihedrals, PeriodicTorsion(dih.i, dih.j, dih.k, dih.l, [
            FourierTerm(3, 1.4, 0.0),
            FourierTerm(1, 0.6, 0.0)
        ]))
    end

    impropers = HarmonicImproper[]
    for imp in mol.impropers
        push!(impropers, HarmonicImproper(imp.central, imp.i, imp.j, imp.k, 0.0, 90.0))
    end

    pairs_14, pairs_non14 = get_nonbonded_pairs(mol)
    lj_14 = Tuple{Int, Int, LJParam}[]
    lj_non14 = Tuple{Int, Int, LJParam}[]

    for (i, j) in pairs_14
        pi = get(CHARMM_LJ_TABLE, mol.atoms[i].symbol, LJParam(3.4, 0.1))
        pj = get(CHARMM_LJ_TABLE, mol.atoms[j].symbol, LJParam(3.4, 0.1))
        push!(lj_14, (i, j, lorentz_berthelot(pi, pj)))
    end

    for (i, j) in pairs_non14
        pi = get(CHARMM_LJ_TABLE, mol.atoms[i].symbol, LJParam(3.4, 0.1))
        pj = get(CHARMM_LJ_TABLE, mol.atoms[j].symbol, LJParam(3.4, 0.1))
        push!(lj_non14, (i, j, lorentz_berthelot(pi, pj)))
    end

    return CHARMM(bonds, angles, dihedrals, impropers,
                  lj_14, lj_non14, pairs_14, pairs_non14, 1.0, 1.0)
end

end # module CHARMMForceField

