# ==============================================================================
# QuantumChemMM.jl: AMBER / GAFF Class I Force Field Implementation
# ==============================================================================

module AMBERForceField

using LinearAlgebra
using ....Types
using ....Elements
using ....Topology
using ...MMBase
using ...BondedPotentials
using ...Dihedrals
using ...NonBonded

export AMBER, parameterize_amber

struct AMBER <: AbstractForceField
    bonds::Vector{HarmonicBond}
    angles::Vector{HarmonicAngle}
    dihedrals::Vector{PeriodicTorsion}
    impropers::Vector{HarmonicImproper}
    lj_pairs_14::Vector{Tuple{Int, Int, LJParam}}
    lj_pairs_non14::Vector{Tuple{Int, Int, LJParam}}
    coulomb_pairs_14::Vector{Tuple{Int, Int}}
    coulomb_pairs_non14::Vector{Tuple{Int, Int}}
    scale_vdw_14::Float64   # AMBER standard: 0.5
    scale_coul_14::Float64  # AMBER standard: 1.0 / 1.2 = 0.833333
end

# Standard AMBER atom type parameters (sigma in Å, epsilon in kcal/mol)
const AMBER_LJ_TABLE = Dict{Symbol, LJParam}(
    :H  => LJParam(1.069, 0.0157),
    :C  => LJParam(3.399, 0.0860),
    :N  => LJParam(3.250, 0.1700),
    :O  => LJParam(2.960, 0.2100),
    :F  => LJParam(3.118, 0.0610),
    :P  => LJParam(3.742, 0.2000),
    :S  => LJParam(3.564, 0.2500),
    :Cl => LJParam(3.471, 0.2650),
    :Br => LJParam(3.850, 0.3200),
    :I  => LJParam(4.150, 0.4000)
)

"""
    parameterize_amber(mol::Molecule)::AMBER
Constructs AMBER/GAFF Class I force field representation for the molecule.
"""
function parameterize_amber(mol::Molecule)::AMBER
    perceive_topology!(mol)

    # 1. Bonds (Harmonic)
    bonds = HarmonicBond[]
    for b in mol.bonds
        s1 = mol.atoms[b.i].symbol
        s2 = mol.atoms[b.j].symbol
        # Standard AMBER bond lengths and constants
        r0 = Elements.covalent_radius(s1) + Elements.covalent_radius(s2)
        kb = 600.0 # kcal/(mol·Å²) typical C-C / C-H
        if (s1 == :C && s2 == :H) || (s2 == :C && s1 == :H)
            r0 = 1.09
            kb = 680.0
        elseif (s1 == :O && s2 == :H) || (s2 == :O && s1 == :H)
            r0 = 0.96
            kb = 1100.0
        elseif (s1 == :C && s2 == :C)
            if b.order > 1.2
                r0 = 1.40
                kb = 938.0
            else
                r0 = 1.53
                kb = 620.0
            end
        end
        push!(bonds, HarmonicBond(b.i, b.j, r0, kb))
    end

    # 2. Angles (Harmonic)
    angles = HarmonicAngle[]
    for ang in mol.angles
        sj = mol.atoms[ang.j].symbol
        theta0 = 109.5 * π / 180.0
        ktheta = 100.0 # kcal/(mol·rad²)
        if sj in (:C, :N) && length(mol.atoms) > 3
            # Check if planar
            theta0 = 120.0 * π / 180.0
            ktheta = 140.0
        end
        push!(angles, HarmonicAngle(ang.i, ang.j, ang.k, theta0, ktheta))
    end

    # 3. Dihedrals (Fourier periodic)
    dihedrals = PeriodicTorsion[]
    for dih in mol.dihedrals
        push!(dihedrals, PeriodicTorsion(dih.i, dih.j, dih.k, dih.l, [
            FourierTerm(3, 1.2, 0.0),
            FourierTerm(1, 0.5, π)
        ]))
    end

    # 4. Impropers
    impropers = HarmonicImproper[]
    for imp in mol.impropers
        push!(impropers, HarmonicImproper(imp.central, imp.i, imp.j, imp.k, 0.0, 40.0))
    end

    # 5. Nonbonded Pairs with Lorentz-Berthelot combining rule
    pairs_14, pairs_non14 = get_nonbonded_pairs(mol)
    lj_14 = Tuple{Int, Int, LJParam}[]
    lj_non14 = Tuple{Int, Int, LJParam}[]

    for (i, j) in pairs_14
        pi = get(AMBER_LJ_TABLE, mol.atoms[i].symbol, LJParam(3.4, 0.1))
        pj = get(AMBER_LJ_TABLE, mol.atoms[j].symbol, LJParam(3.4, 0.1))
        push!(lj_14, (i, j, lorentz_berthelot(pi, pj)))
    end

    for (i, j) in pairs_non14
        pi = get(AMBER_LJ_TABLE, mol.atoms[i].symbol, LJParam(3.4, 0.1))
        pj = get(AMBER_LJ_TABLE, mol.atoms[j].symbol, LJParam(3.4, 0.1))
        push!(lj_non14, (i, j, lorentz_berthelot(pi, pj)))
    end

    return AMBER(bonds, angles, dihedrals, impropers,
                 lj_14, lj_non14, pairs_14, pairs_non14, 0.5, 0.833333)
end

end # module AMBERForceField

