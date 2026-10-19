# ==============================================================================
# QuantumChemMM.jl: Unified Molecular Mechanics Energy and Gradient Evaluator
# ==============================================================================

module MMEnergyGradient

using LinearAlgebra
using ..Types
using ..MMBase
using ..BondedPotentials
using ..Dihedrals
using ..NonBonded
using ..UFFForceField
using ..AMBERForceField
using ..OPLSForceField
using ..CHARMMForceField
using ..MM3ForceField

export evaluate_mm, energy, gradient

"""
    evaluate_mm(ff::AbstractForceField, mol::Molecule; calc_grad::Bool=true)
Computes energy decomposition (EnergyComponents) and analytical Cartesian gradient matrix (3 x N_atoms)
in kcal/(mol·Å).
"""
function evaluate_mm(ff::AbstractForceField, mol::Molecule; calc_grad::Bool=true)
    nat = n_atoms(mol)
    coords = coordinates_matrix(mol)
    grad = zeros(Float64, 3, nat)

    e_bond = 0.0
    e_angle = 0.0
    e_dih = 0.0
    e_imp = 0.0
    e_vdw = 0.0
    e_elec = 0.0
    e_cross = 0.0

    # 1. Bonds
    if hasfield(typeof(ff), :bonds)
        for b in ff.bonds
            ri = coords[:, b.i]
            rj = coords[:, b.j]
            e, gi, gj = BondedPotentials.calc_bond_energy_and_grad(b, ri, rj)
            e_bond += e
            if calc_grad
                grad[:, b.i] .+= gi
                grad[:, b.j] .+= gj
            end
        end
    end

    # 2. Angles
    if hasfield(typeof(ff), :angles)
        for ang in ff.angles
            ri = coords[:, ang.i]
            rj = coords[:, ang.j]
            rk = coords[:, ang.k]
            e, gi, gj, gk = BondedPotentials.calc_angle_energy_and_grad(ang, ri, rj, rk)
            e_angle += e
            if calc_grad
                grad[:, ang.i] .+= gi
                grad[:, ang.j] .+= gj
                grad[:, ang.k] .+= gk
            end
        end
    end

    # 3. Cross terms (MM3 stretch-bend)
    if ff isa MM3ForceField.MM3
        for sb in ff.stretch_bends
            ri = coords[:, sb.i]
            rj = coords[:, sb.j]
            rk = coords[:, sb.k]
            e, gi, gj, gk = MM3ForceField.calc_stretch_bend_energy_and_grad(sb, ri, rj, rk)
            e_cross += e
            if calc_grad
                grad[:, sb.i] .+= gi
                grad[:, sb.j] .+= gj
                grad[:, sb.k] .+= gk
            end
        end
    end

    # 4. Dihedrals
    if hasfield(typeof(ff), :dihedrals)
        for dih in ff.dihedrals
            ri = coords[:, dih.i]
            rj = coords[:, dih.j]
            rk = coords[:, dih.k]
            rl = coords[:, dih.l]
            e, gi, gj, gk, gl = Dihedrals.calc_dihedral_energy_and_grad(dih, ri, rj, rk, rl)
            e_dih += e
            if calc_grad
                grad[:, dih.i] .+= gi
                grad[:, dih.j] .+= gj
                grad[:, dih.k] .+= gk
                grad[:, dih.l] .+= gl
            end
        end
    end

    # 5. Impropers
    if hasfield(typeof(ff), :impropers)
        for imp in ff.impropers
            rc = coords[:, imp.central]
            ri = coords[:, imp.i]
            rj = coords[:, imp.j]
            rk = coords[:, imp.k]
            e, gc, gi, gj, gk = Dihedrals.calc_improper_energy_and_grad(imp, rc, ri, rj, rk)
            e_imp += e
            if calc_grad
                grad[:, imp.central] .+= gc
                grad[:, imp.i] .+= gi
                grad[:, imp.j] .+= gj
                grad[:, imp.k] .+= gk
            end
        end
    end

    # 6. Nonbonded vdW (LJ / Buckingham)
    if hasfield(typeof(ff), :lj_pairs_14)
        for (i, j, param) in ff.lj_pairs_14
            ri = coords[:, i]
            rj = coords[:, j]
            e, gi, gj = NonBonded.calc_lj_energy_and_grad(param, ri, rj; scale=ff.scale_vdw_14)
            e_vdw += e
            if calc_grad
                grad[:, i] .+= gi
                grad[:, j] .+= gj
            end
        end
    end

    if hasfield(typeof(ff), :lj_pairs_non14)
        for (i, j, param) in ff.lj_pairs_non14
            ri = coords[:, i]
            rj = coords[:, j]
            e, gi, gj = NonBonded.calc_lj_energy_and_grad(param, ri, rj; scale=1.0)
            e_vdw += e
            if calc_grad
                grad[:, i] .+= gi
                grad[:, j] .+= gj
            end
        end
    end

    if ff isa MM3ForceField.MM3
        for (i, j, param) in ff.buckingham_pairs
            ri = coords[:, i]
            rj = coords[:, j]
            e, gi, gj = NonBonded.calc_buckingham_energy_and_grad(param, ri, rj; scale=1.0)
            e_vdw += e
            if calc_grad
                grad[:, i] .+= gi
                grad[:, j] .+= gj
            end
        end
    end

    # 7. Nonbonded Coulomb Electrostatics
    if hasfield(typeof(ff), :coulomb_pairs_14)
        for (i, j) in ff.coulomb_pairs_14
            qi = mol.atoms[i].partial_charge
            qj = mol.atoms[j].partial_charge
            if abs(qi * qj) > 1e-8
                ri = coords[:, i]
                rj = coords[:, j]
                e, gi, gj = NonBonded.calc_coulomb_energy_and_grad(qi, qj, ri, rj; scale=ff.scale_coul_14)
                e_elec += e
                if calc_grad
                    grad[:, i] .+= gi
                    grad[:, j] .+= gj
                end
            end
        end
    end

    if hasfield(typeof(ff), :coulomb_pairs_non14)
        for (i, j) in ff.coulomb_pairs_non14
            qi = mol.atoms[i].partial_charge
            qj = mol.atoms[j].partial_charge
            if abs(qi * qj) > 1e-8
                ri = coords[:, i]
                rj = coords[:, j]
                e, gi, gj = NonBonded.calc_coulomb_energy_and_grad(qi, qj, ri, rj; scale=1.0)
                e_elec += e
                if calc_grad
                    grad[:, i] .+= gi
                    grad[:, j] .+= gj
                end
            end
        end
    end

    ec = EnergyComponents(e_bond=e_bond, e_angle=e_angle, e_dihedral=e_dih,
                          e_improper=e_imp, e_vdw=e_vdw, e_elec=e_elec, e_cross=e_cross)
    return ec, grad
end

energy(ff::AbstractForceField, mol::Molecule) = total_energy(evaluate_mm(ff, mol; calc_grad=false)[1])
gradient(ff::AbstractForceField, mol::Molecule) = evaluate_mm(ff, mol; calc_grad=true)[2]

end # module MMEnergyGradient
