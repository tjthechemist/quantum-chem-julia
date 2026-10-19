# ==============================================================================
# QuantumChemMM.jl: QM/MM Hybrid Energy and Interaction Evaluator
# ==============================================================================

module QMMMEnergy

using LinearAlgebra
using Printf
using ..Types
using ..Elements
using ..MolMath
using ..MMBase
using ..MMEnergyGradient
using ..UFFForceField
using ..NonBonded
using ..SemiEmpiricalBase
using ..NDDOSCF
using ..QMMMEmbedding

export evaluate_qmmm, QMMMResult, print_qmmm_summary

struct QMMMResult
    total_energy_kcal::Float64
    e_qm_kcal::Float64
    e_mm_full_kcal::Float64
    e_mm_qm_kcal::Float64
    e_qmmm_interaction_kcal::Float64
    embedding::Symbol
end

"""
    evaluate_qmmm(qmmm::QMMMSystem, qm_method::AbstractSemiEmpiricalMethod=NDDOMethod(:am1);
                  ff::AbstractForceField=parameterize_uff(qmmm.full_molecule))::QMMMResult
Evaluates total hybrid QM/MM energy via ONIOM-style subtractive or additive embedding.
"""
function evaluate_qmmm(qmmm::QMMMSystem, qm_method::AbstractSemiEmpiricalMethod=NDDOMethod(:am1);
                       ff::AbstractForceField=parameterize_uff(qmmm.full_molecule))::QMMMResult
    # 1. Full system MM energy
    ec_full, _ = evaluate_mm(ff, qmmm.full_molecule; calc_grad=false)
    e_mm_full = total_energy(ec_full)

    # 2. Extract QM subsystem (with link hydrogens)
    qm_mol = extract_subsystem(qmmm.full_molecule, qmmm.qm_indices; link_atoms=qmmm.link_atoms)

    # 3. QM calculation on QM subsystem
    qm_res = solve_nddo(qm_mol, qm_method isa NDDOMethod ? qm_method : NDDOMethod(:am1))
    # 1 eV = 23.060548 kcal/mol
    e_qm_kcal = qm_res.total_energy_ev * 23.060548

    # 4. MM calculation on QM subsystem (for subtractive mechanical embedding)
    ff_qm = parameterize_uff(qm_mol)
    ec_qm, _ = evaluate_mm(ff_qm, qm_mol; calc_grad=false)
    e_mm_qm = total_energy(ec_qm)

    # 5. Nonbonded QM/MM electrostatic and vdW interaction
    e_inter_vdw = 0.0
    e_inter_elec = 0.0

    for qm_idx in qmmm.qm_indices
        at_qm = qmmm.full_molecule.atoms[qm_idx]
        q_qm = at_qm.partial_charge
        p_qm = LJParam(Elements.vdw_radius(at_qm.symbol), 0.1)

        for mm_idx in qmmm.mm_indices
            at_mm = qmmm.full_molecule.atoms[mm_idx]
            q_mm = at_mm.partial_charge
            p_mm = LJParam(Elements.vdw_radius(at_mm.symbol), 0.1)

            # vdW interaction
            p_comb = lorentz_berthelot(p_qm, p_mm)
            e_vdw, _, _ = calc_lj_energy_and_grad(p_comb, at_qm.coords, at_mm.coords)
            e_inter_vdw += e_vdw

            # Electrostatic interaction
            if abs(q_qm * q_mm) > 1e-6
                e_coul, _, _ = calc_coulomb_energy_and_grad(q_qm, q_mm, at_qm.coords, at_mm.coords)
                e_inter_elec += e_coul
            end
        end
    end

    e_inter = e_inter_vdw + e_inter_elec

    # 6. Total Energy: E_QMMM = E_QM(QM) + E_MM(Full) - E_MM(QM)
    e_total_kcal = if qmmm.embedding == :mechanical
        e_qm_kcal + e_mm_full - e_mm_qm
    else
        # Electrostatic embedding with explicit QM/MM interaction
        e_qm_kcal + e_mm_full - e_mm_qm + e_inter
    end

    return QMMMResult(e_total_kcal, e_qm_kcal, e_mm_full, e_mm_qm, e_inter, qmmm.embedding)
end

function print_qmmm_summary(res::QMMMResult)
    println("==================================================")
    println("               Hybrid QM/MM Results               ")
    println("==================================================")
    @printf("  Embedding Type:           %s\n", uppercase(string(res.embedding)))
    @printf("  QM Energy (E_QM):         %16.4f kcal/mol\n", res.e_qm_kcal)
    @printf("  MM Full Energy:           %16.4f kcal/mol\n", res.e_mm_full_kcal)
    @printf("  MM QM Subsystem Energy:   %16.4f kcal/mol\n", res.e_mm_qm_kcal)
    @printf("  QM/MM Interaction Energy: %16.4f kcal/mol\n", res.e_qmmm_interaction_kcal)
    println("--------------------------------------------------")
    @printf("  TOTAL HYBRID QM/MM ENERGY:%16.4f kcal/mol\n", res.total_energy_kcal)
    println("==================================================")
end

end # module QMMMEnergy

