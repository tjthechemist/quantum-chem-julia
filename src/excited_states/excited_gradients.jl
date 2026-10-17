# ==============================================================================
# QuantumChemMM.jl: Excited State Potential Energy Surface (PES) Gradients
# Enables Excited-State Geometry Minimization and Stokes Shift Calculations
# ==============================================================================

module ExcitedGradients

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...SemiEmpiricalBase
using ...ZDOSemiEmpirical
using ..CISExcitedStates

export calc_excited_state_energy_and_gradient, optimize_excited_state!

"""
    calc_excited_state_energy_and_gradient(mol::Molecule, state_idx::Int=1;
                                           multiplicity::Symbol=:singlet, delta::Float64=1e-3)
Computes the energy (in eV) and 3 x N_atoms Cartesian gradient matrix of the target excited state.
"""
function calc_excited_state_energy_and_gradient(mol::Molecule, state_idx::Int=1;
                                                multiplicity::Symbol=:singlet, delta::Float64=1e-3)
    nat = n_atoms(mol)

    # Function evaluating total energy of state state_idx
    function eval_state_energy()
        scf_res = solve_cndo2(mol; options=SCFOptions(max_cycles=40, verbose=false))
        cis_res = solve_cis(scf_res, mol; n_states=state_idx, calc_triplets=(multiplicity == :triplet))
        st = (multiplicity == :singlet) ? cis_res.singlets[state_idx] : cis_res.triplets[state_idx]
        return st.total_energy_ev
    end

    e0 = eval_state_energy()
    grad = zeros(Float64, 3, nat)

    # Central finite differences
    for a in 1:nat
        for coord in 1:3
            orig_val = mol.atoms[a].coords[coord]

            mol.atoms[a].coords[coord] = orig_val + delta
            e_plus = eval_state_energy()

            mol.atoms[a].coords[coord] = orig_val - delta
            e_minus = eval_state_energy()

            mol.atoms[a].coords[coord] = orig_val
            grad[coord, a] = (e_plus - e_minus) / (2.0 * delta)
        end
    end

    return e0, grad
end

"""
    optimize_excited_state!(mol::Molecule, state_idx::Int=1;
                            multiplicity::Symbol=:singlet, max_steps::Int=50,
                            tol::Float64=1e-2, verbose::Bool=false)
Performs geometry optimization on the excited state potential energy surface (PES).
"""
function optimize_excited_state!(mol::Molecule, state_idx::Int=1;
                                 multiplicity::Symbol=:singlet, max_steps::Int=50,
                                 tol::Float64=1e-2, verbose::Bool=false)
    nat = n_atoms(mol)
    step_size = 0.1
    converged = false
    last_e = 0.0

    for step in 1:max_steps
        e_curr, grad = calc_excited_state_energy_and_gradient(mol, state_idx; multiplicity=multiplicity)
        max_g = maximum(abs.(grad))

        if verbose && (step % 5 == 1 || step == max_steps)
            @printf("  Excited Step %3d | Energy: %14.6f eV | Max Grad: %10.5f eV/Å\n", step, e_curr, max_g)
        end

        if max_g < tol || (step > 1 && abs(e_curr - last_e) < 1e-4)
            converged = true
            break
        end

        # Steepest descent step on excited PES
        for a in 1:nat
            mol.atoms[a].coords .-= step_size .* clamp.(grad[:, a], -0.2, 0.2)
        end

        last_e = e_curr
    end

    return converged
end

end # module ExcitedGradients

