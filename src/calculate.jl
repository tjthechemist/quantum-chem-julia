# ==============================================================================
# QuantumChemMM.jl: Unified High-Level Calculation Entry Point
# Dispatches across Molecular Mechanics, Semi-Empirical, Ab Initio, DFT, and Post-HF
# Supports Multi-Vendor GPU Acceleration on NVIDIA, AMD, and Intel GPUs
# Implicit solvation (COSMO/GB) available for ALL methods via solvent= keyword
# ==============================================================================

module UnifiedCalculation

using LinearAlgebra
using ..Types
using ..Elements
using ..MMBase
using ..UFFForceField
using ..AMBERForceField
using ..OPLSForceField
using ..CHARMMForceField
using ..MM3ForceField
using ..MMEnergyGradient
using ..SemiEmpiricalBase
using ..HuckelTheory
using ..ExtendedHuckel
using ..ZDOSemiEmpirical
using ..NDDOSCF
using ..TightBinding
using ..SlaterOrbitals
using ..BasisSets
using ..MolecularIntegrals
using ..HartreeFock
using ..KohnShamDFT
using ..MP2Theory
using ..CISDTheory
using ..CoupledCluster
using ..CISExcitedStates
using ..TimeDependentMethods
using ..NMRSpectroscopy
using ..SolvationModels
import ..Spectroscopy: calc_vibrational_frequencies, calc_raman_modes,
                       uv_vis_transitions
using ..GPUInterface
using ..GPUSolvers
import ..OptimizationAlgorithms: OptimizationResult, optimize_geometry!
using ..ResultOutput: format_output, write_output

export calculate, CalculationResult, optimize_geometry!, calc_uv_vis

"""
    CalculationResult
Unified output container for every calculation method.

Fields:
- `method`                    : Symbol identifying the method (e.g. `:hf`, `:b3lyp`, `:mp2`)
- `total_energy_ev`           : Gas-phase total energy in eV
- `total_energy_hartree`      : Gas-phase total energy in Hartree
- `total_energy_kcal`         : Gas-phase total energy in kcal/mol
- `solvation`                 : `SolvationResult` (or `nothing` when `solvent=:vacuum`)
- `total_energy_solv_ev`      : Gas-phase + ΔG_solv in eV  (equals gas-phase when vacuum)
- `total_energy_solv_hartree` : Gas-phase + ΔG_solv in Hartree
- `total_energy_solv_kcal`    : Gas-phase + ΔG_solv in kcal/mol
- `orbital_energies_ev`      : Molecular orbital energies in eV, or `nothing` when unavailable
- `raw_result`                : Full method-specific result object
"""
struct CalculationResult
    method::Symbol
    # Gas-phase energies
    total_energy_ev::Float64
    total_energy_hartree::Float64
    total_energy_kcal::Float64
    # Implicit solvation correction
    solvation::Union{SolvationResult, Nothing}
    # Total energy including solvation (= gas-phase when solvent == :vacuum)
    total_energy_solv_ev::Float64
    total_energy_solv_hartree::Float64
    total_energy_solv_kcal::Float64
    # Molecular orbital energies, normalized to eV across methods.
    orbital_energies_ev::Union{Nothing, Vector{Float64}}
    # Raw method-specific result
    raw_result::Any
end

const HARTREE_TO_EV  = 27.211386245988
const EV_TO_KCAL     = 23.060548
const KCAL_TO_EV     = 1.0 / EV_TO_KCAL
const KCAL_TO_HARTREE = KCAL_TO_EV / HARTREE_TO_EV

function _orbital_energies_ev(raw)::Union{Nothing, Vector{Float64}}
    if hasproperty(raw, :orbital_energies_ev)
        return Float64.(raw.orbital_energies_ev)
    elseif hasproperty(raw, :mo_energies_ev)
        return Float64.(raw.mo_energies_ev)
    elseif hasproperty(raw, :mo_energies_hartree)
        return Float64.(raw.mo_energies_hartree) .* HARTREE_TO_EV
    elseif hasproperty(raw, :mo_energies)
        unit = raw.energy_unit
        if unit == "eV"
            return Float64.(raw.mo_energies)
        elseif unit == "Hartree"
            return Float64.(raw.mo_energies) .* HARTREE_TO_EV
        end
        throw(ArgumentError("Unsupported molecular orbital energy unit: $unit"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Internal helper: apply solvation correction and build CalculationResult
# ---------------------------------------------------------------------------
function _make_result(method::Symbol,
                      e_ev::Float64, e_ha::Float64, e_kcal::Float64,
                      mol::Molecule, solvent::Symbol,
                      charges::Vector{Float64},
                      raw;
                      orbital_energies_ev::Union{Nothing, Vector{Float64}}=_orbital_energies_ev(raw))::CalculationResult
    if solvent == :vacuum
        return CalculationResult(method,
                                 e_ev, e_ha, e_kcal,
                                 nothing,
                                 e_ev, e_ha, e_kcal,
                                 orbital_energies_ev,
                                 raw)
    end

    solv = calc_solvation_energy(mol, solvent; charges=charges)
    dg_kcal  = solv.total_delta_g_solv_kcal
    dg_ev    = dg_kcal * KCAL_TO_EV
    dg_ha    = dg_kcal * KCAL_TO_HARTREE

    return CalculationResult(method,
                             e_ev, e_ha, e_kcal,
                             solv,
                             e_ev  + dg_ev,
                             e_ha  + dg_ha,
                             e_kcal + dg_kcal,
                             orbital_energies_ev,
                             raw)
end

# Extract partial charges from a raw result (best-effort, fallback to zeros)
function _charges(mol::Molecule, raw)::Vector{Float64}
    nat = n_atoms(mol)
    if hasproperty(raw, :mulliken_charges)
        return raw.mulliken_charges
    elseif hasproperty(raw, :partial_charges)
        return raw.partial_charges
    elseif hasproperty(raw, :density_matrix) &&
           hasproperty(raw, :overlap_matrix) && hasproperty(raw, :basis)
        density_overlap = raw.density_matrix * raw.overlap_matrix
        basis_functions = raw.basis.functions
        nuclear_charges = hasproperty(raw, :effective_nuclear_charges) ?
                          raw.effective_nuclear_charges :
                          Float64[ mol.atoms[i].atomic_number for i in 1:nat ]
        return [nuclear_charges[atom] -
                sum(density_overlap[mu, mu] for mu in eachindex(basis_functions)
                    if basis_functions[mu].atom_index == atom)
                for atom in 1:nat]
    else
        return [mol.atoms[i].partial_charge for i in 1:nat]
    end
end

# ---------------------------------------------------------------------------
"""
    calculate(mol::Molecule, method::Symbol;
              basis::String=\"sto-3g\",
              basis_set=nothing,
              functional::Symbol=:b3lyp,
              solvent::Symbol=:vacuum,
              device::AbstractDevice=get_default_device(),
              options::SCFOptions=SCFOptions(),
              kwargs...)::CalculationResult

Unified computational chemistry calculation dispatcher with seamless multi-vendor
GPU acceleration and optional implicit solvation for **all** methods.

Supported Methods:
  - Molecular Mechanics: :uff, :amber, :opls, :charmm, :mm3
  - Semi-Empirical:      :huckel, :eht, :cndo2, :indo, :mndo, :am1, :pm3, :rm1, :pm6, :gfn2_xtb
  - Ab Initio:           :hf, :rhf, :uhf
  - DFT:                 :dft, :b3lyp, :pbe, :pbe0, :lda, :blyp
  - Post-Ab Initio:      :mp2, :scs_mp2, :cisd, :ccsd, :ccsdt
  - Excited States:      :cis, :tdhf, :tddft, :tda
  - NMR Spectroscopy:    :nmr, :nmr_shielding

Supported Solvents (pass as `solvent=:water` etc.):
  :vacuum (default), :water, :acetonitrile, :dmso, :methanol, :ethanol,
  :chloroform, :dichloromethane, :toluene, :hexane, :octanol

Supported Devices:
  - `CPUDevice()`    — Host CPU multithreaded SIMD
  - `CUDADevice()`   — NVIDIA CUDA GPU
  - `ROCmDevice()`   — AMD ROCm GPU
  - `OneAPIDevice()` — Intel oneAPI / Level Zero GPU

When a `solvent` other than `:vacuum` is specified, a COSMO/Generalized-Born
implicit solvation free energy (ΔG_solv) is computed and added to the total.
Gas-phase energies are always preserved in `result.total_energy_*`; solvated
totals are in `result.total_energy_solv_*`.

Set `ecp_name="qcm-prototype"` to opt in to the built-in ECP prototype for supported
atoms. Its effective charges and one-electron matrix are included in HF/DFT SCF.
"""
function _calculate(mol::Molecule, method::Symbol;
                   basis::String="sto-3g",
                   basis_set::Union{Nothing, BasisSet}=nothing,
                   ecp_name::Union{Nothing, String}=nothing,
                   functional::Symbol=:b3lyp,
                   solvent::Symbol=:vacuum,
                   device::AbstractDevice=get_default_device(),
                   options::SCFOptions=SCFOptions(),
                   n_states::Int=6,
                   calc_triplets::Union{Nothing, Bool}=nothing,
                   kwargs...)::CalculationResult

    m = Symbol(lowercase(string(method)))
    if ecp_name !== nothing &&
       !(m in (:hf, :rhf, :uhf, :dft, :b3lyp, :pbe, :pbe0, :lda, :blyp,
               :mp2, :scs_mp2, :cisd, :ccsd, :ccsdt, :ccsd_t,
               :tdhf, :tddft, :tda, :nmr, :nmr_shielding))
        throw(ArgumentError("ECPs are only supported for HF/DFT-based methods; '$m' does not use that SCF path."))
    end

    # ── 1. Molecular Mechanics ──────────────────────────────────────────────
    if m in (:uff, :amber, :opls, :charmm, :mm3)
        ff = if m == :uff;    parameterize_uff(mol)
             elseif m == :amber;  parameterize_amber(mol)
             elseif m == :opls;   parameterize_opls(mol)
             elseif m == :charmm; parameterize_charmm(mol)
             else                 parameterize_mm3(mol)
             end
        ec, _ = is_gpu(device) ? evaluate_mm_gpu(ff, mol; device=device) :
                                  evaluate_mm(ff, mol; calc_grad=false)
        e_kcal = total_energy(ec)
        e_ev   = e_kcal * KCAL_TO_EV
        e_ha   = e_ev   / HARTREE_TO_EV
        charges = m == :uff ? estimate_uff_charges(mol) :
                  [mol.atoms[i].partial_charge for i in 1:n_atoms(mol)]
        return _make_result(m, e_ev, e_ha, e_kcal, mol, solvent, charges, ec)

    # ── 2a. Hückel (pi-electron only) ──────────────────────────────────────
    elseif m == :huckel
        res    = solve_huckel(mol)
        e_ev   = res.total_pi_energy_ev
        charges = _charges(mol, res)
        return _make_result(:huckel, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 2b. Extended Hückel ─────────────────────────────────────────────────
    elseif m == :eht
        res    = solve_extended_huckel(mol)
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(:eht, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 2c. CNDO/2 ──────────────────────────────────────────────────────────
    elseif m in (:cndo, :cndo2)
        res = is_gpu(device) ? solve_scf_gpu(mol, :cndo2; device=device, options=options) :
                                solve_cndo2(mol; options=options)
        e_ev   = res.total_energy
        charges = _charges(mol, res)
        return _make_result(:cndo2, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 2d. INDO ────────────────────────────────────────────────────────────
    elseif m == :indo
        res    = solve_indo(mol; options=options)
        e_ev   = res.total_energy
        charges = _charges(mol, res)
        return _make_result(:indo, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 2e. NDDO family (MNDO, AM1, PM3, RM1, PM6) ─────────────────────────
    elseif m in (:mndo, :am1, :pm3, :rm1, :pm6)
        res    = solve_nddo(mol, NDDOMethod(m); options=options)
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(m, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 2f. DFTB / xTB tight-binding ────────────────────────────────────────
    elseif m in (:dftb, :xtb, :gfn0_xtb, :gfn1_xtb, :gfn2_xtb)
        v   = (m in (:dftb, :xtb)) ? :gfn2_xtb : m
        res = solve_tight_binding(mol, TightBindingModel(v); options=options)
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(v, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 3. Hartree-Fock ─────────────────────────────────────────────────────
    elseif m in (:hf, :rhf)
        res    = solve_hf(mol; basis_name=basis, basis_set=basis_set, ecp_name=ecp_name,
                          device=device, options=options)
        e_ha   = res.total_energy_hartree
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(:hf, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    elseif m == :uhf
        res = solve_uhf(mol; basis_name=basis, basis_set=basis_set, ecp_name=ecp_name,
                        device=device, options=options)
        e_ha = res.total_energy_hartree
        e_ev = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(:uhf, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res;
                            orbital_energies_ev=res.alpha_mo_energies_hartree .* HARTREE_TO_EV)

    # ── 4. Density Functional Theory ────────────────────────────────────────
    elseif m in (:dft, :b3lyp, :pbe, :pbe0, :lda, :blyp)
        func   = (m == :dft) ? functional : m
        res    = solve_dft(mol; functional=func, basis_name=basis, basis_set=basis_set,
                           ecp_name=ecp_name,
                           device=device, options=options)
        e_ha   = res.total_energy_hartree
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(Symbol("dft_", func), e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 5a. MP2 / SCS-MP2 ───────────────────────────────────────────────────
    elseif m in (:mp2, :scs_mp2)
        hf_res  = solve_hf(mol; basis_name=basis, basis_set=basis_set, ecp_name=ecp_name,
                           device=device, options=options)
        mp2_res = solve_mp2(hf_res; device=device)
        e_ha    = (m == :scs_mp2) ?
                  (hf_res.total_energy_hartree + mp2_res.scs_mp2_correlation_hartree) :
                  mp2_res.total_mp2_energy_hartree
        e_ev    = e_ha * HARTREE_TO_EV
        charges = _charges(mol, hf_res)
        return _make_result(m, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, mp2_res;
                            orbital_energies_ev=hf_res.mo_energies_hartree .* HARTREE_TO_EV)

    # ── 5b. CISD ────────────────────────────────────────────────────────────
    elseif m == :cisd
        hf_res   = solve_hf(mol; basis_name=basis, basis_set=basis_set, ecp_name=ecp_name,
                            device=device, options=options)
        cisd_res = solve_cisd(hf_res; device=device)
        e_ha     = cisd_res.total_cisd_energy_hartree
        e_ev     = cisd_res.total_cisd_energy_ev
        charges  = _charges(mol, hf_res)
        return _make_result(:cisd, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, cisd_res;
                            orbital_energies_ev=hf_res.mo_energies_hartree .* HARTREE_TO_EV)

    # ── 5c. CCSD / CCSD(T) ──────────────────────────────────────────────────
    elseif m in (:ccsd, :ccsdt, :ccsd_t)
        hf_res = solve_hf(mol; basis_name=basis, basis_set=basis_set, ecp_name=ecp_name,
                          device=device, options=options)
        cc_res = solve_ccsd(hf_res; compute_triples=(m in (:ccsdt, :ccsd_t)), device=device)
        e_ha   = (m in (:ccsdt, :ccsd_t)) ?
                 cc_res.total_ccsdt_energy_hartree :
                 cc_res.total_ccsd_energy_hartree
        e_ev   = e_ha * HARTREE_TO_EV
        charges = _charges(mol, hf_res)
        return _make_result(m, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, cc_res;
                            orbital_energies_ev=hf_res.mo_energies_hartree .* HARTREE_TO_EV)

    # ── 6. Excited States (CIS) ─────────────────────────────────────────────
    elseif m == :cis
        scf_res = is_gpu(device) ?
                  solve_scf_gpu(mol, :cndo2; device=device, options=options) :
                  solve_cndo2(mol; options=options)
        cis_res = solve_cis(scf_res, mol; n_states=n_states,
                            calc_triplets=calc_triplets === nothing ? true : calc_triplets,
                            device=device)
        e_ev    = cis_res.ground_energy_ev
        charges = _charges(mol, scf_res)
        return _make_result(:cis, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, cis_res;
                            orbital_energies_ev=_orbital_energies_ev(scf_res))

    # ── 6b. Tamm-Dancoff response for RHF and restricted DFT references ──────
    elseif m == :tdhf
        hf_res = solve_hf(mol; basis_name=basis, basis_set=basis_set,
                          ecp_name=ecp_name, device=device, options=options)
        td_res = solve_tdhf(hf_res, mol; n_states=n_states,
                            calc_triplets=calc_triplets === nothing ? true : calc_triplets)
        e_ha = hf_res.total_energy_hartree
        e_ev = hf_res.total_energy_ev
        charges = _charges(mol, hf_res)
        return _make_result(:tdhf, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, td_res;
                            orbital_energies_ev=hf_res.mo_energies_hartree .* HARTREE_TO_EV)

    elseif m in (:tddft, :tda)
        func = functional
        func in (:lda, :pbe, :pbe0, :blyp, :b3lyp) ||
            throw(ArgumentError("TDDFT supports :lda, :pbe, :pbe0, :blyp, and :b3lyp."))
        calc_triplets === true &&
            throw(ArgumentError("Triplet TDDFT response is not implemented yet; use :tdhf for triplet TDA."))
        dft_res = solve_dft(mol; functional=func, basis_name=basis,
                            basis_set=basis_set, ecp_name=ecp_name,
                            device=device, options=options)
        td_res = solve_tddft(dft_res, mol; n_states=n_states, functional=func,
                             calc_triplets=calc_triplets === nothing ? false : calc_triplets)
        e_ha = dft_res.total_energy_hartree
        e_ev = dft_res.total_energy_ev
        charges = _charges(mol, dft_res)
        return _make_result(:tddft, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, td_res;
                            orbital_energies_ev=dft_res.mo_energies_hartree .* HARTREE_TO_EV)

    # ── 7. NMR Chemical Shifts & Shieldings ─────────────────────────────────
    elseif m in (:nmr, :nmr_shielding)
        hf_res  = solve_hf(mol; basis_name=basis, basis_set=basis_set, ecp_name=ecp_name,
                           device=device, options=options)
        nmr_res = calc_nmr_shieldings(hf_res, mol; device=device, method=:hf)
        e_ha    = hf_res.total_energy_hartree
        e_ev    = hf_res.total_energy_ev
        charges = _charges(mol, hf_res)
        return _make_result(:nmr, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, nmr_res;
                            orbital_energies_ev=_orbital_energies_ev(hf_res))

    else
        throw(ArgumentError("Unknown calculation method: $method. " *
              "Supported: :uff, :amber, :opls, :charmm, :mm3, :huckel, :eht, " *
              ":cndo2, :indo, :mndo, :am1, :pm3, :rm1, :pm6, :gfn2_xtb, " *
              ":hf, :dft, :b3lyp, :pbe, :pbe0, :lda, :blyp, " *
              ":mp2, :scs_mp2, :cisd, :ccsd, :ccsdt, :cis, :tdhf, :tddft, :tda, :nmr"))
    end
end

"""
    calculate(mol, method; output_file=nothing, kwargs...)

Run a calculation and return its `CalculationResult`. When `output_file` is
provided, save a formatted plain-text report using that exact filename and
extension. All other keywords are forwarded to the calculation method.
"""
function calculate(mol::Molecule, method::Symbol;
                   output_file::Union{Nothing, AbstractString}=nothing,
                   kwargs...)::CalculationResult
    result = _calculate(mol, method; kwargs...)
    isnothing(output_file) || write_output(output_file, result)
    return result
end

"""
    optimize_geometry!(mol, method::Symbol; kwargs...)

Optimize a semi-empirical, HF/UHF, or DFT geometry using central finite-difference
energy gradients. Coordinates are updated in place. `finite_difference_step` is
in Angstrom and `grad_tol` is in Hartree/Angstrom.
"""
function optimize_geometry!(mol::Molecule, method::Symbol;
                            basis::String="sto-3g",
                            basis_set::Union{Nothing, BasisSet}=nothing,
                            ecp_name::Union{Nothing, String}=nothing,
                            functional::Symbol=:b3lyp,
                            solvent::Symbol=:vacuum,
                            device::AbstractDevice=get_default_device(),
                            options::SCFOptions=SCFOptions(),
                            optimization_method::Symbol=:lbfgs,
                            max_steps::Int=200,
                            grad_tol::Float64=1e-4,
                            finite_difference_step::Float64=1e-3,
                            verbose::Bool=false,
                            kwargs...)::OptimizationResult
    m = Symbol(lowercase(string(method)))
    supported = (:huckel, :eht, :cndo, :cndo2, :indo, :mndo, :am1, :pm3,
                 :rm1, :pm6, :dftb, :xtb, :gfn0_xtb, :gfn1_xtb, :gfn2_xtb,
                 :hf, :rhf, :uhf, :dft, :b3lyp, :pbe, :pbe0, :lda, :blyp)
    m in supported ||
        throw(ArgumentError("Quantum geometry optimization supports semi-empirical, HF/UHF, and DFT methods; got '$m'."))
    finite_difference_step > 0.0 && isfinite(finite_difference_step) ||
        throw(ArgumentError("finite_difference_step must be finite and positive."))

    energy_function = function (current_mol::Molecule)
        result = calculate(current_mol, m; basis=basis, basis_set=basis_set,
                           ecp_name=ecp_name, functional=functional,
                           solvent=solvent, device=device, options=options,
                           kwargs...)
        return result.total_energy_solv_hartree
    end
    gradient_function = function (current_mol::Molecule)
        coords = coordinates_matrix(current_mol)
        gradient_matrix = zeros(Float64, size(coords))
        try
            for atom_idx in axes(coords, 2), axis in axes(coords, 1)
                plus = copy(coords)
                minus = copy(coords)
                plus[axis, atom_idx] += finite_difference_step
                minus[axis, atom_idx] -= finite_difference_step
                set_coordinates!(current_mol, plus)
                e_plus = energy_function(current_mol)
                set_coordinates!(current_mol, minus)
                e_minus = energy_function(current_mol)
                gradient_matrix[axis, atom_idx] =
                    (e_plus - e_minus) / (2.0 * finite_difference_step)
            end
        finally
            set_coordinates!(current_mol, coords)
        end
        return gradient_matrix
    end

    return optimize_geometry!(mol, energy_function, gradient_function;
                              method=optimization_method, max_steps=max_steps,
                              grad_tol=grad_tol, verbose=verbose)
end

const VIBRATIONAL_QM_METHODS = (:eht, :cndo, :cndo2, :indo, :mndo, :am1,
                                :pm3, :rm1, :pm6, :dftb, :xtb, :gfn0_xtb,
                                :gfn1_xtb, :gfn2_xtb, :hf, :rhf, :uhf, :dft,
                                :b3lyp, :pbe, :pbe0, :lda, :blyp)

function _vibrational_qm_callbacks(mol::Molecule, method::Symbol;
                                   basis::String, basis_set::Union{Nothing, BasisSet},
                                   ecp_name::Union{Nothing, String},
                                   functional::Symbol, solvent::Symbol,
                                   device::AbstractDevice, options::SCFOptions)
    m = Symbol(lowercase(string(method)))
    m in VIBRATIONAL_QM_METHODS ||
        throw(ArgumentError("QM vibrations support semi-empirical, HF/UHF, and DFT methods; got '$m'."))

    function singlepoint(current_mol::Molecule)
        calculate(current_mol, m; basis=basis, basis_set=basis_set,
                  ecp_name=ecp_name, functional=functional, solvent=solvent,
                  device=device, options=options)
    end
    energy_function = current_mol ->
        singlepoint(current_mol).total_energy_solv_hartree
    dipole_function = function (current_mol::Molecule)
        result = singlepoint(current_mol)
        charges = _charges(current_mol, result.raw_result)
        com = center_of_mass(current_mol)
        dipole = zeros(Float64, 3)
        for atom_idx in 1:n_atoms(current_mol)
            dipole .+= charges[atom_idx] .*
                       (current_mol.atoms[atom_idx].coords .- com)
        end
        return dipole .* 4.80320427
    end
    polarizability_function = current_mol -> begin
        result = singlepoint(current_mol)
        _independent_particle_polarizability(current_mol, result.raw_result)
    end
    return energy_function, dipole_function, polarizability_function
end

function _mo_energies_hartree(raw)
    hasproperty(raw, :mo_energies_hartree) &&
        return Float64.(raw.mo_energies_hartree)
    hasproperty(raw, :mo_energies_ev) &&
        return Float64.(raw.mo_energies_ev) ./ HARTREE_TO_EV
    if hasproperty(raw, :mo_energies)
        unit = raw.energy_unit
        unit == "eV" && return Float64.(raw.mo_energies) ./ HARTREE_TO_EV
        unit == "Hartree" && return Float64.(raw.mo_energies)
        throw(ArgumentError("Unsupported MO energy unit '$unit' for Raman response."))
    end
    throw(ArgumentError("Method result does not expose molecular orbital energies for Raman response."))
end

function _mo_dipole_matrices(mol::Molecule, raw, nmo::Int)
    if hasproperty(raw, :basis) && hasproperty(raw.basis, :functions)
        basis_functions = raw.basis.functions
        length(basis_functions) == nmo ||
            throw(DimensionMismatch("AO basis size does not match MO count."))
        centers = [Float64.(basis_functions[mu].center) for mu in 1:nmo]
        overlap = hasproperty(raw, :overlap_matrix) ?
                  Matrix{Float64}(raw.overlap_matrix) :
                  compute_overlap(raw.basis)
    else
        orbitals = SlaterOrbitals.build_valence_basis(mol)
        length(orbitals) == nmo ||
            throw(DimensionMismatch("Semi-empirical AO basis size does not match MO count."))
        centers = [mol.atoms[orbital.atom_index].coords .*
                   BasisSets.ANGSTROM_TO_BOHR for orbital in orbitals]
        overlap = hasproperty(raw, :overlap_matrix) ?
                  Matrix{Float64}(raw.overlap_matrix) : Matrix{Float64}(I, nmo, nmo)
    end

    dipole_matrices = [zeros(Float64, nmo, nmo) for _ in 1:3]
    for mu in 1:nmo, nu in 1:nmo
        midpoint = 0.5 .* (centers[mu] .+ centers[nu])
        for axis in 1:3
            dipole_matrices[axis][mu, nu] = midpoint[axis] * overlap[mu, nu]
        end
    end
    return dipole_matrices
end

function _spin_polarizability!(alpha::Matrix{Float64}, energies::Vector{Float64},
                               coefficients::Matrix{Float64}, n_occ::Int,
                               dipole_matrices)
    0 <= n_occ <= length(energies) ||
        throw(ArgumentError("Invalid occupied-orbital count in Raman response."))
    size(coefficients, 2) == length(energies) ||
        throw(DimensionMismatch("MO coefficients and orbital energies are inconsistent."))
    for i in 1:n_occ, a in (n_occ + 1):length(energies)
        gap = energies[a] - energies[i]
        gap > 1e-8 ||
            throw(ArgumentError("Non-positive occupied-virtual gap in Raman response."))
        transition = [dot(@view(coefficients[:, i]),
                          dipole_matrices[axis] * @view(coefficients[:, a]))
                      for axis in 1:3]
        alpha .+= (2.0 / gap) .* (transition * transition')
    end
    return alpha
end

function _independent_particle_polarizability(mol::Molecule, raw)
    if hasproperty(raw, :reference) && raw.reference == :uhf
        nmo = length(raw.alpha_mo_energies_hartree)
        dipole_matrices = _mo_dipole_matrices(mol, raw, nmo)
        alpha = zeros(Float64, 3, 3)
        _spin_polarizability!(alpha, raw.alpha_mo_energies_hartree,
                              raw.alpha_mo_coefficients, raw.n_alpha,
                              dipole_matrices)
        _spin_polarizability!(alpha, raw.beta_mo_energies_hartree,
                              raw.beta_mo_coefficients, raw.n_beta,
                              dipole_matrices)
        return alpha
    end

    energies = _mo_energies_hartree(raw)
    nmo = length(energies)
    coefficients = hasproperty(raw, :mo_coefficients) ?
                   Matrix{Float64}(raw.mo_coefficients) :
                   hasproperty(raw, :coefficients) ?
                   Matrix{Float64}(raw.coefficients) :
                   throw(ArgumentError("Method result does not expose MO coefficients for Raman response."))
    n_electrons = sum(Elements.valence_electrons(atom.symbol) for atom in mol.atoms) -
                  mol.total_charge
    iseven(n_electrons) ||
        throw(ArgumentError("Raman response requires a restricted closed-shell reference or a UHF result."))
    n_occ = hasproperty(raw, :n_occ) ? raw.n_occ : div(n_electrons, 2)
    dipole_matrices = _mo_dipole_matrices(mol, raw, nmo)
    alpha = zeros(Float64, 3, 3)
    _spin_polarizability!(alpha, energies, coefficients, n_occ, dipole_matrices)
    _spin_polarizability!(alpha, energies, coefficients, n_occ, dipole_matrices)
    return alpha
end

"""
    calc_uv_vis(mol, method::Symbol; kwargs...)::Vector{UVVisTransition}

Run an electronic excited-state calculation and return UV-Vis singlet transitions.
Supported methods are `:cis`/`:cndo2` (CNDO/2 CIS), `:tdhf`/`:hf`/`:rhf`
(RHF TDA-TDHF), and `:tddft`/`:tda`/`:dft` or a supported DFT functional
(restricted DFT TDA). Other semi-empirical excited-state response models are not
implemented and are rejected rather than approximated silently.
"""
function calc_uv_vis(mol::Molecule, method::Symbol;
                     functional::Symbol=:b3lyp,
                     n_states::Int=6,
                     basis::String="sto-3g",
                     basis_set::Union{Nothing, BasisSet}=nothing,
                     ecp_name::Union{Nothing, String}=nothing,
                     device::AbstractDevice=get_default_device(),
                     options::SCFOptions=SCFOptions(),
                     calc_triplets::Union{Nothing, Bool}=false)
    n_states > 0 || throw(ArgumentError("n_states must be positive."))
    m = Symbol(lowercase(string(method)))
    response_method = if m in (:cis, :cndo2)
        :cis
    elseif m in (:hf, :rhf, :tdhf)
        :tdhf
    elseif m in (:tddft, :tda)
        :tddft
    elseif m in (:dft, :b3lyp, :pbe, :pbe0, :lda, :blyp)
        functional = m == :dft ? functional : m
        :tddft
    else
        throw(ArgumentError("UV-Vis supports CNDO/2 CIS, RHF TDA-TDHF, and restricted DFT TDA; method '$m' has no implemented excited-state response."))
    end
    result = calculate(mol, response_method; functional=functional,
                       n_states=n_states, basis=basis, basis_set=basis_set,
                       ecp_name=ecp_name, device=device, options=options,
                       calc_triplets=calc_triplets)
    return uv_vis_transitions(result.raw_result; max_states=n_states)
end

"""
    calc_vibrational_frequencies(mol, method::Symbol; ...)

Calculate QM harmonic vibrational modes by finite differences of single-point
energies, with approximate IR intensities from finite-difference Mulliken-charge
dipoles. `delta` and `ir_delta` are in Angstrom. This is computationally costly.
"""
function calc_vibrational_frequencies(mol::Molecule, method::Symbol;
                                     basis::String="sto-3g",
                                     basis_set::Union{Nothing, BasisSet}=nothing,
                                     ecp_name::Union{Nothing, String}=nothing,
                                     functional::Symbol=:b3lyp,
                                     solvent::Symbol=:vacuum,
                                     device::AbstractDevice=get_default_device(),
                                     options::SCFOptions=SCFOptions(),
                                     delta::Float64=0.005,
                                     ir_delta::Float64=0.005,
                                     project_trivials::Bool=true)
    energy_function, dipole_function, _ =
        _vibrational_qm_callbacks(mol, method; basis=basis, basis_set=basis_set,
                                  ecp_name=ecp_name, functional=functional,
                                  solvent=solvent, device=device,
                                  options=options)
    return calc_vibrational_frequencies(mol, energy_function, dipole_function;
                                        delta=delta, ir_delta=ir_delta,
                                        project_trivials=project_trivials)
end

"""
    calc_raman_modes(mol, method::Symbol; ...)

Calculate QM harmonic modes and estimate relative Raman activities using the
finite-difference independent-particle sum-over-states polarizability. The
transition dipole operator is approximated at AO midpoints; this is not a
self-consistent field-response calculation.
"""
function calc_raman_modes(mol::Molecule, method::Symbol;
                          basis::String="sto-3g",
                          basis_set::Union{Nothing, BasisSet}=nothing,
                          ecp_name::Union{Nothing, String}=nothing,
                          functional::Symbol=:b3lyp,
                          solvent::Symbol=:vacuum,
                          device::AbstractDevice=get_default_device(),
                          options::SCFOptions=SCFOptions(),
                          delta::Float64=0.005,
                          ir_delta::Float64=0.005,
                          project_trivials::Bool=true,
                          raman_delta::Float64=1e-3)
    _, _, polarizability_function =
        _vibrational_qm_callbacks(
        mol, method; basis=basis, basis_set=basis_set, ecp_name=ecp_name,
        functional=functional, solvent=solvent, device=device, options=options)
    modes = calc_vibrational_frequencies(
        mol, method; basis=basis, basis_set=basis_set, ecp_name=ecp_name,
        functional=functional, solvent=solvent, device=device, options=options,
        delta=delta, ir_delta=ir_delta, project_trivials=project_trivials)
    return calc_raman_modes(mol, modes, polarizability_function;
                            raman_delta=raman_delta)
end

Base.show(io::IO, ::MIME"text/plain", result::CalculationResult) =
    print(io, format_output(result))

end # module UnifiedCalculation
