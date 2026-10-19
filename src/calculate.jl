# ==============================================================================
# QuantumChemMM.jl: Unified High-Level Calculation Entry Point
# Dispatches across Molecular Mechanics, Semi-Empirical, Ab Initio, DFT, and Post-HF
# Supports Multi-Vendor GPU Acceleration on NVIDIA, AMD, and Intel GPUs
# Implicit solvation (COSMO/GB) available for ALL methods via solvent= keyword
# ==============================================================================

module UnifiedCalculation

using ..Types
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
using ..BasisSets
using ..HartreeFock
using ..KohnShamDFT
using ..MP2Theory
using ..CISDTheory
using ..CoupledCluster
using ..CISExcitedStates
using ..NMRSpectroscopy
using ..SolvationModels
using ..GPUInterface
using ..GPUSolvers
using ..ResultOutput: format_output, write_output

export calculate, CalculationResult

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
        return [Float64(mol.atoms[atom].atomic_number) -
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
  - Ab Initio:           :hf, :rhf
  - DFT:                 :dft, :b3lyp, :pbe, :pbe0, :lda, :blyp
  - Post-Ab Initio:      :mp2, :scs_mp2, :cisd, :ccsd, :ccsdt
  - Excited States:      :cis
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
"""
function _calculate(mol::Molecule, method::Symbol;
                   basis::String="sto-3g",
                   functional::Symbol=:b3lyp,
                   solvent::Symbol=:vacuum,
                   device::AbstractDevice=get_default_device(),
                   options::SCFOptions=SCFOptions(),
                   kwargs...)::CalculationResult

    m = Symbol(lowercase(string(method)))

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
        res    = solve_hf(mol; basis_name=basis, device=device, options=options)
        e_ha   = res.total_energy_hartree
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(:hf, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 4. Density Functional Theory ────────────────────────────────────────
    elseif m in (:dft, :b3lyp, :pbe, :pbe0, :lda, :blyp)
        func   = (m == :dft) ? functional : m
        res    = solve_dft(mol; functional=func, basis_name=basis, device=device, options=options)
        e_ha   = res.total_energy_hartree
        e_ev   = res.total_energy_ev
        charges = _charges(mol, res)
        return _make_result(Symbol("dft_", func), e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, res)

    # ── 5a. MP2 / SCS-MP2 ───────────────────────────────────────────────────
    elseif m in (:mp2, :scs_mp2)
        hf_res  = solve_hf(mol; basis_name=basis, device=device, options=options)
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
        hf_res   = solve_hf(mol; basis_name=basis, device=device, options=options)
        cisd_res = solve_cisd(hf_res; device=device)
        e_ha     = cisd_res.total_cisd_energy_hartree
        e_ev     = cisd_res.total_cisd_energy_ev
        charges  = _charges(mol, hf_res)
        return _make_result(:cisd, e_ev, e_ha, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, cisd_res;
                            orbital_energies_ev=hf_res.mo_energies_hartree .* HARTREE_TO_EV)

    # ── 5c. CCSD / CCSD(T) ──────────────────────────────────────────────────
    elseif m in (:ccsd, :ccsdt, :ccsd_t)
        hf_res = solve_hf(mol; basis_name=basis, device=device, options=options)
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
        cis_res = solve_cis(scf_res, mol; device=device)
        e_ev    = cis_res.ground_energy_ev
        charges = _charges(mol, scf_res)
        return _make_result(:cis, e_ev, e_ev / HARTREE_TO_EV, e_ev * EV_TO_KCAL,
                            mol, solvent, charges, cis_res;
                            orbital_energies_ev=_orbital_energies_ev(scf_res))

    # ── 7. NMR Chemical Shifts & Shieldings ─────────────────────────────────
    elseif m in (:nmr, :nmr_shielding)
        hf_res  = solve_hf(mol; basis_name=basis, device=device, options=options)
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
              ":mp2, :scs_mp2, :cisd, :ccsd, :ccsdt, :cis, :nmr"))
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

Base.show(io::IO, ::MIME"text/plain", result::CalculationResult) =
    print(io, format_output(result))

end # module UnifiedCalculation
