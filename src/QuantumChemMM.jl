# ==============================================================================
# QuantumChemMM.jl: Unified Quantum Chemistry and Molecular Mechanics
# Semi-Empirical, Molecular Mechanics, Ab Initio, DFT, and Post-Ab Initio Methods
# Hardware acceleration supported on NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI) GPUs
# ==============================================================================

module QuantumChemMM

# 1. Core Molecular Representation & Formats
include("core/elements.jl")
include("core/types.jl")
include("core/math.jl")
include("core/topology.jl")
include("core/io.jl")

using .Elements
using .Types
using .MolMath
using .Topology
using .ChemIO

# 2. Multi-Vendor GPU Device Abstraction Layer & Low-Level Kernels
include("gpu/gpu_interface.jl")
include("gpu/gpu_kernels.jl")

using .GPUInterface
using .GPUKernels

# 3. Molecular Mechanics Engine
include("molecular_mechanics/base.jl")
include("molecular_mechanics/bonded_potentials.jl")
include("molecular_mechanics/dihedrals.jl")
include("molecular_mechanics/nonbonded.jl")
include("molecular_mechanics/forcefields/uff.jl")
include("molecular_mechanics/forcefields/amber.jl")
include("molecular_mechanics/forcefields/opls.jl")
include("molecular_mechanics/forcefields/charmm.jl")
include("molecular_mechanics/forcefields/mm2_mm3.jl")
include("molecular_mechanics/energy_gradient.jl")

using .MMBase
using .BondedPotentials
using .Dihedrals
using .NonBonded
using .UFFForceField
using .AMBERForceField
using .OPLSForceField
using .CHARMMForceField
using .MM3ForceField
using .MMEnergyGradient

# 4. Semi-Empirical Quantum Chemistry Engine
include("semi_empirical/base.jl")
include("semi_empirical/slater_orbitals.jl")
include("semi_empirical/dispersion.jl")
include("semi_empirical/huckel.jl")
include("semi_empirical/extended_huckel.jl")
include("semi_empirical/zdo_cndo_indo.jl")
include("semi_empirical/nddo/parameters.jl")
include("semi_empirical/nddo/base.jl")
include("semi_empirical/nddo/scf.jl")
include("semi_empirical/dftb_xtb.jl")
include("periodic/tight_binding.jl")

using .SemiEmpiricalBase
using .SlaterOrbitals
using .DispersionD3
using .HuckelTheory
using .ExtendedHuckel
using .ZDOSemiEmpirical
using .NDDOParameters
using .NDDOBase
using .NDDOSCF
using .TightBinding
using .PeriodicTightBinding

# 5. Ab Initio Quantum Chemistry Engine (Hartree-Fock)
include("ab_initio/basis.jl")
include("ab_initio/integrals.jl")
include("ab_initio/hartree_fock.jl")

using .BasisSets
using .MolecularIntegrals
using .HartreeFock

# 6. Density Functional Theory (DFT) Engine
include("dft/functionals.jl")
include("dft/grid.jl")
include("dft/kohn_sham.jl")

using .DFTFunctionals
using .DFTGrid
using .KohnShamDFT

# 7. Post-Ab Initio Correlation Methods (MP2, CISD, CCSD, CCSD(T))
include("post_hf/integral_transform.jl")
include("post_hf/mp2.jl")
include("post_hf/cisd.jl")
include("post_hf/ccsd.jl")

using .IntegralTransform
using .MP2Theory
using .CISDTheory
using .CoupledCluster

# 8. Excited State Calculations (CIS Singlets & Triplets, Gradients)
include("excited_states/cis.jl")
include("excited_states/excited_gradients.jl")
include("excited_states/tda.jl")

using .CISExcitedStates
using .ExcitedGradients
using .TimeDependentMethods

# 9. Transition State Calculations (NEB, CI-NEB, P-RFO, IRC)
include("transition_states/neb.jl")
include("transition_states/eigenvector_following.jl")
include("transition_states/irc.jl")

using .NEBMethod
using .EigenvectorFollowing
using .IRCMethod

# 10. Solvation Models (COSMO, PCM, Generalized Born)
include("solvation/cosmo.jl")

using .SolvationModels

# 11. Hybrid QM/MM Engine
include("qmmm/embedding.jl")
include("qmmm/hybrid_energy.jl")

using .QMMMEmbedding
using .QMMMEnergy

# 12. Geometry Optimization & Molecular Dynamics
include("optimization/line_search.jl")
include("optimization/algorithms.jl")
include("optimization/molecular_dynamics.jl")

using .LineSearch
using .OptimizationAlgorithms
using .MolecularDynamics

# 13. Spectroscopic, Reactivity & Conformational Analysis
include("analysis/populations.jl")
include("analysis/dipole.jl")
include("analysis/spectra.jl")
include("analysis/thermochemistry.jl")
include("analysis/reactivity.jl")
include("analysis/conformers.jl")
include("analysis/nmr.jl")

using .PopulationAnalysis
using .DipoleAnalysis
using .Spectroscopy
using .Thermochemistry
using .ReactivityDescriptors
using .ConformationalSearch
using .NMRSpectroscopy

# 14. Human-readable text output for calculation results
include("core/output.jl")
using .ResultOutput

# 15. Multi-Vendor GPU High-Level Solvers (NVIDIA, AMD, Intel)
# NOTE: Loaded after Analysis so GPUSolvers can import NMRSpectroscopy
include("gpu/gpu_solvers.jl")

using .GPUSolvers

# 16. Unified High-Level Dispatcher
include("calculate.jl")
using .UnifiedCalculation

# ==============================================================================
# Public API Exports
# ==============================================================================

# Unified Entry Point
export calculate, CalculationResult
export format_output, write_output

# Core Types & IO
export Atom, Bond, Angle, Dihedral, Improper, Molecule,
       n_atoms, n_bonds, coordinates_matrix, set_coordinates!,
       center_of_mass, translate!, rotate!,
       read_xyz, write_xyz, parse_xyz, read_pdb, write_pdb,
       distance, angle_between, dihedral_angle, out_of_plane_angle,
       perceive_topology!, build_bonds!

# Molecular Mechanics
export AbstractForceField, EnergyComponents, total_energy, print_summary,
       evaluate_mm, energy, gradient,
       UFF, parameterize_uff,
       AMBER, parameterize_amber,
       OPLS, parameterize_opls,
       CHARMM, parameterize_charmm,
       MM3, parameterize_mm3

# Semi-Empirical Quantum Chemistry
export AbstractSemiEmpiricalMethod, SCFOptions, SCFResult,
       HuckelModel, solve_huckel, HuckelResult,
       ExtendedHuckelModel, solve_extended_huckel, EHTResult,
       CNDO2Model, solve_cndo2, INDOModel, solve_indo,
       NDDOMethod, solve_nddo, NDDOResult,
       TightBindingModel, solve_tight_binding, DFTBResult,
       D3Correction, calc_d3_dispersion

# Periodic solid-state tight-binding prototype
export CrystalStructure, PeriodicTBResult, BandStructureResult,
       solve_periodic_tb, periodic_band_structure

# Ab Initio Hartree-Fock
export BasisSet, PrimitiveGaussian, ContractedGaussian,
       EffectiveCorePotential, ECPComponent,
       build_basis, validate_basis, n_basis, available_basis_sets,
       build_ecp, ecp_potential_value, ecp_core_electrons,
       effective_nuclear_charges, apply_ecp_matrix,
       compute_overlap, compute_kinetic, compute_nuclear_attraction,
       compute_core_hamiltonian, compute_eri_tensor,
       HFResult, solve_hf, solve_uhf, print_hf_summary

# Density Functional Theory (DFT)
export XCFunctional, evaluate_xc,
       LDA_FUNCTIONAL, PBE_FUNCTIONAL, BLYP_FUNCTIONAL, B3LYP_FUNCTIONAL, PBE0_FUNCTIONAL,
       MolecularGrid, build_molecular_grid,
       DFTResult, solve_dft, print_dft_summary

# Post-Ab Initio Correlation Methods
export transform_eri_ao_to_mo, antisymmetrize_mo_eri,
       MP2Result, solve_mp2, print_mp2_summary,
       CISDResult, solve_cisd, print_cisd_summary,
       CCSDResult, solve_ccsd, print_ccsd_summary

# Hybrid QM/MM
export QMMMSystem, create_qmmm_system, evaluate_qmmm, QMMMResult, print_qmmm_summary

# Optimization & Dynamics
export optimize_geometry!, OptimizationResult,
       run_md, MDSimulation, TrajectoryFrame

# Multi-Vendor GPU Acceleration
export AbstractDevice, CPUDevice, CUDADevice, ROCmDevice, OneAPIDevice,
       device_name, is_gpu, vendor, detect_available_devices,
       get_default_device, set_default_device!,
       solve_scf_gpu, evaluate_mm_gpu, run_md_gpu,
       solve_hf_gpu, solve_uhf_gpu, solve_dft_gpu, solve_mp2_gpu,
       solve_cisd_gpu, solve_ccsd_gpu, solve_cis_gpu, calc_nmr_gpu,
       gpu_distance_matrix, gpu_fock_coulomb_exchange,
       gpu_dft_density_and_xc, gpu_transform_eri_ao_to_mo,
       gpu_antisymmetrize_mo_eri, gpu_mp2_correlation, gpu_cis_hamiltonian,
       gpu_nmr_paramagnetic_shielding, gpu_nmr_diamagnetic_shielding,
       gpu_simulate_nmr_spectrum

# Excited States
export ExcitedState, CISResult, solve_cis, compute_excited_state_dipole,
       calc_excited_state_energy_and_gradient, optimize_excited_state!
export TDExcitedState, TDResult, solve_tdhf, solve_tddft, xc_density_kernel

# Transition States
export NEBPath, run_neb, NEBResult,
       optimize_transition_state!, TSResult,
       trace_irc, IRCResult, IRCPoint

# Solvation Models
export Solvent, SolvationResult, calc_solvation_energy,
       SOLVENTS, get_solvent, print_solvation_summary

# Analysis, Spectroscopy, Reactivity & Conformers
export mulliken_analysis, lowdin_analysis, PopulationResult,
       calc_molecular_dipole, DipoleResult,
       calc_uv_vis, uv_vis_transitions,
       calc_uv_vis_cis, generate_uv_vis_spectrum,
       calc_vibrational_frequencies, generate_ir_spectrum,
       RamanMode, calc_raman_modes, generate_raman_spectrum,
       UVVisTransition, VibrationalMode, project_translations_and_rotations!,
       calc_thermochemistry, ThermochemistryResult, print_thermo_summary,
       ReactivityIndices, FukuiIndices, calc_reactivity_indices,
       calc_condensed_fukui, print_reactivity_summary,
       Conformer, ConformerEnsemble, search_conformers,
       boltzmann_populations, print_conformer_summary

# Nuclear Magnetic Resonance (NMR) Spectroscopy
export NMRShieldingTensor, NMRNucleus, NMRResult,
       calc_nmr_shieldings, simulate_nmr_spectrum, print_nmr_summary,
       DEFAULT_NMR_REFERENCES

end # module QuantumChemMM
