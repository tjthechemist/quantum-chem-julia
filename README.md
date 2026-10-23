# QuantumChemMM.jl

**QuantumChemMM.jl** is an all-in-one, modular, high-performance computational chemistry package written in pure Julia. It unifies **all major semi-empirical quantum chemical methods**, **molecular mechanics force fields**, **ab initio Hartree-Fock**, **density functional theory (DFT)**, and **post-Hartree-Fock correlation methods** under a consistent, type-stable mathematical and algorithmic architecture, with native **hardware acceleration across NVIDIA, AMD, and Intel GPUs**, plus comprehensive **excited state**, **transition state**, **NMR**, **UV-Vis/IR spectroscopic**, **solvation**, and **reactivity** calculation engines.

---

## 🌟 Key Features

### 🚀 Multi-Vendor GPU Hardware Acceleration (`src/gpu/`, `ext/`)
Hardware acceleration is fully supported across all major vendors via Julia package extensions:
- 🟢 **NVIDIA GPUs** via `CUDA.jl` (`CUDADevice`)
- 🔴 **AMD GPUs** via `AMDGPU.jl` (`ROCmDevice`)
- 🔵 **Intel GPUs** via `oneAPI.jl` (`OneAPIDevice`)
- ⚪ **Host CPU** multithreaded SIMD fallback (`CPUDevice`)

**GPU Accelerated Bottlenecks**:
1. **NMR Spectroscopy**: Parallel paramagnetic tensor contractions across all nuclei, interatomic diamagnetic screening, and Lorentzian spectrum evaluation across device grids.
2. **Ab Initio Hartree-Fock**: Two-electron Fock build ($J$ and $K$ matrices) formulated as high-throughput GPU GEMV matrix-vector products.
3. **Kohn-Sham DFT**: Quadrature grid basis evaluations $\chi_\mu(\mathbf{r}_g)$, electron density $\rho(\mathbf{r}_g)$, and exchange-correlation potential matrix $V_{xc}$ computed via GPU matrix multiplications.
4. **Post-HF Methods (MP2, CISD, CCSD, CCSD(T))**: Four-index AO-to-MO integral transformation $( \mu\nu|\lambda\sigma ) \to ( pq|rs )$ computed using four consecutive GPU GEMMs, plus device tensor contractions for denominators and amplitude residuals.
5. **Excited States (CIS)**: Batched transition-charge contractions and CIS Hamiltonian matrix setup on device arrays. RHF/DFT TDA response is currently assembled on the CPU.
6. **Molecular Mechanics & MD**: $O(N^2)$ pairwise Lennard-Jones, Coulomb electrostatics, and Velocity Verlet integration on GPU.

### 🧵 Multicore CPU Execution
CPU calculations use Julia threads for the independent Gaussian integral, HF/DFT matrix, and DFT integration-grid work. Start Julia with multiple threads to enable multicore execution; `auto` selects the available CPU cores:

```sh
julia --threads auto --project=.
```

Run a calculation as usual (for example, `calculate(mol, :b3lyp; device=CPUDevice())`). To limit CPU usage, replace `auto` with a thread count such as `--threads 4`. Julia's default is one thread, so the `--threads` option must be set when launching Julia.

---

### 🧲 Nuclear Magnetic Resonance (NMR) Spectroscopy (`src/analysis/nmr.jl`)
- **Chemical Shielding Tensors ($\boldsymbol{\sigma}$)**:
  - Full $3 \times 3$ shielding tensor $\boldsymbol{\sigma}_K = \boldsymbol{\sigma}_K^d + \boldsymbol{\sigma}_K^p$ for each active nucleus ($^1\text{H}$, $^{13}\text{C}$, $^{15}\text{N}$, $^{19}\text{F}$, $^{31}\text{P}$, $^{17}\text{O}$).
  - Principal components $\sigma_{11} \le \sigma_{22} \le \sigma_{33}$, isotropic shielding $\sigma_{iso}$, chemical shift anisotropy $\Delta \sigma$, and asymmetry parameter $\eta$.
- **Chemical Shifts ($\delta$ in ppm)**:
  - Referenced against IUPAC primary standards: TMS ($^1\text{H}$, $^{13}\text{C}$), nitromethane ($^{15}\text{N}$), $\text{CFCl}_3$ ($^{19}\text{F}$), and $\text{H}_3\text{PO}_4$ ($^{31}\text{P}$).
- **Indirect Spin-Spin $J$-Coupling (Hz)**:
  - One-bond couplings ($^1J_{CH}$, $^1J_{CC}$), geminal couplings ($^2J_{HH}$), and vicinal couplings ($^3J_{HH}$) with conformational Karplus dihedral dependence.
- **Continuous 1D NMR Spectrum Simulation**:
  - Simulates continuous $^1\text{H}$-NMR and $^{13}\text{C}$-NMR spectra on device arrays with Lorentzian line broadening.

---

### 🔬 Ab Initio & Density Functional Theory (`src/ab_initio/`, `src/dft/`)
- **Restricted Hartree-Fock (RHF)**: Exact Gaussian integrals (overlap, kinetic, electron-nuclear attraction, and 4D two-electron repulsion) with DIIS acceleration.
- **Unrestricted Hartree-Fock (UHF)**: Separate alpha/beta orbitals, densities, and Fock matrices. Molecular total charge and multiplicity determine occupations; closed-shell RHF supports singlet ions, while open-shell calculations use `:uhf`. For example, set `radical.total_charge = 1` and `radical.multiplicity = 2`, then call `calculate(radical, :uhf; basis="sto-3g", device=CPUDevice())`. Both CPU and supported GPU solver paths use the same spin-resolved SCF equations. DFT remains restricted closed-shell; it reports an error for open-shell inputs rather than silently miscounting electrons.
- **Kohn-Sham DFT**:
  - **LDA**: Slater exchange + VWN5 correlation.
  - **GGA**: PBE (Perdew-Burke-Ernzerhof) and BLYP (Becke-88 + Lee-Yang-Parr).
  - **Hybrid DFT**: B3LYP (20% exact HF exchange) and PBE0 (25% exact HF exchange).
  - Molecular numerical quadrature grid using Gauss-Legendre / Euler-Maclaurin radial and Lebedev angular grids with Becke fuzzy Voronoi cell partitioning.
- **ECP SCF integration (prototype)**: HF and DFT accept `ecp_name="qcm-prototype"` and include the prototype effective charges, nuclear repulsion, and one-electron correction in the shared CPU/GPU SCF path. You may pass a caller-supplied `BasisSet` with `basis_set=...` when the built-in basis tables do not cover the ECP atoms:

  ```julia
  result = calculate(mol, :hf; basis_set=my_basis,
                     ecp_name="qcm-prototype", device=CPUDevice())
  ```

  This is not a published def2 ECP implementation: the bundled coefficients and matrix model are provisional and are not suitable for quantitative production chemistry. GPU SCF transfers the assembled ECP matrix to the selected device; matrix construction itself currently runs on the CPU.

---

### ⚛️ Post-Hartree-Fock Correlation Methods (`src/post_hf/`)
- **Fast 4-Index Integral Transformation**: $O(N^5)$ quarter-transformations optimized for GPU tensor cores.
- **MP2 & Spin-Component-Scaled (SCS-MP2)**: Second-order Møller-Plesset perturbation theory with opposite-spin and same-spin decompositions.
- **CISD**: Full Configuration Interaction Singles and Doubles with variational correlation energy.
- **Coupled Cluster (CCSD & CCSD(T))**: The "Gold Standard" of quantum chemistry with non-linear exponential cluster operator $e^{\hat{T}_1 + \hat{T}_2}$ and perturbative non-iterative triples $(T)$.

---

### 🌈 Excited State Calculations (`src/excited_states/`)
- **Configuration Interaction Singles (CIS)**:
  - Computes both **Singlet** ($S_1, S_2, \dots$) and **Triplet** ($T_1, T_2, \dots$) excited states from converged SCF determinants.
  - Transition dipole moments $\boldsymbol{\mu}_{0 \to n}$ and oscillator strengths $f_{osc}$.
  - State electric dipole moments and leading orbital excitation percentages.
- **Tamm-Dancoff response (`:tdhf`, `:tddft` / `:tda`)**:
  - `:tdhf` computes closed-shell RHF-based singlet and triplet TDA states; set `calc_triplets=false` to omit triplets.
  - `:tddft` and `:tda` compute restricted DFT singlet TDA states for LDA, PBE, PBE0, BLYP, and B3LYP. For example:

    ```julia
    result = calculate(mol, :tddft; functional=:b3lyp, n_states=6)
    result.raw_result.singlets[1].excitation_energy_ev
    result.raw_result.singlets[1].oscillator_strength
    ```

  - The current DFT response uses a numerical derivative of the package's local XC potential. GGA gradient response terms are not implemented, so this is an approximate local-kernel TDA, not production-grade full TDDFT. DFT triplet response, open-shell TDHF/TDDFT, and analytic excited-state gradients are not available.
  - Existing `:cis` remains the semi-empirical CNDO/2 CIS path. MM has classical molecular dynamics, not electronic TD response; MP2/CI/CC excited-state response and CIS for other semi-empirical models are not implemented.
  - The reference SCF honors `device`; the new RHF/DFT TDA matrix construction currently runs on CPU.
- **Excited State PES Gradients & Optimization**:
  - Analytical and finite-difference gradients on excited state potential energy surfaces.
  - Excited state geometry optimization to locate fluorescence emission minima and calculate Stokes shifts.

---

### ⚡ Transition State (TS) & Reaction Path Engine (`src/transition_states/`)
- **Nudged Elastic Band (NEB) & Climbing-Image NEB (CI-NEB)**:
  - Connects reactant and product with elastic replica bands.
  - Henkelman-Jónsson improved tangent vectors and perpendicular force projections.
  - **Climbing Image (CI)** modification drives the peak image directly up to the exact first-order saddle point.
  - Computes activation barriers ($E_a$) and reaction energies ($\Delta E_{rxn}$).
- **Eigenvector-Following / P-RFO TS Optimization**:
  - Local saddle-point optimizer that follows the negative Hessian mode uphill while minimizing orthogonal modes downhill.
- **Intrinsic Reaction Coordinate (IRC)**:
  - Downhill minimum energy path (MEP) integration from the transition state into both the reactant and product basins.

---

### 📊 Advanced Spectroscopy (`src/analysis/spectra.jl`)
- **UV-Vis Electronic Absorption Spectroscopy**:
  - `calc_uv_vis(mol, :cndo2)` uses the existing CNDO/2 CIS model; `:hf`/`:tdhf` uses RHF TDA-TDHF; `:b3lyp`, `:pbe`, `:pbe0`, `:lda`, and `:blyp` use the restricted DFT TDA response.
  - `generate_uv_vis_spectrum(transitions)` generates a Gaussian-broadened spectrum from computed singlet oscillator strengths.
  - Other semi-empirical methods do not yet have a dedicated excited-state response model and are rejected rather than routed through an unrelated approximation.

  ```julia
  transitions = calc_uv_vis(mol, :b3lyp; basis="sto-3g", n_states=6)
  wavelengths_nm, relative_absorbance =
      generate_uv_vis_spectrum(transitions; lambda_min=150.0, lambda_max=400.0)
  ```
- **Vibrational / Infrared (IR) Spectroscopy**:
  - MM force-field vibrations remain available through `calc_vibrational_frequencies(mol, ff)`.
  - Semi-empirical, HF/UHF, and restricted DFT vibrations are available through `calc_vibrational_frequencies(mol, :method; ...)`. The QM Hessian uses finite differences of total energies; IR intensities use finite differences of dipoles assembled from Mulliken charges.
  - **Eckart Frame Projection**: Eliminates trivial translational and rotational degrees of freedom.
  - Simulates continuous IR transmittance/absorbance spectra with Lorentzian peak broadening.
- **Raman Spectroscopy**:
  - `calc_raman_modes(mol, ff)` uses MM normal modes; `calc_raman_modes(mol, :method; ...)` uses semi-empirical/HF/DFT normal modes.
  - `generate_raman_spectrum(modes; laser_wavelength_nm=532.0)` produces a broadened Stokes spectrum with thermal population and scattered-frequency weighting.
  - QM-method Raman activities use a finite-difference derivative of an approximate independent-particle sum-over-states polarizability. Transition dipoles use an AO midpoint approximation; DFT XC response kernels and self-consistent finite-field response are not included, so these are exploratory estimates rather than production-quality Raman predictions. MM-method Raman activities continue to use the empirical bond-polarizability model.

  ```julia
  ir_modes = calc_vibrational_frequencies(mol, :b3lyp; basis="sto-3g")
  wavenumbers_cm1, ir_intensity = generate_ir_spectrum(ir_modes)

  # Raman modes from QM-level Hessian and approximate electronic response
  raman_modes = calc_raman_modes(mol, :b3lyp; basis="sto-3g")
  shifts_cm1, raman_intensity = generate_raman_spectrum(
      raman_modes; laser_wavelength_nm=532.0, temperature_k=298.15)

  # MM-only variant
  modes = calc_raman_modes(mol, parameterize_uff(mol))
  shifts_cm1, intensity = generate_raman_spectrum(
      modes; laser_wavelength_nm=532.0, temperature_k=298.15)
  ```
- QM harmonic Hessians require roughly `1 + 2(3N)^2` single-point energies for `N` atoms. IR and Raman intensities require additional displaced single points and can be expensive. `delta`, `ir_delta`, and `raman_delta` are in Angstrom.
- **RRHO Thermochemistry**:
  - Zero-Point Vibrational Energy (ZPVE), thermal internal energy $U(T)$, enthalpy $H(T)$, entropy $S(T)$, and Gibbs free energy $G(T)$.

---

### 💧 Implicit Solvation for **All Methods** (`src/solvation/`, `src/calculate.jl`)
Every calculation method supports a `solvent=` keyword for implicit COSMO/Generalized-Born solvation:

```julia
# Works identically for :uff, :am1, :pm3, :hf, :b3lyp, :mp2, :cisd, :ccsd, :nmr, …
result = calculate(mol, :b3lyp; basis="def2-svp", solvent=:water)

result.total_energy_hartree        # gas-phase energy
result.total_energy_solv_hartree   # E_gas + ΔG_solv
result.solvation                   # full SolvationResult (nothing when solvent=:vacuum)
```

**Pre-parameterized solvents** (pass as `solvent=:symbol`):

| Symbol | Solvent | ε |
|---|---|---|
| `:water` | Water | 78.39 |
| `:dmso` | Dimethylsulfoxide | 46.70 |
| `:acetonitrile` | Acetonitrile | 35.69 |
| `:methanol` | Methanol | 32.70 |
| `:ethanol` | Ethanol | 24.55 |
| `:dichloromethane` | DCM | 8.93 |
| `:octanol` | 1-Octanol | 10.30 |
| `:chloroform` | Chloroform | 4.81 |
| `:toluene` | Toluene | 2.38 |
| `:hexane` | Hexane | 1.88 |
| `:vacuum` | Vacuum (default) | 1.00 |

**Solvation model details:**
- **SASA (Solvent Accessible Surface Area)** via Fibonacci sphere tessellation (Shrake-Rupley).
- **Non-polar cavitation** $\Delta G_{cav} = \gamma \cdot \text{SASA}$.
- **Electrostatic** via Generalized Born / COSMO: effective Born radii derived from SASA, pairwise GB Still formula.
- Partial charges taken automatically from the SCF density (Mulliken), or from MM force field atom charges.

---

### 🎯 Chemical Reactivity & Conceptual DFT (`src/analysis/reactivity.jl`)
- **Parr-Pearson Global Reactivity Descriptors**:
  - Chemical Hardness $\eta$, Softness $S$, Chemical Potential $\mu$, Electronegativity $\chi$, and Electrophilicity Index $\omega$.
- **Condensed Fukui Functions**:
  - $f_k^+$: Site susceptibility to nucleophilic attack.
  - $f_k^-$: Site susceptibility to electrophilic attack.
  - $f_k^0$: Site susceptibility to radical attack.
  - **Dual Descriptor ($\Delta f_k = f_k^+ - f_k^-$)**: Distinguishes electrophilic ($> 0$) vs nucleophilic ($< 0$) centers.

---

### 🔄 Automated Conformational Search (`src/analysis/conformers.jl`)
- Automatic perception of flexible rotatable bonds.
- Systematic and stochastic torsional sampling with L-BFGS force field energy minimization.
- Duplicate clustering via RMS coordinate deviation.
- **Boltzmann Population Distribution** at temperature $T$.

---

### ⚛️ Semi-Empirical Quantum Chemistry (`src/semi_empirical/`)
- **Simple Hückel Theory (HMO)**: $\pi$-conjugation, aromaticity, Coulson bond orders.
- **Extended Hückel Theory (EHT)**: Full valence basis with analytical STO-3G overlap matrix.
- **ZDO Family**: **CNDO/2**, **INDO**, and **CNDO/S** with DIIS acceleration.
- **NDDO Family**: **MNDO**, **AM1**, **PM3**, **RM1**, and **PM6** with heats of formation.
- **Tight-Binding**: **GFN0-xTB**, **GFN1-xTB / SCC-DFTB**, and **GFN2-xTB**.
- **Grimme D3 / D3(BJ)**: Analytical dispersion corrections.

### 🌐 Periodic Solid-State Prototype (`src/periodic/tight_binding.jl`)
- A simple periodic Bloch tight-binding model for small 3D crystals and 2D lattices, with one orbital per atom and user-defined onsite/hopping parameters.
- Computes sampled band energies, Gaussian-broadened density of states, a spin-degenerate filling-based Fermi level and band gap, and band structures along fractional reciprocal-space paths.
- This is an exploratory model, not periodic HF/DFT, self-consistent DFTB/xTB, or a validated parameter set for general materials. Total energies are model band sums per cell and do not include an ion-ion repulsion term. Geometry optimization, phonons, forces, and self-consistent charge transfer are not included.

  ```julia
  crystal = CrystalStructure(
      [2.46 1.23 0.0; 0.0 2.13 0.0; 0.0 0.0 20.0],
      [0.0 1/3; 0.0 1/3; 0.0 0.0],
      [:C, :C]; name="Graphene", periodic_dimensions=[1, 2])

  result = solve_periodic_tb(crystal; electrons_per_cell=2,
                             kmesh=(12, 12, 1), hopping_ev=-2.7,
                             cutoff_angstrom=1.6)
  bands = periodic_band_structure(
      crystal, [[0.0, 0.0, 0.0], [1/3, 1/3, 0.0], [0.5, 0.0, 0.0]];
      labels=["Γ", "K", "M"], hopping_ev=-2.7,
      cutoff_angstrom=1.6)
  ```

---

### 🔬 Molecular Mechanics Force Fields (`src/molecular_mechanics/`)
- **Class I**: **UFF** (Z=1..103), **AMBER/GAFF**, **OPLS-AA**, **CHARMM** (Urey-Bradley).
- **Class II**: **MM2 / MM3** (Allinger) with anharmonic stretches, sextic bends, and stretch-bend cross terms.
- **Hybrid QM/MM**: Mechanical and electrostatic embedding with link-atom capping.

### 📐 Geometry Optimization (`src/optimization/`)
- Steepest descent, conjugate gradient, and L-BFGS are available for MM force fields.
- Semi-empirical methods, RHF/UHF, and restricted DFT can be optimized through `optimize_geometry!(mol, method; ...)`, for example:

  ```julia
  result = optimize_geometry!(mol, :b3lyp; basis="sto-3g",
                              optimization_method=:lbfgs)
  ```

- Electronic-structure gradients currently use central finite differences of total energy (including the requested solvation correction). This requires approximately `6N` single-point calculations per gradient for `N` atoms and can be expensive. The step defaults to `0.001 Å`; `grad_tol` is in Hartree/Å. Pass `device` to select the device used by each single-point SCF calculation. Analytic HF/DFT gradients are not yet implemented.

---

## ⚡ Code Examples

### 1. Nuclear Magnetic Resonance (NMR) on GPUs
```julia
using QuantumChemMM

mol = parse_xyz("ethanol.xyz")

# Calculate NMR Chemical Shieldings on NVIDIA, AMD, or Intel GPU
nmr_res = calculate(mol, :nmr; device=CUDADevice())
print_nmr_summary(nmr_res.raw_result)

# Simulate continuous 1H-NMR spectrum on GPU
ppm_axis, spectrum = simulate_nmr_spectrum(nmr_res.raw_result; nucleus=:H, device=CUDADevice())
```

### 2. Unified Calculation with Multi-Vendor GPU Target
```julia
using QuantumChemMM

mol = parse_xyz("water.xyz")

# Run on NVIDIA GPU (CUDA)
res_cuda = calculate(mol, :hf; device=CUDADevice())

# Run on AMD GPU (ROCm)
res_rocm = calculate(mol, :b3lyp; device=ROCmDevice())

# Run on Intel GPU (oneAPI)
res_intel = calculate(mol, :mp2; device=OneAPIDevice())
```

### 3. Excited States (CIS) & UV-Vis Spectrum
```julia
using QuantumChemMM

mol = parse_xyz("formaldehyde.xyz")
scf_res = solve_cndo2(mol)

# Solve CIS excited states with GPU acceleration
cis_res = solve_cis(scf_res, mol; n_states=5, calc_triplets=true, device=CUDADevice())
singlet_transitions = uv_vis_transitions(cis_res)
wavelengths, absorbance = generate_uv_vis_spectrum(singlet_transitions; lambda_min=50.0, lambda_max=400.0)
```

### 4. In-Solvent Calculations for All Methods
```julia
using QuantumChemMM

mol = parse_xyz("formaldehyde.xyz")

# MM in water
res_mm   = calculate(mol, :uff;   solvent=:water)

# Semi-empirical in DMSO
res_am1  = calculate(mol, :am1;   solvent=:dmso)

# HF/STO-3G in multiple solvents — compare ΔG_solv
res_hf_aq  = calculate(mol, :hf;  solvent=:water,      basis="sto-3g")
res_hf_chl = calculate(mol, :hf;  solvent=:chloroform, basis="sto-3g")
res_hf_hex = calculate(mol, :hf;  solvent=:hexane,     basis="sto-3g")

# DFT/B3LYP in water
res_dft  = calculate(mol, :b3lyp; solvent=:water, basis="sto-3g")

# MP2 / CCSD in water
res_mp2  = calculate(mol, :mp2;   solvent=:water, basis="sto-3g")
res_ccsd = calculate(mol, :ccsd;  solvent=:water, basis="sto-3g")

# Access solvated total energy
println("CCSD/water total: ", res_ccsd.total_energy_solv_hartree, " Ha")
println("ΔG_solv: ", res_ccsd.solvation.total_delta_g_solv_kcal, " kcal/mol")

# Detailed solvation breakdown
print_solvation_summary(res_dft.solvation)
```

### Formatted reports and text output

Calculation results can be formatted and saved consistently for every method.
Pass `output_file` to `calculate` to save a plain-text report automatically; the
filename and extension are used as provided:

```julia
result = calculate(mol, :b3lyp; basis="sto-3g", output_file="water.out")
```

Unified quantum-chemistry results expose `result.orbital_energies_ev` as a
vector of molecular orbital energies in eV. This is `nothing` for molecular
mechanics methods. For post-HF methods, the values are from the underlying HF
reference calculation (not correlated quasiparticle energies).

Standalone solver results, such as `solve_huckel` or `solve_hf`, use the same
formatter and writer:

```julia
raw_result = solve_huckel(mol)
println(format_output(raw_result))
write_output("benzene.result", raw_result; title="Benzene Hückel Results")
```

`format_output(result)` returns a plain-text string. `write_output(filename,
result)` writes that report to the exact filename, with any extension (for
example `.txt`, `.out`, or `.log`). Hückel results use a readable report with
orbital energies/occupations, pi charges, and sorted bond orders; large matrices
and tensors from other methods are represented by their dimensions so reports
remain readable.

Examples 01–15 write a plain-text `.out` report for their calculation results
in the current working directory. Each example uses its own filename; running
it again replaces that example's prior report.
Run an example from the repository root:

```sh
julia examples/01_huckel_benzene.jl
```

---

## 📂 Repository Structure

```
QuantumChemMM/
├── Project.toml                     # Package dependencies & GPU extensions
├── README.md                        # Package documentation & guide
├── docs/
│   └── ARCHITECTURE.md              # Mathematical theory & derivations
├── ext/                             # Hardware extensions (CUDA, AMDGPU, oneAPI)
├── src/
│   ├── core/                        # Types, Periodic Table, IO, Math, Topology
│   ├── gpu/                         # Multi-vendor GPU kernels, interfaces & solvers
│   ├── ab_initio/                   # Basis sets, Analytical Gaussian Integrals, HF
│   ├── dft/                         # XC Functionals, Molecular Grid, Kohn-Sham DFT
│   ├── post_hf/                     # 4-Index Transform, MP2, CISD, CCSD, CCSD(T)
│   ├── excited_states/              # CIS Singlets & Triplets, PES Gradients
│   ├── transition_states/           # CI-NEB, Eigenvector-Following, IRC
│   ├── solvation/                   # COSMO & Generalized Born continuum models
│   ├── molecular_mechanics/         # UFF, AMBER, OPLS, CHARMM, MM3
│   ├── semi_empirical/              # HMO, EHT, CNDO/2, INDO, NDDO, DFTB/xTB
│   ├── qmmm/                        # QM/MM hybrid embedding
│   ├── optimization/                # L-BFGS, CG, SD, MD engine
│   ├── analysis/                    # Spectra, NMR, Thermochemistry, Reactivity, Conformers
│   └── calculate.jl                 # Unified multi-device calculation entry point
├── examples/                        # 15 runnable examples
└── test/
    └── runtests.jl                  # Comprehensive unit test suite (14 test sets)
```

---

## 📜 License
MIT License.
