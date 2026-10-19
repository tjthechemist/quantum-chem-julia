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
5. **Excited States (CIS)**: Batched transition-charge contractions and CIS Hamiltonian matrix setup on device arrays.
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
- **Kohn-Sham DFT**:
  - **LDA**: Slater exchange + VWN5 correlation.
  - **GGA**: PBE (Perdew-Burke-Ernzerhof) and BLYP (Becke-88 + Lee-Yang-Parr).
  - **Hybrid DFT**: B3LYP (20% exact HF exchange) and PBE0 (25% exact HF exchange).
  - Molecular numerical quadrature grid using Gauss-Legendre / Euler-Maclaurin radial and Lebedev angular grids with Becke fuzzy Voronoi cell partitioning.

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
  - Simulates continuous absorption curves $\epsilon(\lambda)$ with Gaussian line broadening.
  - Identifies $\lambda_{\max}$ and peak molar extinction coefficients.
- **Vibrational / Infrared (IR) Spectroscopy**:
  - Full numerical Cartesian Hessian matrix evaluation.
  - **Eckart Frame Projection**: Eliminates trivial translational and rotational degrees of freedom.
  - Infrared absorption intensities ($I_k$ in km/mol) from normal mode dipole moment derivatives.
  - Simulates continuous IR transmittance/absorbance spectra with Lorentzian peak broadening.
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

---

### 🔬 Molecular Mechanics Force Fields (`src/molecular_mechanics/`)
- **Class I**: **UFF** (Z=1..103), **AMBER/GAFF**, **OPLS-AA**, **CHARMM** (Urey-Bradley).
- **Class II**: **MM2 / MM3** (Allinger) with anharmonic stretches, sextic bends, and stretch-bend cross terms.
- **Hybrid QM/MM**: Mechanical and electrostatic embedding with link-atom capping.

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
wavelengths, absorbance = generate_uv_vis_spectrum(cis_res.singlets; lambda_min=150.0, lambda_max=400.0)
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
example `.txt`, `.out`, or `.log`). Large matrices and tensors are represented
by their dimensions so reports remain readable.

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
