# Architecture and Theoretical Foundations of QuantumChemMM.jl

`QuantumChemMM.jl` is a high-performance, modular computational chemistry package written in pure Julia. It unifies **all major semi-empirical quantum chemical methods** and **molecular mechanics force fields** within a single, type-stable mathematical and computational framework.

---

## 1. Semi-Empirical Quantum Chemistry Methods

Semi-empirical methods approximate the full Hartree-Fock (HF) or Density Functional Theory (DFT) equations by:
1. Treating only **valence electrons** explicitly (core electrons are absorbed into an effective nuclear charge $Z_A$).
2. Using a **minimal atomic orbital basis** (Slater-type orbitals, STOs).
3. Neglecting or approximating computationally intensive two-electron multi-center integrals.
4. Parameterizing one-center and two-center integrals against high-level ab initio calculations and experimental thermodynamic data.

### 1.1 Summary of Implemented Semi-Empirical Methods

| Generation / Family | Method Name | Integral Approximation | Orbitals / Basis | Core Repulsion / Corrections | Key Application |
|---|---|---|---|---|---|
| **Topological $\pi$** | **HMO (Simple Hückel)** | Complete neglect of $\sigma$ framework | $p_z$ per conjugated atom | Purely electronic | Conjugated $\pi$-systems, aromaticity, resonance energy |
| **All-Valence Non-Iterative** | **EHT (Extended Hückel)** | Non-SCF Wolfsberg-Helmholz $H_{\mu\nu} = \frac{1}{2} K S_{\mu\nu} (H_{\mu\mu} + H_{\nu\nu})$ | Valence STO (1s, 2s, 2p) | Non-SCF sum of orbital energies | Qualitative MO analysis, organometallics, band theory |
| **ZDO Family** | **CNDO/2** | Zero Differential Overlap: $(\mu\nu\|\lambda\sigma) = \delta_{\mu\nu} \delta_{\lambda\sigma} \gamma_{AB}$ | Valence STO | Coulomb core-core repulsion | General charge distributions, dipoles |
| | **INDO** | Intermediate Neglect: keeps 1-center exchange $(s_A p_A \| s_A p_A)$ | Valence STO | Coulomb core-core | Singlet-triplet splittings, radical spin density |
| | **CNDO/S & ZINDO/S** | Spectroscopic parameterization + CI-Singles | Valence STO | Spectroscopic resonance $\beta$ | UV-Vis absorption spectra, excited states |
| **NDDO Family** | **MNDO** | Neglect of Diatomic Differential Overlap: $(\mu_A \nu_A \| \lambda_B \sigma_B)$ multipoles | Valence $s, p$ STO | $Z_A Z_B (ss\|ss)[1 + e^{-\alpha_A R} + e^{-\alpha_B R}]$ | General ground-state organic thermochemistry |
| | **AM1** | Austin Model 1 | Valence $s, p$ STO | MNDO + Gaussian core repulsion functions | Hydrogen bonding, heats of formation |
| | **PM3** | Parameterized Model 3 | Valence $s, p$ STO | Re-optimized parameters + 2 Gaussian terms | Thermochemistry, biochemical intermediates |
| | **RM1** | Recife Model 1 | Valence $s, p$ STO | Reparameterized AM1 for organics | Improved biomolecular geometries |
| | **PM6 / PM7** | Parametric Methods 6 & 7 | Valence $s, p, d$ STO | Pairwise core parameters + d-orbitals | Broad periodic table coverage |
| **Tight-Binding / DFTB** | **GFN0-xTB** | Non-self-consistent tight-binding | Minimal valence AO | Pairwise repulsive potential + D3 dispersion | Fast screening, ultralarge systems |
| | **SCC-DFTB / GFN1-xTB** | Self-Consistent Charge (2nd-order expansion) | Minimal valence AO | Hubbard $\gamma_{AB}^U$ charge fluctuation + D3 | Geometries, vibrational frequencies, reaction paths |
| | **GFN2-xTB** | Multi-pole electrostatics + anisotropic charge | Minimal valence AO | Blom-Hubbard derivatives + D4 dispersion | Noncovalent complexes, conformational search |

---

### 1.2 Mathematical Formulations

#### 1. Extended Hückel Theory (EHT)
- **Overlap Matrix**: Computed analytically via Gaussian STO-3G contractions:
  $$S_{\mu\nu} = \langle \chi_\mu | \chi_\nu \rangle$$
- **Diagonal Hamiltonian Elements**: Set to experimental Valence State Ionization Potentials (VSIP):
  $$H_{\mu\mu} = -I_\mu$$
- **Off-Diagonal Hamiltonian Elements**:
  $$H_{\mu\nu} = \frac{1}{2} K S_{\mu\nu} (H_{\mu\mu} + H_{\nu\nu}) \quad (\text{with standard } K = 1.75)$$
- **Generalized Secular Equation**:
  $$\mathbf{H} \mathbf{c}_i = \epsilon_i \mathbf{S} \mathbf{c}_i$$
  Solved via symmetric orthogonalization: $\mathbf{X} = \mathbf{S}^{-1/2}$.

#### 2. CNDO/2 and INDO (Roothaan-Hall SCF)
- **Fock Matrix Elements**:
  $$F_{\mu\mu} = H_{\mu\mu} - \frac{1}{2} P_{\mu\mu} \gamma_{AA} + P_{AA} \gamma_{AA} + \sum_{B \neq A} (P_{BB} - Z_B) \gamma_{AB}$$
  $$F_{\mu\nu} = H_{\mu\nu} - \frac{1}{2} P_{\mu\nu} \gamma_{AB} \quad (\mu \in A, \nu \in B)$$
  $$F_{\mu\nu} = -\frac{1}{2} P_{\mu\nu} \gamma_{AA} \quad (\mu \neq \nu \in A)$$
  where $\gamma_{AB}$ is the Klopman-Ohno or Mataga-Nishimoto two-center repulsion integral:
  $$\gamma_{AB} = \frac{14.3996}{\sqrt{R_{AB}^2 + \frac{1}{4} \left( \frac{14.3996}{\gamma_{AA}} + \frac{14.3996}{\gamma_{BB}} \right)^2}}$$

#### 3. NDDO Family (MNDO, AM1, PM3, RM1, PM6)
- **Two-Electron Integrals**: Retains all one-center and two-center charge distributions:
  $$(\mu_A \nu_A | \lambda_B \sigma_B)$$
  represented via semi-empirical multipole expansions (monopole-monopole, monopole-dipole, dipole-dipole, quadrupole).
- **Core-Core Repulsion**:
  $$E_{core}^{AB} = Z_A Z_B (s_A s_A | s_B s_B) \left[ 1 + e^{-\alpha_A R_{AB}} + e^{-\alpha_B R_{AB}} \right] + \Delta V_{AB}^{Gauss}$$
  with Gaussian core correction functions:
  $$\Delta V_{AB} = \frac{Z_A Z_B}{R_{AB}} \left[ \sum_k a_{k, A} e^{-b_{k, A} (R_{AB} - c_{k, A})^2} + \sum_l a_{l, B} e^{-b_{l, B} (R_{AB} - c_{l, B})^2} \right]$$
- **Total Energy and Heat of Formation**:
  $$E_{tot} = E_{elec} + E_{core}$$
  $$\Delta H_f = E_{tot} - \sum_A E_{isol, A} + \sum_A \Delta H_{f, A}^{atom}$$

#### 4. Tight-Binding (DFTB & GFN-xTB)
- **Total Energy**:
  $$E_{DFTB} = \sum_{i}^{occ} 2\epsilon_i - \frac{1}{2} \sum_{A, B} \Delta q_A \Delta q_B \gamma_{AB}^U + E_{rep} + E_{disp}^{D3}$$
- **Hubbard Chemical Hardness**:
  $$\gamma_{AB}^U = \frac{1}{\sqrt{R_{AB}^2 + \left[\frac{1}{2} (U_A^{-1} + U_B^{-1})\right]^2}}$$

---

## 2. Molecular Mechanics (MM) Engine

The package implements both **Class I** and **Class II** force fields:

### 2.1 Implemented Force Fields

1. **UFF (Universal Force Field)**:
   - Covers the entire periodic table from $Z = 1$ (H) to $Z = 103$ (Lr).
   - Rule-based derivation of equilibrium bond lengths $r_0$, force constants $k_b$, natural angles $\theta_0$, and inversion potentials from atomic electronegativity $\chi$ and effective charges $Z^*$.
2. **AMBER / GAFF**:
   - Class I biomolecular force field with standard harmonic bonds, harmonic angles, Fourier periodic dihedrals, and 1-4 scaling factors ($s_{vdW} = 0.5, s_{elec} = 0.8333$).
3. **OPLS-AA**:
   - Optimized Potentials for Liquid Simulations with geometric combining rules ($\sigma_{ij} = \sqrt{\sigma_i \sigma_j}$) and Fourier/Ryckaert-Bellemans dihedrals.
4. **CHARMM**:
   - Class I force field with Urey-Bradley 1-3 cross-angle distance harmonic terms.
5. **MM2 / MM3 (Allinger)**:
   - Class II force field featuring:
     - Cubic and quartic bond stretching.
     - Sextic angle bending.
     - Stretch-bend cross terms: $E_{sb} = k_{sb} (\Delta r_{ij} + \Delta r_{kj}) \Delta \theta_{ijk}$.
     - Buckingham $\exp-6$ non-bonded interactions.

---

## 3. Hybrid QM/MM Engine

- **Partitioning**: Molecular system is divided into QM region and MM environment.
- **Link Atoms**: Automatic detection of severed covalent bonds with hydrogen capping ($r_{C-H} = 1.09$ Å).
- **Subtractive (Mechanical) Embedding**:
  $$E_{QMMM} = E_{QM}(QM + \text{link}) + E_{MM}(\text{Full}) - E_{MM}(QM + \text{link})$$
- **Additive (Electrostatic) Embedding**:
  Includes explicit classical Coulomb interaction between MM partial charges and QM Mulliken charges.

---

## 4. Geometry Optimization and Molecular Dynamics

- **Line Search**: Backtracking Armijo condition with polynomial interpolation.
- **Optimizers**:
  - Steepest Descent (`:sd`)
  - Polak-Ribiere Conjugate Gradient (`:cg`)
  - Limited-memory BFGS (`:lbfgs`, 6-vector memory subspace)
- **Molecular Dynamics**:
  - Velocity Verlet integrator ($\Delta t \approx 1$ fs).
  - Maxwell-Boltzmann initial velocity distribution with linear momentum removal.
---

## 5. Multi-Vendor GPU Acceleration Architecture

`QuantumChemMM.jl` leverages Julia's compiler infrastructure and conditional extensions to target **NVIDIA**, **AMD**, and **Intel** GPUs from a single unified API without code duplication.

### 5.1 Architecture & Extensions

```
                 +-------------------------------+
                 |        QuantumChemMM.jl       |
                 |      (AbstractDevice API)     |
                 +---------------+---------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
  [using CUDA]             [using AMDGPU]          [using oneAPI]
         |                       |                       |
         v                       v                       v
+-----------------+     +-----------------+     +-----------------+
|   NVIDIA CUDA   |     |    AMD ROCm     |     |  Intel oneAPI   |
|   (CUDADevice)  |     |   (ROCmDevice)  |     |  (OneAPIDevice) |
|  cuBLAS/cuSOLVER|     | rocBLAS/rocSOLVE|     | oneMKL / Level0 |
+-----------------+     +-----------------+     +-----------------+
```

### 5.2 GPU Accelerated Bottlenecks

1. **Non-Bonded $O(N^2)$ Pairwise Matrices**:
   - Vectorized pairwise Euclidean distance calculation:
     $$\mathbf{D}^2 = \mathbf{r}^2 \mathbf{1}^T + \mathbf{1} (\mathbf{r}^2)^T - 2 \mathbf{R}^T \mathbf{R}$$
   - Parallel evaluation of Lennard-Jones $12-6$, Coulomb electrostatics, and Buckingham potentials across streaming multiprocessors (SMs) / compute units (CUs).
2. **Two-Center Electron Repulsion Matrix $\gamma_{AB}$**:
   - Evaluated on device memory using Klopman-Ohno / Mataga-Nishimoto formulations.
---

## 6. Excited State Calculations (Configuration Interaction Singles)

The package implements Configuration Interaction Singles (CIS) on top of converged SCF reference states for both **Singlet** and **Triplet** multiplicities:

### 6.1 CIS Hamiltonian Matrix
Let $i, j$ index occupied molecular orbitals and $a, b$ index virtual orbitals. The dimension of the single excitation space is $N_{ex} = N_{occ} \times N_{virt}$.

- **Singlet Excitations**:
  $$A_{ia, jb} = (\epsilon_a - \epsilon_i) \delta_{ij} \delta_{ab} + 2 (ia|jb) - (ib|ja)$$
- **Triplet Excitations**:
  $$A_{ia, jb} = (\epsilon_a - \epsilon_i) \delta_{ij} \delta_{ab} - (ib|ja)$$

In semi-empirical zero differential overlap approximations:
$$(ia|jb) = \sum_{\mu \in A} \sum_{\nu \in B} C_{\mu i} C_{\mu a} C_{\nu j} C_{\nu b} \gamma_{AB}$$

### 6.2 Transition Dipoles and Oscillator Strengths
- **Transition Dipole**:
  $$\boldsymbol{\mu}_{0 \to k} = \sqrt{2} \sum_{i, a} X_{ia}^{(k)} \langle \psi_i | \mathbf{\hat{r}} | \psi_a \rangle$$
- **Oscillator Strength**:
  $$f_k = \frac{2}{3} \Delta E_k (\text{a.u.}) \cdot |\boldsymbol{\mu}_{0 \to k} (\text{a.u.})|^2$$

---

## 7. Transition State and Reaction Path Methods

### 7.1 Climbing Image Nudged Elastic Band (CI-NEB)
- A discrete chain of $K$ molecular replicas $\mathbf{R}_0, \mathbf{R}_1, \dots, \mathbf{R}_K$ is optimized simultaneously between reactant $\mathbf{R}_0$ and product $\mathbf{R}_K$.
- **Perpendicular True Force**:
  $$\mathbf{F}_k^\perp = \mathbf{F}_k - (\mathbf{F}_k \cdot \hat{\boldsymbol{\tau}}_k) \hat{\boldsymbol{\tau}}_k$$
- **Parallel Spring Force**:
  $$\mathbf{F}_k^{S\parallel} = k \left( \|\mathbf{R}_{k+1} - \mathbf{R}_k\| - \|\mathbf{R}_k - \mathbf{R}_{k-1}\| \right) \hat{\boldsymbol{\tau}}_k$$
- **Climbing Image (CI)**:
  For the highest energy image $k_{max}$, the potential force along the tangent is inverted:
  $$\mathbf{F}_{climb} = \mathbf{F}_{k_{max}} - 2(\mathbf{F}_{k_{max}} \cdot \hat{\boldsymbol{\tau}}_k) \hat{\boldsymbol{\tau}}_k$$
  driving it directly onto the exact first-order saddle point.

### 7.2 Intrinsic Reaction Coordinate (IRC)
From the optimized transition state, the minimum energy path (MEP) is integrated forward and reverse:
$$\frac{d\mathbf{x}}{ds} = -\frac{\mathbf{g}(\mathbf{x})}{\|\mathbf{g}(\mathbf{x})\|}$$
confirming the reaction profile connecting reactant, transition state, and product.

---

## 8. Advanced Spectroscopy & Peak Broadening

- **UV-Vis Gaussian Line Broadening**:
  $$\epsilon(\lambda) = \sum_k 1.306 \times 10^5 \cdot f_k \frac{1}{\sqrt{2\pi}\sigma} \exp\left( -\frac{(\lambda - \lambda_k)^2}{2\sigma^2} \right)$$
  with $\sigma = \frac{\text{FWHM}}{2\sqrt{2\ln 2}}$.
- **Vibrational Eckart Frame Projection**:
  Eliminates the 6 rigid translations and rotations from the Cartesian mass-weighted Hessian:
  $$\mathbf{H}_{proj} = \mathbf{P} \mathbf{H}_{MW} \mathbf{P} \quad \text{with } \mathbf{P} = \mathbf{I} - \sum_{k=1}^6 \mathbf{u}_k \mathbf{u}_k^T$$
- **Infrared Absorption Intensity**:
  $$I_k \propto \left| \sum_i q_i \mathbf{L}_{ik} \right|^2 \quad (\text{km/mol})$$

---

## 9. Implicit Solvation Models (COSMO & Generalized Born)

- **Total Solvation Free Energy**:
  $$\Delta G_{solv} = \Delta G_{elec} + \Delta G_{cav} + \Delta G_{disp}$$
- **Electrostatic Solvation Free Energy**:
  $$\Delta G_{elec} = -\frac{1}{2} f(\epsilon_s) \sum_{i, j} \frac{q_i q_j}{f_{GB}(r_{ij}, R_i^{eff}, R_j^{eff})}$$
  where $f(\epsilon_s) = \frac{\epsilon_s - 1}{\epsilon_s + 0.5}$ (COSMO factor) and $R_i^{eff}$ is the effective Born radius derived from numerical SASA tessellation.
- **Non-Polar Cavitation**:
  $$\Delta G_{cav} = \gamma \cdot \text{SASA}$$

---

## 10. Conceptual DFT Reactivity Indices & Fukui Functions

- **Chemical Hardness**: $\eta = \frac{1}{2}(\epsilon_{LUMO} - \epsilon_{HOMO})$
- **Chemical Potential**: $\mu = \frac{1}{2}(\epsilon_{HOMO} + \epsilon_{LUMO})$
- **Global Electrophilicity Index**: $\omega = \frac{\mu^2}{2\eta}$
- **Condensed Fukui Functions**:
  - $f_k^+ \approx \sum_{\mu \in k} |C_{\mu, LUMO}|^2$ (nucleophilic attack susceptibility)
  - $f_k^- \approx \sum_{\mu \in k} |C_{\mu, HOMO}|^2$ (electrophilic attack susceptibility)
  - $f_k^0 = \frac{1}{2}(f_k^+ + f_k^-)$ (radical attack susceptibility)
---

## 11. Ab Initio Quantum Chemistry (Hartree-Fock)

The ab initio Hartree-Fock engine implements closed-shell Restricted Hartree-Fock (RHF) and open-shell Unrestricted Hartree-Fock (UHF) using contracted Gaussian basis sets (STO-3G, 6-31G):

### 11.1 Molecular Gaussian Integrals
- **Overlap Matrix $\mathbf{S}$**:
  $$S_{\mu\nu} = \int \chi_\mu(\mathbf{r}) \chi_\nu(\mathbf{r}) d\mathbf{r}$$
- **Kinetic Energy Matrix $\mathbf{T}$**:
  $$T_{\mu\nu} = \int \chi_\mu(\mathbf{r}) \left(-\frac{1}{2} \nabla^2\right) \chi_\nu(\mathbf{r}) d\mathbf{r}$$
- **Electron-Nuclear Attraction $\mathbf{V}$**:
  $$V_{\mu\nu} = \int \chi_\mu(\mathbf{r}) \left( -\sum_A \frac{Z_A}{|\mathbf{r} - \mathbf{R}_A|} \right) \chi_\nu(\mathbf{r}) d\mathbf{r}$$
  evaluated via Boys function $F_n(T) = \int_0^1 t^{2n} e^{-T t^2} dt$.
- **Two-Electron Repulsion Integrals (ERIs)**:
  $$(\mu\nu|\lambda\sigma) = \iint \frac{\chi_\mu(1)\chi_\nu(1)\chi_\lambda(2)\chi_\sigma(2)}{r_{12}} d\mathbf{r}_1 d\mathbf{r}_2$$

### 11.2 Roothaan-Hall SCF Equations
$$\mathbf{F} \mathbf{C} = \mathbf{S} \mathbf{C} \boldsymbol{\epsilon}$$
$$F_{\mu\nu} = H_{\mu\nu}^{core} + \sum_{\lambda, \sigma} P_{\lambda\sigma} \left[ (\mu\nu|\lambda\sigma) - \frac{1}{2} (\mu\lambda|\nu\sigma) \right]$$
$$E_{HF} = \frac{1}{2} \text{Tr}(\mathbf{P}(\mathbf{H}^{core} + \mathbf{F})) + \sum_{A < B} \frac{Z_A Z_B}{R_{AB}}$$

---

## 12. Density Functional Theory (DFT)

The Kohn-Sham DFT solver solves for the electron density $\rho(\mathbf{r})$ that minimizes the electronic energy:
$$E_{DFT}[\rho] = T_s[\rho] + V_{ext}[\rho] + J[\rho] + E_{XC}[\rho] + E_{nuc}$$

### 12.1 Kohn-Sham Matrix
$$F_{\mu\nu}^{KS} = H_{\mu\nu}^{core} + J_{\mu\nu} - c_x K_{\mu\nu} + V_{\mu\nu}^{XC}$$
where $c_x$ is the fraction of exact Hartree-Fock exchange ($0.20$ for B3LYP, $0.25$ for PBE0, $0.0$ for pure GGA).

### 12.2 Exchange-Correlation Potential
$$V_{\mu\nu}^{XC} = \int \chi_\mu(\mathbf{r}) v_{xc}(\mathbf{r}) \chi_\nu(\mathbf{r}) d\mathbf{r} = \sum_g w_g \chi_\mu(\mathbf{r}_g) v_{xc}(\rho(\mathbf{r}_g)) \chi_\nu(\mathbf{r}_g)$$
integrated numerically on a multi-center molecular grid using Becke atomic fuzzy Voronoi cell partitioning weights.

---

## 13. Post-Ab Initio Correlation Methods

### 13.1 Second-Order Møller-Plesset Perturbation (MP2)
Transforms two-electron integrals to the canonical molecular orbital (MO) basis:
$$(ia|jb) = \sum_{\mu, \nu, \lambda, \sigma} C_{\mu i} C_{\nu a} C_{\lambda j} C_{\sigma b} (\mu\nu|\lambda\sigma)$$
$$E_{MP2} = \sum_{i, j}^{occ} \sum_{a, b}^{virt} \frac{(ia|jb) [2(ia|jb) - (ib|ja)]}{\epsilon_i + \epsilon_j - \epsilon_a - \epsilon_b}$$

### 13.2 Configuration Interaction Singles and Doubles (CISD)
Diagonalization of the CI Hamiltonian in the space of $|\Phi_0\rangle$, $|\Phi_i^a\rangle$, and $|\Phi_{ij}^{ab}\rangle$:
$$E_{corr}^{CISD} = \lambda_1 - E_{HF}$$

### 13.3 Coupled Cluster Singles and Doubles (CCSD & CCSD(T))
Solves the non-linear coupled cluster amplitude equations for single $t_i^a$ and double $t_{ij}^{ab}$ excitations:
$$|\Psi_{CC}\rangle = e^{\hat{T}_1 + \hat{T}_2} |\Phi_0\rangle$$
$$E_{corr}^{CCSD} = \frac{1}{4} \sum_{ijab} \langle ij || ab \rangle (t_{ij}^{ab} + 2 t_i^a t_j^b)$$
with non-iterative perturbative triples correction $E_{(T)}$ yielding the "gold standard" CCSD(T) total energy.


---

## 14. Multi-Vendor GPU Acceleration Architecture (NVIDIA, AMD, Intel)

`QuantumChemMM.jl` employs an array-agnostic, multi-vendor GPU acceleration layer designed to offload the $O(N^4)$ and $O(N^5)$ computational bottlenecks to hardware accelerators:
- **NVIDIA GPUs** via `CUDA.jl` (`CuArray`)
- **AMD ROCm GPUs** via `AMDGPU.jl` (`ROCArray`)
- **Intel oneAPI GPUs** via `oneAPI.jl` (`oneArray`)
- **Host CPU** fallback via multithreaded SIMD `Array`

### 14.1 GPU Two-Electron Fock Matrix Construction
In canonical Hartree-Fock and hybrid DFT, evaluating the Coulomb $\mathbf{J}$ and Exchange $\mathbf{K}$ matrices requires contracting the 4-index Two-Electron Repulsion Integral (ERI) tensor with the 2-index density matrix $\mathbf{P}$:
$$J_{\mu\nu} = \sum_{\lambda\sigma} (\mu\nu | \lambda\sigma) P_{\lambda\sigma}$$
$$K_{\mu\nu} = \sum_{\lambda\sigma} (\mu\lambda | \nu\sigma) P_{\lambda\sigma}$$

Instead of 4 nested loops, the ERI tensor is reshaped into static $(N^2 \times N^2)$ device matrices:
$$\mathbf{E}_J = \text{reshape}(\text{ERI}, N^2, N^2)$$
$$\mathbf{E}_K = \text{reshape}(\text{permutedims}(\text{ERI}, (1, 3, 2, 4)), N^2, N^2)$$

Building $\mathbf{J}$ and $\mathbf{K}$ during each SCF iteration becomes two single **GPU GEMV / GEMM** matrix-vector operations:
$$\text{vec}(\mathbf{J}) = \mathbf{E}_J \times \text{vec}(\mathbf{P})$$
$$\text{vec}(\mathbf{K}) = \mathbf{E}_K \times \text{vec}(\mathbf{P})$$
achieving peak memory bandwidth and tensor core utilization without host-device synchronization overhead.

### 14.2 GPU Numerical Quadrature for Kohn-Sham DFT
The basis evaluation matrix $\mathbf{B} \in \mathbb{R}^{N_{grid} \times N_{basis}}$ ($B_{g\mu} = \chi_\mu(\mathbf{r}_g)$) and quadrature weights $\mathbf{w}$ are placed in device VRAM. During each KS-SCF cycle:
1. Electron density on grid:
   $$\mathbf{M} = \mathbf{B} \times \mathbf{P} \quad (\text{GPU GEMM})$$
   $$\rho_g = \sum_{\mu} M_{g\mu} B_{g\mu}$$
2. Exchange-correlation potential $v_{xc}(\rho_g)$ evaluated vectorially across all grid points on device.
3. XC potential matrix formed via device matrix multiplications:
   $$\mathbf{V}_{xc} = \mathbf{B}^T \times \left( \text{Diagonal}(\mathbf{w} \odot \mathbf{v}_{xc}) \times \mathbf{B} \right)$$
   $$E_{xc} = \sum_g w_g \rho_g \epsilon_{xc}(\rho_g)$$

### 14.3 GPU 4-Index AO-to-MO Integral Transformation
Post-Hartree-Fock correlation methods (MP2, CISD, CCSD, CCSD(T)) bottleneck at transforming integrals from the AO basis to the MO basis. `QuantumChemMM.jl` computes this as four successive GPU GEMMs without scalar loops:
1. $\mathbf{A}_1 = \text{reshape}(\text{ERI}, N^3, N) \times \mathbf{C}$
2. $\mathbf{A}_2 = \text{reshape}(\text{permutedims}(\mathbf{A}_1, (1, 2, 4, 3)), N^2 M, N) \times \mathbf{C}$
3. $\mathbf{A}_3 = \text{reshape}(\text{permutedims}(\mathbf{A}_2, (1, 3, 4, 2)), N M^2, N) \times \mathbf{C}$
4. $\mathbf{A}_4 = \text{reshape}(\text{permutedims}(\mathbf{A}_3, (2, 3, 4, 1)), M^3, N) \times \mathbf{C}$
5. $\text{ERI}_{MO} = \text{permutedims}(\text{reshape}(\mathbf{A}_4, M, M, M, M), (4, 3, 2, 1))$

### 14.4 GPU CIS Excited States Matrix Setup
In Configuration Interaction Singles (CIS), two-electron matrix elements $(ia|jb)$ are evaluated on GPU by constructing the transition-charge matrix $\mathbf{T} \in \mathbb{R}^{N_{atoms} \times (N_{occ} N_{virt})}$:
$$T_{A, ia} = \sum_{\mu \in A} C_{\mu i} C_{\mu a}$$
All $(ia|jb)$ integrals are computed simultaneously via a single device matrix sandwich product:
$$\mathbf{G} = \mathbf{T}^T \times \boldsymbol{\Gamma} \times \mathbf{T}$$
where $\boldsymbol{\Gamma}$ is the Klopman-Ohno two-center repulsion matrix on device.
