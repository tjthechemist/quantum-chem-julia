# ==============================================================================
# QuantumChemMM.jl: Kohn-Sham Density Functional Theory (KS-DFT) SCF Solver
# Supports B3LYP, PBE, PBE0, BLYP, and LDA with Multi-Vendor GPU Acceleration
# ==============================================================================

module KohnShamDFT

using LinearAlgebra
using Base.Threads: @threads
using Printf
using ..Types
using ..Elements
using ..MolMath
using ..SemiEmpiricalBase
using ..BasisSets
using ..MolecularIntegrals
using ..DFTFunctionals
using ..DFTGrid
using ..GPUInterface
using ..GPUKernels

export DFTResult, solve_dft, print_dft_summary

const HARTREE_TO_EV = 27.211386245988

struct DFTResult
    functional::String
    converged::Bool
    cycles::Int
    total_energy_hartree::Float64
    total_energy_ev::Float64
    xc_energy_hartree::Float64
    coulomb_energy_hartree::Float64
    exact_exchange_hartree::Float64
    nuclear_repulsion_hartree::Float64
    mo_energies_hartree::Vector{Float64}
    mo_coefficients::Matrix{Float64}
    density_matrix::Matrix{Float64}
    basis::BasisSet
    effective_nuclear_charges::Vector{Float64}
    n_occ::Int
    n_virt::Int
end

"""
    solve_dft(mol::Molecule; functional::Symbol=:b3lyp, basis_name::String="sto-3g",
              basis_set=nothing, ecp_name=nothing,
              device::AbstractDevice=get_default_device(),
              options::SCFOptions=SCFOptions())::DFTResult
Executes Kohn-Sham Density Functional Theory calculation with multi-vendor GPU acceleration
on NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI) GPUs. A caller-supplied `BasisSet`
overrides `basis_name`; `ecp_name="qcm-prototype"` opts into the experimental ECP model.
Supported functionals:
  - :b3lyp (Hybrid Becke 3-parameter Lee-Yang-Parr, 20% exact exchange)
  - :pbe0  (Hybrid PBE, 25% exact exchange)
  - :pbe   (Generalized Gradient Approximation)
  - :blyp  (GGA Becke-88 + LYP)
  - :lda   (Local Density Approximation SVWN)
"""
function solve_dft(mol::Molecule; functional::Symbol=:b3lyp, basis_name::String="sto-3g",
                   basis_set::Union{Nothing, BasisSet}=nothing,
                   ecp_name::Union{Nothing, String}=nothing,
                   device::AbstractDevice=get_default_device(),
                   options::SCFOptions=SCFOptions())::DFTResult
    nat = n_atoms(mol)
    bs = basis_set === nothing ? build_basis(mol, basis_name) :
         validate_basis(mol, basis_set)
    nb = n_basis(bs)
    ecps = ecp_name === nothing ? Dict{Int, EffectiveCorePotential}() :
           build_ecp(mol; name=ecp_name)
    nuclear_charges = effective_nuclear_charges(mol, ecps)

    # 1. Total electrons and number of occupied orbitals
    total_electrons = sum(nuclear_charges) - mol.total_charge
    isinteger(total_electrons) ||
        throw(ArgumentError("Effective electron count must be an integer."))
    total_electrons = Int(round(total_electrons))
    mol.multiplicity == 1 ||
        throw(ArgumentError("This DFT solver is restricted closed-shell only; unrestricted DFT is not implemented."))
    iseven(total_electrons) ||
        throw(ArgumentError("Closed-shell DFT requires an even effective electron count."))
    total_electrons >= 0 ||
        throw(ArgumentError("Molecular charge gives a negative electron count."))
    n_occ = div(total_electrons, 2)
    n_occ <= nb || throw(ArgumentError("Basis set has $nb functions, fewer than the $n_occ occupied orbitals required."))
    n_virt = nb - n_occ

    # 2. Select functional
    func = if functional == :pbe
        PBE_FUNCTIONAL
    elseif functional == :pbe0
        PBE0_FUNCTIONAL
    elseif functional == :blyp
        BLYP_FUNCTIONAL
    elseif functional == :lda
        LDA_FUNCTIONAL
    else
        B3LYP_FUNCTIONAL
    end

    c_x = func.exact_exchange_fraction

    # 3. 1-electron integrals and nuclear repulsion
    S = compute_overlap(bs)
    T = compute_kinetic(bs)
    V = compute_nuclear_attraction(bs, mol; nuclear_charges=nuclear_charges)
    H_core = T .+ V
    if ecp_name !== nothing
        H_core .+= apply_ecp_matrix(mol, bs; name=ecp_name)
    end
    eri = compute_eri_tensor(bs)

    coords_bohr = [mol.atoms[i].coords .* BasisSets.ANGSTROM_TO_BOHR for i in 1:nat]
    e_nuc = 0.0
    for a in 1:nat
        za = nuclear_charges[a]
        for b in (a+1):nat
            zb = nuclear_charges[b]
            rab = MolMath.distance(coords_bohr[a], coords_bohr[b])
            e_nuc += (za * zb) / max(rab, 1e-4)
        end
    end

    # 4. Build numerical integration grid
    grid = build_molecular_grid(mol; n_radial=18, n_angular=26)
    n_pts = length(grid.points)

    # 5. Symmetric Orthogonalization
    eig_s = eigen(Symmetric(S))
    s_inv_sqrt = [val > 1e-10 ? 1.0 / sqrt(val) : 0.0 for val in eig_s.values]
    X = eig_s.vectors * Diagonal(s_inv_sqrt) * eig_s.vectors'

    # Initial density matrix from core Hamiltonian
    H_prime = X' * H_core * X
    eig_init = eigen(Symmetric(H_prime))
    C = X * eig_init.vectors
    P = zeros(Float64, nb, nb)
    for mu in 1:nb
        for nu in 1:nb
            P[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
        end
    end

    # 6. Kohn-Sham SCF Loop
    diis = DIISController(options.diis_size)
    eps = zeros(Float64, nb)
    converged = false
    last_e = 0.0
    cycles = 0

    E_xc = 0.0
    E_coulomb = 0.0
    E_exact_x = 0.0

    if is_gpu(device)
        @info "Accelerating Kohn-Sham DFT ($(uppercase(string(functional)))) on $(device_name(device))..."

        # Precompute basis values at grid points
        basis_values = zeros(Float64, n_pts, nb)
        @threads for g in 1:n_pts
            rg = grid.points[g]
            for mu in 1:nb
                basis_values[g, mu] = DFTGrid.eval_basis_function(bs.functions[mu], rg.coords)
            end
        end

        # Reshaped ERI tensors for GPU GEMV contractions
        eri_j_mat = reshape(eri, nb^2, nb^2)
        eri_k_mat = reshape(permutedims(eri, (1, 3, 2, 4)), nb^2, nb^2)

        H_core_dev = to_device(H_core, device)
        X_dev = to_device(X, device)
        P_dev = to_device(P, device)
        eri_j_dev = to_device(eri_j_mat, device)
        eri_k_dev = to_device(eri_k_mat, device)
        basis_values_dev = to_device(basis_values, device)
        weights_dev = to_device([point.weight for point in grid.points], device)
        F_ks_dev = copy(H_core_dev)
        C_dev = to_device(C, device)

        for iter in 1:options.max_cycles
            cycles = iter

            # GPU GEMV Coulomb J and exact Exchange K
            J_dev, K_dev = GPUKernels.gpu_fock_coulomb_exchange(P_dev, eri_j_dev, eri_k_dev, c_x)

            # GPU Grid evaluation for V_xc matrix and E_xc
            V_xc_dev, E_xc = GPUKernels.gpu_dft_density_and_xc(basis_values_dev, weights_dev, P_dev)

            # Form Kohn-Sham matrix on GPU
            F_ks_dev .= H_core_dev .+ J_dev .- (0.5 * c_x) .* K_dev .+ V_xc_dev

            # DIIS Acceleration
            F_host = to_host(F_ks_dev)
            P_host = to_host(P_dev)
            F_work_host = options.use_diis && iter > 1 ? diis_update!(diis, F_host, P_host, S) : F_host
            F_work_dev = to_device(F_work_host, device)

            # Diagonalize on device / host
            F_prime_dev = X_dev' * F_work_dev * X_dev
            eig_f = eigen(Symmetric(to_host(F_prime_dev)))
            eps = eig_f.values
            C_host = X * eig_f.vectors
            C_dev = to_device(C_host, device)

            # New density on GPU
            C_occ_dev = C_dev[:, 1:n_occ]
            P_new_dev = 2.0 .* (C_occ_dev * C_occ_dev')

            E_one_elec = sum(P_dev .* H_core_dev)
            E_coulomb = 0.5 * sum(P_dev .* J_dev)
            E_exact_x = c_x > 0.0 ? -0.25 * c_x * sum(P_dev .* K_dev) : 0.0
            tot_e = E_one_elec + E_coulomb + E_exact_x + E_xc + e_nuc

            d_p = maximum(abs.(to_host(P_new_dev .- P_dev)))
            d_e = abs(tot_e - last_e)

            if options.damping > 0.0
                P_dev .= (1.0 - options.damping) .* P_new_dev .+ options.damping .* P_dev
            else
                P_dev .= P_new_dev
            end

            last_e = tot_e

            if iter > 1 && (d_p < options.convergence_threshold || d_e < options.convergence_threshold)
                converged = true
                break
            end
        end

        synchronize_device(device)
        P_final = to_host(P_dev)
        C_final = to_host(C_dev)
        tot_hartree = last_e
        tot_ev = tot_hartree * HARTREE_TO_EV

        return DFTResult(func.name, converged, cycles, tot_hartree, tot_ev,
                         E_xc, E_coulomb, E_exact_x, e_nuc, eps, C_final, P_final, bs,
                         nuclear_charges, n_occ, n_virt)

    else
        # Standard Host CPU Execution
        F_ks = copy(H_core)

        for iter in 1:options.max_cycles
            cycles = iter

            J = zeros(Float64, nb, nb)
            K = zeros(Float64, nb, nb)

            @threads for mu in 1:nb
                for nu in mu:nb
                    val_j = 0.0
                    val_k = 0.0
                    for lam in 1:nb
                        for sig in 1:nb
                            p_ls = P[lam, sig]
                            abs(p_ls) < 1e-12 && continue
                            val_j += p_ls * eri[mu, nu, lam, sig]
                            if c_x > 0.0
                                val_k += p_ls * eri[mu, lam, nu, sig]
                            end
                        end
                    end
                    J[mu, nu] = val_j
                    if c_x > 0.0
                        K[mu, nu] = val_k
                    end
                end
            end
            for mu in 1:nb
                for nu in (mu + 1):nb
                    J[nu, mu] = J[mu, nu]
                    K[nu, mu] = K[mu, nu]
                end
            end

            V_xc, E_xc = integrate_xc_matrix_and_energy(grid, bs, P, func)

            F_ks .= H_core .+ J .- (0.5 * c_x) .* K .+ V_xc

            F_work = options.use_diis && iter > 1 ? diis_update!(diis, F_ks, P, S) : F_ks

            F_prime = X' * F_work * X
            eig_f = eigen(Symmetric(F_prime))
            eps = eig_f.values
            C = X * eig_f.vectors

            P_new = zeros(Float64, nb, nb)
            for mu in 1:nb
                for nu in 1:nb
                    P_new[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
                end
            end

            E_one_elec = sum(P .* H_core)
            E_coulomb = 0.5 * sum(P .* J)
            E_exact_x = c_x > 0.0 ? -0.25 * c_x * sum(P .* K) : 0.0
            tot_e = E_one_elec + E_coulomb + E_exact_x + E_xc + e_nuc

            d_p = maximum(abs.(P_new .- P))
            d_e = abs(tot_e - last_e)

            if options.damping > 0.0
                P .= (1.0 - options.damping) .* P_new .+ options.damping .* P
            else
                P .= P_new
            end

            last_e = tot_e

            if iter > 1 && (d_p < options.convergence_threshold || d_e < options.convergence_threshold)
                converged = true
                break
            end
        end

        tot_hartree = last_e
        tot_ev = tot_hartree * HARTREE_TO_EV

        return DFTResult(func.name, converged, cycles, tot_hartree, tot_ev,
                         E_xc, E_coulomb, E_exact_x, e_nuc, eps, C, P, bs,
                         nuclear_charges, n_occ, n_virt)
    end
end

function print_dft_summary(res::DFTResult)
    println("==================================================")
    @printf("       Kohn-Sham DFT Method: %-8s Results        \n", uppercase(res.functional))
    println("==================================================")
    @printf("  Convergence:              %s (%d iterations)\n", res.converged ? "CONVERGED" : "FAILED", res.cycles)
    @printf("  Coulomb Energy (J):       %16.8f a.u.\n", res.coulomb_energy_hartree)
    @printf("  Exchange-Correlation(XC): %16.8f a.u.\n", res.xc_energy_hartree)
    if abs(res.exact_exchange_hartree) > 1e-10
        @printf("  Exact Exchange (HF):      %16.8f a.u.\n", res.exact_exchange_hartree)
    end
    @printf("  Nuclear Repulsion Energy: %16.8f a.u.\n", res.nuclear_repulsion_hartree)
    @printf("  TOTAL DFT ENERGY:         %16.8f a.u. (%12.4f eV)\n", res.total_energy_hartree, res.total_energy_ev)
    println("--------------------------------------------------")
    println("  Frontier Molecular Orbitals (a.u. / eV):")
    for i in 1:n_basis(res.basis)
        tag = (i == res.n_occ) ? " [HOMO]" : (i == res.n_occ + 1) ? " [LUMO]" : ""
        ev = res.mo_energies_hartree[i] * HARTREE_TO_EV
        @printf("    MO #%2d:  %12.6f a.u. (%9.3f eV)%s\n", i, res.mo_energies_hartree[i], ev, tag)
    end
    println("==================================================")
end

end # module KohnShamDFT
