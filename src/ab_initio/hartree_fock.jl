# ==============================================================================
# QuantumChemMM.jl: Ab Initio Hartree-Fock (RHF and UHF) SCF Solver
# Full Roothaan-Hall Equations with DIIS, Exact Gaussian Integrals, and GPU Acceleration
# ==============================================================================

module HartreeFock

using LinearAlgebra
using Base.Threads: @threads
using Printf
using ..Types
using ..Elements
using ..MolMath
using ..SemiEmpiricalBase
using ..BasisSets
using ..MolecularIntegrals
using ..GPUInterface
using ..GPUKernels

export HFResult, solve_hf, solve_uhf, print_hf_summary

const HARTREE_TO_EV = 27.211386245988
const HARTREE_TO_KCAL = 627.509474

struct HFResult
    converged::Bool
    cycles::Int
    total_energy_hartree::Float64
    total_energy_ev::Float64
    electronic_energy_hartree::Float64
    nuclear_repulsion_hartree::Float64
    mo_energies_hartree::Vector{Float64}
    mo_coefficients::Matrix{Float64}
    density_matrix::Matrix{Float64}
    fock_matrix::Matrix{Float64}
    core_hamiltonian::Matrix{Float64}
    overlap_matrix::Matrix{Float64}
    eri_tensor::Array{Float64, 4}
    basis::BasisSet
    effective_nuclear_charges::Vector{Float64}
    reference::Symbol
    multiplicity::Int
    n_alpha::Int
    n_beta::Int
    alpha_mo_energies_hartree::Vector{Float64}
    beta_mo_energies_hartree::Vector{Float64}
    alpha_mo_coefficients::Matrix{Float64}
    beta_mo_coefficients::Matrix{Float64}
    alpha_density_matrix::Matrix{Float64}
    beta_density_matrix::Matrix{Float64}
    alpha_fock_matrix::Matrix{Float64}
    beta_fock_matrix::Matrix{Float64}
    n_occ::Int
    n_virt::Int
end

"""
    solve_hf(mol::Molecule; basis_name::String="sto-3g", basis_set=nothing,
             ecp_name=nothing,
             device::AbstractDevice=get_default_device(),
             options::SCFOptions=SCFOptions())::HFResult
Solves the ab initio Restricted Hartree-Fock (RHF) Roothaan-Hall equations
using exact analytical Gaussian integrals, with multi-vendor GPU acceleration
on NVIDIA (CUDA), AMD (ROCm), and Intel (oneAPI) GPUs. A caller-supplied
`BasisSet` overrides `basis_name`. `ecp_name="qcm-prototype"` opts into the
experimental built-in ECP model.
"""
function solve_hf(mol::Molecule; basis_name::String="sto-3g",
                  basis_set::Union{Nothing, BasisSet}=nothing,
                  ecp_name::Union{Nothing, String}=nothing,
                  device::AbstractDevice=get_default_device(),
                  options::SCFOptions=SCFOptions())::HFResult
    isempty(mol.atoms) && throw(ArgumentError("Cannot run RHF for a molecule with no atoms."))
    mol.multiplicity == 1 ||
        throw(ArgumentError("RHF requires multiplicity 1; use solve_uhf or calculate(mol, :uhf) for open-shell molecules."))
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
    total_electrons >= 0 || throw(ArgumentError("Molecular charge gives a negative electron count."))
    iseven(total_electrons) ||
        throw(ArgumentError("RHF requires an even electron count; use solve_uhf for open-shell molecules."))
    n_occ = div(total_electrons, 2)
    n_occ <= nb || throw(ArgumentError("Basis set has $nb functions, fewer than the $n_occ occupied orbitals required."))
    n_virt = nb - n_occ

    # 2. Compute 1-electron integrals
    S = compute_overlap(bs)
    T = compute_kinetic(bs)
    V = compute_nuclear_attraction(bs, mol; nuclear_charges=nuclear_charges)
    H_core = T .+ V
    if ecp_name !== nothing
        H_core .+= apply_ecp_matrix(mol, bs; name=ecp_name)
    end

    # 3. Compute 2-electron repulsion integrals (ERIs)
    eri = compute_eri_tensor(bs)

    # 4. Nuclear Repulsion Energy in atomic units (Hartrees)
    coords_bohr = [mol.atoms[i].coords .* BasisSets.ANGSTROM_TO_BOHR for i in 1:nat]
    e_nuc = 0.0
    for a in 1:nat
        za = nuclear_charges[a]
        for b in (a+1):nat
        zb = nuclear_charges[b]
            rab_bohr = MolMath.distance(coords_bohr[a], coords_bohr[b])
            e_nuc += (za * zb) / max(rab_bohr, 1e-4)
        end
    end

    # 5. Symmetric Orthogonalization Matrix X = S^(-1/2)
    eig_s = eigen(Symmetric(S))
    s_inv_sqrt = [val > 1e-10 ? 1.0 / sqrt(val) : 0.0 for val in eig_s.values]
    X = eig_s.vectors * Diagonal(s_inv_sqrt) * eig_s.vectors'

    # 6. Initial Guess Density Matrix from Core Hamiltonian
    H_prime = X' * H_core * X
    eig_init = eigen(Symmetric(H_prime))
    C = X * eig_init.vectors
    P = zeros(Float64, nb, nb)
    for mu in 1:nb
        for nu in 1:nb
            P[mu, nu] = 2.0 * sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
        end
    end

    # 7. Self-Consistent Field (SCF) Iteration Loop
    diis = DIISController(options.diis_size)
    eps = zeros(Float64, nb)
    converged = false
    last_e = 0.0
    cycles = 0

    if is_gpu(device)
        @info "Accelerating Hartree-Fock SCF on $(device_name(device))..."
        # Reshaped ERI tensors for GPU GEMV contractions
        eri_j_mat = reshape(eri, nb^2, nb^2)
        eri_k_mat = reshape(permutedims(eri, (1, 3, 2, 4)), nb^2, nb^2)

        H_core_dev = to_device(H_core, device)
        X_dev = to_device(X, device)
        P_dev = to_device(P, device)
        eri_j_dev = to_device(eri_j_mat, device)
        eri_k_dev = to_device(eri_k_mat, device)
        F_dev = copy(H_core_dev)
        C_dev = to_device(C, device)

        for iter in 1:options.max_cycles
            cycles = iter

            # GPU GEMV contractions for J and K
            J_dev, K_dev = GPUKernels.gpu_fock_coulomb_exchange(P_dev, eri_j_dev, eri_k_dev, 1.0)
            F_dev .= H_core_dev .+ J_dev .- 0.5 .* K_dev

            # DIIS update
            F_host = to_host(F_dev)
            P_host = to_host(P_dev)
            F_work_host = options.use_diis && iter > 1 ? diis_update!(diis, F_host, P_host, S) : F_host
            F_work_dev = to_device(F_work_host, device)

            # Diagonalization
            F_prime_dev = X_dev' * F_work_dev * X_dev
            eig_f = eigen(Symmetric(to_host(F_prime_dev)))
            eps = eig_f.values
            C_host = X * eig_f.vectors
            C_dev = to_device(C_host, device)

            # GPU matrix multiply for new density: P = 2 * C_occ * C_occ'
            C_occ_dev = C_dev[:, 1:n_occ]
            P_new_dev = 2.0 .* (C_occ_dev * C_occ_dev')

            e_elec = 0.5 * sum(P_dev .* (H_core_dev .+ F_dev))
            tot_e = e_elec + e_nuc

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
        F_final = to_host(F_dev)
        C_final = to_host(C_dev)
        e_elec_final = 0.5 * sum(P_final .* (H_core .+ F_final))
        tot_hartree = e_elec_final + e_nuc
        tot_ev = tot_hartree * HARTREE_TO_EV

        return HFResult(converged, cycles, tot_hartree, tot_ev, e_elec_final, e_nuc,
                        eps, C_final, P_final, F_final, H_core, S, eri, bs,
                        nuclear_charges, :rhf, mol.multiplicity, n_occ, n_occ,
                        eps, eps, C_final, C_final, P_final ./ 2, P_final ./ 2,
                        F_final, F_final, n_occ, n_virt)

    else
        # Standard Host CPU Execution
        F = copy(H_core)

        for iter in 1:options.max_cycles
            cycles = iter

            F .= H_core
            @threads for mu in 1:nb
                for nu in mu:nb
                    val_2e = 0.0
                    for lam in 1:nb
                        for sig in 1:nb
                            p_ls = P[lam, sig]
                            abs(p_ls) < 1e-12 && continue
                            coul = eri[mu, nu, lam, sig]
                            exch = eri[mu, lam, nu, sig]
                            val_2e += p_ls * (coul - 0.5 * exch)
                        end
                    end
                    F[mu, nu] += val_2e
                end
            end
            for mu in 1:nb
                for nu in (mu + 1):nb
                    F[nu, mu] = F[mu, nu]
                end
            end

            F_work = options.use_diis && iter > 1 ? diis_update!(diis, F, P, S) : F

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

            e_elec = 0.5 * sum(P .* (H_core .+ F))
            tot_e = e_elec + e_nuc

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

        e_elec = 0.5 * sum(P .* (H_core .+ F))
        tot_hartree = e_elec + e_nuc
        tot_ev = tot_hartree * HARTREE_TO_EV

        return HFResult(converged, cycles, tot_hartree, tot_ev, e_elec, e_nuc,
                        eps, C, P, F, H_core, S, eri, bs, nuclear_charges,
                        :rhf, mol.multiplicity, n_occ, n_occ,
                        eps, eps, C, C, P ./ 2, P ./ 2, F, F, n_occ, n_virt)
    end
end

function _density_matrix(C::AbstractMatrix{<:Real}, n_occ::Int)
    n_occ == 0 && return zeros(Float64, size(C, 1), size(C, 1))
    C_occ = @view C[:, 1:n_occ]
    return Matrix{Float64}(C_occ * C_occ')
end

function _build_jk_cpu(P::Matrix{Float64}, eri::Array{Float64, 4})
    nb = size(P, 1)
    J = zeros(Float64, nb, nb)
    K = zeros(Float64, nb, nb)
    @threads for mu in 1:nb
        for nu in 1:nb
            j_val = 0.0
            k_val = 0.0
            for lam in 1:nb, sig in 1:nb
                p_val = P[lam, sig]
                abs(p_val) < 1e-12 && continue
                j_val += p_val * eri[mu, nu, lam, sig]
                k_val += p_val * eri[mu, lam, nu, sig]
            end
            J[mu, nu] = j_val
            K[mu, nu] = k_val
        end
    end
    return J, K
end

function _nuclear_repulsion(mol::Molecule, charges::Vector{Float64})
    coords_bohr = [at.coords .* BasisSets.ANGSTROM_TO_BOHR for at in mol.atoms]
    energy = 0.0
    for a in eachindex(charges), b in (a + 1):length(charges)
        rab = MolMath.distance(coords_bohr[a], coords_bohr[b])
        energy += charges[a] * charges[b] / max(rab, 1e-4)
    end
    return energy
end

function _spin_occupations(mol::Molecule, nuclear_charges::Vector{Float64})
    total = sum(nuclear_charges) - mol.total_charge
    isfinite(total) && isinteger(total) ||
        throw(ArgumentError("Molecular charge and ECP core counts must yield an integer electron count."))
    n_electrons = Int(round(total))
    n_electrons >= 0 || throw(ArgumentError("Molecular charge gives a negative electron count."))
    multiplicity = mol.multiplicity
    multiplicity >= 1 || throw(ArgumentError("Molecular multiplicity must be a positive integer."))

    spin_difference = multiplicity - 1
    iseven(n_electrons + spin_difference) ||
        throw(ArgumentError("Electron count $n_electrons is incompatible with multiplicity $multiplicity."))
    n_alpha = div(n_electrons + spin_difference, 2)
    n_beta = div(n_electrons - spin_difference, 2)
    n_beta >= 0 ||
        throw(ArgumentError("Multiplicity $multiplicity exceeds the available electron count $n_electrons."))
    return n_alpha, n_beta
end

"""
    solve_uhf(mol::Molecule; basis_name="sto-3g", basis_set=nothing,
              ecp_name=nothing, device=get_default_device(),
              options=SCFOptions())::HFResult

Solve unrestricted Hartree-Fock for the electron count implied by molecular
charge and multiplicity. The result exposes total density in `density_matrix`
and spin-resolved orbitals, densities, and Fock matrices in the `alpha_*` and
`beta_*` fields. CPU and GPU paths use the same Hamiltonian and SCF equations.
"""
function solve_uhf(mol::Molecule; basis_name::String="sto-3g",
                   basis_set::Union{Nothing, BasisSet}=nothing,
                   ecp_name::Union{Nothing, String}=nothing,
                   device::AbstractDevice=get_default_device(),
                   options::SCFOptions=SCFOptions())::HFResult
    isempty(mol.atoms) && throw(ArgumentError("Cannot run UHF for a molecule with no atoms."))
    bs = basis_set === nothing ? build_basis(mol, basis_name) :
         validate_basis(mol, basis_set)
    nb = n_basis(bs)
    ecps = ecp_name === nothing ? Dict{Int, EffectiveCorePotential}() :
           build_ecp(mol; name=ecp_name)
    nuclear_charges = effective_nuclear_charges(mol, ecps)
    n_alpha, n_beta = _spin_occupations(mol, nuclear_charges)
    max(n_alpha, n_beta) <= nb ||
        throw(ArgumentError("Basis set has $nb functions, insufficient for $n_alpha alpha and $n_beta beta electrons."))

    S = compute_overlap(bs)
    T = compute_kinetic(bs)
    V = compute_nuclear_attraction(bs, mol; nuclear_charges=nuclear_charges)
    H_core = T .+ V
    if ecp_name !== nothing
        H_core .+= apply_ecp_matrix(mol, bs; name=ecp_name)
    end
    eri = compute_eri_tensor(bs)
    e_nuc = _nuclear_repulsion(mol, nuclear_charges)

    eig_s = eigen(Symmetric(S))
    eig_s.values[1] > 1e-10 ||
        throw(ArgumentError("Overlap matrix is singular or linearly dependent for this basis."))
    X = eig_s.vectors * Diagonal(1.0 ./ sqrt.(eig_s.values)) * eig_s.vectors'
    eig_init = eigen(Symmetric(X' * H_core * X))
    C_alpha = X * eig_init.vectors
    C_beta = copy(C_alpha)
    P_alpha = _density_matrix(C_alpha, n_alpha)
    P_beta = _density_matrix(C_beta, n_beta)

    diis_alpha = DIISController(options.diis_size)
    diis_beta = DIISController(options.diis_size)
    eps_alpha = zeros(Float64, nb)
    eps_beta = zeros(Float64, nb)
    F_alpha = copy(H_core)
    F_beta = copy(H_core)
    converged = false
    cycles = 0
    last_energy = Inf

    if is_gpu(device)
        @info "Accelerating Unrestricted Hartree-Fock SCF on $(device_name(device))..."
        eri_j = to_device(reshape(eri, nb^2, nb^2), device)
        eri_k = to_device(reshape(permutedims(eri, (1, 3, 2, 4)), nb^2, nb^2), device)
        H_dev = to_device(H_core, device)
        X_dev = to_device(X, device)
        S_dev = S
        Pa_dev = to_device(P_alpha, device)
        Pb_dev = to_device(P_beta, device)
        Ca_dev = to_device(C_alpha, device)
        Cb_dev = to_device(C_beta, device)
        Fa_dev = copy(H_dev)
        Fb_dev = copy(H_dev)

        for iter in 1:options.max_cycles
            cycles = iter
            Ptot_dev = Pa_dev .+ Pb_dev
            J_dev, _ = GPUKernels.gpu_fock_coulomb_exchange(Ptot_dev, eri_j, eri_k, 1.0)
            _, Ka_dev = GPUKernels.gpu_fock_coulomb_exchange(Pa_dev, eri_j, eri_k, 1.0)
            _, Kb_dev = GPUKernels.gpu_fock_coulomb_exchange(Pb_dev, eri_j, eri_k, 1.0)
            Fa_dev .= H_dev .+ J_dev .- Ka_dev
            Fb_dev .= H_dev .+ J_dev .- Kb_dev

            Fa_host, Fb_host = to_host(Fa_dev), to_host(Fb_dev)
            Pa_host, Pb_host = to_host(Pa_dev), to_host(Pb_dev)
            Fa_work = options.use_diis && iter > 1 ?
                diis_update!(diis_alpha, Fa_host, Pa_host, S_dev) : Fa_host
            Fb_work = options.use_diis && iter > 1 ?
                diis_update!(diis_beta, Fb_host, Pb_host, S_dev) : Fb_host

            eig_a = eigen(Symmetric(X' * Fa_work * X))
            eig_b = eigen(Symmetric(X' * Fb_work * X))
            eps_alpha, eps_beta = eig_a.values, eig_b.values
            C_alpha, C_beta = X * eig_a.vectors, X * eig_b.vectors
            Ca_dev, Cb_dev = to_device(C_alpha, device), to_device(C_beta, device)
            Pa_new = to_device(_density_matrix(C_alpha, n_alpha), device)
            Pb_new = to_device(_density_matrix(C_beta, n_beta), device)

            energy = sum(Pa_dev .* (H_dev .+ Fa_dev)) / 2 +
                     sum(Pb_dev .* (H_dev .+ Fb_dev)) / 2 + e_nuc
            d_p = max(maximum(abs.(to_host(Pa_new .- Pa_dev))),
                      maximum(abs.(to_host(Pb_new .- Pb_dev))))
            d_e = isfinite(last_energy) ? abs(Float64(energy) - last_energy) : Inf
            if options.damping > 0
                Pa_dev .= (1 - options.damping) .* Pa_new .+ options.damping .* Pa_dev
                Pb_dev .= (1 - options.damping) .* Pb_new .+ options.damping .* Pb_dev
            else
                Pa_dev .= Pa_new
                Pb_dev .= Pb_new
            end
            last_energy = Float64(energy)
            if iter > 1 && d_p < options.convergence_threshold &&
               d_e < options.convergence_threshold
                converged = true
                break
            end
        end
        synchronize_device(device)
        P_alpha, P_beta = to_host(Pa_dev), to_host(Pb_dev)
        C_alpha, C_beta = to_host(Ca_dev), to_host(Cb_dev)
        F_alpha, F_beta = to_host(Fa_dev), to_host(Fb_dev)
    else
        for iter in 1:options.max_cycles
            cycles = iter
            J, _ = _build_jk_cpu(P_alpha + P_beta, eri)
            _, K_alpha = _build_jk_cpu(P_alpha, eri)
            _, K_beta = _build_jk_cpu(P_beta, eri)
            F_alpha .= H_core .+ J .- K_alpha
            F_beta .= H_core .+ J .- K_beta

            Fa_work = options.use_diis && iter > 1 ?
                diis_update!(diis_alpha, F_alpha, P_alpha, S) : F_alpha
            Fb_work = options.use_diis && iter > 1 ?
                diis_update!(diis_beta, F_beta, P_beta, S) : F_beta
            eig_a = eigen(Symmetric(X' * Fa_work * X))
            eig_b = eigen(Symmetric(X' * Fb_work * X))
            eps_alpha, eps_beta = eig_a.values, eig_b.values
            C_alpha, C_beta = X * eig_a.vectors, X * eig_b.vectors
            Pa_new = _density_matrix(C_alpha, n_alpha)
            Pb_new = _density_matrix(C_beta, n_beta)

            energy = sum(P_alpha .* (H_core .+ F_alpha)) / 2 +
                     sum(P_beta .* (H_core .+ F_beta)) / 2 + e_nuc
            d_p = max(maximum(abs.(Pa_new .- P_alpha)),
                      maximum(abs.(Pb_new .- P_beta)))
            d_e = isfinite(last_energy) ? abs(energy - last_energy) : Inf
            if options.damping > 0
                P_alpha .= (1 - options.damping) .* Pa_new .+ options.damping .* P_alpha
                P_beta .= (1 - options.damping) .* Pb_new .+ options.damping .* P_beta
            else
                P_alpha .= Pa_new
                P_beta .= Pb_new
            end
            last_energy = energy
            if iter > 1 && d_p < options.convergence_threshold &&
               d_e < options.convergence_threshold
                converged = true
                break
            end
        end
    end

    J_final, _ = _build_jk_cpu(P_alpha + P_beta, eri)
    _, K_alpha_final = _build_jk_cpu(P_alpha, eri)
    _, K_beta_final = _build_jk_cpu(P_beta, eri)
    F_alpha .= H_core .+ J_final .- K_alpha_final
    F_beta .= H_core .+ J_final .- K_beta_final
    electronic_energy = 0.5 * sum(P_alpha .* (H_core .+ F_alpha)) +
                        0.5 * sum(P_beta .* (H_core .+ F_beta))
    total_energy = electronic_energy + e_nuc
    P_total = P_alpha + P_beta
    n_occ = n_alpha
    return HFResult(converged, cycles, total_energy, total_energy * HARTREE_TO_EV,
                    electronic_energy, e_nuc, eps_alpha, C_alpha, P_total,
                    F_alpha, H_core, S, eri, bs, nuclear_charges, :uhf,
                    mol.multiplicity, n_alpha, n_beta, eps_alpha, eps_beta,
                    C_alpha, C_beta, P_alpha, P_beta, F_alpha, F_beta,
                    n_occ, nb - n_alpha)
end

function print_hf_summary(res::HFResult)
    println("==================================================")
    println("           Ab Initio Hartree-Fock (RHF)           ")
    println("==================================================")
    @printf("  Basis Set:                %s (%d basis functions)\n", res.basis.name, n_basis(res.basis))
    @printf("  Reference / multiplicity:  %s / %d\n", uppercase(string(res.reference)), res.multiplicity)
    @printf("  Alpha / beta electrons:    %d / %d\n", res.n_alpha, res.n_beta)
    @printf("  Convergence:              %s (%d iterations)\n", res.converged ? "CONVERGED" : "FAILED", res.cycles)
    @printf("  Electronic Energy:        %16.8f a.u.\n", res.electronic_energy_hartree)
    @printf("  Nuclear Repulsion Energy: %16.8f a.u.\n", res.nuclear_repulsion_hartree)
    @printf("  TOTAL HARTREE-FOCK ENERGY:%16.8f a.u. (%12.4f eV)\n", res.total_energy_hartree, res.total_energy_ev)
    println("--------------------------------------------------")
    println("  Frontier Molecular Orbitals (a.u. / eV):")
    for i in 1:n_basis(res.basis)
        tag = (i == res.n_occ) ? " [HOMO]" : (i == res.n_occ + 1) ? " [LUMO]" : ""
        ev = res.mo_energies_hartree[i] * HARTREE_TO_EV
        @printf("    MO #%2d:  %12.6f a.u. (%9.3f eV)%s\n", i, res.mo_energies_hartree[i], ev, tag)
    end
    println("==================================================")
end

end # module HartreeFock
