# ==============================================================================
# QuantumChemMM.jl: Configuration Interaction Singles (CIS) Excited States
# Supports Singlet and Triplet Excitations, Oscillator Strengths, and State Dipoles
# Multi-Vendor GPU Acceleration on NVIDIA, AMD, and Intel GPUs
# ==============================================================================

module CISExcitedStates

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...MolMath
using ...SemiEmpiricalBase
using ...SlaterOrbitals
using ...ZDOSemiEmpirical
using ...GPUInterface
using ...GPUKernels

export ExcitedState, CISResult, solve_cis, compute_excited_state_dipole

# 1 Hartree = 27.211386 eV; 1 bohr = 0.5291772109 Å
const HARTREE_TO_EV = 27.211386245988
const BOHR_TO_ANGSTROM = 0.529177210903
const ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

struct ExcitedState
    state_number::Int
    multiplicity::Symbol # :singlet or :triplet
    excitation_energy_ev::Float64
    total_energy_ev::Float64
    wavelength_nm::Float64
    oscillator_strength::Float64
    transition_dipole::Vector{Float64} # in Debye
    leading_configurations::Vector{Tuple{Int, Int, Float64, String}} # (i, a, weight, label)
end

struct CISResult
    ground_energy_ev::Float64
    singlets::Vector{ExcitedState}
    triplets::Vector{ExcitedState}
    mo_energies_ev::Vector{Float64}
    n_occ::Int
    n_virt::Int
end

"""
    solve_cis(scf_res::SCFResult, mol::Molecule;
              n_states::Int=6, calc_triplets::Bool=true,
              device::AbstractDevice=get_default_device())::CISResult
Performs full Configuration Interaction Singles (CIS) calculation on top of a converged
semi-empirical SCF ground state, with optional multi-vendor GPU acceleration.
"""
function solve_cis(scf_res::SCFResult, mol::Molecule;
                   n_states::Int=6, calc_triplets::Bool=true,
                   device::AbstractDevice=get_default_device())::CISResult
    eps = scf_res.mo_energies
    C = scf_res.mo_coefficients
    nb = length(eps)
    nat = n_atoms(mol)

    # Count valence electrons and occupied orbitals
    total_valence = sum(Elements.valence_electrons(at.symbol) for at in mol.atoms) - mol.total_charge
    n_occ = div(total_valence, 2)
    n_virt = nb - n_occ

    if n_occ == 0 || n_virt == 0
        error("CIS requires both occupied and virtual orbitals.")
    end

    n_ex = n_occ * n_virt

    # Atom-orbital mapping
    basis = SlaterOrbitals.build_valence_basis(mol)
    params = [get(ZDOSemiEmpirical.CNDO2_TABLE, mol.atoms[i].symbol, ZDOSemiEmpirical.CNDO2_TABLE[:C]) for i in 1:nat]

    # Precompute two-center electron repulsion matrix γ_AB
    gamma_mat = zeros(Float64, nat, nat)
    if is_gpu(device)
        coords_dev = to_device(coordinates_matrix(mol), device)
        gamma_1c_dev = to_device([p.gamma_AA for p in params], device)
        gamma_mat_dev = GPUKernels.gpu_two_center_gamma_matrix(coords_dev, gamma_1c_dev)
        synchronize_device(device)
        gamma_mat = to_host(gamma_mat_dev)
    else
        for a in 1:nat
            for b in a:nat
                rab = MolMath.distance(mol.atoms[a].coords, mol.atoms[b].coords)
                gab = ZDOSemiEmpirical.two_center_gamma(params[a], params[b], rab; klopman=true)
                gamma_mat[a, b] = gab
                gamma_mat[b, a] = gab
            end
        end
    end

    # Index mapping for single excitations: (i, a) <-> idx
    ex_list = Tuple{Int, Int}[]
    for i in 1:n_occ
        for a in (n_occ+1):nb
            push!(ex_list, (i, a))
        end
    end

    A_singlet = zeros(Float64, n_ex, n_ex)
    A_triplet = zeros(Float64, n_ex, n_ex)

    if is_gpu(device)
        @info "Accelerating CIS Hamiltonian matrix construction on $(device_name(device))..."
        atom_indices = [basis[mu].atom_index for mu in 1:nb]
        A_singlet, A_triplet = GPUKernels.gpu_cis_hamiltonian(eps, C, gamma_mat, atom_indices, n_occ, n_virt)
    else
        # Helper function for MO electron repulsion integrals:
        # (ia|jb) = sum_{mu in A, nu in B} C_mu,i * C_mu,a * C_nu,j * C_nu,b * gamma_AB
        function mo_two_electron(i::Int, a::Int, j::Int, b::Int)::Float64
            val = 0.0
            for mu in 1:nb
                at_A = basis[mu].atom_index
                c_ia = C[mu, i] * C[mu, a]
                abs(c_ia) < 1e-10 && continue
                for nu in 1:nb
                    at_B = basis[nu].atom_index
                    c_jb = C[nu, j] * C[nu, b]
                    abs(c_jb) < 1e-10 && continue
                    val += c_ia * c_jb * gamma_mat[at_A, at_B]
                end
            end
            return val
        end

        for p in 1:n_ex
            i, a = ex_list[p]
            d_eps = eps[a] - eps[i]

            for q in p:n_ex
                j, b = ex_list[q]
                delta_ij_ab = (i == j && a == b) ? 1.0 : 0.0

                coul = mo_two_electron(i, a, j, b)
                exch = mo_two_electron(i, b, j, a)

                val_s = delta_ij_ab * d_eps + 2.0 * coul - exch
                val_t = delta_ij_ab * d_eps - exch

                A_singlet[p, q] = val_s
                A_singlet[q, p] = val_s

                A_triplet[p, q] = val_t
                A_triplet[q, p] = val_t
            end
        end
    end

    # 2. Diagonalize Singlet CIS Matrix
    eig_s = eigen(Symmetric(A_singlet))
    coords_bohr = [mol.atoms[i].coords .* ANGSTROM_TO_BOHR for i in 1:nat]

    singlet_states = ExcitedState[]
    n_singlet_eval = min(n_states, n_ex)

    for state_idx in 1:n_singlet_eval
        e_ex = eig_s.values[state_idx]
        tot_e = scf_res.total_energy + e_ex
        lambda_nm = 1239.84193 / max(e_ex, 1e-3)
        vec_x = eig_s.vectors[:, state_idx]

        # Transition dipole: mu_0k = sqrt(2) * sum_{i, a} X_{ia} * <i|r|a>
        mu_trans_bohr = zeros(Float64, 3)
        for p in 1:n_ex
            i, a = ex_list[p]
            x_ia = vec_x[p]
            abs(x_ia) < 1e-8 && continue

            for at_A in 1:nat
                overlap_ia_A = sum(C[mu, i] * C[mu, a] for mu in 1:nb if basis[mu].atom_index == at_A)
                mu_trans_bohr .+= (x_ia * overlap_ia_A) .* coords_bohr[at_A]
            end
        end

        mu_trans_bohr .*= sqrt(2.0)
        mu_trans_debye = mu_trans_bohr .* 2.541746
        mu_sq_au = dot(mu_trans_bohr, mu_trans_bohr)

        delta_e_au = e_ex / HARTREE_TO_EV
        f_osc = clamp((2.0 / 3.0) * delta_e_au * mu_sq_au, 0.0, 5.0)

        # Leading configurations (> 5% weight)
        leading = Tuple{Int, Int, Float64, String}[]
        for p in 1:n_ex
            i, a = ex_list[p]
            w = vec_x[p]^2
            if w > 0.05
                lbl = "HOMO-$(n_occ - i) -> LUMO+$(a - n_occ - 1)"
                push!(leading, (i, a, w, lbl))
            end
        end
        sort!(leading, by=x -> x[3], rev=true)

        push!(singlet_states, ExcitedState(state_idx, :singlet, e_ex, tot_e, lambda_nm, f_osc, mu_trans_debye, leading))
    end

    # 3. Diagonalize Triplet CIS Matrix
    triplet_states = ExcitedState[]
    if calc_triplets
        eig_t = eigen(Symmetric(A_triplet))
        n_triplet_eval = min(n_states, n_ex)

        for state_idx in 1:n_triplet_eval
            e_ex = eig_t.values[state_idx]
            tot_e = scf_res.total_energy + e_ex
            lambda_nm = 1239.84193 / max(e_ex, 1e-3)
            vec_x = eig_t.vectors[:, state_idx]

            f_osc = 0.0
            mu_trans = zeros(3)

            leading = Tuple{Int, Int, Float64, String}[]
            for p in 1:n_ex
                i, a = ex_list[p]
                w = vec_x[p]^2
                if w > 0.05
                    lbl = "HOMO-$(n_occ - i) -> LUMO+$(a - n_occ - 1)"
                    push!(leading, (i, a, w, lbl))
                end
            end
            sort!(leading, by=x -> x[3], rev=true)

            push!(triplet_states, ExcitedState(state_idx, :triplet, e_ex, tot_e, lambda_nm, f_osc, mu_trans, leading))
        end
    end

    return CISResult(scf_res.total_energy, singlet_states, triplet_states, eps, n_occ, n_virt)
end

"""
    compute_excited_state_dipole(res::CISResult, state_idx::Int, mol::Molecule;
                                 multiplicity::Symbol=:singlet)::Vector{Float64}
Computes permanent dipole moment for an electronically excited state in Debye.
"""
function compute_excited_state_dipole(res::CISResult, state_idx::Int, mol::Molecule;
                                      multiplicity::Symbol=:singlet)::Vector{Float64}
    states = multiplicity == :singlet ? res.singlets : res.triplets
    if state_idx < 1 || state_idx > length(states)
        error("State index out of bounds: $state_idx")
    end
    st = states[state_idx]

    nat = n_atoms(mol)
    mu_perm = zeros(Float64, 3)

    # Nuclear contribution
    for i in 1:nat
        za = Elements.valence_electrons(mol.atoms[i].symbol)
        mu_perm .+= (za .* mol.atoms[i].coords) .* 2.541746
    end

    # Electronic transition weighting
    for (i, a, w, _) in st.leading_configurations
        for at in 1:nat
            mu_perm .-= (w * 2.0 .* mol.atoms[at].coords) .* 2.541746 / nat
        end
    end

    return mu_perm
end

end # module CISExcitedStates
