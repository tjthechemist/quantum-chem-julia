# ==============================================================================
# QuantumChemMM.jl: Tamm-Dancoff TDHF and adiabatic TDDFT response
# ==============================================================================

module TimeDependentMethods

using LinearAlgebra
using ..Types
using ..BasisSets
using ..HartreeFock
using ..KohnShamDFT
using ..DFTFunctionals
using ..DFTGrid
using ..IntegralTransform
using ..MolecularIntegrals

export TDExcitedState, TDResult, solve_tdhf, solve_tddft, xc_density_kernel

const HARTREE_TO_EV = 27.211386245988
struct TDExcitedState
    state_number::Int
    multiplicity::Symbol
    excitation_energy_hartree::Float64
    excitation_energy_ev::Float64
    total_energy_hartree::Float64
    wavelength_nm::Float64
    oscillator_strength::Float64
    transition_dipole_debye::Vector{Float64}
    leading_configurations::Vector{Tuple{Int, Int, Float64, String}}
end

struct TDResult
    method::Symbol
    functional::Union{Nothing, String}
    ground_energy_hartree::Float64
    singlets::Vector{TDExcitedState}
    triplets::Vector{TDExcitedState}
    n_occ::Int
    n_virt::Int
end

function xc_density_kernel(functional::Symbol, rho::Float64)::Float64
    rho <= 1e-10 && return 0.0
    xc = if functional == :lda
        LDA_FUNCTIONAL
    elseif functional == :pbe
        PBE_FUNCTIONAL
    elseif functional == :blyp
        BLYP_FUNCTIONAL
    elseif functional == :pbe0
        PBE0_FUNCTIONAL
    elseif functional == :b3lyp
        B3LYP_FUNCTIONAL
    else
        throw(ArgumentError("Unsupported TDDFT functional: $functional"))
    end
    step = max(1e-6, rho * 1e-4)
    rho_lo = max(rho - step, 1e-10)
    rho_hi = rho + step
    v_lo = DFTFunctionals.evaluate_xc(xc, rho_lo)[2]
    v_hi = DFTFunctionals.evaluate_xc(xc, rho_hi)[2]
    return (v_hi - v_lo) / (rho_hi - rho_lo)
end

function _functional_symbol(name::String)
    key = lowercase(replace(name, r"[^a-zA-Z0-9]" => ""))
    occursin("pbe0", key) && return :pbe0
    occursin("b3lyp", key) && return :b3lyp
    occursin("blyp", key) && return :blyp
    occursin("pbe", key) && return :pbe
    occursin("lda", key) && return :lda
    throw(ArgumentError("Cannot map DFT functional '$name' to an implemented response kernel."))
end

function _integrate_transition_properties(mol::Molecule, basis::BasisSet,
                                          C::Matrix{Float64}, n_occ::Int, excitations,
                                          functional::Union{Nothing, Symbol})
    grid = DFTGrid.build_molecular_grid(mol; n_radial=18, n_angular=26)
    ntrans = length(excitations)
    kernel = zeros(Float64, ntrans, ntrans)
    transition_dipoles = zeros(Float64, 3, ntrans)
    chi = zeros(Float64, n_basis(basis))
    phi_i = zeros(Float64, ntrans)
    phi_a = zeros(Float64, ntrans)
    transition_density = zeros(Float64, ntrans)

    for point in grid.points
        for mu in eachindex(chi)
            chi[mu] = DFTGrid.eval_basis_function(basis.functions[mu], point.coords)
        end
        for (t, (i, a)) in enumerate(excitations)
            phi_i[t] = dot(chi, @view C[:, i])
            phi_a[t] = dot(chi, @view C[:, a])
            transition_density[t] = phi_i[t] * phi_a[t]
            transition_dipoles[:, t] .+= point.weight .* point.coords .* transition_density[t]
        end
        if functional !== nothing
            rho = 0.0
            for mu in eachindex(chi), nu in eachindex(chi)
                rho += 2.0 * chi[mu] * chi[nu] *
                       sum(C[mu, i] * C[nu, i] for i in 1:n_occ)
            end
            fxc = xc_density_kernel(functional, max(rho, 1e-10))
            kernel .+= (2.0 * point.weight * fxc) .*
                       (transition_density * transition_density')
        end
    end

    return kernel, transition_dipoles
end

function _response_matrices(eps::Vector{Float64},
                            eri::Array{Float64, 4}, n_occ::Int,
                            exact_exchange::Float64, kernel::Matrix{Float64},
                            calc_triplets::Bool)
    nmo = length(eps)
    excitations = [(i, a) for i in 1:n_occ for a in (n_occ + 1):nmo]
    ntrans = length(excitations)
    ntrans > 0 || throw(ArgumentError("TDA requires both occupied and virtual orbitals."))
    A_singlet = zeros(Float64, ntrans, ntrans)
    A_triplet = calc_triplets ? zeros(Float64, ntrans, ntrans) : zeros(Float64, 0, 0)

    for p in 1:ntrans
        i, a = excitations[p]
        for q in p:ntrans
            j, b = excitations[q]
            diagonal = p == q ? eps[a] - eps[i] : 0.0
            coulomb = eri[i, a, j, b]
            exchange = eri[i, j, a, b]
            singlet = diagonal + 2.0 * coulomb - exact_exchange * exchange + kernel[p, q]
            A_singlet[p, q] = singlet
            A_singlet[q, p] = singlet
            if calc_triplets
                triplet = diagonal - exact_exchange * exchange
                A_triplet[p, q] = triplet
                A_triplet[q, p] = triplet
            end
        end
    end
    return excitations, A_singlet, A_triplet
end

function _make_states(eig, excitations, dipoles, ground_energy::Float64,
                      multiplicity::Symbol, n_states::Int; optical::Bool)
    states = TDExcitedState[]
    n_eval = min(n_states, length(eig.values))
    for state_idx in 1:n_eval
        energy = eig.values[state_idx]
        energy > 0 ||
            throw(ArgumentError("TDA response has a non-positive excitation energy ($energy Hartree); inspect SCF stability/reference."))
        vector = eig.vectors[:, state_idx]
        transition_dipole_au = zeros(3)
        if optical
            transition_dipole_au .= sqrt(2.0) .* (dipoles * vector)
        end
        transition_dipole_debye = transition_dipole_au .* 2.541746
        oscillator = optical ?
            (2.0 / 3.0) * energy * dot(transition_dipole_au, transition_dipole_au) : 0.0
        leading = Tuple{Int, Int, Float64, String}[]
        for (config_idx, (i, a)) in enumerate(excitations)
            weight = vector[config_idx]^2
            if weight > 0.05
                push!(leading, (i, a, weight, "MO $i -> MO $a"))
            end
        end
        if isempty(leading)
            config_idx = argmax(abs2.(vector))
            i, a = excitations[config_idx]
            push!(leading, (i, a, vector[config_idx]^2, "MO $i -> MO $a"))
        end
        sort!(leading, by=x -> x[3], rev=true)
        push!(states, TDExcitedState(state_idx, multiplicity, energy,
                                    energy * HARTREE_TO_EV,
                                    ground_energy + energy,
                                    1239.84193 / (energy * HARTREE_TO_EV),
                                    oscillator,
                                    transition_dipole_debye, leading))
    end
    return states
end

function _solve_tda(mol::Molecule, eps::Vector{Float64}, C::Matrix{Float64},
                    eri_ao::Array{Float64, 4}, basis::BasisSet, n_occ::Int,
                    ground_energy::Float64; exact_exchange::Float64,
                    functional::Union{Nothing, Symbol}, method::Symbol,
                    n_states::Int, calc_triplets::Bool)
    n_states > 0 || throw(ArgumentError("n_states must be positive."))
    n_occ > 0 && n_occ < length(eps) ||
        throw(ArgumentError("TDA requires at least one occupied and one virtual orbital."))
    eri_mo = transform_eri_ao_to_mo(eri_ao, C)
    excitations = [(i, a) for i in 1:n_occ for a in (n_occ + 1):length(eps)]
    kernel, dipoles =
        _integrate_transition_properties(mol, basis, C, n_occ, excitations, functional)
    excitation_list, A_singlet, A_triplet =
        _response_matrices(eps, eri_mo, n_occ, exact_exchange, kernel, calc_triplets)

    singlet_eig = eigen(Symmetric(A_singlet))
    singlets = _make_states(singlet_eig, excitation_list, dipoles,
                            ground_energy, :singlet, n_states; optical=true)
    triplets = TDExcitedState[]
    if calc_triplets
        triplet_eig = eigen(Symmetric(A_triplet))
        triplets = _make_states(triplet_eig, excitation_list, dipoles,
                                ground_energy, :triplet, n_states; optical=false)
    end
    return TDResult(method, functional === nothing ? nothing : string(functional),
                    ground_energy, singlets, triplets, n_occ,
                    length(eps) - n_occ)
end

function solve_tdhf(res::HFResult, mol::Molecule; n_states::Int=6,
                    calc_triplets::Bool=true)
    res.converged ||
        throw(ArgumentError("TDHF requires a converged RHF reference."))
    res.reference == :rhf ||
        throw(ArgumentError("TDHF currently requires a closed-shell RHF reference."))
    return _solve_tda(mol, res.mo_energies_hartree, res.mo_coefficients,
                      res.eri_tensor, res.basis, res.n_occ,
                      res.total_energy_hartree; exact_exchange=1.0,
                      functional=nothing, method=:tdhf, n_states=n_states,
                      calc_triplets=calc_triplets)
end

function solve_tddft(res::DFTResult, mol::Molecule; n_states::Int=6,
                     functional::Symbol=_functional_symbol(res.functional),
                     calc_triplets::Bool=false)
    res.converged ||
        throw(ArgumentError("TDDFT requires a converged restricted DFT reference."))
    _functional_symbol(res.functional) == functional ||
        throw(ArgumentError("Response functional '$functional' does not match the ground-state functional '$(res.functional)'."))
    calc_triplets &&
        throw(ArgumentError("Triplet TDDFT response is not implemented yet; use TDHF for triplet TDA."))
    c_x = functional == :b3lyp ? 0.20 :
          functional == :pbe0 ? 0.25 : 0.0
    return _solve_tda(mol, res.mo_energies_hartree, res.mo_coefficients,
                      compute_eri_tensor(res.basis),
                      res.basis, res.n_occ, res.total_energy_hartree;
                      exact_exchange=c_x, functional=functional,
                      method=:tddft, n_states=n_states,
                      calc_triplets=calc_triplets)
end

end # module TimeDependentMethods
