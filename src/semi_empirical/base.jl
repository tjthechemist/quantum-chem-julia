# ==============================================================================
# QuantumChemMM.jl: Semi-Empirical Base Types, SCF Options, DIIS Engine
# ==============================================================================

module SemiEmpiricalBase

using LinearAlgebra
using Printf

export AbstractSemiEmpiricalMethod, SCFOptions, SCFResult,
       DIISController, diis_update!, print_scf_summary

abstract type AbstractSemiEmpiricalMethod end

struct SCFOptions
    max_cycles::Int
    convergence_threshold::Float64 # in Hartrees or eV
    use_diis::Bool
    diis_size::Int
    damping::Float64               # 0.0 = no damping, 0.5 = 50% new + 50% old
    verbose::Bool
end

function SCFOptions(; max_cycles=100, convergence_threshold=1e-7,
                      use_diis=true, diis_size=6, damping=0.0, verbose=false)
    SCFOptions(max_cycles, convergence_threshold, use_diis, diis_size, damping, verbose)
end

struct SCFResult
    converged::Bool
    cycles::Int
    total_energy::Float64          # in eV (or Hartrees)
    electronic_energy::Float64
    nuclear_repulsion::Float64
    mo_energies::Vector{Float64}
    mo_coefficients::Matrix{Float64}
    density_matrix::Matrix{Float64}
    fock_matrix::Matrix{Float64}
    overlap_matrix::Matrix{Float64}
    mulliken_charges::Vector{Float64}
    dipole_moment::Vector{Float64} # in Debye
    energy_unit::String            # "eV" or "Hartree"
end

# Direct Inversion in the Iterative Subspace (Pulay's DIIS)
mutable struct DIISController
    fock_history::Vector{Matrix{Float64}}
    error_history::Vector{Matrix{Float64}}
    max_subspace::Int
end

DIISController(max_subspace::Int=6) = DIISController(Matrix{Float64}[], Matrix{Float64}[], max_subspace)

"""
    diis_update!(diis::DIISController, F::Matrix{Float64}, P::Matrix{Float64}, S::Matrix{Float64})
Updates DIIS subspace and returns extrapolated Fock matrix F_diis.
Error vector e = F*P*S - S*P*F (commutator).
"""
function diis_update!(diis::DIISController, F::Matrix{Float64}, P::Matrix{Float64},
                      S::AbstractMatrix{<:Real})::Matrix{Float64}
    # Commutator error matrix e = F*P*S - S*P*F
    e = F * P * S - S * P * F

    push!(diis.fock_history, copy(F))
    push!(diis.error_history, copy(e))

    # Maintain maximum subspace size
    if length(diis.fock_history) > diis.max_subspace
        popfirst!(diis.fock_history)
        popfirst!(diis.error_history)
    end

    n = length(diis.fock_history)
    if n < 2
        return F
    end

    # Build Pulay's B matrix
    # [ B   -1 ] [ c ]   [ 0 ]
    # [-1'   0 ] [ λ ] = [-1 ]
    B = zeros(Float64, n + 1, n + 1)
    for i in 1:n
        for j in 1:n
            B[i, j] = dot(diis.error_history[i], diis.error_history[j])
        end
        B[i, n + 1] = -1.0
        B[n + 1, i] = -1.0
    end
    B[n + 1, n + 1] = 0.0

    rhs = zeros(Float64, n + 1)
    rhs[n + 1] = -1.0

    # Solve for DIIS weights
    coeffs = try
        B \ rhs
    catch
        # If singular or ill-conditioned, fallback to most recent Fock matrix
        return F
    end

    # Extrapolate Fock matrix: F_diis = sum_i c_i * F_i
    F_diis = zeros(Float64, size(F))
    for i in 1:n
        F_diis .+= coeffs[i] .* diis.fock_history[i]
    end

    return F_diis
end

function print_scf_summary(res::SCFResult)
    println("==================================================")
    println("           Semi-Empirical SCF Results             ")
    println("==================================================")
    @printf("  Convergence Status:       %s (in %d iterations)\n", res.converged ? "CONVERGED" : "FAILED", res.cycles)
    @printf("  Electronic Energy:        %16.8f %s\n", res.electronic_energy, res.energy_unit)
    @printf("  Nuclear Repulsion Energy: %16.8f %s\n", res.nuclear_repulsion, res.energy_unit)
    @printf("  TOTAL ENERGY:             %16.8f %s\n", res.total_energy, res.energy_unit)
    dip_norm = norm(res.dipole_moment)
    @printf("  Dipole Moment:            %16.4f Debye (x: %.3f, y: %.3f, z: %.3f)\n",
            dip_norm, res.dipole_moment[1], res.dipole_moment[2], res.dipole_moment[3])
    println("--------------------------------------------------")
    println("  Frontier Molecular Orbital Energies:")
    n_mo = length(res.mo_energies)
    n_show = min(8, n_mo)
    for i in 1:n_show
        @printf("    MO #%2d:  %12.4f %s\n", i, res.mo_energies[i], res.energy_unit)
    end
    println("==================================================")
end

end # module SemiEmpiricalBase
