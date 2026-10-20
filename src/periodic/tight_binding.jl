# ==============================================================================
# Periodic one-orbital tight-binding prototype for crystal band structures
# ==============================================================================

module PeriodicTightBinding

using LinearAlgebra
using ..Types

export CrystalStructure, PeriodicTBResult, BandStructureResult,
       solve_periodic_tb, periodic_band_structure

const HARTREE_TO_EV = 27.211386245988

struct CrystalStructure
    name::String
    lattice_vectors::Matrix{Float64}
    fractional_positions::Matrix{Float64}
    species::Vector{Symbol}
    periodic_dimensions::Vector{Int}

    function CrystalStructure(name::String, lattice_vectors::AbstractMatrix{<:Real},
                              fractional_positions::AbstractMatrix{<:Real},
                              species::AbstractVector{Symbol};
                              periodic_dimensions::AbstractVector{<:Integer}=[1, 2, 3])
        size(lattice_vectors) == (3, 3) ||
            throw(DimensionMismatch("lattice_vectors must be a 3x3 matrix with lattice vectors in columns."))
        size(fractional_positions, 1) == 3 ||
            throw(DimensionMismatch("fractional_positions must have three rows."))
        size(fractional_positions, 2) == length(species) ||
            throw(DimensionMismatch("Provide one species for each fractional position."))
        isempty(species) && throw(ArgumentError("A crystal must contain at least one atom in its unit cell."))
        all(isfinite, lattice_vectors) && all(isfinite, fractional_positions) ||
            throw(ArgumentError("Lattice vectors and fractional positions must be finite."))
        abs(det(Matrix{Float64}(lattice_vectors))) > 1e-10 ||
            throw(ArgumentError("The 3D lattice-vector matrix must be nonsingular, including for a 2D slab with vacuum."))
        dimensions = sort!(unique!(Int.(periodic_dimensions)))
        !isempty(dimensions) && all(d -> 1 <= d <= 3, dimensions) ||
            throw(ArgumentError("periodic_dimensions must contain one or more dimensions selected from 1, 2, and 3."))
        new(name, Matrix{Float64}(lattice_vectors),
            Matrix{Float64}(fractional_positions), collect(species), dimensions)
    end
end

CrystalStructure(lattice_vectors::AbstractMatrix{<:Real},
                 fractional_positions::AbstractMatrix{<:Real},
                 species::AbstractVector{Symbol}; name::String="Crystal",
                 periodic_dimensions::AbstractVector{<:Integer}=[1, 2, 3]) =
    CrystalStructure(name, lattice_vectors, fractional_positions, species;
                     periodic_dimensions=periodic_dimensions)

struct PeriodicTBResult
    crystal_name::String
    kpoints_fractional::Matrix{Float64}
    bands_ev::Matrix{Float64}
    electrons_per_cell::Int
    fermi_level_ev::Float64
    band_gap_ev::Float64
    total_energy_ev_per_cell::Float64
    dos_energies_ev::Vector{Float64}
    dos_states_per_ev_per_cell::Vector{Float64}
    kmesh::NTuple{3, Int}
end

struct BandStructureResult
    crystal_name::String
    kpoints_fractional::Matrix{Float64}
    path_distance_per_angstrom::Vector{Float64}
    bands_ev::Matrix{Float64}
    high_symmetry_indices::Vector{Int}
    labels::Vector{String}
end

function _validate_model(crystal::CrystalStructure, hopping_ev::Float64,
                         cutoff_angstrom::Float64, decay_per_angstrom::Float64,
                         reference_distance_angstrom::Float64,
                         dos_sigma_ev::Float64, dos_points::Int)
    all(isfinite, (hopping_ev, cutoff_angstrom, decay_per_angstrom,
                   reference_distance_angstrom, dos_sigma_ev)) ||
        throw(ArgumentError("Tight-binding parameters must be finite."))
    cutoff_angstrom > 0.0 ||
        throw(ArgumentError("cutoff_angstrom must be positive."))
    decay_per_angstrom >= 0.0 ||
        throw(ArgumentError("decay_per_angstrom must be non-negative."))
    reference_distance_angstrom > 0.0 ||
        throw(ArgumentError("reference_distance_angstrom must be positive."))
    dos_sigma_ev > 0.0 ||
        throw(ArgumentError("dos_sigma_ev must be positive."))
    dos_points >= 2 ||
        throw(ArgumentError("dos_points must be at least 2."))
    return nothing
end

function _translation_vectors(crystal::CrystalStructure, cutoff_angstrom::Float64)
    periodic_lattice = crystal.lattice_vectors[:, crystal.periodic_dimensions]
    min_spacing = minimum(svdvals(periodic_lattice))
    nmax = ceil(Int, cutoff_angstrom / min_spacing) + 1
    nmax <= 12 ||
        throw(ArgumentError("Cutoff and lattice spacing require more than 12 periodic images per axis; reduce the cutoff or use a larger unit cell."))
    translations = NTuple{3, Int}[]
    ranges = ntuple(axis -> axis in crystal.periodic_dimensions ? (-nmax:nmax) : (0:0), 3)
    for i in ranges[1], j in ranges[2], k in ranges[3]
        push!(translations, (i, j, k))
    end
    return translations
end

function _pair_hopping(species_i::Symbol, species_j::Symbol, hopping_ev::Float64,
                       hopping_by_pair::AbstractDict)
    key = (species_i, species_j)
    reverse_key = (species_j, species_i)
    value = haskey(hopping_by_pair, key) ? hopping_by_pair[key] :
            haskey(hopping_by_pair, reverse_key) ? hopping_by_pair[reverse_key] :
            hopping_ev
    value isa Real && isfinite(value) ||
        throw(ArgumentError("Pair hopping parameters must be finite real values."))
    return Float64(value)
end

function _hamiltonian(crystal::CrystalStructure, kpoint::AbstractVector{<:Real};
                      onsite_ev::AbstractDict{Symbol, <:Real}=Dict{Symbol, Float64}(),
                      hopping_ev::Float64=-2.7,
                      hopping_by_pair::AbstractDict=Dict{Tuple{Symbol, Symbol}, Float64}(),
                      cutoff_angstrom::Float64=3.0,
                      decay_per_angstrom::Float64=0.0,
                      reference_distance_angstrom::Float64=1.42,
                      translations=_translation_vectors(crystal, cutoff_angstrom))
    length(kpoint) == 3 ||
        throw(DimensionMismatch("Each fractional k-point must have three components."))
    all(isfinite, kpoint) ||
        throw(ArgumentError("k-point coordinates must be finite."))
    nat = length(crystal.species)
    hamiltonian = zeros(ComplexF64, nat, nat)
    for atom in 1:nat
        level = get(onsite_ev, crystal.species[atom], 0.0)
        level isa Real && isfinite(level) ||
            throw(ArgumentError("Onsite energies must be finite real values."))
        hamiltonian[atom, atom] += Float64(level)
    end

    for translation in translations
        translation_vector = collect(translation)
        lattice_shift = crystal.lattice_vectors * translation_vector
        phase = cis(2.0 * π * dot(kpoint, translation_vector))
        for i in 1:nat, j in 1:nat
            (i == j && all(iszero, translation_vector)) && continue
            fractional_delta = crystal.fractional_positions[:, j] .+
                              translation_vector .-
                              crystal.fractional_positions[:, i]
            distance = norm(crystal.lattice_vectors * fractional_delta)
            (distance <= 1e-10 || distance > cutoff_angstrom) && continue
            base_hopping = _pair_hopping(crystal.species[i], crystal.species[j],
                                         hopping_ev, hopping_by_pair)
            hopping = base_hopping *
                      exp(-decay_per_angstrom *
                          (distance - reference_distance_angstrom))
            hamiltonian[i, j] += hopping * phase
        end
    end
    return Hermitian(hamiltonian)
end

function _eigenvalues_at_k(crystal::CrystalStructure, kpoints::Matrix{Float64};
                           kwargs...)
    bands = zeros(Float64, size(kpoints, 2), length(crystal.species))
    for k in axes(kpoints, 2)
        bands[k, :] .= eigvals(_hamiltonian(crystal, @view(kpoints[:, k]); kwargs...))
    end
    return bands
end

function _gamma_kmesh(kmesh::NTuple{3, Int}, periodic_dimensions::Vector{Int})
    all(n -> n > 0, kmesh) ||
        throw(ArgumentError("All kmesh dimensions must be positive."))
    points = Vector{NTuple{3, Float64}}()
    ranges = ntuple(axis -> axis in periodic_dimensions ?
                    ((i - 1) / kmesh[axis] for i in 1:kmesh[axis]) :
                    (0.0,), 3)
    for x in ranges[1], y in ranges[2], z in ranges[3]
        push!(points, (x, y, z))
    end
    return reduce(hcat, (collect(point) for point in points))
end

function _fill_bands(bands_ev::Matrix{Float64}, electrons_per_cell::Int)
    nk, nbands = size(bands_ev)
    0 <= electrons_per_cell <= 2 * nbands ||
        throw(ArgumentError("electrons_per_cell must be between 0 and $(2 * nbands) for a spin-degenerate one-orbital model."))
    electrons_per_cell == 0 &&
        return (0.0, minimum(bands_ev), Inf)
    electrons_per_cell == 2 * nbands &&
        return (2.0 * sum(bands_ev) / nk, maximum(bands_ev), Inf)

    states = sort(vec(copy(bands_ev)))
    electrons_to_fill = electrons_per_cell * nk
    full_states = div(electrons_to_fill, 2)
    remainder = rem(electrons_to_fill, 2)
    energy_sum = 2.0 * sum(@view states[1:full_states])
    remainder > 0 && (energy_sum += remainder * states[full_states + 1])
    total_energy = energy_sum / nk
    fermi_level = remainder > 0 ? states[full_states + 1] :
                  0.5 * (states[full_states] + states[full_states + 1])

    gap = Inf
    if iseven(electrons_per_cell)
        n_occupied_bands = div(electrons_per_cell, 2)
        valence_max = maximum(@view bands_ev[:, n_occupied_bands])
        conduction_min = minimum(@view bands_ev[:, n_occupied_bands + 1])
        gap = max(0.0, conduction_min - valence_max)
    end
    return total_energy, fermi_level, gap
end

"""
    solve_periodic_tb(crystal; electrons_per_cell, kmesh=(6,6,6), ...)

Solve a periodic, spin-degenerate, one-orbital-per-site tight-binding model on a
uniform Gamma-centered k mesh. Energies are in eV, distances in Angstrom, and
the reported energy is per unit cell. This prototype is not periodic DFT and is
not quantitatively parameterized for general materials.
"""
function solve_periodic_tb(crystal::CrystalStructure;
                           electrons_per_cell::Int,
                           kmesh::NTuple{3, Int}=(6, 6, 6),
                           onsite_ev::AbstractDict{Symbol, <:Real}=Dict{Symbol, Float64}(),
                           hopping_ev::Float64=-2.7,
                           hopping_by_pair::AbstractDict=Dict{Tuple{Symbol, Symbol}, Float64}(),
                           cutoff_angstrom::Float64=3.0,
                           decay_per_angstrom::Float64=0.0,
                           reference_distance_angstrom::Float64=1.42,
                           dos_points::Int=500,
                           dos_sigma_ev::Float64=0.08)::PeriodicTBResult
    _validate_model(crystal, hopping_ev, cutoff_angstrom, decay_per_angstrom,
                    reference_distance_angstrom, dos_sigma_ev, dos_points)
    all(n -> n > 0, kmesh) ||
        throw(ArgumentError("All kmesh dimensions must be positive."))
    for axis in 1:3
        !(axis in crystal.periodic_dimensions) && kmesh[axis] != 1 &&
            throw(ArgumentError("Non-periodic kmesh dimensions must equal 1."))
    end
    translations = _translation_vectors(crystal, cutoff_angstrom)
    kpoints = _gamma_kmesh(kmesh, crystal.periodic_dimensions)
    bands = _eigenvalues_at_k(crystal, kpoints; onsite_ev=onsite_ev,
                              hopping_ev=hopping_ev, hopping_by_pair=hopping_by_pair,
                              cutoff_angstrom=cutoff_angstrom,
                              decay_per_angstrom=decay_per_angstrom,
                              reference_distance_angstrom=reference_distance_angstrom,
                              translations=translations)
    total_energy, fermi, gap = _fill_bands(bands, electrons_per_cell)

    energy_min = minimum(bands) - 5.0 * dos_sigma_ev
    energy_max = maximum(bands) + 5.0 * dos_sigma_ev
    dos_energies = collect(range(energy_min, energy_max, length=dos_points))
    dos = zeros(Float64, dos_points)
    prefactor = 2.0 / (size(kpoints, 2) * dos_sigma_ev * sqrt(2.0 * π))
    for band_energy in bands
        for idx in eachindex(dos_energies)
            delta_e = dos_energies[idx] - band_energy
            dos[idx] += prefactor * exp(-0.5 * (delta_e / dos_sigma_ev)^2)
        end
    end

    return PeriodicTBResult(crystal.name, kpoints, bands, electrons_per_cell,
                            fermi, gap, total_energy, dos_energies, dos, kmesh)
end

"""
    periodic_band_structure(crystal, kpath; points_per_segment=30, ...)

Evaluate the tight-binding bands along a list of fractional reciprocal-space
vertices. `kpath` is a vector of 3-component k-point vectors and all coordinates
are fractional reciprocal-lattice coordinates.
"""
function periodic_band_structure(crystal::CrystalStructure,
                                 kpath::AbstractVector{<:AbstractVector};
                                 labels::Vector{String}=fill("", length(kpath)),
                                 points_per_segment::Int=30,
                                 onsite_ev::AbstractDict{Symbol, <:Real}=Dict{Symbol, Float64}(),
                                 hopping_ev::Float64=-2.7,
                                 hopping_by_pair::AbstractDict=Dict{Tuple{Symbol, Symbol}, Float64}(),
                                 cutoff_angstrom::Float64=3.0,
                                 decay_per_angstrom::Float64=0.0,
                                 reference_distance_angstrom::Float64=1.42)
    length(kpath) >= 2 ||
        throw(ArgumentError("kpath must contain at least two vertices."))
    length(labels) == length(kpath) ||
        throw(DimensionMismatch("Provide one label for each kpath vertex."))
    points_per_segment >= 2 ||
        throw(ArgumentError("points_per_segment must be at least 2."))
    all(point -> length(point) == 3 && all(isfinite, point), kpath) ||
        throw(ArgumentError("Each kpath vertex must be a finite 3-component vector."))
    _validate_model(crystal, hopping_ev, cutoff_angstrom, decay_per_angstrom,
                    reference_distance_angstrom, 0.08, 2)
    translations = _translation_vectors(crystal, cutoff_angstrom)

    points = Vector{Vector{Float64}}()
    high_symmetry_indices = Int[]
    for segment in 1:(length(kpath) - 1)
        start = Float64.(kpath[segment])
        stop = Float64.(kpath[segment + 1])
        for point_idx in 0:points_per_segment
            point_idx == 0 && segment > 1 && continue
            fraction = point_idx / points_per_segment
            push!(points, (1.0 - fraction) .* start .+ fraction .* stop)
        end
        push!(high_symmetry_indices, length(points))
    end
    pushfirst!(high_symmetry_indices, 1)
    kpoints = reduce(hcat, points)
    bands = _eigenvalues_at_k(crystal, kpoints; onsite_ev=onsite_ev,
                              hopping_ev=hopping_ev, hopping_by_pair=hopping_by_pair,
                              cutoff_angstrom=cutoff_angstrom,
                              decay_per_angstrom=decay_per_angstrom,
                              reference_distance_angstrom=reference_distance_angstrom,
                              translations=translations)

    reciprocal_lattice = 2.0 * π * inv(crystal.lattice_vectors)'
    distance = zeros(Float64, length(points))
    for idx in 2:length(points)
        cart_delta = reciprocal_lattice * (points[idx] - points[idx - 1])
        distance[idx] = distance[idx - 1] + norm(cart_delta)
    end
    return BandStructureResult(crystal.name, kpoints, distance, bands,
                               high_symmetry_indices, labels)
end

end # module PeriodicTightBinding
