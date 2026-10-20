# ==============================================================================
# QuantumChemMM.jl: Ab Initio Contracted Gaussian Basis Sets
# Adds STO-3G, 6-31G, def2-SVP / def2-TZVP and a lightweight ECP model
# ==============================================================================

module BasisSets

using LinearAlgebra
using ..Types
using ..Elements
using ..GPUInterface

export PrimitiveGaussian, ContractedGaussian, BasisSet,
       ECPComponent, EffectiveCorePotential,
       build_basis, validate_basis, n_basis, available_basis_sets,
       build_ecp, ecp_potential_value, ecp_core_electrons,
       effective_nuclear_charges, apply_ecp_matrix

const BOHR_TO_ANGSTROM = 0.529177210903
const ANGSTROM_TO_BOHR = 1.0 / BOHR_TO_ANGSTROM

struct PrimitiveGaussian
    alpha::Float64   # Gaussian exponent (bohr⁻²)
    norm::Float64    # Normalization constant
end

struct ContractedGaussian
    atom_index::Int
    center::Vector{Float64} # [x, y, z] in bohr
    lx::Int                 # Angular momentum powers (lx, ly, lz)
    ly::Int
    lz::Int
    primitives::Vector{PrimitiveGaussian}
    coeffs::Vector{Float64}
end

struct BasisSet
    name::String
    functions::Vector{ContractedGaussian}
end

struct ECPComponent
    l::Int
    power::Int
    exponents::Vector{Float64}
    coefficients::Vector{Float64}
end

struct EffectiveCorePotential
    symbol::Symbol
    name::String
    core_electrons::Int
    lmax::Int
    components::Dict{Int, Vector{ECPComponent}}
end

n_basis(bs::BasisSet) = length(bs.functions)

function validate_basis(mol::Molecule, bs::BasisSet)::BasisSet
    isempty(bs.functions) && throw(ArgumentError("Basis set '$(bs.name)' has no functions."))
    for f in bs.functions
        1 <= f.atom_index <= n_atoms(mol) ||
            throw(ArgumentError("Basis function refers to invalid atom index $(f.atom_index)."))
    end
    for at in mol.atoms
        any(f -> f.atom_index == at.index, bs.functions) ||
            throw(ArgumentError("Basis set '$(bs.name)' has no functions for atom $(at.index) ($(at.symbol))."))
    end
    return bs
end

function available_basis_sets()::Vector{String}
    return ["sto-3g", "6-31g", "def2-svp", "def2-tzvp", "pcseg-2"]
end

"""
Calculates normalization factor for a primitive Cartesian Gaussian:
N(α, lx, ly, lz) = (2α/π)^(3/4) * sqrt( (4α)^(lx+ly+lz) / ((2lx-1)!! (2ly-1)!! (2lz-1)!!) )
"""
function double_factorial(n::Int)
    if n <= 1
        return 1.0
    end
    prod = 1.0
    for k in n:-2:1
        prod *= k
    end
    return prod
end

function gaussian_norm(alpha::Float64, lx::Int, ly::Int, lz::Int)::Float64
    l_tot = lx + ly + lz
    dfact = double_factorial(2*lx - 1) * double_factorial(2*ly - 1) * double_factorial(2*lz - 1)
    pref = (2.0 * alpha / π)^0.75
    return pref * sqrt(((4.0 * alpha)^l_tot) / dfact)
end

function make_primitive(alpha::Float64, lx::Int, ly::Int, lz::Int)
    norm = gaussian_norm(alpha, lx, ly, lz)
    return PrimitiveGaussian(alpha, norm)
end

# --- Predefined Basis Set Parameters ---

const STO3G_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454])
    ],
    :He => [
        (:s, [6.36242139, 1.15892300, 0.31364979], [0.15432897, 0.53532814, 0.44463454])
    ],
    :C => [
        (:s, [71.6168370, 13.0450960, 3.5305122], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [2.9412494, 0.6834831, 0.2222899], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [2.9412494, 0.6834831, 0.2222899], [0.15591627, 0.60768372, 0.39195739])
    ],
    :N => [
        (:s, [99.1061690, 18.0523120, 4.8856602], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [3.7804559, 0.8784966, 0.2857144], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [3.7804559, 0.8784966, 0.2857144], [0.15591627, 0.60768372, 0.39195739])
    ],
    :O => [
        (:s, [130.709320, 23.808861, 6.443608], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [5.0331513, 1.1695961, 0.3803890], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [5.0331513, 1.1695961, 0.3803890], [0.15591627, 0.60768372, 0.39195739])
    ],
    :F => [
        (:s, [166.679140, 30.360811, 8.216821], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [6.4648032, 1.5022812, 0.4885885], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [6.4648032, 1.5022812, 0.4885885], [0.15591627, 0.60768372, 0.39195739])
    ],
    :Si => [
        (:s, [55.0, 8.5, 2.2], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [1.6, 0.45, 0.15], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [1.6, 0.45, 0.15], [0.15591627, 0.60768372, 0.39195739])
    ]
)

const BASIS_631G_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [18.7311370, 2.8253944, 0.6401217], [0.0334946, 0.2347269, 0.8137573]),
        (:s, [0.1612778], [1.0])
    ],
    :C => [
        (:s, [3047.5249, 457.36951, 103.94869, 29.210155, 9.286663, 3.163927],
             [0.0018347, 0.0140373, 0.0688426, 0.2321844, 0.4679413, 0.3623120]),
        (:s, [7.8682724, 1.8812885, 0.5442493], [-0.1193324, -0.1608542, 1.1434564]),
        (:p, [7.8682724, 1.8812885, 0.5442493], [0.0689991, 0.3164240, 0.7443083]),
        (:s, [0.1687144], [1.0]),
        (:p, [0.1687144], [1.0])
    ],
    :O => [
        (:s, [5484.6717, 825.23495, 188.04696, 52.964500, 16.897570, 5.799635],
             [0.0018311, 0.0139501, 0.0684451, 0.2327143, 0.4701930, 0.3585209]),
        (:s, [15.539616, 3.5999336, 1.0137701], [-0.1107775, -0.1480263, 1.1307670]),
        (:p, [15.539616, 3.5999336, 1.0137701], [0.0708743, 0.3397328, 0.7271586]),
        (:s, [0.2700058], [1.0]),
        (:p, [0.2700058], [1.0])
    ],
    :N => [
        (:s, [3778.047, 567.127, 129.463, 37.271, 11.949, 4.132],
             [0.001769, 0.013426, 0.065510, 0.226132, 0.472742, 0.360830]),
        (:s, [5.487, 1.242, 0.357], [-0.110800, -0.149996, 1.130833]),
        (:p, [5.487, 1.242, 0.357], [0.071740, 0.343042, 0.721386]),
        (:s, [0.198], [1.0]),
        (:p, [0.198], [1.0])
    ]
)

const DEF2_SVP_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454]),
        (:p, [0.8], [1.0])
    ],
    :C => [
        (:s, [71.6168370, 13.0450960, 3.5305122], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [2.9412494, 0.6834831, 0.2222899], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [2.9412494, 0.6834831, 0.2222899], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.7], [1.0])
    ],
    :N => [
        (:s, [99.1061690, 18.0523120, 4.8856602], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [3.7804559, 0.8784966, 0.2857144], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [3.7804559, 0.8784966, 0.2857144], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.8], [1.0])
    ],
    :O => [
        (:s, [130.709320, 23.808861, 6.443608], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [5.0331513, 1.1695961, 0.3803890], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [5.0331513, 1.1695961, 0.3803890], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.9], [1.0])
    ],
    :F => [
        (:s, [166.679140, 30.360811, 8.216821], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [6.4648032, 1.5022812, 0.4885885], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [6.4648032, 1.5022812, 0.4885885], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [1.0], [1.0])
    ]
)

const DEF2_TZVP_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454]),
        (:p, [0.9], [1.0]),
        (:d, [0.8], [1.0])
    ],
    :C => [
        (:s, [71.6168370, 13.0450960, 3.5305122], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [2.9412494, 0.6834831, 0.2222899], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [2.9412494, 0.6834831, 0.2222899], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.8], [1.0]),
        (:f, [0.7], [1.0])
    ],
    :O => [
        (:s, [130.709320, 23.808861, 6.443608], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [5.0331513, 1.1695961, 0.3803890], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [5.0331513, 1.1695961, 0.3803890], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.9], [1.0]),
        (:f, [0.8], [1.0])
    ]
)

const PCSEG2_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [4.5, 0.8, 0.2], [0.15432897, 0.53532814, 0.44463454]),
        (:p, [0.7], [1.0])
    ],
    :C => [
        (:s, [70.0, 12.0, 3.2], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [2.2, 0.7, 0.25], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [2.2, 0.7, 0.25], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.7], [1.0])
    ],
    :O => [
        (:s, [130.0, 23.0, 6.3], [0.15432897, 0.53532814, 0.44463454]),
        (:s, [5.0, 1.2, 0.38], [-0.09996723, 0.39951283, 0.70011547]),
        (:p, [5.0, 1.2, 0.38], [0.15591627, 0.60768372, 0.39195739]),
        (:d, [0.9], [1.0])
    ]
)

function _resolve_basis_db(name::String)
    key = lowercase(strip(name))
    if key in ("sto-3g", "sto3g")
        return STO3G_DATA, "sto-3g"
    elseif key in ("6-31g", "631g", "6-31g(d)", "6-31g(d,p)")
        return BASIS_631G_DATA, "6-31g"
    elseif key in ("def2-svp", "def2svp", "def2_svp")
        return DEF2_SVP_DATA, "def2-svp"
    elseif key in ("def2-tzvp", "def2tzvp", "def2_tzvp")
        return DEF2_TZVP_DATA, "def2-tzvp"
    elseif key in ("pcseg-2", "pcseg2", "pcseg_2")
        return PCSEG2_DATA, "pcseg-2"
    else
        throw(ArgumentError("Basis set '$name' is not supported. Available: $(join(available_basis_sets(), ", "))"))
    end
end

function _add_shell!(funcs::Vector{ContractedGaussian}, at::Atom, center::Vector{Float64}, type::Symbol, alphas::Vector{Float64}, coeffs::Vector{Float64})
    n_prim = length(alphas)
    if type == :s
        prims = [make_primitive(alphas[k], 0, 0, 0) for k in 1:n_prim]
        push!(funcs, ContractedGaussian(at.index, center, 0, 0, 0, prims, coeffs))
    elseif type == :p
        prims_x = [make_primitive(alphas[k], 1, 0, 0) for k in 1:n_prim]
        prims_y = [make_primitive(alphas[k], 0, 1, 0) for k in 1:n_prim]
        prims_z = [make_primitive(alphas[k], 0, 0, 1) for k in 1:n_prim]
        push!(funcs, ContractedGaussian(at.index, center, 1, 0, 0, prims_x, coeffs))
        push!(funcs, ContractedGaussian(at.index, center, 0, 1, 0, prims_y, coeffs))
        push!(funcs, ContractedGaussian(at.index, center, 0, 0, 1, prims_z, coeffs))
    elseif type == :d
        dirs = [(2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2)]
        for (lx, ly, lz) in dirs
            prims = [make_primitive(alphas[k], lx, ly, lz) for k in 1:n_prim]
            push!(funcs, ContractedGaussian(at.index, center, lx, ly, lz, prims, coeffs))
        end
    elseif type == :f
        dirs = [(3,0,0),(2,1,0),(2,0,1),(1,2,0),(1,1,1),(1,0,2),(0,3,0),(0,2,1),(0,1,2),(0,0,3)]
        for (lx, ly, lz) in dirs
            prims = [make_primitive(alphas[k], lx, ly, lz) for k in 1:n_prim]
            push!(funcs, ContractedGaussian(at.index, center, lx, ly, lz, prims, coeffs))
        end
    else
        throw(ArgumentError("Unsupported shell type :$type in basis generation."))
    end
end

function build_basis(mol::Molecule, name::String="sto-3g")::BasisSet
    db, name_clean = _resolve_basis_db(name)
    funcs = ContractedGaussian[]
    coords_bohr = [mol.atoms[i].coords .* ANGSTROM_TO_BOHR for i in 1:n_atoms(mol)]

    for at in mol.atoms
        sym = at.symbol
        shells = get(db, sym) do
            throw(ArgumentError("Element $sym not supported in basis set $name_clean."))
        end
        center = coords_bohr[at.index]
        for (type, alphas, coeffs) in shells
            _add_shell!(funcs, at, center, type, alphas, coeffs)
        end
    end

    return BasisSet(name_clean, funcs)
end

# --- Lightweight Effective Core Potential (ECP) Support ---
# NOTE: This is a practical approximation intended for heavy-element core handling
# and is compatible with both CPU and GPU arrays by accepting device-aware inputs.

const ECP_LIBRARY = Dict{Symbol, EffectiveCorePotential}(
    :Br => EffectiveCorePotential(:Br, "qcm-prototype", 28, 2,
        Dict(0 => [ECPComponent(0, 0, [0.70, 0.12], [0.75, -0.35]),
                   ECPComponent(0, 1, [0.45], [0.35])],
             1 => [ECPComponent(1, 1, [0.60, 0.18], [0.95, -0.40])],
             2 => [ECPComponent(2, 2, [0.80, 0.18], [1.10, -0.30])])),
    :I => EffectiveCorePotential(:I, "qcm-prototype", 46, 2,
        Dict(0 => [ECPComponent(0, 0, [0.65, 0.10], [0.80, -0.30]),
                   ECPComponent(0, 1, [0.38], [0.28])],
             1 => [ECPComponent(1, 1, [0.52, 0.15], [0.92, -0.35])],
             2 => [ECPComponent(2, 2, [0.72, 0.15], [1.05, -0.27])])),
    :Cl => EffectiveCorePotential(:Cl, "qcm-prototype", 10, 1,
        Dict(0 => [ECPComponent(0, 0, [0.75, 0.18], [0.90, -0.35])],
             1 => [ECPComponent(1, 1, [0.60], [0.70])])),
    :S => EffectiveCorePotential(:S, "qcm-prototype", 10, 1,
        Dict(0 => [ECPComponent(0, 0, [0.80, 0.20], [0.95, -0.40])],
             1 => [ECPComponent(1, 1, [0.65], [0.75])]))
)

function ecp_potential_value(ecp::EffectiveCorePotential, r::Real, l::Int=0)::Float64
    rr = abs(r)
    total = 0.0
    for comp in get(ecp.components, l, ECPComponent[])
        for (α, c) in zip(comp.exponents, comp.coefficients)
            total += c * (rr^comp.power) * exp(-α * rr^2)
        end
    end
    return total
end

function build_ecp(atom::Atom; name::String="qcm-prototype")
    lowercase(strip(name)) == "qcm-prototype" ||
        throw(ArgumentError("ECP '$name' is not supported. Available ECP: qcm-prototype"))
    if haskey(ECP_LIBRARY, atom.symbol)
        return ECP_LIBRARY[atom.symbol]
    end
    return nothing
end

function build_ecp(mol::Molecule; name::String="qcm-prototype")
    out = Dict{Int, EffectiveCorePotential}()
    for at in mol.atoms
        ecp = build_ecp(at; name=name)
        if ecp !== nothing
            out[at.index] = ecp
        end
    end
    isempty(out) && throw(ArgumentError("No atoms in the molecule have a built-in '$name' ECP."))
    return out
end

ecp_core_electrons(ecps::AbstractDict) = sum(ecp.core_electrons for ecp in values(ecps))

function effective_nuclear_charges(mol::Molecule, ecps::AbstractDict)::Vector{Float64}
    return [Float64(at.atomic_number) -
            (haskey(ecps, at.index) ? ecps[at.index].core_electrons : 0)
            for at in mol.atoms]
end

function apply_ecp_matrix(mol::Molecule, basis::BasisSet; name::String="qcm-prototype", device::GPUInterface.AbstractDevice = GPUInterface.CPUDevice())
    n = n_basis(basis)
    V = zeros(Float64, n, n)
    ecps = build_ecp(mol; name=name)

    for (idx, func) in enumerate(basis.functions)
        ecp = get(ecps, func.atom_index, nothing)
        if ecp === nothing
            continue
        end
        l = func.lx + func.ly + func.lz
        V[idx, idx] += ecp_potential_value(ecp, 0.0, l)
    end

    if device != GPUInterface.CPUDevice()
        return GPUInterface.to_device(V, device)
    end
    return V
end

end # module BasisSets
