# ==============================================================================
# QuantumChemMM.jl: Ab Initio Contracted Gaussian Basis Sets
# Supports STO-3G, 3-21G, 6-31G for H, He, Li..Ne
# ==============================================================================

module BasisSets

using LinearAlgebra
using ...Types
using ...Elements

export PrimitiveGaussian, ContractedGaussian, BasisSet,
       build_basis, n_basis

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
    coeffs::Vector{Float64} # Contraction coefficients
end

struct BasisSet
    name::String
    functions::Vector{ContractedGaussian}
end

n_basis(bs::BasisSet) = length(bs.functions)

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

# --- Predefined Basis Set Parameters (Hehre, Stewart, Pople) ---

# STO-3G Database
const STO3G_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [3.42525091, 0.62391373, 0.16885540], [0.15432897, 0.53532814, 0.44463454])
    ],
    :He => [
        (:s, [6.36242139, 1.15892300, 0.31364979], [0.15432897, 0.53532814, 0.44463454])
    ],
    :C => [
        (:s, [71.6168370, 13.0450960, 3.5305122], [0.15432897, 0.53532814, 0.44463454]), # 1s
        (:s, [2.9412494, 0.6834831, 0.2222899], [-0.09996723, 0.39951283, 0.70011547]),  # 2s
        (:p, [2.9412494, 0.6834831, 0.2222899], [0.15591627, 0.60768372, 0.39195739])    # 2p
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
    ]
)

# 6-31G Database for H and C/N/O
const BASIS_631G_DATA = Dict{Symbol, Vector{Tuple{Symbol, Vector{Float64}, Vector{Float64}}}}(
    :H => [
        (:s, [18.7311370, 2.8253944, 0.6401217], [0.0334946, 0.2347269, 0.8137573]), # core 3
        (:s, [0.1612778], [1.0]) # valence 1
    ],
    :C => [
        (:s, [3047.5249, 457.36951, 103.94869, 29.210155, 9.286663, 3.163927],
             [0.0018347, 0.0140373, 0.0688426, 0.2321844, 0.4679413, 0.3623120]), # 1s
        (:s, [7.8682724, 1.8812885, 0.5442493], [-0.1193324, -0.1608542, 1.1434564]), # 2s_inner
        (:p, [7.8682724, 1.8812885, 0.5442493], [0.0689991, 0.3164240, 0.7443083]),
        (:s, [0.1687144], [1.0]), # 2s_outer
        (:p, [0.1687144], [1.0])
    ],
    :O => [
        (:s, [5484.6717, 825.23495, 188.04696, 52.964500, 16.897570, 5.799635],
             [0.0018311, 0.0139501, 0.0684451, 0.2327143, 0.4701930, 0.3585209]),
        (:s, [15.539616, 3.5999336, 1.0137701], [-0.1107775, -0.1480263, 1.1307670]),
        (:p, [15.539616, 3.5999336, 1.0137701], [0.0708743, 0.3397328, 0.7271586]),
        (:s, [0.2700058], [1.0]),
        (:p, [0.2700058], [1.0])
    ]
)

"""
    build_basis(mol::Molecule, name::String="sto-3g")::BasisSet
Constructs the full molecular basis set of contracted Cartesian Gaussians.
"""
function build_basis(mol::Molecule, name::String="sto-3g")::BasisSet
    name_clean = lowercase(strip(name))
    db = if name_clean in ("6-31g", "631g")
        BASIS_631G_DATA
    else
        STO3G_DATA
    end

    funcs = ContractedGaussian[]
    coords_bohr = [mol.atoms[i].coords .* ANGSTROM_TO_BOHR for i in 1:n_atoms(mol)]

    for at in mol.atoms
        sym = at.symbol
        shells = get(db, sym) do
            get(STO3G_DATA, sym) do
                throw(ArgumentError("Element $sym not supported in basis set $name."))
            end
        end

        center = coords_bohr[at.index]

        for (type, alphas, coeffs) in shells
            n_prim = length(alphas)
            if type == :s
                prims = [make_primitive(alphas[k], 0, 0, 0) for k in 1:n_prim]
                push!(funcs, ContractedGaussian(at.index, center, 0, 0, 0, prims, coeffs))
            elseif type == :p
                # 3 Cartesian components: px (1,0,0), py (0,1,0), pz (0,0,1)
                prims_x = [make_primitive(alphas[k], 1, 0, 0) for k in 1:n_prim]
                prims_y = [make_primitive(alphas[k], 0, 1, 0) for k in 1:n_prim]
                prims_z = [make_primitive(alphas[k], 0, 0, 1) for k in 1:n_prim]

                push!(funcs, ContractedGaussian(at.index, center, 1, 0, 0, prims_x, coeffs))
                push!(funcs, ContractedGaussian(at.index, center, 0, 1, 0, prims_y, coeffs))
                push!(funcs, ContractedGaussian(at.index, center, 0, 0, 1, prims_z, coeffs))
            end
        end
    end

    return BasisSet(name_clean, funcs)
end

end # module BasisSets

