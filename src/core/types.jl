# ==============================================================================
# QuantumChemMM.jl: Core Molecular Data Types
# ==============================================================================

module Types

using ..Elements
using LinearAlgebra

export Atom, Bond, Angle, Dihedral, Improper, Molecule,
       n_atoms, n_bonds, coordinates_matrix, set_coordinates!,
       center_of_mass, translate!, rotate!

mutable struct Atom
    index::Int
    symbol::Symbol
    atomic_number::Int
    coords::Vector{Float64}    # [x, y, z] in Angstroms
    mass::Float64              # in amu
    partial_charge::Float64    # in e (elementary charge)
    atom_type::String          # force field atom type (e.g., "C_3", "ca", "ct")
    formal_charge::Int
    hybridization::Symbol      # :s, :sp, :sp2, :sp3, :unknown
end

function Atom(index::Int, symbol::Symbol, coords::AbstractVector{<:Real};
              partial_charge::Float64=0.0, atom_type::String="", formal_charge::Int=0,
              hybridization::Symbol=:unknown)
    ed = Elements.get_element(symbol)
    Atom(index, symbol, ed.atomic_number, Vector{Float64}(coords),
         ed.mass, partial_charge, isempty(atom_type) ? String(symbol) : atom_type,
         formal_charge, hybridization)
end

struct Bond
    i::Int
    j::Int
    order::Float64   # 1.0, 2.0, 3.0, 1.5 (aromatic)
end

struct Angle
    i::Int
    j::Int           # vertex atom
    k::Int
end

struct Dihedral
    i::Int
    j::Int
    k::Int
    l::Int
end

struct Improper
    central::Int
    i::Int
    j::Int
    k::Int
end

mutable struct Molecule
    name::String
    atoms::Vector{Atom}
    bonds::Vector{Bond}
    angles::Vector{Angle}
    dihedrals::Vector{Dihedral}
    impropers::Vector{Improper}
    total_charge::Int
    multiplicity::Int          # 1 = singlet, 2 = doublet, 3 = triplet, etc.
end

function Molecule(name::String="Molecule"; total_charge::Int=0, multiplicity::Int=1)
    Molecule(name, Atom[], Bond[], Angle[], Dihedral[], Improper[], total_charge, multiplicity)
end

n_atoms(mol::Molecule) = length(mol.atoms)
n_bonds(mol::Molecule) = length(mol.bonds)

function coordinates_matrix(mol::Molecule)::Matrix{Float64}
    nat = n_atoms(mol)
    mat = zeros(Float64, 3, nat)
    for (idx, at) in enumerate(mol.atoms)
        mat[:, idx] = at.coords
    end
    return mat
end

function set_coordinates!(mol::Molecule, mat::AbstractMatrix{<:Real})
    @assert size(mat, 1) == 3 && size(mat, 2) == n_atoms(mol) "Matrix size must be 3 x N_atoms"
    for (idx, at) in enumerate(mol.atoms)
        at.coords .= mat[:, idx]
    end
    return mol
end

function center_of_mass(mol::Molecule)::Vector{Float64}
    nat = n_atoms(mol)
    if nat == 0
        return zeros(3)
    end
    total_m = 0.0
    com = zeros(3)
    for at in mol.atoms
        com .+= at.mass .* at.coords
        total_m += at.mass
    end
    return com ./ total_m
end

function translate!(mol::Molecule, shift::AbstractVector{<:Real})
    for at in mol.atoms
        at.coords .+= shift
    end
    return mol
end

function rotate!(mol::Molecule, R::AbstractMatrix{<:Real})
    com = center_of_mass(mol)
    translate!(mol, -com)
    for at in mol.atoms
        at.coords .= R * at.coords
    end
    translate!(mol, com)
    return mol
end

end # module Types

