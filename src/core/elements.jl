# ==============================================================================
# QuantumChemMM.jl: Core Element Definitions and Periodic Table Data
# ==============================================================================

module Elements

export ElementData, get_element, symbol_to_Z, Z_to_symbol, atomic_mass,
       covalent_radius, vdw_radius, valence_electrons, electronegativity

struct ElementData
    symbol::Symbol
    atomic_number::Int
    name::String
    mass::Float64               # in amu
    valence_electrons::Int      # standard valence count
    covalent_radius::Float64    # in Angstroms
    vdw_radius::Float64         # in Angstroms
    electronegativity::Float64  # Pauling scale
end

# Periodic table database up to Kr (extended as needed)
const PERIODIC_TABLE = Dict{Symbol, ElementData}(
    :H  => ElementData(:H,  1,  "Hydrogen",    1.008,    1, 0.31, 1.20, 2.20),
    :He => ElementData(:He, 2,  "Helium",      4.0026,   2, 0.28, 1.40, 0.00),
    :Li => ElementData(:Li, 3,  "Lithium",     6.94,     1, 1.28, 1.82, 0.98),
    :Be => ElementData(:Be, 4,  "Beryllium",   9.0122,   2, 0.96, 1.53, 1.57),
    :B  => ElementData(:B,  5,  "Boron",       10.81,    3, 0.84, 1.92, 2.04),
    :C  => ElementData(:C,  6,  "Carbon",      12.011,   4, 0.76, 1.70, 2.55),
    :N  => ElementData(:N,  7,  "Nitrogen",    14.007,   5, 0.71, 1.55, 3.04),
    :O  => ElementData(:O,  8,  "Oxygen",      15.999,   6, 0.66, 1.52, 3.44),
    :F  => ElementData(:F,  9,  "Fluorine",    18.998,   7, 0.57, 1.47, 3.98),
    :Ne => ElementData(:Ne, 10, "Neon",        20.180,   8, 0.58, 1.54, 0.00),
    :Na => ElementData(:Na, 11, "Sodium",      22.990,   1, 1.66, 2.27, 0.93),
    :Mg => ElementData(:Mg, 12, "Magnesium",   24.305,   2, 1.41, 1.73, 1.31),
    :Al => ElementData(:Al, 13, "Aluminum",    26.982,   3, 1.21, 1.84, 1.61),
    :Si => ElementData(:Si, 14, "Silicon",     28.085,   4, 1.11, 2.10, 1.90),
    :P  => ElementData(:P,  15, "Phosphorus",  30.974,   5, 1.07, 1.80, 2.19),
    :S  => ElementData(:S,  16, "Sulfur",      32.06,    6, 1.05, 1.80, 2.58),
    :Cl => ElementData(:Cl, 17, "Chlorine",    35.45,    7, 1.02, 1.75, 3.16),
    :Ar => ElementData(:Ar, 18, "Argon",       39.948,   8, 1.06, 1.88, 0.00),
    :K  => ElementData(:K,  19, "Potassium",   39.098,   1, 2.03, 2.75, 0.82),
    :Ca => ElementData(:Ca, 20, "Calcium",     40.078,   2, 1.76, 2.31, 1.00),
    :Br => ElementData(:Br, 35, "Bromine",     79.904,   7, 1.20, 1.85, 2.96),
    :I  => ElementData(:I,  53, "Iodine",      126.90,   7, 1.39, 1.98, 2.66)
)

const Z_TABLE = Dict{Int, ElementData}(
    ed.atomic_number => ed for (_, ed) in PERIODIC_TABLE
)

get_element(s::Symbol) = get(PERIODIC_TABLE, s) do
    throw(ArgumentError("Element symbol :$s not found in periodic table database."))
end

get_element(Z::Int) = get(Z_TABLE, Z) do
    throw(ArgumentError("Element with atomic number $Z not found in periodic table database."))
end

symbol_to_Z(s::Symbol) = get_element(s).atomic_number
Z_to_symbol(Z::Int) = get_element(Z).symbol
atomic_mass(s::Symbol) = get_element(s).mass
covalent_radius(s::Symbol) = get_element(s).covalent_radius
vdw_radius(s::Symbol) = get_element(s).vdw_radius
valence_electrons(s::Symbol) = get_element(s).valence_electrons
electronegativity(s::Symbol) = get_element(s).electronegativity

end # module Elements

