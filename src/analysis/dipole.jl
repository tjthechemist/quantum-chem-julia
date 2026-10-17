# ==============================================================================
# QuantumChemMM.jl: Molecular Dipole Moment & Center of Charge Analysis
# ==============================================================================

module DipoleAnalysis

using LinearAlgebra
using Printf
using ...Types

export calc_molecular_dipole, DipoleResult

# Conversion factor: 1 elementary charge · Å = 4.803204 Debye
const E_ANGSTROM_TO_DEBYE = 4.80320427

struct DipoleResult
    dipole_vector::Vector{Float64} # in Debye
    magnitude::Float64             # in Debye
    center_of_charge::Vector{Float64}
end

"""
    calc_molecular_dipole(mol::Molecule; charges::Vector{Float64}=Float64[])::DipoleResult
Computes the molecular electric dipole moment vector in Debye.
If charges vector is provided, it is used; otherwise atom.partial_charge is used.
"""
function calc_molecular_dipole(mol::Molecule; charges::Vector{Float64}=Float64[])::DipoleResult
    nat = n_atoms(mol)
    q_vec = isempty(charges) ? [at.partial_charge for at in mol.atoms] : charges

    # Center of mass / reference origin
    com = center_of_mass(mol)

    dip = zeros(Float64, 3)
    pos_q = 0.0
    center_of_pos = zeros(3)

    for i in 1:nat
        r_rel = mol.atoms[i].coords .- com
        dip .+= (q_vec[i] * E_ANGSTROM_TO_DEBYE) .* r_rel
        if q_vec[i] > 0.0
            pos_q += q_vec[i]
            center_of_pos .+= q_vec[i] .* mol.atoms[i].coords
        end
    end

    center_of_q = (pos_q > 1e-6) ? (center_of_pos ./ pos_q) : com
    mag = norm(dip)

    return DipoleResult(dip, mag, center_of_q)
end

end # module DipoleAnalysis

