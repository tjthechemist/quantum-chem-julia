# ==============================================================================
# QuantumChemMM.jl: Grimme D3 Dispersion Correction (Becke-Johnson Damping)
# ==============================================================================

module DispersionD3

using LinearAlgebra
using ...Types
using ...Elements
using ...MolMath

export D3Correction, calc_d3_dispersion

struct D3AtomParam
    r0::Float64 # vdW cutoff radius in Å
    c6::Float64 # C6 coefficient in J·nm⁶/mol (converted to kcal·Å⁶/mol)
    c8::Float64 # C8 coefficient
end

# Atomic C6 parameters in kcal·Å⁶/mol
const D3_PARAMS = Dict{Symbol, D3AtomParam}(
    :H  => D3AtomParam(1.00, 20.0, 300.0),
    :C  => D3AtomParam(1.45, 140.0, 2800.0),
    :N  => D3AtomParam(1.40, 100.0, 1800.0),
    :O  => D3AtomParam(1.35, 75.0, 1200.0),
    :F  => D3AtomParam(1.30, 50.0, 750.0),
    :P  => D3AtomParam(1.70, 380.0, 8500.0),
    :S  => D3AtomParam(1.68, 350.0, 7800.0),
    :Cl => D3AtomParam(1.65, 300.0, 6500.0)
)

struct D3Correction
    s6::Float64   # scaling factor for C6 (usually 1.0)
    s8::Float64   # scaling factor for C8
    a1::Float64   # BJ damping parameter
    a2::Float64   # BJ damping parameter in Å
end

# Default parameters optimized for semi-empirical methods (GFN-xTB / AM1)
D3Correction(; s6=1.0, s8=1.2, a1=0.55, a2=4.5) = D3Correction(s6, s8, a1, a2)

"""
    calc_d3_dispersion(mol::Molecule, d3::D3Correction=D3Correction(); calc_grad::Bool=true)
Computes Grimme D3(BJ) dispersion energy in kcal/mol and analytical gradient (3 x N_atoms).
"""
function calc_d3_dispersion(mol::Molecule, d3::D3Correction=D3Correction(); calc_grad::Bool=true)
    nat = n_atoms(mol)
    energy = 0.0
    grad = zeros(Float64, 3, nat)

    for i in 1:nat
        sym_i = mol.atoms[i].symbol
        pi = get(D3_PARAMS, sym_i, D3AtomParam(1.5, 100.0, 2000.0))
        ri = mol.atoms[i].coords

        for j in (i+1):nat
            sym_j = mol.atoms[j].symbol
            pj = get(D3_PARAMS, sym_j, D3AtomParam(1.5, 100.0, 2000.0))
            rj = mol.atoms[j].coords

            r = MolMath.distance(ri, rj)
            if r < 0.1
                continue
            end

            c6_ij = sqrt(pi.c6 * pj.c6)
            c8_ij = sqrt(pi.c8 * pj.c8)
            r0_ij = sqrt(pi.r0 * pj.r0)

            # Becke-Johnson damping radii: R_cut = a1 * R0 + a2
            r_bj = d3.a1 * r0_ij + d3.a2
            denom6 = r^6 + r_bj^6
            denom8 = r^8 + r_bj^8

            e_pair = -(d3.s6 * c6_ij / denom6 + d3.s8 * c8_ij / denom8)
            energy += e_pair

            if calc_grad
                # Analytical gradient dE/dr
                dE_dr = d3.s6 * c6_ij * (6.0 * r^5) / (denom6^2) +
                        d3.s8 * c8_ij * (8.0 * r^7) / (denom8^2)
                gi, gj = MolMath.distance_gradient(ri, rj)
                grad[:, i] .+= dE_dr .* gi
                grad[:, j] .+= dE_dr .* gj
            end
        end
    end

    return energy, grad
end

end # module DispersionD3

