# ==============================================================================
# QuantumChemMM.jl: Chemical Reactivity Descriptors & Conceptual DFT Indices
# Hardness, Softness, Electrophilicity, and Condensed Fukui Functions
# ==============================================================================

module ReactivityDescriptors

using LinearAlgebra
using Printf
using ..Types
using ..Elements
using ..SemiEmpiricalBase
using ..NDDOSCF
using ..SlaterOrbitals

export ReactivityIndices, FukuiIndices, calc_reactivity_indices,
       calc_condensed_fukui, print_reactivity_summary

struct ReactivityIndices
    homo_energy_ev::Float64
    lumo_energy_ev::Float64
    homo_lumo_gap_ev::Float64
    ionization_potential_ev::Float64
    electron_affinity_ev::Float64
    chemical_hardness_ev::Float64
    chemical_softness_ev_inv::Float64
    chemical_potential_ev::Float64
    electronegativity_ev::Float64
    electrophilicity_index_ev::Float64
end

struct FukuiIndices
    f_plus::Vector{Float64}   # Nucleophilic attack susceptibility (f+)
    f_minus::Vector{Float64}  # Electrophilic attack susceptibility (f-)
    f_zero::Vector{Float64}   # Radical attack susceptibility (f0)
    dual_descriptor::Vector{Float64} # Delta f = f+ - f-
end

"""
    calc_reactivity_indices(scf_res::Union{SCFResult, NDDOResult},
                            mol::Molecule)::ReactivityIndices
Computes global Parr-Pearson conceptual DFT reactivity indices from HOMO/LUMO energies.
"""
function calc_reactivity_indices(scf_res::Union{SCFResult, NDDOResult},
                                 mol::Molecule)::ReactivityIndices
    eps = scf_res isa NDDOResult ? scf_res.mo_energies_ev : scf_res.mo_energies
    nb = length(eps)
    total_val = sum(Elements.valence_electrons(at.symbol) for at in mol.atoms) - mol.total_charge
    n_occ = div(total_val, 2)

    homo = eps[n_occ]
    lumo = eps[n_occ + 1]
    gap = lumo - homo

    # Koopmans' theorem approximations:
    # I = -HOMO, A = -LUMO
    ip = -homo
    ea = -lumo

    hardness = 0.5 * (ip - ea) # eta = 0.5 * gap
    softness = 1.0 / max(2.0 * hardness, 1e-4)
    chem_pot = -0.5 * (ip + ea)
    electroneg = -chem_pot
    omega = (chem_pot^2) / max(2.0 * hardness, 1e-4) # electrophilicity index

    return ReactivityIndices(homo, lumo, gap, ip, ea, hardness, softness, chem_pot, electroneg, omega)
end

"""
    calc_condensed_fukui(scf_res::Union{SCFResult, NDDOResult},
                         mol::Molecule)::FukuiIndices
Computes condensed Fukui functions f+, f-, f0 and the Dual Descriptor (Δf)
to identify reactive centers for electrophilic, nucleophilic, and radical attacks.
"""
function calc_condensed_fukui(scf_res::Union{SCFResult, NDDOResult},
                              mol::Molecule)::FukuiIndices
    C = scf_res.mo_coefficients
    nb = size(C, 1)
    nat = n_atoms(mol)

    total_val = sum(Elements.valence_electrons(at.symbol) for at in mol.atoms) - mol.total_charge
    n_occ = div(total_val, 2)
    i_homo = n_occ
    i_lumo = n_occ + 1

    basis = SlaterOrbitals.build_valence_basis(mol)

    f_plus = zeros(Float64, nat)
    f_minus = zeros(Float64, nat)

    for mu in 1:nb
        at_idx = basis[mu].atom_index
        f_plus[at_idx] += C[mu, i_lumo]^2   # f+ ~ |C_LUMO|^2
        f_minus[at_idx] += C[mu, i_homo]^2  # f- ~ |C_HOMO|^2
    end

    # Normalize sums to 1.0
    sum_plus = sum(f_plus)
    sum_minus = sum(f_minus)
    if sum_plus > 1e-6
        f_plus ./= sum_plus
    end
    if sum_minus > 1e-6
        f_minus ./= sum_minus
    end

    f_zero = 0.5 .* (f_plus .+ f_minus)
    dual = f_plus .- f_minus

    return FukuiIndices(f_plus, f_minus, f_zero, dual)
end

function print_reactivity_summary(indices::ReactivityIndices, fukui::FukuiIndices, mol::Molecule)
    println("==================================================")
    println("        Conceptual DFT Reactivity Descriptors     ")
    println("==================================================")
    @printf("  HOMO Energy:              %12.4f eV\n", indices.homo_energy_ev)
    @printf("  LUMO Energy:              %12.4f eV\n", indices.lumo_energy_ev)
    @printf("  HOMO-LUMO Energy Gap:     %12.4f eV\n", indices.homo_lumo_gap_ev)
    @printf("  Ionization Potential (I): %12.4f eV\n", indices.ionization_potential_ev)
    @printf("  Electron Affinity (A):    %12.4f eV\n", indices.electron_affinity_ev)
    @printf("  Chemical Hardness (η):    %12.4f eV\n", indices.chemical_hardness_ev)
    @printf("  Chemical Softness (S):    %12.4f eV⁻¹\n", indices.chemical_softness_ev_inv)
    @printf("  Electronegativity (χ):    %12.4f eV\n", indices.electronegativity_ev)
    @printf("  Electrophilicity Index(ω):%12.4f eV\n", indices.electrophilicity_index_ev)
    println("--------------------------------------------------")
    println("  Condensed Fukui Functions & Dual Descriptor:")
    println("  Atom  Element      f+ (Nu-site)  f- (E+-site)  f0 (Radical)  Dual (Δf)")
    for i in 1:n_atoms(mol)
        @printf("   %2d      %-2s         %8.4f      %8.4f      %8.4f    %+8.4f\n",
                i, string(mol.atoms[i].symbol), fukui.f_plus[i], fukui.f_minus[i],
                fukui.f_zero[i], fukui.dual_descriptor[i])
    end
    println("==================================================")
end

end # module ReactivityDescriptors
