# ==============================================================================
# QuantumChemMM.jl: Hybrid QM/MM Partitioning & Embedding
# ==============================================================================

module QMMMEmbedding

using ...Types
using ...Elements
using ...MolMath

export QMMMSystem, create_qmmm_system, extract_subsystem

struct QMMMSystem
    full_molecule::Molecule
    qm_indices::Vector{Int}
    mm_indices::Vector{Int}
    link_atoms::Vector{Tuple{Int, Int, Vector{Float64}}} # (qm_boundary_atom, mm_boundary_atom, link_h_coords)
    embedding_type::Symbol                               # :mechanical or :electrostatic
end

"""
    create_qmmm_system(mol::Molecule, qm_indices::Vector{Int};
                       embedding::Symbol=:mechanical)::QMMMSystem
Partitions the system into QM and MM regions and detects boundary bonds for link-atom capping.
"""
function create_qmmm_system(mol::Molecule, qm_indices::Vector{Int};
                            embedding::Symbol=:mechanical)::QMMMSystem
    nat = n_atoms(mol)
    qm_set = Set(qm_indices)
    mm_indices = [i for i in 1:nat if !(i in qm_set)]

    # Detect boundary bonds cut between QM and MM
    link_atoms = Tuple{Int, Int, Vector{Float64}}[]
    for b in mol.bonds
        is_i_qm = b.i in qm_set
        is_j_qm = b.j in qm_set
        if is_i_qm != is_j_qm
            qm_at = is_i_qm ? b.i : b.j
            mm_at = is_i_qm ? b.j : b.i
            # Place Link Hydrogen along the bond vector at typical C-H distance ~ 1.09 Å
            r_qm = mol.atoms[qm_at].coords
            r_mm = mol.atoms[mm_at].coords
            v_bond = r_mm .- r_qm
            d_bond = norm(v_bond)
            if d_bond > 1e-6
                r_link = r_qm .+ (1.09 / d_bond) .* v_bond
                push!(link_atoms, (qm_at, mm_at, r_link))
            end
        end
    end

    return QMMMSystem(mol, sort(qm_indices), mm_indices, link_atoms, embedding)
end

"""
    extract_subsystem(mol::Molecule, indices::Vector{Int}; add_link_h::Bool=true)::Molecule
Extracts a sub-molecule containing only the specified atoms (plus optional link hydrogens).
"""
function extract_subsystem(mol::Molecule, indices::Vector{Int};
                           link_atoms::Vector{Tuple{Int, Int, Vector{Float64}}}=Tuple{Int, Int, Vector{Float64}}[])::Molecule
    sub = Molecule("Subsystem_$(mol.name)")
    old_to_new = Dict{Int, Int}()

    # Add core atoms
    for (new_idx, old_idx) in enumerate(indices)
        at = mol.atoms[old_idx]
        push!(sub.atoms, Atom(new_idx, at.symbol, copy(at.coords);
                              partial_charge=at.partial_charge, atom_type=at.atom_type))
        old_to_new[old_idx] = new_idx
    end

    # Add link hydrogens
    for link in link_atoms
        new_idx = length(sub.atoms) + 1
        push!(sub.atoms, Atom(new_idx, :H, copy(link[3]); partial_charge=0.0, atom_type="H"))
    end

    return sub
end

end # module QMMMEmbedding

