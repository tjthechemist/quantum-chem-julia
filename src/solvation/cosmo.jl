# ==============================================================================
# QuantumChemMM.jl: Implicit Solvation Models (COSMO, PCM, and Generalized Born)
# Supports Water, Alcohols, Nonpolar Solvents, SASA Cavitation, and Solvation Free Energy
# ==============================================================================

module SolvationModels

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...MolMath

export Solvent, SolvationResult, calc_solvation_energy,
       SOLVENTS, get_solvent

struct Solvent
    name::String
    dielectric::Float64   # relative permittivity epsilon
    refractive_index::Float64
    probe_radius::Float64 # in Å (default 1.4 for water)
    surface_tension::Float64 # in kcal/(mol·Å²)
end

const SOLVENTS = Dict{Symbol, Solvent}(
    :water        => Solvent("Water", 78.39, 1.333, 1.40, 0.0072),
    :acetonitrile => Solvent("Acetonitrile", 35.69, 1.344, 2.10, 0.0050),
    :dmso         => Solvent("Dimethylsulfoxide", 46.70, 1.479, 2.50, 0.0058),
    :methanol     => Solvent("Methanol", 32.70, 1.328, 1.80, 0.0051),
    :ethanol      => Solvent("Ethanol", 24.55, 1.361, 2.00, 0.0048),
    :chloroform   => Solvent("Chloroform", 4.81, 1.446, 2.70, 0.0042),
    :dichloromethane => Solvent("Dichloromethane", 8.93, 1.424, 2.50, 0.0045),
    :toluene      => Solvent("Toluene", 2.38, 1.496, 3.00, 0.0035),
    :hexane       => Solvent("Hexane", 1.88, 1.375, 3.20, 0.0030),
    :octanol      => Solvent("1-Octanol", 10.30, 1.429, 3.40, 0.0045),
    :vacuum       => Solvent("Vacuum", 1.00, 1.000, 0.00, 0.0000)
)

get_solvent(sym::Symbol) = get(SOLVENTS, sym, SOLVENTS[:water])

struct SolvationResult
    solvent_name::String
    dielectric::Float64
    total_delta_g_solv_kcal::Float64
    electrostatic_solv_kcal::Float64
    cavitation_dispersion_kcal::Float64
    total_sasa_ang2::Float64
    atomic_sasa::Vector{Float64}
end

"""
    estimate_atomic_sasa(mol::Molecule, probe_radius::Float64)::Vector{Float64}
Estimates Solvent Accessible Surface Area (SASA) for each atom in Å²
using Shrake-Rupley / Lee-Richards numerical sphere tessellation.
"""
function estimate_atomic_sasa(mol::Molecule, probe_radius::Float64)::Vector{Float64}
    nat = n_atoms(mol)
    sasa = zeros(Float64, nat)

    # 50 Fibonacci sphere test points
    n_points = 50
    points = [zeros(3) for _ in 1:n_points]
    golden_ratio = (1.0 + sqrt(5.0)) / 2.0

    for i in 1:n_points
        theta = 2.0 * π * i / golden_ratio
        phi = acos(1.0 - 2.0 * (i - 0.5) / n_points)
        points[i] = [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]
    end

    for i in 1:nat
        r_i = Elements.vdw_radius(mol.atoms[i].symbol) + probe_radius
        accessible_count = 0

        for pt in points
            test_pos = mol.atoms[i].coords .+ r_i .* pt
            occluded = false

            for j in 1:nat
                if i != j
                    r_j = Elements.vdw_radius(mol.atoms[j].symbol) + probe_radius
                    if MolMath.distance(test_pos, mol.atoms[j].coords) < r_j
                        occluded = true
                        break
                    end
                end
            end

            if !occluded
                accessible_count += 1
            end
        end

        sphere_area = 4.0 * π * (r_i^2)
        sasa[i] = sphere_area * (accessible_count / n_points)
    end

    return sasa
end

"""
    calc_solvation_energy(mol::Molecule, solvent_sym::Symbol=:water;
                          charges::Vector{Float64}=Float64[])::SolvationResult
Computes the free energy of solvation ΔG_solv using the Generalized Born / COSMO
surface area continuum solvation model.
"""
function calc_solvation_energy(mol::Molecule, solvent_sym::Symbol=:water;
                              charges::Vector{Float64}=Float64[])::SolvationResult
    solvent = get_solvent(solvent_sym)
    nat = n_atoms(mol)

    if solvent.dielectric <= 1.0
        return SolvationResult(solvent.name, 1.0, 0.0, 0.0, 0.0, 0.0, zeros(nat))
    end

    q_vec = isempty(charges) ? [at.partial_charge for at in mol.atoms] : charges

    # 1. Cavitation & Non-polar dispersion free energy from SASA
    atomic_sasa = estimate_atomic_sasa(mol, solvent.probe_radius)
    total_sasa = sum(atomic_sasa)
    # ΔG_cav = gamma * SASA + offset
    delta_g_cav = solvent.surface_tension * total_sasa

    # 2. Electrostatic Solvation Free Energy (Generalized Born / COSMO)
    # f(eps) = (eps - 1) / (eps + 0.5) [COSMO scaling]
    f_eps = (solvent.dielectric - 1.0) / (solvent.dielectric + 0.5)

    # Effective Born radii R_i_born ~ r_vdw / (1 - C * (r_vdw/SASA_sphere))
    born_radii = zeros(Float64, nat)
    for i in 1:nat
        r_vdw = Elements.vdw_radius(mol.atoms[i].symbol)
        ratio = atomic_sasa[i] / (4.0 * π * (r_vdw + solvent.probe_radius)^2)
        born_radii[i] = r_vdw / max(ratio, 0.15)
    end

    # Coulomb constant in kcal·Å/mol
    C_ELEC = 332.063711
    delta_g_elec = 0.0

    for i in 1:nat
        qi = q_vec[i]
        # Self-polarization (Born term)
        delta_g_elec -= 0.5 * f_eps * (C_ELEC * qi^2) / born_radii[i]

        # Pairwise polarization
        for j in (i+1):nat
            qj = q_vec[j]
            abs(qi * qj) < 1e-8 && continue
            rij = MolMath.distance(mol.atoms[i].coords, mol.atoms[j].coords)
            # Still-formula Generalized Born distance: f_GB = sqrt(rij^2 + alpha_i * alpha_j * exp(-rij^2 / (4*alpha_i*alpha_j)))
            alpha_prod = born_radii[i] * born_radii[j]
            f_gb = sqrt(rij^2 + alpha_prod * exp(- (rij^2) / (4.0 * alpha_prod)))

            delta_g_elec -= f_eps * (C_ELEC * qi * qj) / f_gb
        end
    end

    total_delta_g = delta_g_elec + delta_g_cav

    return SolvationResult(solvent.name, solvent.dielectric, total_delta_g,
                           delta_g_elec, delta_g_cav, total_sasa, atomic_sasa)
end

function print_solvation_summary(res::SolvationResult)
    println("==================================================")
    println("         Implicit Solvation Model (COSMO/GB)      ")
    println("==================================================")
    @printf("  Solvent:                  %s (ε = %.2f)\n", res.solvent_name, res.dielectric)
    @printf("  Total SASA:               %14.2f Å²\n", res.total_sasa_ang2)
    @printf("  Electrostatic Solvation:  %14.4f kcal/mol\n", res.electrostatic_solv_kcal)
    @printf("  Cavitation / Dispersion:  %14.4f kcal/mol\n", res.cavitation_dispersion_kcal)
    println("--------------------------------------------------")
    @printf("  TOTAL ΔG_solv:            %14.4f kcal/mol\n", res.total_delta_g_solv_kcal)
    println("==================================================")
end

end # module SolvationModels

