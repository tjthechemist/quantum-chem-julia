# ==============================================================================
# QuantumChemMM.jl: Molecular Mechanics Base Types & Energy Components
# ==============================================================================

module MMBase

using Printf

export AbstractForceField, EnergyComponents, total_energy, print_summary

abstract type AbstractForceField end

struct EnergyComponents
    e_bond::Float64
    e_angle::Float64
    e_dihedral::Float64
    e_improper::Float64
    e_vdw::Float64
    e_elec::Float64
    e_cross::Float64   # For Class II force fields (MM2/MM3/COMPASS)
end

function EnergyComponents(; e_bond=0.0, e_angle=0.0, e_dihedral=0.0,
                            e_improper=0.0, e_vdw=0.0, e_elec=0.0, e_cross=0.0)
    EnergyComponents(e_bond, e_angle, e_dihedral, e_improper, e_vdw, e_elec, e_cross)
end

total_energy(ec::EnergyComponents) = ec.e_bond + ec.e_angle + ec.e_dihedral +
                                     ec.e_improper + ec.e_vdw + ec.e_elec + ec.e_cross

function print_summary(ec::EnergyComponents; unit::String="kcal/mol")
    tot = total_energy(ec)
    println("---------------------------------------------")
    println("  Molecular Mechanics Energy Breakdown ($unit)")
    println("---------------------------------------------")
    @printf("  Bond Stretching:     %14.6f\n", ec.e_bond)
    @printf("  Angle Bending:       %14.6f\n", ec.e_angle)
    @printf("  Proper Torsions:     %14.6f\n", ec.e_dihedral)
    @printf("  Improper/Inversion:  %14.6f\n", ec.e_improper)
    @printf("  Van der Waals (vdW): %14.6f\n", ec.e_vdw)
    @printf("  Electrostatic:       %14.6f\n", ec.e_elec)
    if abs(ec.e_cross) > 1e-10
        @printf("  Cross-terms:         %14.6f\n", ec.e_cross)
    end
    println("---------------------------------------------")
    @printf("  TOTAL ENERGY:        %14.6f\n", tot)
    println("---------------------------------------------")
end

end # module MMBase

