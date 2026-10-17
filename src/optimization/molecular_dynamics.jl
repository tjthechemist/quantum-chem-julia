# ==============================================================================
# QuantumChemMM.jl: Molecular Dynamics Engine (Velocity Verlet, NVE, NVT)
# ==============================================================================

module MolecularDynamics

using LinearAlgebra
using Printf
using ...Types
using ...Elements
using ...MMBase
using ...MMEnergyGradient

export MDSimulation, run_md, TrajectoryFrame

# Boltzmann constant in (kcal·Å²)/(mol·amu·fs²)
# kB = 1.380649e-23 J/K; 1 kcal = 4184 J; 1 amu = 1.660539e-27 kg; 1 Å = 1e-10 m; 1 fs = 1e-15 s
# 1 kcal/(mol·amu·fs²) = 418.4 Å²/fs²
const KB_MD = 0.0019872041 # kcal/(mol·K)
# Conversion factor from force (-grad in kcal/(mol·Å)) to acceleration (Å/fs²):
# a = - (grad / mass) * 4.184e-4
const FORCE_TO_ACCEL = 4.184e-4

struct TrajectoryFrame
    step::Int
    time_fs::Float64
    potential_energy::Float64
    kinetic_energy::Float64
    total_energy::Float64
    temperature_k::Float64
    coords::Matrix{Float64}
end

struct MDSimulation
    dt_fs::Float64
    n_steps::Int
    target_temp_k::Float64
    thermostat::Symbol # :nve or :berendsen
    tau_thermostat_fs::Float64
    output_freq::Int
end

function MDSimulation(; dt_fs=1.0, n_steps=1000, target_temp_k=300.0,
                        thermostat=:berendsen, tau_thermostat_fs=100.0, output_freq=50)
    MDSimulation(dt_fs, n_steps, target_temp_k, thermostat, tau_thermostat_fs, output_freq)
end

"""
    initialize_velocities!(velocities::Matrix{Float64}, mol::Molecule, target_temp::Float64)
Initializes Maxwell-Boltzmann distributed velocities at target temperature with zero net momentum.
"""
function initialize_velocities!(velocities::Matrix{Float64}, mol::Molecule, target_temp::Float64)
    nat = n_atoms(mol)
    total_mass = sum(at.mass for at in mol.atoms)

    # Sample standard Gaussian
    net_p = zeros(3)
    for i in 1:nat
        m = mol.atoms[i].mass
        std_v = sqrt(KB_MD * target_temp / m) * sqrt(4.184e-4) # in Å/fs
        v = randn(3) .* std_v
        velocities[:, i] = v
        net_p .+= m .* v
    end

    # Remove net center-of-mass momentum
    v_com = net_p ./ total_mass
    for i in 1:nat
        velocities[:, i] .-= v_com
    end

    # Scale to exact target temperature
    current_temp = calc_temperature(velocities, mol)
    if current_temp > 1e-4
        scale = sqrt(target_temp / current_temp)
        velocities .*= scale
    end
end

function calc_temperature(velocities::Matrix{Float64}, mol::Molecule)::Float64
    nat = n_atoms(mol)
    dof = max(3 * nat - 6, 1) # subtract translational and rotational degrees of freedom
    ke = calc_kinetic_energy(velocities, mol)
    return (2.0 * ke) / (dof * KB_MD)
end

function calc_kinetic_energy(velocities::Matrix{Float64}, mol::Molecule)::Float64
    nat = n_atoms(mol)
    ke = 0.0
    for i in 1:nat
        m = mol.atoms[i].mass
        v_sq = dot(velocities[:, i], velocities[:, i])
        ke += 0.5 * m * v_sq / FORCE_TO_ACCEL
    end
    return ke
end

"""
    run_md(mol::Molecule, ff::AbstractForceField, sim::MDSimulation=MDSimulation())::Vector{TrajectoryFrame}
Runs molecular dynamics simulation using Velocity Verlet integration.
"""
function run_md(mol::Molecule, ff::AbstractForceField, sim::MDSimulation=MDSimulation())::Vector{TrajectoryFrame}
    nat = n_atoms(mol)
    velocities = zeros(Float64, 3, nat)
    initialize_velocities!(velocities, mol, sim.target_temp_k)

    # Initial force and acceleration
    ec, grad = evaluate_mm(ff, mol; calc_grad=true)
    pe = total_energy(ec)
    accel = zeros(Float64, 3, nat)
    for i in 1:nat
        accel[:, i] = (-grad[:, i] ./ mol.atoms[i].mass) .* FORCE_TO_ACCEL
    end

    trajectory = TrajectoryFrame[]
    dt = sim.dt_fs
    dt_half = 0.5 * dt

    for step in 1:sim.n_steps
        # 1. Update positions: r(t + dt) = r(t) + v(t)*dt + 0.5*a(t)*dt^2
        for i in 1:nat
            mol.atoms[i].coords .+= velocities[:, i] .* dt .+ 0.5 .* accel[:, i] .* (dt^2)
        end

        # Half-step velocity update
        velocities .+= accel .* dt_half

        # 2. Compute new forces and accelerations at r(t + dt)
        ec, grad = evaluate_mm(ff, mol; calc_grad=true)
        pe = total_energy(ec)
        for i in 1:nat
            accel[:, i] = (-grad[:, i] ./ mol.atoms[i].mass) .* FORCE_TO_ACCEL
        end

        # Second half-step velocity update
        velocities .+= accel .* dt_half

        # 3. Apply Thermostat
        if sim.thermostat == :berendsen
            t_curr = calc_temperature(velocities, mol)
            if t_curr > 1e-4
                lambda_scale = sqrt(1.0 + (dt / sim.tau_thermostat_fs) * (sim.target_temp_k / t_curr - 1.0))
                velocities .*= lambda_scale
            end
        end

        # 4. Record trajectory frame
        if step % sim.output_freq == 0 || step == 1 || step == sim.n_steps
            ke = calc_kinetic_energy(velocities, mol)
            temp = calc_temperature(velocities, mol)
            coords = coordinates_matrix(mol)
            push!(trajectory, TrajectoryFrame(step, step * dt, pe, ke, pe + ke, temp, coords))
        end
    end

    return trajectory
end

end # module MolecularDynamics

