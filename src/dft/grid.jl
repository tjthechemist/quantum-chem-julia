# ==============================================================================
# QuantumChemMM.jl: Molecular Numerical Quadrature Grids for DFT Integration
# Becke Fuzzy Voronoi Partitioning & Gauss-Legendre Quadrature
# ==============================================================================

module DFTGrid

using LinearAlgebra
using ..Types
using ..Elements
using ..BasisSets
using ..DFTFunctionals

export MolecularGrid, build_molecular_grid, evaluate_density,
       integrate_xc_matrix_and_energy

struct GridPoint
    coords::Vector{Float64} # [x, y, z] in bohr
    weight::Float64
end

struct MolecularGrid
    points::Vector{GridPoint}
    n_points::Int
end

"""
Evaluates contracted Gaussian basis function mu at point r (in bohrs).
"""
function eval_basis_function(cg::ContractedGaussian, r::Vector{Float64})::Float64
    dr = r .- cg.center
    r_sq = dot(dr, dr)
    # Cartesian prefactor: (x-xA)^lx * (y-yA)^ly * (z-zA)^lz
    poly = (dr[1]^cg.lx) * (dr[2]^cg.ly) * (dr[3]^cg.lz)

    val = 0.0
    for (idx, p) in enumerate(cg.primitives)
        val += cg.coeffs[idx] * p.norm * exp(-p.alpha * r_sq)
    end
    return poly * val
end

"""
    build_molecular_grid(mol::Molecule; n_radial::Int=20, n_angular::Int=26)::MolecularGrid
Constructs atom-centered numerical integration grid with Becke atomic partitioning weights.
"""
function build_molecular_grid(mol::Molecule; n_radial::Int=20, n_angular::Int=26)::MolecularGrid
    nat = n_atoms(mol)
    coords_bohr = [mol.atoms[i].coords .* BasisSets.ANGSTROM_TO_BOHR for i in 1:nat]
    grid_points = GridPoint[]

    # Fibonacci sphere points for angular quadrature
    golden_ratio = (1.0 + sqrt(5.0)) / 2.0
    angular_dirs = [zeros(3) for _ in 1:n_angular]
    angular_weight = 4.0 * π / n_angular

    for i in 1:n_angular
        theta = 2.0 * π * i / golden_ratio
        phi = acos(1.0 - 2.0 * (i - 0.5) / n_angular)
        angular_dirs[i] = [sin(phi)*cos(theta), sin(phi)*sin(theta), cos(phi)]
    end

    for at_idx in 1:nat
        r_c = coords_bohr[at_idx]
        r_vdw_bohr = Elements.vdw_radius(mol.atoms[at_idx].symbol) * BasisSets.ANGSTROM_TO_BOHR

        # Radial grid (Euler-Maclaurin / logarithmic): r_i = r_vdw * (x / (1 - x))
        for rad_step in 1:n_radial
            x = (rad_step - 0.5) / n_radial
            r_dist = r_vdw_bohr * (x / (1.0 - x))
            dr = r_vdw_bohr / ((1.0 - x)^2 * n_radial)
            rad_weight = r_dist^2 * dr

            for ang_idx in 1:n_angular
                pt_coords = r_c .+ r_dist .* angular_dirs[ang_idx]

                # Becke fuzzy Voronoi atomic partition weight w_A(r)
                w_becke = 1.0
                for other in 1:nat
                    if other != at_idx
                        r_other = coords_bohr[other]
                        d_at = norm(pt_coords .- r_c)
                        d_ot = norm(pt_coords .- r_other)
                        R_bond = norm(r_c .- r_other)
                        mu = (d_at - d_ot) / max(R_bond, 1e-4)
                        # Becke cutoff polynomial p(mu):
                        p1 = 1.5 * mu - 0.5 * mu^3
                        p2 = 1.5 * p1 - 0.5 * p1^3
                        p3 = 1.5 * p2 - 0.5 * p2^3
                        s_cutoff = 0.5 * (1.0 - p3)
                        w_becke *= s_cutoff
                    end
                end

                total_w = rad_weight * angular_weight * w_becke
                if total_w > 1e-12
                    push!(grid_points, GridPoint(pt_coords, total_w))
                end
            end
        end
    end

    return MolecularGrid(grid_points, length(grid_points))
end

"""
    integrate_xc_matrix_and_energy(grid::MolecularGrid, bs::BasisSet,
                                   P::Matrix{Float64}, func::XCFunctional)
Computes the Exchange-Correlation potential matrix V_xc and total XC energy E_xc.
"""
function integrate_xc_matrix_and_energy(grid::MolecularGrid, bs::BasisSet,
                                        P::Matrix{Float64}, func::XCFunctional)
    nb = n_basis(bs)
    V_xc = zeros(Float64, nb, nb)
    E_xc = 0.0

    # Evaluate on grid
    chi_vals = zeros(Float64, nb)

    for pt in grid.points
        r = pt.coords
        w = pt.weight

        # Basis functions at r
        for mu in 1:nb
            chi_vals[mu] = eval_basis_function(bs.functions[mu], r)
        end

        # Electron density rho(r) = sum_{mu, nu} P_mu,nu * chi_mu(r) * chi_nu(r)
        rho = dot(chi_vals, P * chi_vals)
        if rho > 1e-10
            eps_xc, v_xc = DFTFunctionals.evaluate_xc(func, rho)
            E_xc += w * rho * eps_xc

            # V_xc_mu,nu += w * v_xc * chi_mu * chi_nu
            pref = w * v_xc
            for mu in 1:nb
                c_mu = chi_vals[mu]
                for nu in mu:nb
                    v_val = pref * c_mu * chi_vals[nu]
                    V_xc[mu, nu] += v_val
                    if mu != nu
                        V_xc[nu, mu] += v_val
                    end
                end
            end
        end
    end

    return V_xc, E_xc
end

end # module DFTGrid

