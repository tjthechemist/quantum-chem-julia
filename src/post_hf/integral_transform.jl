# ==============================================================================
# QuantumChemMM.jl: AO-to-MO 4-Index Integral Transformation
# Fast O(N^5) Quarter-Transformation for Post-Hartree-Fock Correlation Methods
# ==============================================================================

module IntegralTransform

using LinearAlgebra

export transform_eri_ao_to_mo, antisymmetrize_mo_eri

"""
    transform_eri_ao_to_mo(eri_ao::Array{Float64, 4}, C::Matrix{Float64})::Array{Float64, 4}
Transforms Two-Electron Repulsion Integrals (ERIs) from Atomic Orbital (AO) basis to
Molecular Orbital (MO) basis using four successive quarter transformations:
(pq|rs) = sum_{mu, nu, lam, sig} C_mu,p * C_nu,q * C_lam,r * C_sig,s * (mu nu | lam sig).
"""
function transform_eri_ao_to_mo(eri_ao::Array{Float64, 4}, C::Matrix{Float64})::Array{Float64, 4}
    nb, _, _, _ = size(eri_ao)
    n_mo = size(C, 2)

    # 1. First quarter transformation: (mu, nu, lam, s) = sum_sig C_sig,s * (mu, nu, lam, sig)
    step1 = zeros(Float64, nb, nb, nb, n_mo)
    for s in 1:n_mo
        for lam in 1:nb
            for nu in 1:nb
                for mu in 1:nb
                    val = 0.0
                    for sig in 1:nb
                        val += C[sig, s] * eri_ao[mu, nu, lam, sig]
                    end
                    step1[mu, nu, lam, s] = val
                end
            end
        end
    end

    # 2. Second quarter transformation: (mu, nu, r, s) = sum_lam C_lam,r * (mu, nu, lam, s)
    step2 = zeros(Float64, nb, nb, n_mo, n_mo)
    for s in 1:n_mo
        for r in 1:n_mo
            for nu in 1:nb
                for mu in 1:nb
                    val = 0.0
                    for lam in 1:nb
                        val += C[lam, r] * step1[mu, nu, lam, s]
                    end
                    step2[mu, nu, r, s] = val
                end
            end
        end
    end

    # 3. Third quarter transformation: (mu, q, r, s) = sum_nu C_nu,q * (mu, nu, r, s)
    step3 = zeros(Float64, nb, n_mo, n_mo, n_mo)
    for s in 1:n_mo
        for r in 1:n_mo
            for q in 1:n_mo
                for mu in 1:nb
                    val = 0.0
                    for nu in 1:nb
                        val += C[nu, q] * step2[mu, nu, r, s]
                    end
                    step3[mu, q, r, s] = val
                end
            end
        end
    end

    # 4. Fourth quarter transformation: (p, q, r, s) = sum_mu C_mu,p * (mu, q, r, s)
    eri_mo = zeros(Float64, n_mo, n_mo, n_mo, n_mo)
    for s in 1:n_mo
        for r in 1:n_mo
            for q in 1:n_mo
                for p in 1:n_mo
                    val = 0.0
                    for mu in 1:nb
                        val += C[mu, p] * step3[mu, q, r, s]
                    end
                    eri_mo[p, q, r, s] = val
                end
            end
        end
    end

    return eri_mo
end

"""
    antisymmetrize_mo_eri(eri_mo::Array{Float64, 4})::Array{Float64, 4}
Converts chemists' notation (pq|rs) into antisymmetrized physicists' notation:
<pq||rs> = (pr|qs) - (ps|qr).
"""
function antisymmetrize_mo_eri(eri_mo::Array{Float64, 4})::Array{Float64, 4}
    n = size(eri_mo, 1)
    anti = zeros(Float64, n, n, n, n)
    for p in 1:n
        for q in 1:n
            for r in 1:n
                for s in 1:n
                    anti[p, q, r, s] = eri_mo[p, r, q, s] - eri_mo[p, s, q, r]
                end
            end
        end
    end
    return anti
end

end # module IntegralTransform

