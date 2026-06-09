"""
    dssf_SP(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ;
        options_μ = Optim.Options(show_trace=false, iterations=100),
        tol = 1e-12,
        max_iters = 1000,
        include_condensation::Bool = true,
    )

Compute the zero-temperature saddle-point dynamical spin structure factor using
the path-integral Green-function trace formula.

Returns

    ret_normal, ret_condensate

where both arrays have size `3 × length(energies)`.

This version matches the canonical finite-size condensate convention

    ik == aux.conden_index.

The pinned normal contribution is removed only from the negative-pole side of
G(k) in that sector. The mixed condensate contribution is then added from

    G_condensed(k = qc) × G_normal(k + q = qc + q).
"""
function dssf_SP(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    options_μ = Optim.Options(show_trace=false, iterations=100),
    tol = 1e-12,
    max_iters = 1000,
    include_condensation::Bool = true,
)
    num_energies = length(energies)

    ret_normal = zeros(Float64, 3, num_energies)
    ret_condensate = zeros(Float64, 3, num_energies)

    aux = CondensationAux(0.0, 0.0, nothing, 0.0, 0)

    if include_condensation
        μ0 = copy(real(sbs.mean_fields[13:15]))
        optimize_μ0!(sbs, μ0, aux; options=options_μ, tol=tol, max_iters=max_iters)
        condensation_results!(sbs, aux)
    end

    (; L) = sbs
    Ns = 3L^2

    q_reshaped = to_reshaped_rlu(q)

    Uq = [external_vertex(μ, q) for μ in 1:3]
    Umq = [external_vertex(μ, -q) for μ in 1:3]

    k_grid = [Vec3(i/L, j/L, 0.0) for i in 0:L-1, j in 0:L-1, _ in 1:1]

    has_condensate = include_condensation && aux.conden_index !== nothing

    # -------------------------------------------------------------------------
    # Normal-normal contribution.
    #
    # Important:
    #
    # We call Green_SP_normal_residues with aux = nothing here, so it does not
    # globally remove pinned poles. We remove the pinned normal contribution
    # locally, matching the canonical convention:
    #
    #     ik == aux.conden_index
    #
    # and only on the negative-pole side of G(k), which corresponds to the
    # canonical V2 / -k line.
    # -------------------------------------------------------------------------
    for (ik, k) in enumerate(k_grid)
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k =
            Green_SP_normal_residues(sbs, k, nothing)

        ϵs_kq, Vkq, weights_kq =
            Green_SP_normal_residues(sbs, kq, nothing)

        if has_condensate && ik == aux.conden_index
            for a in 1:6
                lneg = 6 + a

                if isapprox(abs(ϵs_k[lneg]), sbs.condensation_ϵ; atol=1e-8)
                    weights_k[lneg] = 0.0
                end
            end
        end

        for a in 1:6
            lneg = 6 + a
            iszero(weights_k[lneg]) && continue

            ω1 = -ϵs_k[lneg]

            for b in 1:6
                lpos = b
                iszero(weights_kq[lpos]) && continue

                ω2 = ϵs_kq[lpos]
                ΔE = ω1 + ω2

                for μ in 1:3
                    v1 = @view Vk[:, lneg]
                    v2 = @view Vkq[:, lpos]

                    pref =
                        weights_k[lneg] *
                        weights_kq[lpos] *
                        Ĩ[lneg, lneg] *
                        Ĩ[lpos, lpos]

                    trace_weight =
                        pref *
                        dot(v1, Uq[μ] * v2) *
                        dot(v2, Umq[μ] * v1)

                    weight = -real(trace_weight) / (8Ns)

                    for (ie, energy) in enumerate(energies)
                        ret_normal[μ, ie] +=
                            weight *
                            lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end
    end

    # -------------------------------------------------------------------------
    # Condensate-normal contribution.
    #
    # This matches the canonical convention ik == aux.conden_index.
    #
    # The condensed canonical line is the V2 / -k line, which corresponds to the
    # negative pole of G(k). Therefore we set
    #
    #     k = qc,
    #     k + q = qc + q.
    #
    # Green_SP_condensed_residues already assigns the pinned poles weight
    # aux.ξ + 1. The extra L^2 below is the collapsed finite-size momentum sum.
    # -------------------------------------------------------------------------
    if has_condensate
        qc = k_grid[aux.conden_index]

        k = qc
        kq = qc + q_reshaped

        ϵs_c, Vc, weights_c =
            Green_SP_condensed_residues(sbs, k, aux)

        # Use aux = nothing here. The normal line should not have pinned poles
        # removed globally.
        ϵs_n, Vn, weights_n =
            Green_SP_normal_residues(sbs, kq, nothing)

        for a in 1:6
            lcond_neg = 6 + a
            iszero(weights_c[lcond_neg]) && continue

            for b in 1:6
                lpos = b
                iszero(weights_n[lpos]) && continue

                # Condensed line has zero physical energy.
                ΔE = ϵs_n[lpos]

                for μ in 1:3
                    v1 = @view Vc[:, lcond_neg]
                    v2 = @view Vn[:, lpos]

                    pref =
                        weights_c[lcond_neg] *
                        weights_n[lpos] *
                        Ĩ[lcond_neg, lcond_neg] *
                        Ĩ[lpos, lpos]

                    trace_weight =
                        pref *
                        dot(v1, Uq[μ] * v2) *
                        dot(v2, Umq[μ] * v1)

                    weight = -L^2 * real(trace_weight) / (8Ns)

                    for (ie, energy) in enumerate(energies)
                        ret_condensate[μ, ie] +=
                            weight *
                            lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end
    end

    return ret_normal, ret_condensate
end