# src/PathIntegral/DSSF.jl

"""
    dssf_SP(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ;
        aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
    )

Compute the saddle-point dynamical spin structure factor using the
path-integral Green-function residue formula.

Returns `ret_normal, ret_condensate`, where both arrays have size
`3 × length(energies)`.

The normal-normal part uses `Green_SP_normal_residues`, so the selected
soft-mode poles stored in `aux` are removed from the normal sector.

The selected-normal part uses `Green_SP_condensed_residues`, so the same
selected soft-mode poles are inserted into the selected sector with the total
Green-function pole weights stored in `aux.condensate_weights`.

The branch-dependent collapsed momentum-sum factor is handled by
`_condensate_sum_factor(aux, L)`:

    finite_size_minimum: 1
    pinned:              L^2

Thus a finite-size selected pole is only split out as an ordinary BdG pole,
while an active pinned soft-min pole receives the macroscopic collapsed-sum
enhancement.

For the selected-normal terms, the transition energy keeps the finite selected
soft-mode energy:

    Orientation 1: ΔE = E_normal(k + q) - E_selected(k)
    Orientation 2: ΔE = E_selected(k + q) - E_normal(k)

This is important for finite-size selected modes whose soft-mode energy is not
exactly zero.

The purely elastic selected-selected contribution is not included.
"""
function dssf_SP(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
)
    num_energies = length(energies)

    ret_normal = zeros(Float64, 3, num_energies)
    ret_condensate = zeros(Float64, 3, num_energies)

    (; L) = sbs

    Ns = 3L^2
    βtemp = _inverse_temperature(sbs)

    q_ext = Vec3(q[1], q[2], q[3])
    q_reshaped = to_reshaped_rlu(q_ext)

    Uq = [external_vertex(μ, q_ext) for μ in 1:3]
    Umq = [external_vertex(μ, -q_ext) for μ in 1:3]

    k_grid = Vec3[]

    for i in 1:L, j in 1:L
        push!(k_grid, Vec3([(i - 1) / L, (j - 1) / L, 0.0]))
    end

    # ------------------------------------------------------------------
    # Normal-normal contribution.
    # ------------------------------------------------------------------

    for k in k_grid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for m in eachindex(ϵs_k)
            iszero(weights_k[m]) && continue

            Em = ϵs_k[m]

            for n in eachindex(ϵs_kq)
                iszero(weights_kq[n]) && continue

                En = ϵs_kq[n]
                ΔE = real(En - Em)

                transition_factor = _dssf_transition_factor(
                    Em,
                    En,
                    βtemp,
                )

                iszero(transition_factor) && continue

                for μ in 1:3
                    trace_weight = _residue_vertex_trace(
                        Vkq,
                        weights_kq,
                        n,
                        Umq[μ],
                        Vk,
                        weights_k,
                        m,
                        Uq[μ],
                    )

                    weight =
                        -real(trace_weight) * transition_factor / (8Ns)

                    for (ie, energy) in enumerate(energies)
                        ret_normal[μ, ie] +=
                            weight * lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Selected-normal contribution.
    #
    # Selected-normal terms are collected in `ret_condensate`, while the
    # purely selected-selected elastic piece is omitted.
    # ------------------------------------------------------------------

    qc = _spectral_condensation_momentum(aux, L)
    condensate_sum_factor = _condensate_sum_factor(aux, L)

    # --------------------------------------------------------------
    # Orientation 1:
    #
    #     G_n(k + q) G_c(k),
    #
    # with k = qc. The selected pole is on the second line.
    # --------------------------------------------------------------

    k = qc
    kq = qc + q_reshaped

    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, k, aux)
    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kq, aux)

    for a in 1:6
        lcond_neg = 6 + a
        iszero(weights_c[lcond_neg]) && continue

        Em = ϵs_c[lcond_neg]

        for b in 1:6
            lpos = b
            iszero(weights_n[lpos]) && continue

            En = ϵs_n[lpos]
            ΔE = real(En - Em)

            transition_factor = _dssf_transition_factor(
                Em,
                En,
                βtemp,
            )

            iszero(transition_factor) && continue

            for μ in 1:3
                trace_weight = _residue_vertex_trace(
                    Vn,
                    weights_n,
                    lpos,
                    Umq[μ],
                    Vc,
                    weights_c,
                    lcond_neg,
                    Uq[μ],
                )

                weight = transition_factor * (
                    -condensate_sum_factor * real(trace_weight) / (8Ns)
                )

                for (ie, energy) in enumerate(energies)
                    ret_condensate[μ, ie] +=
                        weight * lorentzian(energy - ΔE, Γ)
                end
            end
        end
    end

    # --------------------------------------------------------------
    # Orientation 2:
    #
    #     G_c(k + q) G_n(k),
    #
    # with k + q = qc. The selected pole is on the first line.
    # --------------------------------------------------------------

    k = qc - q_reshaped
    kq = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, k, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kq, aux)

    for a in 1:6
        lneg = 6 + a
        iszero(weights_n[lneg]) && continue

        Em = ϵs_n[lneg]

        for b in 1:6
            lcond_pos = b
            iszero(weights_c[lcond_pos]) && continue

            En = ϵs_c[lcond_pos]
            ΔE = real(En - Em)

            transition_factor = _dssf_transition_factor(
                Em,
                En,
                βtemp,
            )

            iszero(transition_factor) && continue

            for μ in 1:3
                trace_weight = _residue_vertex_trace(
                    Vc,
                    weights_c,
                    lcond_pos,
                    Umq[μ],
                    Vn,
                    weights_n,
                    lneg,
                    Uq[μ],
                )

                weight = transition_factor * (
                    -condensate_sum_factor * real(trace_weight) / (8Ns)
                )

                for (ie, energy) in enumerate(energies)
                    ret_condensate[μ, ie] +=
                        weight * lorentzian(energy - ΔE, Γ)
                end
            end
        end
    end

    return ret_normal, ret_condensate
end

"""
    dssf_FL(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ;
        aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
        Nflavor::Real = 2,
        κtol::Real = 1e-12,
        return_components::Bool = false,
    )

Compute the Gaussian-fluctuation Fig. 1(b) contribution to the dynamical
spin structure factor.

The Lorentzian helper `lorentzian(x, Γ)` uses Γ as the full width at half
maximum. Therefore the retarded frequency broadening used here is η = Γ / 2.

The external-bubble decomposition is

    S_full = S_normal + S_mixed,

where `S_mixed` denotes the mixed normal-condensate external-internal bubble,
not a condensate-condensate elastic bubble.

The returned backward-compatible pair is

    ret_FL_normal, ret_FL_condensate,

with

    ret_FL_normal      = S_normal D S_normal,
    ret_FL_condensate = S_mixed D S_mixed
                       + S_normal D S_mixed
                       + S_mixed D S_normal.

If `return_components=true`, a named tuple is returned with the separated
pieces:

    normal,
    mixed_mixed,
    normal_mixed,
    mixed_normal,
    cross,
    condensate,
    total.
"""
function dssf_FL(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
    Nflavor::Real = 2,
    κtol::Real = 1e-12,
    return_components::Bool = false,
)
    num_energies = length(energies)

    ret_FL_normal = zeros(Float64, 3, num_energies)
    ret_FL_mixed_mixed = zeros(Float64, 3, num_energies)
    ret_FL_normal_mixed = zeros(Float64, 3, num_energies)
    ret_FL_mixed_normal = zeros(Float64, 3, num_energies)

    (; L) = sbs

    q_ext = Vec3(q[1], q[2], q[3])
    q_reshaped = to_reshaped_rlu(q_ext)

    k_grid = Vec3[]

    for i in 1:L, j in 1:L
        push!(k_grid, Vec3([(i - 1) / L, (j - 1) / L, 0.0]))
    end

    fields_all = internal_field_basis()
    nϕ_all = length(fields_all)

    Π0_all = zeros(ComplexF64, nϕ_all, nϕ_all)
    Pi0!(Π0_all, sbs, fields_all)

    active_indices = Int[]

    for i in eachindex(fields_all)
        field = fields_all[i]

        if field.kind === :λ || abs(Π0_all[i, i]) > κtol
            push!(active_indices, i)
        end
    end

    fields = fields_all[active_indices]
    nϕ = length(fields)

    Π0 = zeros(ComplexF64, nϕ, nϕ)
    Π = zeros(ComplexF64, nϕ, nϕ)
    K = zeros(ComplexF64, nϕ, nϕ)

    Scol_normal = zeros(ComplexF64, nϕ)
    Srow_normal = zeros(ComplexF64, nϕ)
    Scol_full = zeros(ComplexF64, nϕ)
    Srow_full = zeros(ComplexF64, nϕ)
    Scol_mixed = zeros(ComplexF64, nϕ)
    Srow_mixed = zeros(ComplexF64, nϕ)

    Pi0!(Π0, sbs, fields)

    # Γ is the FWHM used by `lorentzian`; η is the retarded HWHM.
    η = Γ / 2

    for (ie, energy) in enumerate(energies)
        z = energy + im * η

        fill!(Π, 0.0 + 0.0im)
        fill!(K, 0.0 + 0.0im)

        polarization!(
            Π,
            sbs,
            fields,
            k_grid,
            q_reshaped,
            z;
            Nflavor = Nflavor,
            aux = aux,
        )

        rpa_kernel!(K, Π0, Π)

        for μ in 1:3
            external_internal_bubble!(
                Scol_normal,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = η,
                aux = aux,
                include_condensate = false,
            )

            external_internal_bubble_row!(
                Srow_normal,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = η,
                aux = aux,
                include_condensate = false,
            )

            external_internal_bubble!(
                Scol_full,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = η,
                aux = aux,
                include_condensate = true,
            )

            external_internal_bubble_row!(
                Srow_full,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = η,
                aux = aux,
                include_condensate = true,
            )

            @. Scol_mixed = Scol_full - Scol_normal
            @. Srow_mixed = Srow_full - Srow_normal

            D_Srow_normal = K \ Srow_normal
            D_Srow_mixed = K \ Srow_mixed

            χ_FL_normal =
                (1 / Nflavor) *
                (transpose(Scol_normal) * D_Srow_normal)

            χ_FL_normal_mixed =
                (1 / Nflavor) *
                (transpose(Scol_normal) * D_Srow_mixed)

            χ_FL_mixed_normal =
                (1 / Nflavor) *
                (transpose(Scol_mixed) * D_Srow_normal)

            χ_FL_mixed_mixed =
                (1 / Nflavor) *
                (transpose(Scol_mixed) * D_Srow_mixed)

            ret_FL_normal[μ, ie] = imag(χ_FL_normal) / π
            ret_FL_normal_mixed[μ, ie] = imag(χ_FL_normal_mixed) / π
            ret_FL_mixed_normal[μ, ie] = imag(χ_FL_mixed_normal) / π
            ret_FL_mixed_mixed[μ, ie] = imag(χ_FL_mixed_mixed) / π
        end
    end

    ret_FL_cross = ret_FL_normal_mixed .+ ret_FL_mixed_normal
    ret_FL_condensate = ret_FL_mixed_mixed .+ ret_FL_cross
    ret_FL_total = ret_FL_normal .+ ret_FL_condensate

    if return_components
        return (
            normal = ret_FL_normal,
            mixed_mixed = ret_FL_mixed_mixed,
            normal_mixed = ret_FL_normal_mixed,
            mixed_normal = ret_FL_mixed_normal,
            cross = ret_FL_cross,
            condensate = ret_FL_condensate,
            total = ret_FL_total,
        )
    end

    return ret_FL_normal, ret_FL_condensate
end