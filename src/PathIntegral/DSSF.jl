# src/PathIntegral/DSSF.jl

"""
    dssf_SP(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ;
        aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
    )

Compute the positive-frequency diagonal saddle-point DSSF using the
path-integral Green-function spectral representation.

The result is a named tuple with fields `ordinary_normal`,
`ordinary_condensed`, `active_constraint`, and `total`. Each field has size
`3 × length(energies)`.

The ordinary terms use the complete unit-residue saddle-point Green function.
`ordinary_condensed` contains both mixed orientations involving the selected
sector; the selected-selected elastic contribution is omitted. All signed-pole
transitions allowed at positive frequency are retained at finite temperature.

For an active soft-minimum constraint, `active_constraint` contains the
separate fixed-`ξ` source-source contribution. The enhanced occupation is not
inserted into an ordinary Green-function residue or Bose factor.

`Γ` is the full width at half maximum of the Lorentzian broadening.
"""
function dssf_SP(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
)
    num_energies = length(energies)

    ret_ordinary_normal = zeros(Float64, 3, num_energies)
    ret_ordinary_condensed = zeros(Float64, 3, num_energies)
    ret_active_constraint = zeros(Float64, 3, num_energies)

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

    qc = _spectral_condensation_momentum(aux, L)
    selected_band_mask = falses(12)

    for band in aux.conden_band_indices
        selected_band_mask[band] = true
    end

    # Complete ordinary unit-residue response from S_eff.
    for k in k_grid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = _full_sp_residues(sbs, k)
        ϵs_kq, Vkq, weights_kq = _full_sp_residues(sbs, kq)

        k_is_selected = _same_momentum_mod1(k, qc)
        kq_is_selected = _same_momentum_mod1(kq, qc)

        for m in eachindex(ϵs_k)
            Em = ϵs_k[m]
            m_selected = k_is_selected && selected_band_mask[m]

            for n in eachindex(ϵs_kq)
                En = ϵs_kq[n]
                n_selected = kq_is_selected && selected_band_mask[n]

                m_selected && n_selected && continue

                ΔE = real(En - Em)
                transition_factor =
                    _dssf_transition_factor(Em, En, βtemp)

                iszero(transition_factor) && continue

                ret_sector =
                    m_selected != n_selected ?
                    ret_ordinary_condensed :
                    ret_ordinary_normal

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
                        -real(trace_weight) *
                        transition_factor /
                        (8Ns)

                    for (ie, energy) in enumerate(energies)
                        ret_sector[μ, ie] +=
                            weight * lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end
    end

    # Separate fixed-ξ source-source curvature.
    if aux.selection_kind === :pinned
        ϵs_c, Vc, _ = _selected_unit_residues(sbs, qc, aux)

        active_weights = _active_positive_weights(ϵs_c, aux)
        active_mask = _active_positive_mask(ϵs_c, aux)
        unit_active_weights = zeros(Float64, length(ϵs_c))

        Nflavor = 2.0

        # First ordering: qc -> qc + q -> qc.
        kn = qc + q_reshaped
        ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)

        exclude_active_intermediate =
            _same_momentum_mod1(kn, qc)

        for i in eachindex(ϵs_c)
            ξi = active_weights[i]
            iszero(ξi) && continue

            Ei = ϵs_c[i]

            fill!(unit_active_weights, 0.0)
            unit_active_weights[i] = 1.0

            for n in eachindex(ϵs_n)
                if exclude_active_intermediate && active_mask[n]
                    continue
                end

                En = ϵs_n[n]
                ΔE = real(En - Ei)
                dssf_factor =
                    _dssf_fluctuation_dissipation_factor(ΔE, βtemp)

                iszero(dssf_factor) && continue

                for μ in 1:3
                    coherence = _residue_vertex_trace(
                        Vn,
                        weights_n,
                        n,
                        Umq[μ],
                        Vc,
                        unit_active_weights,
                        i,
                        Uq[μ],
                    )

                    weight =
                        Nflavor *
                        ξi *
                        real(coherence) *
                        dssf_factor /
                        4

                    for (ie, energy) in enumerate(energies)
                        ret_active_constraint[μ, ie] +=
                            weight * lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end

        # Second ordering: qc -> qc - q -> qc.
        kn = qc - q_reshaped
        ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)

        exclude_active_intermediate =
            _same_momentum_mod1(kn, qc)

        for i in eachindex(ϵs_c)
            ξi = active_weights[i]
            iszero(ξi) && continue

            Ei = ϵs_c[i]

            fill!(unit_active_weights, 0.0)
            unit_active_weights[i] = 1.0

            for m in eachindex(ϵs_n)
                if exclude_active_intermediate && active_mask[m]
                    continue
                end

                Em = ϵs_n[m]
                ΔE = real(Ei - Em)
                dssf_factor =
                    _dssf_fluctuation_dissipation_factor(ΔE, βtemp)

                iszero(dssf_factor) && continue

                for μ in 1:3
                    coherence = _residue_vertex_trace(
                        Vc,
                        unit_active_weights,
                        i,
                        Umq[μ],
                        Vn,
                        weights_n,
                        m,
                        Uq[μ],
                    )

                    weight =
                        -Nflavor *
                        ξi *
                        real(coherence) *
                        dssf_factor /
                        4

                    for (ie, energy) in enumerate(energies)
                        ret_active_constraint[μ, ie] +=
                            weight * lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end
    end

    ret_total =
        ret_ordinary_normal .+
        ret_ordinary_condensed .+
        ret_active_constraint

    return (
        ordinary_normal = ret_ordinary_normal,
        ordinary_condensed = ret_ordinary_condensed,
        active_constraint = ret_active_constraint,
        total = ret_total,
    )
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
