# src/SpectralCondensation.jl

"""
    spectral_condensation_aux(
        sbs::SchwingerBosonSystem;
        check_constraint::Bool = true,
        constraint_atol::Float64 = 1e-4,
        pin_atol::Float64 = 1e-6,
        degeneracy_atol::Float64 = 1e-6,
        weight_atol::Float64 = 1e-10,
    ) -> SpectralCondensationAux

Construct the spectral-condensation auxiliary object used to split normal and
condensed contributions in DSSF calculations.

This function assumes that `sbs` is already an optimized saddle-point solution.
By default, it checks the local constraint by evaluating `expectation_values(sbs)`
and requiring `expectation_values(sbs)[2] ≈ 2S` on all sublattices. This
check can be skipped with `check_constraint = false`.

The condensed sector is selected as follows:

  - If BdG modes pinned near `±sbs.condensation_ϵ` are found, those modes are
    treated as the condensed sector and `selection_kind = :pinned`.

  - If no pinned mode is found, the smallest positive-energy finite-size BdG
    mode is treated as the condensed sector, together with any degenerate
    partners within `degeneracy_atol`. In this case
    `selection_kind = :finite_size_minimum`.

The returned `SpectralCondensationAux` stores the condensed momentum index,
the selected BdG pole indices, their condensate weights `nc_i`, the number of
positive-energy condensed modes, the selection kind, and the raw finite-size
minimum gap.

For pinned modes, the condensate weights are computed as `nc_i = ξ_i + 1`,
where `ξ_i` is the soft-min condensate fraction inferred from the constraint
sum rule. For finite-size-gap modes, the weights are computed from the regular
finite-size BdG expression, not from the soft-min `ξ_i`.

Keyword arguments:

  - `check_constraint`: whether to check the optimized constraint before
    selecting the condensed sector.

  - `constraint_atol`: absolute tolerance for the constraint check.

  - `pin_atol`: absolute tolerance used to identify pinned modes near
    `±sbs.condensation_ϵ`.

  - `degeneracy_atol`: absolute tolerance used to include degenerate partners
    of the smallest finite-size gap.

  - `weight_atol`: tolerance below which small negative condensate weights are
    treated as numerical roundoff.
"""
function spectral_condensation_aux(
    sbs::SchwingerBosonSystem;
    check_constraint::Bool = true,
    constraint_atol::Float64 = 1e-3,
    pin_atol::Float64 = 1e-6,
    degeneracy_atol::Float64 = 1e-6,
    weight_atol::Float64 = 1e-10,
)
    (; L, S, condensation_ϵ) = sbs

    if check_constraint
        res = expectation_values(sbs)
        constraint_values = real.(res[2])
        target = 2S

        if !all(x -> isapprox(x, target; atol = constraint_atol), constraint_values)
            error(
                "The input SchwingerBosonSystem does not appear to satisfy the constraint. " *
                "Expected expectation_values(sbs)[2] ≈ $(target), got $(constraint_values). " *
                "Pass check_constraint = false to skip this check.",
            )
        end
    end

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    pinned_index = 0
    pinned_gap = Inf
    pinned_band_indices = Int[]

    best_index = 0
    best_gap = Inf
    best_E = zeros(Float64, 12)

    for i in 1:L, j in 1:L
        ik = (i - 1) * L + j
        q = Vec3([(i - 1) / L, (j - 1) / L, 0.0])

        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)

        gap = minimum(@view E[1:6])

        if gap < best_gap
            best_index = ik
            best_gap = gap
            best_E .= E
        end

        pinned = findall(
            l -> isapprox(abs(E[l]), condensation_ϵ; atol = pin_atol),
            eachindex(E),
        )

        if !isempty(pinned) && gap < pinned_gap
            pinned_index = ik
            pinned_gap = gap
            pinned_band_indices = pinned
        end
    end

    if pinned_index != 0
        conden_index = pinned_index
        selection_kind = :pinned
        conden_band_indices = pinned_band_indices
        min_gap = pinned_gap
    else
        conden_index = best_index
        selection_kind = :finite_size_minimum
        conden_band_indices = findall(
            l -> isapprox(abs(best_E[l]), best_gap; atol = degeneracy_atol),
            eachindex(best_E),
        )
        min_gap = best_gap
    end

    num_conden = count(l -> 1 <= l <= 6, conden_band_indices)

    if conden_index == 0 || num_conden == 0
        error("Failed to select a spectral condensate sector.")
    end

    i = (conden_index - 1) ÷ L + 1
    j = (conden_index - 1) % L + 1
    q_condensed = Vec3([(i - 1) / L, (j - 1) / L, 0.0])

    dynamical_matrix!(D, sbs, q_condensed)
    bogoliubov!(V, D)

    ∂ID∂μ = zeros(ComplexF64, 12, 12)
    ∂ID∂μ0!(∂ID∂μ, 1)

    condensate_weights = zeros(Float64, 12)
    positive_band_indices = filter(l -> 1 <= l <= 6, conden_band_indices)

    if selection_kind === :pinned
        Nu = L^2

        P_tmp = zeros(ComplexF64, 12, 12)
        D_tmp = zeros(ComplexF64, 12, 12)
        V_tmp = zeros(ComplexF64, 12, 12)
        tmp = zeros(ComplexF64, 12, 12)

        N_normal = 0.0

        for i in 1:L, j in 1:L
            q_tmp = Vec3([(i - 1) / L, (j - 1) / L, 0.0])
            single_particle_density_matrix!(P_tmp, D_tmp, V_tmp, tmp, sbs, q_tmp)
            N_normal += -real(tr(P_tmp * ∂ID∂μ)) / Nu
        end

        qcsum = 0.0

        for band in positive_band_indices
            v = view(V, :, band)
            qcsum += -real(v' * Ĩ * ∂ID∂μ * v)
        end

        ξ = (2S + 1 - N_normal) / qcsum

        if ξ < -weight_atol
            error(
                "Negative soft-min condensate fraction ξ = $ξ for pinned modes. " *
                "This indicates that the input saddle point is inconsistent with the pinned condensation treatment.",
            )
        end

        nc = max(ξ, 0.0) + 1.0

        for band in conden_band_indices
            condensate_weights[band] = nc
        end
    elseif selection_kind === :finite_size_minimum
        qsum = 0.0

        for band in positive_band_indices
            v = view(V, :, band)
            q = -real(v' * Ĩ * ∂ID∂μ * v)

            if q < -weight_atol
                error(
                    "Negative regular finite-size condensate weight q = $q " *
                    "for band $band at q = $q_condensed.",
                )
            end

            qsum += max(q, 0.0)
        end

        nc = qsum / num_conden

        for band in conden_band_indices
            condensate_weights[band] = nc
        end
    end

    return SpectralCondensationAux(
        conden_index = conden_index,
        num_conden = num_conden,
        selection_kind = selection_kind,
        conden_band_indices = conden_band_indices,
        condensate_weights = condensate_weights,
        min_gap = min_gap,
    )
end