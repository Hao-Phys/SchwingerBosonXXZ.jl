# src/SpectralCondensation.jl

"""
    spectral_condensation_aux(
        sbs::SchwingerBosonSystem;
        check_constraint::Bool = true,
        constraint_atol::Float64 = 1e-3,
        pin_atol::Float64 = 1e-6,
        degeneracy_atol::Float64 = 1e-6,
        weight_atol::Float64 = 1e-10,
    ) -> SpectralCondensationAux

Construct the spectral-condensation auxiliary object used to split the
saddle-point Green function into normal and selected soft-mode sectors.

This function assumes that `sbs` is already an optimized saddle-point solution.

The returned `condensate_weights` are total pole multipliers for the split
Green function. They are consumed directly by `Green_SP_condensed_residues`.

Conventions:

- For `selection_kind === :finite_size_minimum`, the selected mode is only an
  ordinary finite-size BdG pole split out from the normal sector. Therefore its
  total Green-function pole multiplier is exactly `1`.

- For `selection_kind === :pinned`, the selected pole is removed from the normal
  sector and reinserted in the condensed sector together with the extra soft-min
  occupation. If `ξ` denotes the extra soft-min occupation inferred from the
  constraint sum rule, the total Green-function pole multiplier is `1 + ξ`.

The finite-size branch therefore does not compute a condensate fraction from the
mode charge. It only selects the pole and assigns unit weight.
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
                "The input SchwingerBosonSystem does not appear to satisfy the constraint.\n" *
                "Expected expectation_values(sbs)[2] ≈ $(target), got $(constraint_values).\n" *
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

        if condensation_ϵ > 0
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

    condensate_weights = zeros(Float64, 12)
    positive_band_indices = filter(l -> 1 <= l <= 6, conden_band_indices)

    if selection_kind === :finite_size_minimum
        # Exact finite-size split:
        #
        #     G_full = G_normal + G_selected
        #
        # The selected mode is an ordinary BdG pole. Its total pole multiplier
        # in the split Green function is exactly one.
        for band in conden_band_indices
            condensate_weights[band] = 1.0
        end

    elseif selection_kind === :pinned
        # Active soft-min treatment:
        #
        # `single_particle_density_matrix!` already contains the ordinary BdG
        # pole with unit multiplier. The canonical soft-min correction adds an
        # extra occupation ξ. Since the split Green function removes the
        # selected pole from the normal sector, the condensed sector must carry
        # the total multiplier 1 + ξ.
        Nu = L^2

        ∂ID∂μ = zeros(ComplexF64, 12, 12)
        ∂ID∂μ0!(∂ID∂μ, 1)

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

        if abs(qcsum) < weight_atol
            error(
                "Selected pinned mode has vanishing constraint charge. " *
                "qcsum = $qcsum at q = $q_condensed.",
            )
        end

        ξ = (2S + 1 - N_normal) / qcsum

        if ξ < -weight_atol
            error(
                "Negative soft-min condensate fraction ξ = $ξ for pinned modes.\n" *
                "This indicates that the input saddle point is inconsistent " *
                "with the pinned condensation treatment.",
            )
        end

        total_weight = 1.0 + max(ξ, 0.0)

        for band in conden_band_indices
            condensate_weights[band] = total_weight
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