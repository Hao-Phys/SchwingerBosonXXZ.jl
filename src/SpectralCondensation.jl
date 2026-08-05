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

Construct the auxiliary object used to identify the algebraically selected
soft-mode sector and the separate active soft-minimum occupation.

The input `sbs` is treated as read-only. When `check_constraint=true`,
`expectation_values` is evaluated on a deep copy because that routine
internally optimizes the constraint fields.

The returned data follow these conventions:

- `conden_band_indices` contains the signed BdG poles belonging to the
  selected sector. Their ordinary Green-function residues remain exactly one.
- `active_positive_weights[l]` contains only the additional enhanced
  occupation `ξₗ`.
- For `selection_kind === :finite_size_minimum`,
  `active_positive_weights` is identically zero.
- For `selection_kind === :pinned`, nonzero active weights occur only on
  selected positive-energy poles. Negative-energy Nambu partners do not carry
  independent active occupations.
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

    # `expectation_values` optimizes μ₀ internally. Perform the check on a
    # private copy so metadata construction cannot mutate the supplied saddle.
    if check_constraint
        sbs_check = deepcopy(sbs)
        res = expectation_values(sbs_check)

        constraint_values = real.(res[2])
        target = 2S

        if !all(
            x -> isapprox(
                x,
                target;
                atol = constraint_atol,
            ),
            constraint_values,
        )
            error(
                "The input SchwingerBosonSystem does not appear to " *
                "satisfy the constraint.\n" *
                "Expected expectation_values(sbs)[2] ≈ $(target), " *
                "got $(constraint_values).\n" *
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
        q = Vec3([
            (i - 1) / L,
            (j - 1) / L,
            0.0,
        ])

        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)

        gap = minimum(@view E[1:6])

        if gap < best_gap
            best_index = ik
            best_gap = gap
            best_E .= E
        end

        if condensation_ϵ > 0.0
            pinned =
                findall(
                    l -> isapprox(
                        abs(E[l]),
                        condensation_ϵ;
                        atol = pin_atol,
                    ),
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
        conden_band_indices =
            sort(unique(pinned_band_indices))
        min_gap = pinned_gap
    else
        conden_index = best_index
        selection_kind = :finite_size_minimum
        conden_band_indices =
            findall(
                l -> isapprox(
                    abs(best_E[l]),
                    best_gap;
                    atol = degeneracy_atol,
                ),
                eachindex(best_E),
            )
        conden_band_indices =
            sort(unique(conden_band_indices))
        min_gap = best_gap
    end

    positive_band_indices =
        filter(
            l -> 1 <= l <= 6,
            conden_band_indices,
        )

    if conden_index == 0 ||
       isempty(positive_band_indices)

        error(
            "Failed to select a spectral soft-mode sector.",
        )
    end

    active_positive_weights =
        zeros(Float64, 12)

    if selection_kind === :pinned
        i = (conden_index - 1) ÷ L + 1
        j = (conden_index - 1) % L + 1

        q_condensed = Vec3([
            (i - 1) / L,
            (j - 1) / L,
            0.0,
        ])

        dynamical_matrix!(D, sbs, q_condensed)
        bogoliubov!(V, D)

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
                "Selected pinned mode has vanishing constraint " *
                "charge. qcsum = $(qcsum) at " *
                "q = $(q_condensed).",
            )
        end

        ξ = (2S + 1 - N_normal) / qcsum

        if ξ < -weight_atol
            error(
                "Negative active soft-minimum weight ξ = $(ξ).\n" *
                "This indicates that the input saddle is " *
                "inconsistent with the pinned treatment.",
            )
        end

        ξ = max(ξ, 0.0)

        for band in positive_band_indices
            active_positive_weights[band] = ξ
        end
    end

    return SpectralCondensationAux(
        conden_index = conden_index,
        selection_kind = selection_kind,
        conden_band_indices = conden_band_indices,
        active_positive_weights =
            active_positive_weights,
        min_gap = min_gap,
    )
end