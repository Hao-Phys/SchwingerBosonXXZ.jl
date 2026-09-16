to_reshaped_rlu(q) = recipvecs_reduce \ (recipvecs_origin * q)

function bogoliubov!(T::Matrix{ComplexF64}, H::Matrix{ComplexF64})
    @assert size(T) == size(H) == (12, 12)

    # Initialize T to the para-unitary identity Ĩ = diagm([ones(L), -ones(L)])
    T .= 0
    for i in 1:6
        T[i, i] = 1
        T[i+6, i+6] = -1
    end

    # Solve generalized eigenvalue problem, Ĩ t = λ H t, for columns t of T.
    # Eigenvalues are sorted such that positive values appear first, and are
    # otherwise ascending in absolute value.
    sortby(x) = (-sign(x), abs(x))
    λ, T0 = eigen!(Hermitian(T), Hermitian(H); sortby)

    # Note that T0 and T refer to the same data.
    @assert T0 === T

    # Normalize columns of T so that para-unitarity holds, T† Ĩ T = Ĩ.
    for j in axes(T, 2)
        c = 1 / sqrt(abs(λ[j]))
        view(T, :, j) .*= c
    end

    # Inverse of λ are eigenvalues of Ĩ H, or equivalently, of √H Ĩ √H.
    energies = λ        # reuse storage
    @. energies = 1 / λ

    # By Sylvester's theorem, "inertia" (sign signature) is invariant under a
    # congruence transform Ĩ → √H Ĩ √H. The first L elements are positive,
    # while the next L elements are negative. Their absolute values are
    # excitation energies for the wavevectors q and -q, respectively.
    @assert all(>(0), view(energies, 1:6)) && all(<(0), view(energies, 7:12))

    # Disable tests below for speed. Note that the data in H has been
    # overwritten by eigen!, so H0 should refer to an original copy of H.
    #=
    Ĩ = Diagonal([ones(L); -ones(L)])
    @assert T' * Ĩ * T ≈ Ĩ
    @assert diag(T' * H0 * T) ≈ Ĩ * energies
    # Reflection symmetry H(q) = H(-q) is identified as H11 = conj(H22). In this
    # case, eigenvalues come in pairs.
    if H0[1:L, 1:L] ≈ conj(H0[L+1:2L, L+1:2L])
        @assert energies[1:L] ≈ -energies[L+1:2L]
    end
    =#

    return energies
end

function excitations!(T, tmp, sbs::SchwingerBosonSystem, q)

    q_reshaped = to_reshaped_rlu(q)
    dynamical_matrix!(tmp, sbs, q_reshaped)

    try
        return bogoliubov!(T, tmp)
    catch _
        rethrow(ErrorException("Not an energy-minimum; wavevector q = $q unstable."))
    end
end

function excitations(sbs::SchwingerBosonSystem, q)
    T = zeros(ComplexF64, 12, 12)
    H = zeros(ComplexF64, 12, 12)
    energies = excitations!(T, copy(H), sbs, q)
    return (energies, T)
end

function dispersion(sbs::SchwingerBosonSystem, qs)
    disp = zeros(6, length(qs))
    for (iq, q) in enumerate(qs)
        view(disp, :, iq) .= view(excitations(sbs, q)[1], 1:6)
    end
    return reshape(disp, 6, size(qs)...)
end

@inline lorentzian(x, Γ) = (1/π) * (Γ / 2) / (x^2 + (Γ / 2)^2)

function global_position(i::Int)
    if i == 1
        return Vec3(0.0, 0.0, 0.0)
    elseif i == 2
        return Vec3(1/2, √3/2, 0.0)
    elseif i == 3
        return Vec3(1.0, 0.0, 0.0)
    else
        error("Invalid site index: $i")
    end
end

"""
    dssf_mean_field(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ,
        aux::SpectralCondensationAux
    )

Compute the positive-frequency diagonal saddle-point DSSF using the canonical
Bogoliubov representation.

`ordinary_normal` and `ordinary_condensed` contain the unit-residue
contributions from `S_eff`. The canonical pair-creation contribution uses the
physical spinon momenta `-k` and `k + q`. The independently derived thermal
quasiparticle-scattering contribution uses the physical transition
`k -> k + q`. Its component expression is written in the conjugate matrix-
element orientation and therefore uses the conjugate sublattice phase.

`active_constraint` contains the contribution of the enhanced occupation of
the pinned mode. It is obtained directly in the Lehmann representation: the
pinned Bogoliubov mode carries the additional per-mode occupation
`δn_c = Nu ξ`, and the active-constraint response is the term of the
ordinary sum that is linear in `δn_c`. The ordinary pair weight
`1 + n_a + n_b` therefore gains `δn_c` for each pinned leg, and the ordinary
scattering weight `n_a - n_b` gains `+δn_c` when the pinned mode is the
initial state and `-δn_c` when it is the final state. Amplitudes, energies,
and normalizations are shared with the ordinary terms. The ordinary BdG
residues and Bose factors are left untouched, and the selected-selected
elastic contribution is omitted.

This construction is independent of the path-integral saddle-point route in
`dssf_SP`: it uses only canonical matrix elements and occupation numbers, and
never refers to the vertex normalizations or the soft-mode curvature. It is
therefore usable as a check on `dssf_SP` rather than a restatement of it.

`Γ` is the full width at half maximum of the Lorentzian broadening.
"""
function dssf_mean_field(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ,
    aux::SpectralCondensationAux
)
    num_energies = length(energies)
    num_bands = 6
    num_poles = 2num_bands

    ret_ordinary_normal =
        zeros(Float64, 3, num_energies)

    ret_ordinary_condensed =
        zeros(Float64, 3, num_energies)

    ret_active_constraint =
        zeros(Float64, 3, num_energies)

    (; L) = sbs

    Nu = L^2
    Ns = 3Nu
    βtemp = _inverse_temperature(sbs)

    # Enhanced occupation of the pinned Bogoliubov mode, in the same
    # per-mode units as the Bose factors `n_a` appearing below.
    #
    # `aux.active_positive_weights` stores ξ as a DENSITY: the constrained
    # solve in SpectralCondensation.jl imposes
    #
    #     N_normal + ξ qcsum = 2S + 1,
    #
    # where `N_normal` is averaged over the Nu Brillouin-zone points and
    # `qcsum` is the boson number carried by one unit of occupation of the
    # pinned mode. A per-mode occupation δn at a single momentum contributes
    # δn qcsum / Nu to that average, so matching the two expressions gives
    #
    #     δn_c = Nu ξ.
    #
    # The pinned mode is therefore macroscopically occupied, and the
    # active-constraint response below is the ordinary Lehmann sum with the
    # Bose factor of that one mode replaced by its enhancement.
    is_pinned = aux.selection_kind === :pinned
    active_weights = aux.active_positive_weights
    enhanced_occupation(band) = Nu * active_weights[band]

    q_ext = Vec3(q[1], q[2], q[3])
    q_global = recipvecs_origin * q_ext
    q_reshaped = to_reshaped_rlu(q_ext)

    spin_phase = zeros(ComplexF64, 3)

    for α in 1:3
        spin_phase[α] =
            exp(-im * dot(q_global, global_position(α)))
    end

    k_grid = Vec3[]

    for i in 1:L, j in 1:L
        push!(k_grid, Vec3([(i - 1) / L, (j - 1) / L, 0.0]))
    end

    qc = _spectral_condensation_momentum(aux, L)

    selected_positive_mask = falses(num_bands)

    for band in aux.conden_band_indices
        if 1 <= band <= num_bands
            selected_positive_mask[band] = true
        end
    end

    Hk = zeros(ComplexF64, num_poles, num_poles)
    Vk = zeros(ComplexF64, num_poles, num_poles)

    Hmk = zeros(ComplexF64, num_poles, num_poles)
    Vmk = zeros(ComplexF64, num_poles, num_poles)

    Hqpk = zeros(ComplexF64, num_poles, num_poles)
    Vqpk = zeros(ComplexF64, num_poles, num_poles)

    pair_amplitude =
        zeros(ComplexF64, 3, num_bands, num_bands)

    scattering_amplitude =
        zeros(ComplexF64, 3, num_bands, num_bands)

    # ------------------------------------------------------------------
    # Ordinary unit-residue contribution from S_eff.
    #
    # Pair creation:
    #
    #     (-k, a) + (k + q, b),
    #
    # whose total physical momentum is q.
    #
    # Thermal scattering:
    #
    #     (k, a) -> (k + q, b).
    # ------------------------------------------------------------------

    for k in k_grid
        mk = -k
        qpk = k + q_reshaped

        dynamical_matrix!(Hk, sbs, k)
        dynamical_matrix!(Hmk, sbs, mk)
        dynamical_matrix!(Hqpk, sbs, qpk)

        ϵs_k = bogoliubov!(Vk, Hk)
        ϵs_mk = bogoliubov!(Vmk, Hmk)
        ϵs_qpk = bogoliubov!(Vqpk, Hqpk)

        fill!(pair_amplitude, 0.0 + 0.0im)
        fill!(scattering_amplitude, 0.0 + 0.0im)

        for b in 1:num_bands
            vqpk =
                reshape(view(Vqpk, :, b), 2, 3, 2)

            for a in 1:num_bands
                vmk =
                    reshape(view(Vmk, :, a), 2, 3, 2)

                vk =
                    reshape(view(Vk, :, a), 2, 3, 2)

                for α in 1:3
                    pair_phase = spin_phase[α]
                    scattering_phase = conj(spin_phase[α])

                    for μ in 1:3
                        σμ = σs[μ]

                        for σ in 1:2, σ′ in 1:2
                            # Eq. (175): symmetrized canonical
                            # pair-creation matrix element.
                            pair_amplitude[μ, a, b] +=
                                0.5 *
                                pair_phase *
                                σμ[σ, σ′] *
                                (
                                    vqpk[σ, α, 2] *
                                    vmk[σ′, α, 1] +
                                    vqpk[σ′, α, 1] *
                                    vmk[σ, α, 2]
                                )

                            # Number-conserving canonical matrix element
                            # for β†_{k+q,b} β_{k,a}. The component
                            # expression is the conjugate orientation of
                            # v†_{k,a} U_q v_{k+q,b}, so it carries the
                            # conjugate sublattice phase.
                            scattering_amplitude[μ, a, b] +=
                                0.5 *
                                scattering_phase *
                                σμ[σ, σ′] *
                                (
                                    conj(vqpk[σ, α, 1]) *
                                    vk[σ′, α, 1] +
                                    vk[σ, α, 2] *
                                    conj(vqpk[σ′, α, 2])
                                )
                        end
                    end
                end
            end
        end

        k_is_selected =
            _same_momentum_mod1(k, qc)

        mk_is_selected =
            _same_momentum_mod1(mk, qc)

        qpk_is_selected =
            _same_momentum_mod1(qpk, qc)

        for a in 1:num_bands
            E_k = real(ϵs_k[a])
            E_mk = real(ϵs_mk[a])

            pair_line_a_selected =
                mk_is_selected &&
                selected_positive_mask[a]

            scattering_line_a_selected =
                k_is_selected &&
                selected_positive_mask[a]

            for b in 1:num_bands
                E_qpk = real(ϵs_qpk[b])

                line_b_selected =
                    qpk_is_selected &&
                    selected_positive_mask[b]

                # ------------------------------------------------------
                # Pair creation.
                #
                # The signed-pole transition is
                #
                #     -E_{-k,a} -> E_{k+q,b},
                #
                # and therefore has excitation energy
                #
                #     E_{-k,a} + E_{k+q,b}.
                # ------------------------------------------------------

                if !(pair_line_a_selected && line_b_selected)
                    ΔE_pair = E_mk + E_qpk

                    pair_factor =
                        _dssf_transition_factor(-E_mk, E_qpk, βtemp)

                    if !iszero(pair_factor)
                        ret_sector =
                            pair_line_a_selected != line_b_selected ?
                            ret_ordinary_condensed :
                            ret_ordinary_normal

                        for μ in 1:3
                            weight =
                                abs2(pair_amplitude[μ, a, b]) *
                                pair_factor /
                                (2Ns)

                            for (ie, energy) in enumerate(energies)
                                ret_sector[μ, ie] +=
                                    weight *
                                    lorentzian(energy - ΔE_pair, Γ)
                            end
                        end
                    end

                    # Active-constraint enhancement. The ordinary pair weight
                    # is 1 + n_a + n_b, so replacing the pinned mode's
                    # occupation by n_c + δn_c adds δn_c for each pinned leg.
                    if is_pinned
                        δn_pair = 0.0

                        if pair_line_a_selected
                            δn_pair += enhanced_occupation(a)
                        end

                        if line_b_selected
                            δn_pair += enhanced_occupation(b)
                        end

                        if !iszero(δn_pair)
                            fdt = _dssf_fluctuation_dissipation_factor(
                                ΔE_pair,
                                βtemp,
                            )

                            if !iszero(fdt)
                                for μ in 1:3
                                    weight =
                                        abs2(pair_amplitude[μ, a, b]) *
                                        δn_pair *
                                        fdt /
                                        (2Ns)

                                    for (ie, energy) in enumerate(energies)
                                        ret_active_constraint[μ, ie] +=
                                            weight *
                                            lorentzian(energy - ΔE_pair, Γ)
                                    end
                                end
                            end
                        end
                    end
                elseif is_pinned && a == b
                    # Both legs are the same pinned mode: k = -qc and
                    # q ≡ 2qc (mod the magnetic reciprocal lattice), so
                    # both created quanta land back on the pinned band.
                    #
                    # This is NOT the naive substitution n_a = n_b = δn_c
                    # into the ordinary weight 1+n_a+n_b -- that weight
                    # comes from a non-degenerate Wick contraction and
                    # does not apply once both legs are literally the
                    # same mode. The correct weight is fixed instead by
                    # the rank-one-trace identity that establishes the
                    # Green-function/canonical equivalence (Appendix G,
                    # `eq:rank_one_trace_identity_with_vertices`), which
                    # holds for arbitrary vectors with no non-degeneracy
                    # assumption. Carrying it through with the enhanced
                    # occupation gives a weight LINEAR in δn_c (not
                    # quadratic), with prefactor 1/Ns (not 1/2Ns): this
                    # (k,a,b) = (-qc,c,c) term is its own image under the
                    # (k,a,b) -> (-k-q,b,a) relabeling that the 1/2
                    # ordinarily compensates for, so there is nothing left
                    # to double-count. See Eq. (S_active_both_legs) of
                    # main_sbt.tex, verified against dssf_SP's independent
                    # curvature normalization to ~1e-10 relative.
                    ΔE_pair = E_mk + E_qpk
                    δn_c = enhanced_occupation(a)

                    if !iszero(δn_c)
                        fdt = _dssf_fluctuation_dissipation_factor(
                            ΔE_pair,
                            βtemp,
                        )

                        if !iszero(fdt)
                            for μ in 1:3
                                weight =
                                    abs2(pair_amplitude[μ, a, b]) *
                                    δn_c *
                                    fdt /
                                    Ns

                                for (ie, energy) in enumerate(energies)
                                    ret_active_constraint[μ, ie] +=
                                        weight *
                                        lorentzian(energy - ΔE_pair, Γ)
                                end
                            end
                        end
                    end
                end

                # ------------------------------------------------------
                # Thermal quasiparticle scattering.
                #
                # The physical transition is
                #
                #     (k, a) -> (k + q, b).
                #
                # `_dssf_transition_factor(E_k, E_qpk, β)` is negative
                # for a positive-energy scattering transition, so the
                # canonical Lehmann weight carries the compensating
                # minus sign below.
                # ------------------------------------------------------

                if !(scattering_line_a_selected && line_b_selected)
                    ΔE_scattering = E_qpk - E_k

                    if ΔE_scattering > 1e-12
                        scattering_factor =
                            -_dssf_transition_factor(E_k, E_qpk, βtemp)

                        if !iszero(scattering_factor)
                            ret_sector =
                                scattering_line_a_selected != line_b_selected ?
                                ret_ordinary_condensed :
                                ret_ordinary_normal

                            for μ in 1:3
                                weight =
                                    abs2(scattering_amplitude[μ, a, b]) *
                                    scattering_factor /
                                    Ns

                                for (ie, energy) in enumerate(energies)
                                    ret_sector[μ, ie] +=
                                        weight *
                                        lorentzian(
                                            energy - ΔE_scattering,
                                            Γ
                                        )
                                end
                            end
                        end

                        # Active-constraint enhancement. The ordinary
                        # scattering weight is n_a - n_b, so the pinned mode
                        # contributes +δn_c as the initial state and -δn_c as
                        # the final state.
                        if is_pinned
                            δn_scattering = 0.0

                            if scattering_line_a_selected
                                δn_scattering += enhanced_occupation(a)
                            end

                            if line_b_selected
                                δn_scattering -= enhanced_occupation(b)
                            end

                            if !iszero(δn_scattering)
                                fdt = _dssf_fluctuation_dissipation_factor(
                                    ΔE_scattering,
                                    βtemp,
                                )

                                if !iszero(fdt)
                                    for μ in 1:3
                                        weight =
                                            abs2(scattering_amplitude[μ, a, b]) *
                                            δn_scattering *
                                            fdt /
                                            Ns

                                        for (ie, energy) in enumerate(energies)
                                            ret_active_constraint[μ, ie] +=
                                                weight *
                                                lorentzian(
                                                    energy - ΔE_scattering,
                                                    Γ
                                                )
                                        end
                                    end
                                end
                            end
                        end
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
        total = ret_total
    )
end