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
        Γ;
        options_μ = Optim.Options(show_trace=false, iterations=100),
        tol = 1e-12,
        max_iters = 1000,
        include_condensation::Bool = true,
        aux::Union{Nothing,SpectralCondensationAux} = nothing,
    )

Compute the diagonal components of the mean-field dynamical spin structure
factor from the Matsubara-summed BdG Green-function residue formula.

Returns `ret_normal, ret_condensate`, where both arrays have size
`3 × length(energies)`.

The normal contribution uses the full BdG pole-pair sum. Therefore, at finite
temperature, it includes both the pair-creation channels and the thermal
scattering channels required by detailed balance.

If `include_condensation = true`, selected soft-mode poles are split out using
`SpectralCondensationAux`. If `aux === nothing`, it is constructed by
`spectral_condensation_aux(sbs)`.

The selected-pole convention is the same as the saddle-point Green function and
the single-particle density matrix:

    finite_size_minimum: selected pole weight = 1
    pinned:              selected pole weight = 1 + ξ

Thus the selected pole is removed from the normal sector and reinserted through
the selected sector with the total selected-pole weight.

The branch-dependent collapsed-sum factor is handled by
`_condensate_sum_factor(aux, L)`:

    finite_size_minimum: 1
    pinned:              L^2

For selected-normal terms, the finite selected soft-mode energy is kept:

    Orientation 1: ΔE = E_normal(k + q) - E_selected(k)
    Orientation 2: ΔE = E_selected(k + q) - E_normal(k)

The purely selected-selected elastic contribution is not included.

The keyword arguments `options_μ`, `tol`, and `max_iters` are retained for
compatibility with older calls using the previous condensation workflow. The
current implementation assumes that `sbs` is already the saddle-point solution
used to construct `SpectralCondensationAux`.
"""
function dssf_mean_field(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    options_μ = Optim.Options(show_trace=false, iterations=100),
    tol = 1e-12,
    max_iters = 1000,
    include_condensation::Bool = true,
    aux::Union{Nothing,SpectralCondensationAux} = nothing,
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

    spectral_aux = if include_condensation
        aux === nothing ? spectral_condensation_aux(sbs) : aux
    else
        nothing
    end

    has_condensate =
        spectral_aux !== nothing &&
        !isempty(spectral_aux.conden_band_indices)

    # ------------------------------------------------------------------
    # Normal-normal contribution.
    #
    # This is the full Matsubara-summed BdG pole-pair expression. At finite
    # temperature, it includes both pair-creation and thermal scattering
    # channels.
    # ------------------------------------------------------------------

    for k in k_grid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(
            sbs,
            k,
            spectral_aux,
        )

        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(
            sbs,
            kq,
            spectral_aux,
        )

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

    has_condensate || return ret_normal, ret_condensate

    # ------------------------------------------------------------------
    # Selected-normal contribution.
    #
    # This uses the same selected-sector convention as the saddle-point
    # Green function:
    #
    #     finite_size_minimum: selected weight = 1
    #     pinned:              selected weight = 1 + ξ
    #
    # The selected pole is removed from the normal sector and reinserted here
    # through `Green_SP_condensed_residues`.
    # ------------------------------------------------------------------

    qc = _spectral_condensation_momentum(spectral_aux, L)
    condensate_sum_factor = _condensate_sum_factor(spectral_aux, L)

    # --------------------------------------------------------------
    # Orientation 1:
    #
    #     G_n(k + q) G_c(k),
    #
    # with k = qc. The selected pole is on the second line.
    # --------------------------------------------------------------

    k = qc
    kq = qc + q_reshaped

    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, k, spectral_aux)
    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kq, spectral_aux)

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

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, k, spectral_aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kq, spectral_aux)

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