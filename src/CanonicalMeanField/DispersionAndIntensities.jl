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
        aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
    )

Compute the diagonal components of the dynamical spin structure factor at the
mean-field level using the canonical Bogoliubov formalism.

Returns

    ret_normal, ret_condensate

where both arrays have size `3 × length(energies)`.

This is the zero-temperature canonical counterpart of `dssf_SP`.

The split follows the Green-function convention

    G = G_n + G_c.

Therefore the normal sector contains only terms where neither canonical line
is condensed. The condensate sector contains terms where exactly one canonical
line is condensed. The elastic condensate-condensate contribution, where both
lines are condensed, is omitted.

The two canonical lines are

    line 1: q + k,
    line 2: -k.

When line 1 is condensed, the peak is placed at the line-2 energy. When line 2
is condensed, the peak is placed at the line-1 energy.
"""
function dssf_mean_field(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
)
    num_energies = length(energies)
    num_bands = 6

    H1 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V1 = zeros(ComplexF64, 2num_bands, 2num_bands)

    H2 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V2 = zeros(ComplexF64, 2num_bands, 2num_bands)

    Avec_pref = zeros(ComplexF64, 3)
    Avec = zeros(ComplexF64, 3, num_bands, num_bands)

    q_ext = Vec3(q[1], q[2], q[3])
    q_global = recipvecs_origin * q_ext

    for α in 1:3
        rα = global_position(α)
        Avec_pref[α] = exp(-im * dot(q_global, rα))
    end

    q_reshaped = to_reshaped_rlu(q_ext)

    (; L) = sbs

    ret_normal = zeros(Float64, 3, num_energies)
    ret_condensate = zeros(Float64, 3, num_energies)

    positive_condensed_bands = filter(
        l -> 1 <= l <= num_bands,
        aux.conden_band_indices,
    )

    if isempty(positive_condensed_bands)
        error("SpectralCondensationAux has no selected positive-energy condensed bands.")
    end

    i_cond = (aux.conden_index - 1) ÷ L + 1
    j_cond = (aux.conden_index - 1) % L + 1
    qc = Vec3([(i_cond - 1) / L, (j_cond - 1) / L, 0.0])

    for i in 1:L, j in 1:L
        ik = (i - 1) * L + j
        k_reshaped = Vec3([(i - 1) / L, (j - 1) / L, 0.0])
        qpk_reshaped = q_reshaped + k_reshaped

        # Canonical line 1: q + k
        # Canonical line 2: -k
        dynamical_matrix!(H1, sbs, qpk_reshaped)
        dynamical_matrix!(H2, sbs, -k_reshaped)

        disp1 = bogoliubov!(V1, H1)
        disp2 = bogoliubov!(V2, H2)

        fill!(Avec, 0.0)

        for band1 in 1:num_bands
            v1 = reshape(view(V1, :, band1), 2, 3, 2)

            for band2 in 1:num_bands
                v2 = reshape(view(V2, :, band2), 2, 3, 2)

                for α in 1:3, μ in 1:3, σ in 1:2, σ′ in 1:2
                    Avec[μ, band1, band2] +=
                        0.5 *
                        Avec_pref[α] *
                        σs[μ][σ, σ′] *
                        (
                            v1[σ, α, 2] * v2[σ′, α, 1] +
                            v1[σ′, α, 1] * v2[σ, α, 2]
                        )
                end
            end
        end

        line1_condensed_momentum = _same_momentum_mod1(qpk_reshaped, qc)
        line2_condensed_momentum = ik == aux.conden_index

        for (ie, energy) in enumerate(energies)
            for μ in 1:3
                for band1 in 1:num_bands, band2 in 1:num_bands
                    band1_condensed =
                        line1_condensed_momentum &&
                        band1 in positive_condensed_bands

                    band2_condensed =
                        line2_condensed_momentum &&
                        band2 in positive_condensed_bands

                    if band1_condensed && band2_condensed
                        # Elastic condensate-condensate contribution.
                        # Omitted in the same convention as dssf_SP.
                        continue
                    elseif band1_condensed
                        # Condensed line 1, normal line 2.
                        #
                        # Line 1 corresponds to a positive BdG pole, so the
                        # condensate weight is aux.condensate_weights[band1].
                        condensate_weight =
                            aux.condensate_weights[band1] * L^2

                        ΔE = disp2[band2]

                        ret_condensate[μ, ie] +=
                            condensate_weight *
                            abs2(Avec[μ, band1, band2]) *
                            lorentzian(energy - ΔE, Γ)
                    elseif band2_condensed
                        # Normal line 1, condensed line 2.
                        #
                        # Line 2 is the V2 / -k line and corresponds to the
                        # negative BdG pole 6 + band2.
                        lcond_neg = num_bands + band2

                        condensate_weight =
                            aux.condensate_weights[lcond_neg] * L^2

                        ΔE = disp1[band1]

                        ret_condensate[μ, ie] +=
                            condensate_weight *
                            abs2(Avec[μ, band1, band2]) *
                            lorentzian(energy - ΔE, Γ)
                    else
                        ΔE = disp1[band1] + disp2[band2]

                        ret_normal[μ, ie] +=
                            abs2(Avec[μ, band1, band2]) *
                            lorentzian(energy - ΔE, Γ)
                    end
                end
            end
        end
    end

    ret_normal ./= 6L^2
    ret_condensate ./= 6L^2

    return ret_normal, ret_condensate
end