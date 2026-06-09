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
    )

Compute the diagonal components of the dynamical spin structure factor at the
mean-field level using the canonical Bogoliubov formalism.

Returns

    ret_normal, ret_condensate

where both arrays have size `3 × length(energies)`.

The split follows the original finite-size condensate convention: the
condensate sector is identified by

    ik == aux.conden_index.

In this convention, the pinned condensate line is the second canonical line,
namely the `V2` / `-k` line. Therefore the condensate contribution is assigned
when `band2` is in the pinned condensate bands.

The pinned `band2` contribution is removed from the normal sector and assigned
to `ret_condensate` with finite-size weight `(aux.ξ + 1) * L^2`.
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
)
    num_energies = length(energies)
    num_bands = 6

    aux = CondensationAux(0.0, 0.0, nothing, 0.0, 0)

    if include_condensation
        μ0 = copy(real(sbs.mean_fields[13:15]))
        optimize_μ0!(sbs, μ0, aux; options=options_μ, tol=tol, max_iters=max_iters)
        condensation_results!(sbs, aux)
    end

    H1 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V1 = zeros(ComplexF64, 2num_bands, 2num_bands)

    H2 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V2 = zeros(ComplexF64, 2num_bands, 2num_bands)

    Avec_pref = zeros(ComplexF64, 3)
    Avec = zeros(ComplexF64, 3, num_bands, num_bands)

    q_global = recipvecs_origin * q

    for α in 1:3
        rα = global_position(α)
        Avec_pref[α] = exp(-im * dot(q_global, rα))
    end

    q_reshaped = to_reshaped_rlu(q)

    (; L) = sbs
    k_reshapes = [Vec3(i/L, j/L, 0.0) for i in 0:L-1, j in 0:L-1, _ in 1:1]

    ret_normal = zeros(Float64, 3, num_energies)
    ret_condensate = zeros(Float64, 3, num_energies)

    has_condensate = include_condensation && aux.conden_index !== nothing

    if has_condensate
        condensed_bands = (num_bands - aux.num_conden + 1):num_bands
        condensate_weight = (aux.ξ + 1) * L^2
    else
        condensed_bands = 1:0
        condensate_weight = 0.0
    end

    for (ik, k_reshaped) in enumerate(k_reshapes)
        qpk_reshaped = q_reshaped + k_reshaped

        # Canonical line 1: q + k
        # Canonical line 2: -k
        dynamical_matrix!(H1, sbs, qpk_reshaped)
        dynamical_matrix!(H2, sbs, -k_reshaped)

        disp1 = bogoliubov!(V1, H1)
        disp2 = bogoliubov!(V2, H2)

        Avec .= 0.0

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

        is_condensed_sector = has_condensate && ik == aux.conden_index

        for (ie, energy) in enumerate(energies)
            for μ in 1:3
                for band1 in 1:num_bands, band2 in 1:num_bands
                    band2_condensed =
                        is_condensed_sector &&
                        band2 in condensed_bands

                    if band2_condensed
                        # The pinned V2 / -k line is removed from the normal
                        # sector and reassigned to the condensate sector.
                        #
                        # The condensed line carries zero physical energy, so
                        # the peak is placed at the normal line energy disp1.
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