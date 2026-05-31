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

# The diagonal element of the dynamical spin structure factor
# Here we perform the finite-size summation over the Brillouin zone, based on the linear system size `L` in `sbs`.
# Warning: To get physically correct results, `include_condensation` should be set to `true`. We leave it as an option for testing purposes, but it is not recommended to set it to `false` when analyzing results.
function dssf_mean_field(sbs::SchwingerBosonSystem, q, energies, Γ; options_μ = Optim.Options(show_trace=false, iterations=100), tol=1e-12, max_iters=1000, include_condensation::Bool=true)

    num_energies = length(energies)
    num_bands = 6

    # Determine the condensation
    if include_condensation
        μ0 = copy(real(sbs.mean_fields[13:15]))
        aux = CondensationAux(0.0, 0.0, nothing, 0.0, 0)
        optimize_μ0!(sbs, μ0, aux; options=options_μ, tol, max_iters)
        condensation_results!(sbs, aux)
    end

    # Buffers for Bogoliubov transformation and dynamical matrix.
    # H1, V1 for q+k
    H1 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V1 = zeros(ComplexF64, 2num_bands, 2num_bands)
    # H2, V2 for -k
    H2 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V2 = zeros(ComplexF64, 2num_bands, 2num_bands)

    Avec_pref = zeros(ComplexF64, 3)
    Avec = zeros(ComplexF64, 3, num_bands, num_bands)

    q_global = recipvecs_origin * q

    for i in 1:3
        r_i = global_position(i)
        Avec_pref[i] = exp(-im * dot(q_global, r_i))
    end

    q_reshaped = to_reshaped_rlu(q)

    (; L) = sbs
    k_reshapes = [Vec3(i/L, j/L, 0.0) for i in 0:L-1, j in 0:L-1, _ in 1:1]
    ret = zeros(3, num_energies)

    for (ik, k_reshaped) in enumerate(k_reshapes)
        qpk_reshaped = q_reshaped + k_reshaped
        dynamical_matrix!(H1, sbs, qpk_reshaped)
        dynamical_matrix!(H2, sbs, -k_reshaped)

        disp1 = bogoliubov!(V1, H1)
        disp2 = bogoliubov!(V2, H2)

        # Fill the buffers with zeros
        Avec .= 0.0

        for band1 in 1:num_bands
            v1 = reshape(view(V1, :, band1), 2, 3, 2)
            for band2 in 1:num_bands
                v2 = reshape(view(V2, :, band2), 2, 3, 2)
                for α in 1:3, μ in 1:3, σ in 1:2, σ′ in 1:2
                    Avec[μ, band1, band2] += 0.5 * Avec_pref[α] * σs[μ][σ, σ′] * (v1[σ, α, 2]*v2[σ′, α, 1] + v1[σ′, α, 1]*v2[σ, α, 2])
                end
            end
        end

        if include_condensation && ik == aux.conden_index
            (; ξ, num_conden) = aux
            for band1 in num_bands-num_conden+1:num_bands
                v1 = reshape(view(V1, :, band1), 2, 3, 2)
                for band2 in num_bands-num_conden+1:num_bands
                    v2 = reshape(view(V2, :, band2), 2, 3, 2)
                    for α in 1:3, μ in 1:3, σ in 1:2, σ′ in 1:2
                        Avec[μ, band1, band2] += ξ * L^2 * 0.5 * Avec_pref[α] * σs[μ][σ, σ′] * (v1[σ, α, 2]*v2[σ′, α, 1] + v1[σ′, α, 1]*v2[σ, α, 2])
                    end
                end
            end
        end

        for (ie, energy) in enumerate(energies)
            for μ in 1:3
                for band1 in 1:num_bands, band2 in 1:num_bands
                    ret[μ, ie] += abs2(Avec[μ, band1, band2]) * lorentzian(energy - disp1[band1] - disp2[band2], Γ)
                end
            end
        end
    end

    # Normalize by the total number of sites, N_s = 3L^2.
    # The extra factor 1/2 avoids double counting identical two-spinon
    # final states in the unrestricted ordered sum over (k, n1, n2).
    ret /= 6L^2

    return ret
end