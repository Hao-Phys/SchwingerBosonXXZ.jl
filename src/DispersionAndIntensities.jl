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

@inline lorentzian(x, Γ) = (Γ / 2) / (x^2 + (Γ / 2)^2)

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
function dssf_mean_field(sbs::SchwingerBosonSystem, q, energies, Γ, mode::Symbol=:sum; opts...)
    @assert mode in (:sum, :integration) "mode must be :sum or :integration"
    num_energies = length(energies)
    num_bands = 6
    # Buffers for Bogoliubov transformation and dynamical matrix.
    # H1, V1 for q+k
    H1 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V1 = zeros(ComplexF64, 2num_bands, 2num_bands)
    # H2, V2 for -k
    H2 = zeros(ComplexF64, 2num_bands, 2num_bands)
    V2 = zeros(ComplexF64, 2num_bands, 2num_bands)

    Avec_pref = zeros(ComplexF64, 3)
    Avec = zeros(ComplexF64, 3, num_bands, num_bands)
    corr_buf = zeros(3, num_energies)

    q_global = recipvecs_origin * q

    for i in 1:3
        r_i = global_position(i)
        Avec_pref[i] = exp(-im * dot(q_global, r_i))
    end

    q_reshaped = to_reshaped_rlu(q)
    if mode == :integration
        ints = hcubature((0,0,0), (1,1,1); opts...) do k_reshaped
            qpk_reshaped = q_reshaped + k_reshaped
            dynamical_matrix!(H1, sbs, qpk_reshaped)
            dynamical_matrix!(H2, sbs, -k_reshaped)

            disp1 = bogoliubov!(V1, H1)
            disp2 = bogoliubov!(V2, H2)

            # Fill the buffers with zeros
            Avec .= 0.0
            corr_buf .= 0.0

            for band1 in 1:num_bands
                v1 = reshape(view(V1, :, band1), 2, 3, 2)
                for band2 in 1:num_bands
                    v2 = reshape(view(V2, :, band2), 2, 3, 2)
                    for α in 1:3, μ in 1:3, σ in 1:2, σ′ in 1:2
                        Avec[μ, band1, band2] += 0.5 * Avec_pref[α] * σs[μ][σ, σ′] * (v1[σ, α, 2]*v2[σ′, α, 1] + v1[σ′, α, 1]*v2[σ, α, 2])
                    end
                end
            end

            for (ie, energy) in enumerate(energies)
                for μ in 1:3
                    for band1 in 1:num_bands, band2 in 1:num_bands
                        corr_buf[μ, ie] += abs2(Avec[μ, band1, band2]) * lorentzian(energy - disp1[band1] - disp2[band2], Γ)
                    end
                end
            end

            return SVector{3num_energies}(vec(corr_buf))
        end

        ret = reshape(ints[1], 3, num_energies)
    else
        (; L) = sbs
        k_reshapes = [Vec3(i/L, j/L, 0.0) for i in 0:L-1, j in 0:L-1]
        ret = zeros(3, num_energies)

        for k_reshaped in k_reshapes
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

            for (ie, energy) in enumerate(energies)
                for μ in 1:3
                    for band1 in 1:num_bands, band2 in 1:num_bands
                        ret[μ, ie] += abs2(Avec[μ, band1, band2]) * lorentzian(energy - disp1[band1] - disp2[band2], Γ)
                    end
                end
            end
        end

        ret /= L^2
    end

    return ret
end