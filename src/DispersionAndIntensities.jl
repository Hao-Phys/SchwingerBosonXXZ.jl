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