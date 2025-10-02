function gauge_transform!(sbs::SchwingerBosonSystem, ϕ, comp::Int)
    # Make sure to set the mean fields to the `sbs` object
    set_ϕ!(sbs, ϕ)

    # Calculate the "temperature*entropy" term of the free energy
    (; L, J, Δ, S, T) = sbs
    Nu = L^2

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    # Maximize the mean-field free energy to find the optimal chemical potential
    # But we need a μ0 such that the dynamical matrix is positive definite
    eigvals_min = Float64[]
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        eigval_min = eigmin(D)
        push!(eigvals_min, eigval_min)
    end

    τ = max(0.0, -minimum(eigvals_min))
    μ0s = sbs.mean_fields[13:15] .- (τ + T)

    optimize_μ0!(sbs, μ0s)

    # Buffers to calculate the gradient
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    tmp2 = zeros(ComplexF64, 12, 12)
    Dmat = zeros(ComplexF64, 12, 12)
    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 27)
    ∂F2α = zeros(27)
    ∂F2αβ = zeros(27, 27)

    # The "interaction" strength
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2
    inv_J₊ = J₊ == 0 ? 0 : 1 / J₊
    inv_J₋ = J₋ == 0 ? 0 : 1 / J₋
    fα = zeros(24)
    inv_fα = zeros(24)
    for α in (1, 2, 3, 13, 14, 15)
        fα[α] = -J₊
        inv_fα[α] = -inv_J₊
    end
    for α in (4, 5, 6, 16, 17, 18)
        fα[α] = J₊
        inv_fα[α] = inv_J₊
    end
    for α in (7, 8, 9, 19, 20, 21)
        fα[α] = J₋
        inv_fα[α] = inv_J₋
    end
    for α in (10, 11, 12, 22, 23, 24)
        fα[α] = -J₋
        inv_fα[α] = -inv_J₋
    end

    @views for α in 1:3
        ∂ID∂μ0!(∂ID∂ϕs[:, :, α+24], tmp, α)
    end

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        E = single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)

        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α],   ∂ID∂ϕs[:, :, α+12], tmp, sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+15], tmp, sbs, q, α)
            ∂ID∂C!(∂ID∂ϕs[:, :, α+6], ∂ID∂ϕs[:, :, α+18], tmp, sbs, q, α)
            ∂ID∂D!(∂ID∂ϕs[:, :, α+9], ∂ID∂ϕs[:, :, α+21], tmp, sbs, q, α)
        end

        # Calculate the first and second derivatives of the F2 (quadratic bosonic free energy)
        @views for α in 1:27
            # Gradient from entropy term
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
            # Calculate the second derivatives of the quadratic free energy
            divided_aux!(tmp, tmp2, Dmat, ∂ID∂ϕs[:, :, α], V, inv_V)
            for β in 1:27
                ∂F2αβ[α, β] += real(tr(tmp * ∂ID∂ϕs[:, :, β])) / (4Nu)
            end
        end
    end

    G_OO = view(∂F2αβ, 1:24, 1:24)
    G_On = view(∂F2αβ, 1:24, 25:27)
    G_nO = view(∂F2αβ, 25:27, 1:24)
    G_nn = view(∂F2αβ, 25:27, 25:27)
    nn = nullspace(G_nn; atol=1e-8)

    S = G_OO - G_On * pinv(G_nn) * G_nO
    n = nullspace(S)

    for i in 1:24
        ϕ[i] += inv_fα[i] / 6 * n[i, comp]
    end

end