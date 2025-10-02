function set_ϕ_AB!(sbs::SchwingerBosonSystem, ϕ)
    if length(ϕ) ≠ 12
        throw(ArgumentError("Mean field vector must have length 12."))
    end
    for i in 1:6
        sbs.mean_fields[i] = ϕ[i] + 1im * ϕ[i+6]
    end
end

# The variational free energy objective function with gradient for Optim
function fg_ϕ_AB!(sbs::SchwingerBosonSystem, f, g, ϕ)
    if isnothing(g)
        g = zero(ϕ)
    end
    # Make sure to set the mean fields to the `sbs` object
    set_ϕ_AB!(sbs, ϕ)

    # Calculate the "temperature*entropy" term of the free energy
    (; L, J, S, T) = sbs
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

    f = 0.0

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        try
            E = bogoliubov!(V, D)
            for n in 1:6
                f += E[n] / (2Nu)
                (T > 1e-8) && (f += real(T * log1p(-exp(-E[n]/T))) / Nu)
            end
        # If the dynamical matrix is not positive definite, return to Inf immediately
        catch _
            @warn "Dynamical matrix is not positive definite, returning Inf. Skipping this configuration..."
            g .= Inf
            f = Inf
            return f
        end
    end

    g .= 0.0
    # Buffers to calculate the gradient
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    tmp2 = zeros(ComplexF64, 12, 12)
    Dmat = zeros(ComplexF64, 12, 12)
    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 15)
    ∂F2α = zeros(15)
    ∂F2αβ = zeros(15, 15)

    # The "interaction" strength
    fα = zeros(12)
    inv_fα = zeros(12)
    for α in (1, 2, 3, 7, 8, 9)
        fα[α] = -J
        inv_fα[α] = -1/J
    end
    for α in (4, 5, 6, 10, 11, 12)
        fα[α] = J
        inv_fα[α] = 1/J
    end

    @views for α in 1:3
        ∂ID∂μ0!(∂ID∂ϕs[:, :, α+12], tmp, α)
    end

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        E = single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)

        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α],   ∂ID∂ϕs[:, :, α+6], tmp, sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+9], tmp, sbs, q, α)
        end

        # Calculate the first and second derivatives of the F2 (quadratic bosonic free energy)
        @views for α in 1:15
            # Gradient from entropy term
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
            # Calculate the second derivatives of the quadratic free energy
            divided_aux!(tmp, tmp2, Dmat, ∂ID∂ϕs[:, :, α], V, inv_V)
            for β in 1:15
                ∂F2αβ[α, β] += real(tr(tmp * ∂ID∂ϕs[:, :, β])) / (4Nu)
            end
        end
    end

    # Now we add the contribution from the "correction" term L"⟨H - H_{MF}⟩_{MF}"
    for α in 1:12
        f += inv_fα[α] * ∂F2α[α]^2 / 12 - ∂F2α[α] * ϕ[α]
        for β in 1:12
            g[α] += ∂F2αβ[α, β] * (inv_fα[β]/6 * ∂F2α[β] - ϕ[β])
        end
    end

    # Now we need to calculate μ and the gradient of f with respect to ϕ
    μ0s = real(sbs.mean_fields[13:15])
    κ0  = zeros(3, 3)
    ∂ΔH∂μ0 = zeros(3)
    for α in 1:3
        f += μ0s[α] * (2S+1)
        for β in 1:3
            κ0[α, β] += -∂F2αβ[α+12, β+12]
        end
        for β in 1:12
            ∂ΔH∂μ0[α] += ∂F2αβ[α+12, β] * (inv_fα[β]/6 * ∂F2α[β] - ϕ[β])
        end
    end

    sbs.Δμs .= pinv(κ0) * ∂ΔH∂μ0

    for α in 1:12
        for β in 1:3
            g[α] += ∂F2αβ[α, β+12] * sbs.Δμs[β]
        end
    end
    
    return f
end

function optimize_mean_fields_AB!(sbs::SchwingerBosonSystem, ϕ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_ϕ_AB!(sbs, ϕ0)
    fg!(f, g, x) = fg_ϕ_AB!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm, options)
    set_ϕ_AB!(sbs, ret.minimizer)
    return ret
end

function gauge_transform_AB!(sbs::SchwingerBosonSystem, ϕ, comp::Int)
    # Make sure to set the mean fields to the `sbs` object
    set_ϕ_AB!(sbs, ϕ)

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
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 15)
    ∂F2α = zeros(15)
    ∂F2αβ = zeros(15, 15)

    # The "interaction" strength
    fα = zeros(12)
    inv_fα = zeros(12)
    for α in (1, 2, 3, 7, 8, 9)
        fα[α] = -J
        inv_fα[α] = -1/J
    end
    for α in (4, 5, 6, 10, 11, 12)
        fα[α] = J
        inv_fα[α] = 1/J
    end

    @views for α in 1:3
        ∂ID∂μ0!(∂ID∂ϕs[:, :, α+12], tmp, α)
    end

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        E = single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)

        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α],   ∂ID∂ϕs[:, :, α+6], tmp, sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+9], tmp, sbs, q, α)
        end

        # Calculate the first and second derivatives of the F2 (quadratic bosonic free energy)
        @views for α in 1:15
            # Gradient from entropy term
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
            # Calculate the second derivatives of the quadratic free energy
            divided_aux!(tmp, tmp2, Dmat, ∂ID∂ϕs[:, :, α], V, inv_V)
            for β in 1:15
                ∂F2αβ[α, β] += real(tr(tmp * ∂ID∂ϕs[:, :, β])) / (4Nu)
            end
        end
    end

    G_OO = view(-∂F2αβ, 1:12, 1:12)
    G_On = view(-∂F2αβ, 1:12, 13:15)
    G_nO = view( ∂F2αβ, 13:15, 1:12)
    G_nn = view( ∂F2αβ, 13:15, 13:15)

    nn = nullspace(G_OO)
    @show nn

    @show eigvals(G_OO)
    S = G_OO - G_On * pinv(G_nn) * G_nO
    n = nullspace(S; atol=1e-8)
    @show eigvals(S)
    @show n
    # @show G_OO
    # @show S
    @show size(n)

    for i in 1:12
        ϕ[i] += inv_fα[i] / 6 * n[i, comp]
    end

end