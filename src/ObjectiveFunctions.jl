function single_particle_density_matrix!(P::Matrix{ComplexF64}, D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, tmp::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)
    P .= 0.0
    (; T) = sbs
    dynamical_matrix!(D, sbs, q_reshaped)
    try
        E = bogoliubov!(V, D)
        for i in eachindex(E)
            # Include the 1/2 factor here
            P[i, i] += coth(E[i] / (2T)) / 4
        end
        # P = V * diag( coth(E / (2T)) ) * inv(V) / 2
        mul!(tmp, V, P)
        mul!(P, tmp, inv(V))
        return E
    catch e
        P .= 0.0
        E = ones(12) * Inf
        return E
    end
end

# Objective function for the chemical potential optimization and its gradient
function fg_μ0!(sbs::SchwingerBosonSystem, f, g, x)
    set_μ0!(sbs, x)

    # Calculate the gradient
    g .= 0.0
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    (; L, S) = sbs
    Nu = L^2

    ∂D∂X_re = zeros(ComplexF64, 12, 12)

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        for α in 1:3
            ∂ID∂μ0!(∂D∂X_re, tmp, α)
            g[α] += -real(tr(P * ∂D∂X_re))
        end
    end

    @. g /= Nu

    for α in 1:3
        g[α] += -(1+2S)
    end

    # The function value
    f = - free_energy_mean_field(sbs)
    return f
end

# Calculate the divided difference matrix as L"\mathbb{D}" defined in the note
function divided_difference!(sbs, Dmat, E; atol=1e-8)
    Dmat .= 0.0
    T = sbs.T
    for i in eachindex(E), j in eachindex(E)
        if isapprox(E[i], E[j]; atol)
            Dmat[i, j] += T > 1e-8 ? -csch(E[i]/(2T))^2 / (2T) : 0.0
        else
            Dmat[i, j] += (coth(E[i]/2T) - coth(E[j]/2T)) / (E[i] - E[j])
        end
    end
end

# The contents of `tmp` and `tmp2` are modified in-place. The remainings are fixed. The final result is stored in `tmp`.
function divided_aux!(tmp, tmp2, Dmat, ∂D∂X, V, inv_V)
    mul!(tmp, inv_V, ∂D∂X)
    mul!(tmp2, tmp, V)
    tmp .= Dmat .* tmp2
    mul!(tmp2, V, tmp)
    mul!(tmp, tmp2, inv_V)
end

function fgh_μ0!(sbs::SchwingerBosonSystem, f, g, h, x)
    set_μ0!(sbs, x)

    # Calculate the gradient
    g .= 0.0
    h .= 0.0

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    tmp2 = zeros(ComplexF64, 12, 12)
    Dmat = zeros(ComplexF64, 12, 12)

    (; L, S) = sbs
    Nu = L^2

    ∂D∂X_re_α = zeros(ComplexF64, 12, 12, 3)

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        E = single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)

        @views for α in 1:3
            ∂ID∂μ0!(∂D∂X_re_α[:, :, α], tmp, α)
        end

        @views for α in 1:3
            g[α] += -real(tr(P * ∂D∂X_re_α[:, :, α])) / Nu
            divided_aux!(tmp, tmp2, Dmat, ∂D∂X_re_α[:, :, α], V, inv_V)
            for β in 1:3
                h[α, β] += -real(tr(tmp * ∂D∂X_re_α[:, :, β])) / (4Nu)
            end
        end
    end

    for α in 1:3
        g[α] += -(1+2S)
    end

    # The function value
    f = - free_energy_mean_field(sbs)
    return f
end

# The variational free energy objective function with gradient for Optim
function fg_ϕ!(sbs::SchwingerBosonSystem, f, g, ϕ)
    if isnothing(g)
        g = zero(ϕ)
    end
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

    optimize_μ0!(sbs, μ0s; algorithm=Optim.GradientDescent(), options = Optim.Options(show_trace=false, iterations=100, extended_trace=false))

    f = 0.0

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        try
            E = bogoliubov!(V, D)
            for n in 1:6
                f += E[n] / (2Nu)
                (T > 1e-8) && (f += real(T * log1mexp_modified(E[n]/T)) / Nu)
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

    # Now we add the contribution from the "correction" term L"⟨H - H_{MF}⟩_{MF}"
    for α in 1:24
        f += inv_fα[α] * ∂F2α[α]^2 / 12 - ∂F2α[α] * ϕ[α]
        for β in 1:24
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
            κ0[α, β] += -∂F2αβ[α+24, β+24]
        end
        for β in 1:24
            ∂ΔH∂μ0[α] += ∂F2αβ[α+24, β] * (inv_fα[β]/6 * ∂F2α[β] - ϕ[β])
        end
    end

    sbs.Δμs .= pinv(κ0) * ∂ΔH∂μ0

    for α in 1:24
        for β in 1:3
            g[α] += ∂F2αβ[α, β+24] * sbs.Δμs[β]
        end
    end

    return f
end