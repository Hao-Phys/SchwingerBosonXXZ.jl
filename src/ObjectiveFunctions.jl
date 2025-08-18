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
    end
end

# Objective function for the chemical potential optimization and its gradient
function fg_μ!(sbs::SchwingerBosonSystem, f, g, x)
    set_μ!(sbs, x)

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
            ∂ID∂μ!(∂D∂X_re, tmp, α)
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

# The variational free energy objective function for Optim
function fg_ϕ!(sbs::SchwingerBosonSystem, f, g, ϕ)
    # Make sure to set the mean fields to the `sbs` object
    set_ϕ!(sbs, ϕ)

    # Maximize the mean-field free energy to find the optimal chemical potential
    μ0 = real(sbs.mean_fields[13:15])
    optimize_μ!(sbs, μ0)

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    # Calculate the "temperature*entropy" term of the free energy
    (; L, J, Δ, S, T) = sbs
    Nu = L^2
    f = 0.0
    nα = zeros(3)

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        try
            E = bogoliubov!(V, D)
            for n in 1:6
                f += E[n] / (2Nu)
                (T > 1e-8) && (f += real(T * log1p(-exp(-E[n]/T))) / Nu)
                for α in 1:3
                    nα[α] += ((abs2(V[combine_index(α,1), n]) + abs2(V[combine_index(α,2), n])) * bose(E[n], T) + (abs2(V[combine_index(α,1), n+6]) + abs2(V[combine_index(α,2), n+6]) * bose(E[n+6], T))) / Nu
                end
            end
        # If the dynamical matrix is not positive definite, return to Inf immediately
        catch _
            @warn "Dynamical matrix is not positive definite, returning Inf. Skipping this configuration..."
            g .= Inf
            f = Inf
            return f
        end
    end

    # Sometimes even though the dynamical matrix is positive definite, the number of Schwinger bosons per site may different from 2S, leading to unphysical mean-fields.
    if prod(isapprox.(nα, 2S; atol=1e-6))
        # The "normal" operation
        # Initialize the gradient
        g .= 0.0
        # Buffers to calculate the gradient
        P = zeros(ComplexF64, 12, 12)
        tmp = zeros(ComplexF64, 12, 12)
        tmp2 = zeros(ComplexF64, 12, 12)
        Dmat = zeros(ComplexF64, 12, 12)
        # Buffers to hold the derivatives of the dynamical matrix
        ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 24)
        ∂F2αβ = zeros(24, 24)

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

        # Here we calculate the gradient contribution from the entropy term and the Πα vector and ∂Παβ matrix
        for i in 1:L, j in 1:L
            q = Vec3([(i-1)/L, (j-1)/L, 0.0])
            E = single_particle_density_matrix!(P, D, V, tmp, sbs, q)
            inv_V = inv(V)
            divided_difference!(sbs, Dmat, E)

            @views for α in 1:3
                ∂ID∂A!(∂ID∂ϕs[:, :, α], ∂ID∂ϕs[:, :, α+12], tmp, sbs, q, α)
                ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+15], tmp, sbs, q, α)
                ∂ID∂C!(∂ID∂ϕs[:, :, α+6], ∂ID∂ϕs[:, :, α+18], tmp, sbs, q, α)
                ∂ID∂D!(∂ID∂ϕs[:, :, α+9], ∂ID∂ϕs[:, :, α+21], tmp, sbs, q, α)
            end

            for α in 1:24
                # Gradient from entropy term
                g[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
                # Calculate the second derivatives of the quadratic free energy
                divided_aux!(tmp, tmp2, Dmat, ∂ID∂ϕs[:, :, α], V, inv_V)
                for β in 1:24
                    ∂F2αβ[α, β] += real(tr(tmp * ∂ID∂ϕs[:, :, β])) / (4Nu)
                end
            end
        end

        # Now we add the contribution from the "correction" term L"⟨H - H_{MF}⟩_{MF}"
        g2 = copy(g)
        for α in 1:24
            f += inv_fα[α] * g2[α]^2 / 12 - g2[α] * ϕ[α]
            for β in 1:24
                g[α] += ∂F2αβ[α, β] * (inv_fα[β]/6 * g2[β] - ϕ[β])
                if β == α
                    g[α] -= g2[β]
                end
            end
        end

        μs = sbs.mean_fields[13:15]
        for α in 1:3
            f += real(μs[α]) * (2S+1)
        end
        return f
    else
        @warn "The number of Schwinger bosons per site is not equal to 2S, returning Inf. Skipping this configuration..."
        g .= Inf
        f = Inf
        return f
    end

end