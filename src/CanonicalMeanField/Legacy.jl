function optimize_μ0_legacy!(sbs::SchwingerBosonSystem, μ0; 
    options = Optim.Options(show_trace=false, iterations=100))
    fg!(f, g, x) = fg_μ0!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), μ0, Optim.LBFGS(), options)
    μ0_minimizer = ret.minimizer
    set_μ0!(sbs, μ0_minimizer)
end

# Variational free energy and its gradient with respect to the mean fields ϕ
# given the Schwinger boson number constraints are satisfied by μ₀⋆ and μ⋆
function fg_ϕ_legacy!(sbs::SchwingerBosonSystem, f, g, ϕ; 
    options = Optim.Options(show_trace=false, iterations=100))

    set_ϕ!(sbs, ϕ)
    (; L, S, T) = sbs
    Nu = L^2

    if isnothing(g)
        g = zero(ϕ)
    else
        g .= 0.0
    end

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
    μ0s = copy(real(sbs.mean_fields[13:15])) .- (τ + T)

    optimize_μ0_legacy!(sbs, μ0s; options)

    # Buffers
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    tmp2 = zeros(ComplexF64, 12, 12)
    Dmat = zeros(ComplexF64, 12, 12)
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 27)
    ∂F2α = zeros(27)
    ∂F2αβ = zeros(27, 27)

    # The bosonic free energy contribution,
    # whose gradient is cancelled by the "correction" term in the variational free energy.
    # See below
    f = bosonic_free_energy!(nothing, V, D, sbs)

    inv_fα = inv_interaction_strengths(sbs)

    # Computes the gradient of Ĩ D with respect to the chemical potentials μ₀
    @views for α in 1:3
        ∂ID∂μ0!(∂ID∂ϕs[:, :, α+24], α)
    end

    # Calculates `∂F2α` and `∂F2αβ`
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        E = single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)

        # Computes the gradient of Ĩ D_q with respect to the mean fields A, B, C, and D.
        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α],   ∂ID∂ϕs[:, :, α+12], sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+15], sbs, q, α)
            ∂ID∂C!(∂ID∂ϕs[:, :, α+6], ∂ID∂ϕs[:, :, α+18], sbs, q, α)
            ∂ID∂D!(∂ID∂ϕs[:, :, α+9], ∂ID∂ϕs[:, :, α+21], sbs, q, α)
        end

        @views for α in 1:27
            # Computes ∂F / ∂ϕ_α (F being the bosonic free energy),
            # which is stored in `∂F2α`.
            # In our convention: ∂F2α = f[α] * ⟨\hat{O}[α]⟩₀,
            # with \hat{O}[α] being the real (α=1:12) or imaginary (α=13:24) part of
            # the corresponding operators (\hat{A}, \hat{B}, \hat{C}, \hat{D})
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
            # Calculate the second derivatives of the bosonic free energy
            # ∂F2αβ = ∂²F / ∂ϕ_α∂ϕ_β = f[α] * f[β] ∂⟨\hat{O}[β]⟩₀ / ∂ϕ_α
            divided_aux!(tmp, tmp2, Dmat, ∂ID∂ϕs[:, :, α], V, inv_V)
            for β in 1:27
                ∂F2αβ[α, β] += real(tr(tmp * ∂ID∂ϕs[:, :, β])) / Nu
            end
        end
    end

    # Now we add the contribution from the "correction" term L"⟨H - H_{MF}⟩₀"
    for α in 1:24
        f += inv_fα[α] * ∂F2α[α]^2 / 12 - ∂F2α[α] * ϕ[α]
        # Accumulate the gradient from the above "correction" term
        # Note that the additional term -δ_{αβ} ∂F2α[β] cancels the contribution
        # from the gradient of the bosonic free energy.
        for β in 1:24
            g[α] += ∂F2αβ[α, β] * (inv_fα[β]/6 * ∂F2α[β] - ϕ[β])
        end
    end

    # Now we add the contribution from - Δμ * ⟨n⟩₀ and its gradient,
    # where Δμ = pinv(κ0) * (ΔH; n)_{KM} = pinv(κ0) * ∂ΔH/∂μ₀.
    μ0s = real(sbs.mean_fields[13:15])
    # Buffer for the compressiblity matrix κ0
    κ0 = zeros(3, 3)
    # Buffer for the term ∂ΔH/∂μ₀
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

function optimize_mean_fields_legacy!(sbs::SchwingerBosonSystem, ϕ0; 
    options_inner = Optim.Options(show_trace=false, iterations=1000),
    options_outer = Optim.Options(show_trace=false, iterations=1000),
    algorithm_outer = Optim.LBFGS())

    set_ϕ!(sbs, ϕ0)
    fg!(f, g, x) = fg_ϕ_legacy!(sbs, f, g, x; options=options_inner)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm_outer, options_outer)
    set_ϕ!(sbs, ret.minimizer)

    return ret
end

function expectation_values_legacy(sbs::SchwingerBosonSystem;
    optimize_μ::Bool=true,
    options = Optim.Options(show_trace=false, iterations=100))
    if optimize_μ
        μ0s = copy(real(sbs.mean_fields[13:15]))
        optimize_μ0_legacy!(sbs, μ0s; options)
    end

    (; L, S) = sbs
    Nu = L^2

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 27)
    ∂D∂Ss = zeros(ComplexF64, 12, 12, 3, 3)

    # Computes the gradient of Ĩ D with respect to the chemical potentials μ₀
    @views for α in 1:3
        ∂ID∂μ0!(∂ID∂ϕs[:, :, α+24], α)
        for μ in 1:3
            ∂ID∂S!(∂D∂Ss[:, :, α, μ], α, μ, sbs)
        end
    end

    ∂F2α = zeros(27)
    Ss_exps = zeros(3, 3)
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)

        # Computes the gradient of Ĩ D_q with respect to the mean fields A, B, C, and D.
        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α],   ∂ID∂ϕs[:, :, α+12], sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+15], sbs, q, α)
            ∂ID∂C!(∂ID∂ϕs[:, :, α+6], ∂ID∂ϕs[:, :, α+18], sbs, q, α)
            ∂ID∂D!(∂ID∂ϕs[:, :, α+9], ∂ID∂ϕs[:, :, α+21], sbs, q, α)
        end

        @views for α in 1:27
            # Computes ∂F / ∂ϕ_α (F being the bosonic free energy),
            # which is stored in `∂F2α`.
            # In our convention: ∂F2α = f[α] * ⟨\hat{O}[α]⟩₀,
            # with \hat{O}[α] being the real (α=1:12) or imaginary (α=13:24) part of
            # the corresponding operators (\hat{A}, \hat{B}, \hat{C}, \hat{D})
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
        end

        for α in 1:3, μ in 1:3
            Ss_exps[α, μ] += real(tr(P * ∂D∂Ss[:, :, α, μ])) / Nu
        end
    end

    mean_fields = zeros(ComplexF64, 12)
    inv_fα = inv_interaction_strengths(sbs)
    for α in 1:12
        mean_fields[α] = inv_fα[α] * ∂F2α[α] / 6 + 1im * inv_fα[α+12] * ∂F2α[α+12] / 6
    end

    ns = zeros(3)
    for α in 1:3
        ns[α] = -∂F2α[α+24] - 1
    end

    return mean_fields, ns, Ss_exps
end

function dssf_mean_field_integration(sbs::SchwingerBosonSystem, q, energies, Γ; opts...)
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

    return ret
end