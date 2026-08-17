function self_consistent_mean_fields!(f, ϕ, sbs::SchwingerBosonSystem; options_μ = Optim.Options(show_trace=false, iterations=100), tol=1e-12, max_iters=1000)
    set_ϕ!(sbs, ϕ)
    (; L) = sbs

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    μ0 = copy(real(sbs.mean_fields[13:15]))
    aux = CondensationAux(0.0, 0.0, nothing, 0.0, 0)
    optimize_μ0!(sbs, μ0, aux; options=options_μ, tol, max_iters)
    den_mat_conden = condensation_results!(sbs, aux)

    Nu = L^2
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 24)

    # Computes the gradient of Ĩ D with respect to the chemical potentials μ₀
    # @views for α in 1:3
    #     ∂ID∂μ0!(∂ID∂ϕs[:, :, α+24], α)
    # end

    ∂F2α = zeros(24)
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)

        linear_idx = (i-1)*L + j
        if linear_idx == aux.conden_index && !isnothing(den_mat_conden)
            P .+= den_mat_conden
        end

        # Computes the gradient of Ĩ D_q with respect to the mean fields A, B, C, and D.
        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α],   ∂ID∂ϕs[:, :, α+12], sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+15], sbs, q, α)
            ∂ID∂C!(∂ID∂ϕs[:, :, α+6], ∂ID∂ϕs[:, :, α+18], sbs, q, α)
            ∂ID∂D!(∂ID∂ϕs[:, :, α+9], ∂ID∂ϕs[:, :, α+21], sbs, q, α)
        end

        @views for α in 1:24
            # Computes ∂F / ∂ϕ_α (F being the bosonic free energy),
            # which is stored in `∂F2α`.
            # In our convention: ∂F2α = f[α] * ⟨\hat{O}[α]⟩₀,
            # with \hat{O}[α] being the real (α=1:12) or imaginary (α=13:24) part of
            # the corresponding operators (\hat{A}, \hat{B}, \hat{C}, \hat{D})
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
        end
    end

    inv_fα = inv_interaction_strengths(sbs)
    for α in 1:24
        f[α] = inv_fα[α]/6 * ∂F2α[α]
    end
end

function solve_self_consistent_mean_fields_condensed!(sbs::SchwingerBosonSystem, x0; nlsolve_opts::NamedTuple=NamedTuple(;), options_μ = Optim.Options(show_trace=false, iterations=1000), tol=1e-8, max_iters=100)
    sce_eqn!(f, ϕ) = self_consistent_mean_fields!(f, ϕ, sbs; options_μ, tol, max_iters)
    ret = fixedpoint(sce_eqn!, x0; nlsolve_opts...)
    !converged(ret) && @warn "Self-consitent equations converged to a solution with residual $(ret.residual_norm)"
    best_mean_fields = zeros(ComplexF64, 15)
    for i in 1:12
        best_mean_fields[i] = ret.zero[i] + 1im * ret.zero[i+12]
    end
    best_mean_fields[13:15] = sbs.mean_fields[13:15]
    set_mean_fields!(sbs, best_mean_fields)

    μ0 = copy(real(sbs.mean_fields[13:15]))
    aux = CondensationAux(0.0, 0.0, nothing, 0.0, 0)

    optimize_μ0!(
        sbs,
        μ0,
        aux;
        options = options_μ,
        tol,
        max_iters,
    )

    return ret.residual_norm
end