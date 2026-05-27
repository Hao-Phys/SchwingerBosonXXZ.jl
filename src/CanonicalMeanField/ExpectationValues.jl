# Mean-field expectation values of observables
# 
# `mean_fields` returns to the mean-field values of As, Bs, Cs, and Ds (12 dim vector).
# `ns` returns to the boson number ⟨n⟩ for three sublattices (3 dim vector).
# `Sμ` returns to ⟨S^μ_a⟩ for μ = x, y, z for on a-th column for sublattice a (3x3 matrix).
function expectation_values(sbs::SchwingerBosonSystem;
    options = Optim.Options(show_trace=false, iterations=1000), tol=1e-8, max_iters=100)

    aux = CondensationAux(0.0, 0.0, nothing, 0.0, 0)
    μ0s = copy(real(sbs.mean_fields[13:15]))
    optimize_μ0!(sbs, μ0s, aux; options, tol, max_iters)
    den_mat_conden = condensation_results!(sbs, aux)

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

        linear_idx = (j-1)*L + i
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