# Search for the condensation shift `c` such that
# the minimum eigenvalue of Ĩ * D is ϵ
function search_for_condensation_shift!(sbs::SchwingerBosonSystem, y;
    tol=1e-8, max_iters=1000)
    set_μ0!(sbs, y)
    (; L, condensation_ϵ) = sbs
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    # Pre-shift μ₀ so that D is positive-definite
    shifts = Float64[]
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        push!(shifts, max(0.0, -eigmin(D)))
    end
    shift = maximum(shifts) + 1e-8
    set_μ0!(sbs, copy(y) .- shift)

    c_shifts = Float64[]
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)[1:6]
        eigval_min = minimum(E)
        if eigval_min < condensation_ϵ
            c_min, c_max = 0.0, 50.0
            find_c = false
            iter = 0
            while !find_c && iter < max_iters
                c = (c_min + c_max) / 2
                dynamical_matrix!(D, sbs, q)
                for k in 1:12
                    D[k, k] += c
                end
                E = bogoliubov!(V, D)[1:6]
                eigval_min = minimum(E)
                if eigval_min - condensation_ϵ > tol
                    c_max = c
                elseif eigval_min - condensation_ϵ < -tol
                    c_min = c
                else
                    find_c = true
                end
                iter += 1
            end
            push!(c_shifts, c)
        else
            c = 0.0
            push!(c_shifts, c)
        end
    end

    c_shift = maximum(c_shifts) + shift
    return c_shift
end

# `sign` is used to control whether we are evaluating the original bosonic free energy 
# (sign=1) or its negative (sign=-1, used in the optimization of μ₀)
function bosonic_free_energy!(g_boson::Union{Nothing, AbstractVector}, V::Matrix{ComplexF64}, D::Matrix{ComplexF64},
    sbs::SchwingerBosonSystem; sign::Int=1)
    (; L, T) = sbs
    Nu = L^2
    f = 0.0
    !isnothing(g_boson) && (g_boson .= 0.0)

    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    ∂ID = zeros(ComplexF64, 12, 12)
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        E = single_particle_density_matrix_normal!(P, D, V, tmp, sbs, q)
        if any(isnan, E)
            isnothing(g_boson) || (g_boson .= NaN)
            f = Inf
            return f
        else
            @inbounds for n in 1:6
                f += sign * E[n] / (2Nu)
                (T > 1e-8) && (f += real(sign * T * log1mexp_modified(E[n]/T)) / Nu)
            end
            if !isnothing(g_boson)
                @inbounds for α in 1:3
                    ∂ID∂μ0!(∂ID, α)
                    g_boson[α] += sign * real(tr(P * ∂ID)) / Nu
                end
            end
        end
    end

    return f
end

# Objective functions for the chemical potential optimization.
# The regular function without a shift
function fg_μ0!(sbs::SchwingerBosonSystem, f, g, x)
    set_μ0!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    (; S, mean_fields, J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2

    # Maximize the bosonic free energy, equivalent to minimizing the negative of it.
    f = bosonic_free_energy!(g, V, D, sbs; sign=-1)

    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]

    for α in 1:3
        f -= -3 * (-J₊ * abs2(As[α]) + J₊ * abs2(Bs[α]) + J₋ * abs2(Cs[α]) - J₋ *abs2(Ds[α]))
        f -= (1+2S) * x[α]
        g[α] -= (1+2S)
    end

    return f
end

# We need to shift μ₀ by `c_shift` to make sure that the minimum mode
# is greater or equal to the condensation threshold ϵ.
function f_y!(sbs::SchwingerBosonSystem, y; tol=1e-8, max_iters=100)
    c_shift = search_for_condensation_shift!(sbs, y; tol, max_iters)
    μ0_shifted = copy(y) .- c_shift
    return f_μ0!(sbs, μ0_shifted)
end

# Variational free energy and its gradient with respect to the mean fields ϕ
# given the Schwinger boson number constraints are satisfied by μ₀⋆ and μ⋆
function fg_ϕ!(sbs::SchwingerBosonSystem, f, g, ϕ; 
    options = Optim.Options(show_trace=false, iterations=100), tol=1e-8, max_iters=100)

    set_ϕ!(sbs, ϕ)
    (; L, S) = sbs
    Nu = L^2

    if isnothing(g)
        g = zero(ϕ)
    end
    g .= 0.0

    μ0s = copy(real(sbs.mean_fields[13:15]))
    optimize_μ0!(sbs, μ0s; options, tol, max_iters)

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
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
        E = single_particle_density_matrix_condensed!(P, D, V, tmp, sbs, q)
        inv_V = inv(V)
        divided_difference!(sbs, Dmat, E)
        divided_difference_condensed!(sbs, Dmat, E, P, V, inv_V)

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