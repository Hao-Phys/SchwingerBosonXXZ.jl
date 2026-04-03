function f_μ0!(sbs::SchwingerBosonSystem, x)
    set_μ0!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    (; L, S, T, mean_fields, J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2
    Nu = L^2

    f = 0.0
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)
        for n in 1:6
            f += -E[n] / (2Nu)
            (T > 1e-8) && (f += -real(T * log1mexp_modified(E[n]/T)) / Nu)
        end
    end

    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]

    for α in 1:3
        f -= -3 * (-J₊ * abs2(As[α]) + J₊ * abs2(Bs[α]) + J₋ * abs2(Cs[α]) - J₋ *abs2(Ds[α]))
        f -= (1+2S) * x[α]
    end

    return f
end

# Use Newton's method to find the diagonal shift c added to the dynamical matrix
# so that the minimum positive Bogoliubov eigenvalue equals ϵ.
#
# Physical basis: adding c·I₁₂ to D changes ĨD → Ĩ(D + cI) = ĨD + cĨ.
# For positive-sector BdG modes the eigenvalue shifts at rate ≈ +1 per unit c,
# so Newton converges in very few iterations (often 1–3).
function search_for_condensation_shift!(sbs::SchwingerBosonSystem, y, ϵ, δ)
    set_μ0!(sbs, y)
    (; L) = sbs

    tol      = 1e-12
    max_iter = 50
    D     = zeros(ComplexF64, 12, 12)
    D_tmp = zeros(ComplexF64, 12, 12)
    V     = zeros(ComplexF64, 12, 12)

    # Pre-shift μ₀ so that D is classically positive-definite
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
        E_min = minimum(bogoliubov!(V, D)[1:6])

        if E_min < ϵ
            # Newton iteration: find c ≥ 0 such that E_min(D + cI) = ϵ.
            # Initial guess: assume unit slope → c₀ = ϵ - E_min.
            c = max(0.0, ϵ - E_min)
            for _ in 1:max_iter
                copyto!(D_tmp, D)
                @inbounds for k in 1:12; D_tmp[k, k] += c; end
                E_c = bogoliubov!(V, D_tmp)[1:6]
                residual = minimum(E_c) - ϵ
                abs(residual) ≤ tol && break

                # Numerical derivative: perturb c by δc and measure ΔE_min
                δc = max(1e-7, abs(c) * 1e-6)
                copyto!(D_tmp, D)
                @inbounds for k in 1:12; D_tmp[k, k] += c + δc; end
                E_p = bogoliubov!(V, D_tmp)[1:6]
                dEdc = (minimum(E_p) - minimum(E_c)) / δc
                dEdc = abs(dEdc) < 1e-15 ? 1.0 : dEdc   # fallback: unit slope
                c -= residual / dEdc
                c = max(0.0, c)
            end
            push!(c_shifts, c)
        else
            push!(c_shifts, 0.0)
        end
    end

    return maximum(c_shifts) + shift
end

function f_tilde_y!(sbs::SchwingerBosonSystem, y, ϵ, δ)
    c = search_for_condensation_shift!(sbs, y, ϵ, δ)
    μ_shifted = y .- c
    return f_μ0!(sbs, μ_shifted)
end

function optimize_μ0_f_tilde!(sbs::SchwingerBosonSystem, μ0, ϵ, δ; algorithm=Optim.NelderMead(), options = Optim.Options(show_trace=false, iterations=100))
    f!(y) = f_tilde_y!(sbs, y, ϵ, δ) 
    ret = optimize(f!, μ0, algorithm, options)
    y_minimizer = ret.minimizer
    set_μ0!(sbs, y_minimizer)
    c = search_for_condensation_shift!(sbs, y_minimizer, ϵ, δ)
    μ_shifted = y_minimizer .- c
    set_μ0!(sbs, μ_shifted)
end

function single_particle_density_matrix_tmp!(P::Matrix{ComplexF64}, D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)
    P .= 0.0
    (; T) = sbs
    dynamical_matrix!(D, sbs, q_reshaped)
    E = bogoliubov!(V, D)
    inv_V = inv(V)
    for i in eachindex(E)
        tmp = coth(E[i] / (2T)) / 4 * V[:, i] * transpose(inv_V[i, :])
        P .+= tmp
    end
end

function modified_single_particle_density_matrix!(P::Matrix{ComplexF64}, D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, tmp::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3, ϵ::Float64)
    P .= 0.0
    (; T, L, S) = sbs
    dynamical_matrix!(D, sbs, q_reshaped)
    E = bogoliubov!(V, D)
    for i in eachindex(E)
        P[i, i] += coth(E[i] / (2T)) / 4
    end
    mul!(tmp, V, P)
    mul!(P, tmp, inv(V))

    condensed_indices = findall(x -> isapprox(x, ϵ; atol=1e-8), E)
    num_ncs = length(condensed_indices)
    if !isempty(condensed_indices)
        # Calculate ξ
        condensed_index1 = condensed_indices[1]
        inv_V = inv(V)
        vr = copy(V[:, condensed_index1])
        vl = transpose(inv_V[condensed_index1, :])
        ∂D∂X_re = zeros(ComplexF64, 12, 12)
        ∂ID∂μ0!(∂D∂X_re, tmp, 1)
        qc = - vl * ∂D∂X_re * vr

        P_tmp = zeros(ComplexF64, 12, 12)
        D_tmp = zeros(ComplexF64, 12, 12)
        V_tmp = zeros(ComplexF64, 12, 12)
        N_normal = 0
        for i in 1:L, j in 1:L
            q_tmp = Vec3([(i-1)/L, (j-1)/L, 0.0])
            dynamical_matrix!(D_tmp, sbs, q_tmp)
            single_particle_density_matrix!(P_tmp, D_tmp, V_tmp, tmp, sbs, q_tmp)
            N_normal += -real(tr(P_tmp * ∂D∂X_re)) / L^2
        end
        ξ = (2S + 1 - N_normal) / (num_ncs *qc)

        for idx in condensed_indices
            vr = copy(V[:, idx])
            vl = transpose(inv_V[idx, :])
            P_condensed = ξ * vr * vl * L^2
            P .+= P_condensed
        end
    end
end

function self_consistent_mean_fields_condensed!(f, ϕ, sbs::SchwingerBosonSystem, ϵ, δ)
    set_ϕ!(sbs, ϕ)
    μ0 = real.(copy(sbs.mean_fields[13:15]))
    optimize_μ0_f_tilde!(sbs, μ0, ϵ, δ; algorithm=Optim.NelderMead(), options=Optim.Options(iterations=500, g_tol=1e-12))

    (; L, J, Δ) = sbs
    Nu = L^2

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 27)
    ∂F2α = zeros(27)

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
        modified_single_particle_density_matrix!(P, D, V, tmp, sbs, q, ϵ)

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
        end
    end

    for α in 1:24
        f[α] = inv_fα[α]/6 * ∂F2α[α]
    end
end

function solve_self_consistent_mean_fields_condensed!(sbs::SchwingerBosonSystem, x0, ϵ, δ; nlsolve_opts::NamedTuple=NamedTuple(;))
    sce_eqn! = (f, x) -> self_consistent_mean_fields_condensed!(f, x, sbs, ϵ, δ)
    ret = fixedpoint(sce_eqn!, x0; nlsolve_opts...)
    !converged(ret) && @warn "Self-consitent equations converged to a solution with residual $(ret.residual_norm)"
    best_mean_fields = zeros(ComplexF64, 15)
    for i in 1:12
        best_mean_fields[i] = ret.zero[i] + 1im * ret.zero[i+12]
    end
    best_mean_fields[13:15] = sbs.mean_fields[13:15]
    set_mean_fields!(sbs, best_mean_fields)
    return ret.residual_norm
end

function expectation_values_condensed(sbs::SchwingerBosonSystem, ϵ, δ)
    μ0 = real.(copy(sbs.mean_fields[13:15]))
    optimize_μ0_f_tilde!(sbs, μ0, ϵ, δ; algorithm=Optim.NelderMead(), options=Optim.Options(iterations=500, g_tol=1e-12))

    (; L, J, Δ) = sbs
    Nu = L^2

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 27)
    f = zeros(27)
    ∂F2α = zeros(27)

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
        modified_single_particle_density_matrix!(P, D, V, tmp, sbs, q, ϵ)

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
        end
    end

    for α in 1:24
        f[α] = inv_fα[α]/6 * ∂F2α[α]
    end
    for α in 25:27
        f[α] = - ∂F2α[α] - 1
    end

    return f
end

function ∂ID∂S!(∂D∂S_re, tmp, α::Int, μ::Int, sbs::SchwingerBosonSystem)
    S = sbs.S
    ∂D∂S_re .= 0.0
    index = 2α-1
    view1 = view(∂D∂S_re, index:index+1, index:index+1)
    view1 .= S * σs[μ]
    view2 = view(∂D∂S_re, index+6:index+7, index+6:index+7)
    view2 .= S * σs[μ]
    mul!(tmp, Ĩ, ∂D∂S_re)
    copyto!(∂D∂S_re, tmp)
end

function spin_expectations_condensed(sbs::SchwingerBosonSystem, ϵ::Float64, δ::Float64)
    μ0 = real.(copy(sbs.mean_fields[13:15]))
    optimize_μ0_f_tilde!(sbs, μ0, ϵ, δ; algorithm=Optim.NelderMead(), options=Optim.Options(iterations=500, g_tol=1e-12))

    (; L) = sbs
    Nu = L^2

    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    ∂D∂S_re = zeros(ComplexF64, 12, 12, 3, 3)
    @views for α in 1:3, μ in 1:3
        ∂ID∂S!(∂D∂S_re[:, :, α, μ], tmp, α, μ, sbs)
    end

    S_exps = zeros(3, 3)
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        modified_single_particle_density_matrix!(P, D, V, tmp, sbs, q, ϵ)

        @views for α in 1:3, μ in 1:3
            S_exps[μ, α] += real(tr(P * ∂D∂S_re[:, :, α, μ])) / Nu
        end
    end

    return S_exps
end