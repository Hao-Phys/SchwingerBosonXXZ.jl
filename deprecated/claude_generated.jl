# ── KM geometric gradient descent ─────────────────────────────────────────────
#
# optimize_mean_fields_km!(sbs, ϕ0; …)
#
# Minimises F^var over the 12 complex mean-field variables using the Wirtinger
# (Kubo-Mori) gradient
#
#   g_w[ν] = ∂F/∂O_ν  (12-element ComplexF64, non-conjugate Wirtinger gradient)
#
# The steepest-descent step is  ΔO_ν = −η · conj(g_w[ν]) = −η · ∂F/∂O*_ν,
# which is exactly the gradient flow on ℂ¹² ≅ ℝ²⁴ with Euclidean metric.
# A backtracking Armijo line search controls the step size.
#
# Arguments
#   ϕ0        – initial 12-element ComplexF64 mean-field vector
#   max_iter  – maximum gradient steps (default 2000)
#   η0        – initial step size (default 0.1)
#   armijo_c  – sufficient-decrease constant (default 1e-4)
#   armijo_ρ  – step-size reduction factor (default 0.5)
#   tol_g     – stop when ‖g_w‖ < tol_g (default 1e-7)
#   tol_f     – stop when |ΔF| < tol_f  (default 1e-12)
#   show_trace – print iteration info (default false)

function optimize_mean_fields_km!(sbs::SchwingerBosonSystem,
                                   ϕ0::Vector{ComplexF64};
                                   max_iter  :: Int     = 2000,
                                   η0        :: Float64 = 0.1,
                                   armijo_c  :: Float64 = 1e-4,
                                   armijo_ρ  :: Float64 = 0.5,
                                   tol_g     :: Float64 = 1e-7,
                                   tol_f     :: Float64 = 1e-12,
                                   show_trace :: Bool   = false)

    ϕ     = copy(ϕ0)
    g     = zeros(ComplexF64, 12)
    f_ref = Ref(0.0)

    # Initial evaluation
    fg_ϕ!(sbs, f_ref, g, ϕ)
    f_cur = f_ref[]

    η = η0   # step size carried across iterations (warm start)

    for iter in 1:max_iter
        gnorm = norm(g)
        if show_trace
            println("  KM iter $(lpad(iter,4))  F = $(f_cur)  ‖g_w‖ = $(gnorm)  η = $(η)")
        end
        gnorm < tol_g && break

        # Descent direction d = -conj(g_w); directional derivative = -2‖g_w‖²
        slope = -2.0 * sum(abs2, g)   # = 2·real(dot(conj(g), -conj(g))) < 0

        # Armijo backtracking line search
        η_try = η
        ϕ_new = ϕ - η_try .* conj.(g)
        fg_ϕ!(sbs, f_ref, g, ϕ_new)
        f_new = f_ref[]

        backtrack_count = 0
        while f_new > f_cur + armijo_c * η_try * slope && backtrack_count < 50
            η_try        *= armijo_ρ
            backtrack_count += 1
            ϕ_new         = ϕ - η_try .* conj.(g)
            fg_ϕ!(sbs, f_ref, g, ϕ_new)
            f_new         = f_ref[]
        end

        ϕ = ϕ_new
        # Mild step-size expansion for next iteration (bold driver)
        η = min(η_try * 1.2, η0)

        abs(f_new - f_cur) < tol_f && break
        f_cur = f_new
    end

    set_ϕ!(sbs, ϕ)
    return ϕ
end

# Computes c-shifts
c_shifts = Float64[]
for i in 1:L, j in 1:L
    q = Vec3([(i-1)/L, (j-1)/L, 0.0])
    dynamical_matrix!(D, sbs, q)
    E = bogoliubov!(V, D)[1:6]
    E_min = minimum(E)
    if E_min < condensation_ϵ
        # Newton iteration: find c ≥ 0 such that E_min(D + cI) = ϵ.
        # Initial guess: assume unit slope → c₀ = ϵ - E_min.
        c = max(0.0, condensation_ϵ - E_min)
        for _ in 1:max_iters
            copyto!(D_tmp, D)
            @inbounds for k in 1:12; D_tmp[k, k] += c; end
            E_c = bogoliubov!(V, D_tmp)[1:6]
            residual = minimum(E_c) - condensation_ϵ
            abs(residual) ≤ tol && break
            # Numerical derivative: perturb c by δc and measure ΔE_min,
            # which gives ∂h/∂c ≈ ΔE_min/δc.
            # We then update c by -residual / (∂h/∂c).
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