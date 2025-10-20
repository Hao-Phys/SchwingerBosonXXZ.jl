function newton_with_backtracking(fgh!, x0; f_reltol=NaN, x_reltol=NaN, g_abstol=NaN, maxiters=20, armijo_c=1e-4, armijo_backoff=0.5, armijo_α_min=1e-4, show_trace=false)
    # Make a copy of the initial guess.
    x = copy(x0)

    # Preallocate buffers
    candidate_x = zero(x0)
    g = zero(x0)
    H = zeros(eltype(x0), length(x0), length(x0))
    p = zero(x0)

    # Evaluate objective function f, gradient, and Hessian.
    f = fgh!(0.0, g, H, x)
    isfinite(f) || error("Initial guess is not feasible")

    maximum(abs, g) < g_abstol && return x

    function has_converged(x, candidate_x, f, candidate_f, g)
        return (!isnan(x_reltol) && isapprox(x, candidate_x; rtol=x_reltol)) ||
               (!isnan(f_reltol) && isapprox(f, candidate_f; rtol=f_reltol)) ||
               (!isnan(g_abstol) && maximum(abs, g) < g_abstol)
    end

    for k in 1:maxiters
        # Newton direction p = H \ g.
        ldiv!(p, cholesky!(Hermitian(H)), g)

        norm_g = norm(g)

        # Start with damped Newton step
        α = 1.0
        # α  = 1 / (1 + sqrt(g_dot_p))
        # println("Original damped α ", α)

        # Candidate updates to x and f.
        @. candidate_x = x - α * p
        candidate_f = fgh!(0.0, g, H, candidate_x)
        candidate_norm_g = norm(g)

        # Backtracking line search until Armijo condition is satisfied:
        while candidate_norm_g ≥ norm_g
            has_converged(x, candidate_x, f, candidate_f, g) && return candidate_x
            α < armijo_α_min && error("Step size limit reached. Consider reducing armijo_α_min=$armijo_α_min or a tolerance parameter.")

            α *= armijo_backoff
            @. candidate_x = x - α * p
            candidate_f = fgh!(0.0, g, H, candidate_x)
            candidate_norm_g = norm(g)
        end

        if show_trace
            println("Iter $k: α=$α, |g|=$(norm(g)), f(x)=$candidate_f, x=$candidate_x")
        end
        has_converged(x, candidate_x, f, candidate_f, g) && return candidate_x

        # Accept candidate updates. Note that g and H will also be reused.
        x .= candidate_x
        f = candidate_f
    end

    error("Failed to converge in maxiters=$maxiters iterations.")
end


"""
    newton_selfconcordant_lm(fgh!, x0; f_reltol=NaN, x_reltol=NaN, g_abstol=NaN,
                             maxiters=50, max_lm_iters=30, lm_growth=2.0,
                             show_trace=false)

Convex solver for objectives with logarithmic singularities using:

1) Self-concordant damped Newton step with α = 1 / (1 + λ), λ² = g' * H⁻¹ * g.
2) Fallback Levenberg–Marquardt regularization (H + λI) if the SC step yields an
   infeasible point or no monotone progress in f or ‖g‖.

`fgh!(t, g, H, x)` should return f(x) (finite in-feasible; +Inf otherwise) and
write the gradient `g` and Hessian `H` at `x` when feasible. `H` must be PD on
the feasible set.

Convergence is declared if any of these hold:
- relative change in x ≤ x_reltol
- relative change in f ≤ f_reltol
- ‖g‖ ≤ g_abstol
"""
function newton_selfconcordant_lm(fgh!, x0;
    f_reltol=NaN, x_reltol=NaN, g_abstol=NaN,
    maxiters=50, max_lm_iters=30, lm_growth=2.0, show_trace=false)

    # Working copies and buffers
    x = copy(x0)
    n = length(x)
    T = eltype(x0)

    g       = zero(x0)
    H       = zeros(T, n, n)
    g_tmp   = similar(g)
    H_tmp   = similar(H)
    H_reg   = similar(H)
    p       = zero(x0)
    cand_x  = zero(x0)

    # Evaluate at the initial point
    f = fgh!(zero(T), g, H, x)
    isfinite(f) || error("Initial point x0 is not feasible (f=Inf).")
    (!isnan(g_abstol) && norm(g) < g_abstol) && return x

    has_converged(x, cand_x, f, cand_f, gvec) =
        (!isnan(x_reltol) && isapprox(x, cand_x; rtol=x_reltol)) ||
        (!isnan(f_reltol) && isapprox(f, cand_f; rtol=f_reltol)) ||
        (!isnan(g_abstol) && norm(gvec) < g_abstol)

    for k in 1:maxiters
        # --------- Self-concordant damped Newton attempt ----------
        F = cholesky(Hermitian(H))             # non-mutating: preserves H
        ldiv!(p, F, g)                         # p = H \ g
        gp = max(dot(g, p), zero(T))           # clamp for numerical safety
        λ  = sqrt(gp)
        α  = one(T) / (one(T) + λ)
        @show α

        @. cand_x = x - α * p
        cand_f = fgh!(zero(T), g_tmp, H_tmp, cand_x)

        # Progress: feasible and (f decreases OR ‖g‖ decreases)
        prog_sc = isfinite(cand_f) && ((isfinite(f) && cand_f <= f) || (norm(g_tmp) < norm(g)))

        if isfinite(cand_f) && (prog_sc || has_converged(x, cand_x, f, cand_f, g_tmp))
            x .= cand_x; f = cand_f
            copyto!(g, g_tmp); copyto!(H, H_tmp)
            if show_trace
                println("Iter $k (SC): α=$(α), λ=$(λ), |g|=$(norm(g)), f=$f")
            end
            continue
        end

        # --------- LM fallback: (H + λI) p = g, x+ = x - p ----------
        λ_lm = tr(H) / T(n)   # scale-aware initialization
        accepted = false
        for j in 1:max_lm_iters
            # H_reg = H + λ_lm I
            @. H_reg = H
            @inbounds for i in 1:n
                H_reg[i,i] += λ_lm
            end

            Freg = cholesky(Hermitian(H_reg))
            ldiv!(p, Freg, g)            # p = (H + λI) \ g
            @. cand_x = x - p
            cand_f = fgh!(zero(T), g_tmp, H_tmp, cand_x)

            # prog_lm = isfinite(cand_f) && ((isfinite(f) && cand_f <= f) || (norm(g_tmp) < norm(g)))
            prog_lm = isfinite(cand_f) && (norm(g_tmp) < norm(g))

            if isfinite(cand_f) && (prog_lm || has_converged(x, cand_x, f, cand_f, g_tmp))
                x .= cand_x; f = cand_f
                copyto!(g, g_tmp); copyto!(H, H_tmp)
                accepted = true
                if show_trace
                    println("Iter $k (LM λ=$(λ_lm)): |g|=$(norm(g)), f=$f")
                end
                break
            end

            λ_lm *= lm_growth
        end

        accepted || error("LM fallback failed to find a feasible improving step in max_lm_iters=$max_lm_iters.")
        (!isnan(g_abstol) && norm(g) < g_abstol) && return x
    end

    error("Failed to converge in maxiters=$maxiters iterations.")
end
