function newton_with_backtracking(fgh!, x0; f_reltol=NaN, x_reltol=NaN, g_abstol=NaN, decrement_tol=NaN,
    maxiters=40, armijo_c=1e-2, armijo_backoff=0.5, armijo_α_min=1e-12, show_trace=false)

    # Preallocate buffers
    T = eltype(x0)
    x = copy(x0)
    candidate_x = fill(NaN, length(x))
    g = zero(x)
    H = zeros(T, length(x), length(x))

    # Evaluate objective function f, gradient, and Hessian.
    f = fgh!(0.0, g, H, x)
    isfinite(f) || error("Initial guess is not feasible")

    function has_converged(x, candidate_x, f, candidate_f, g, decr)
        isfinite(x_reltol) && isapprox(x, candidate_x; rtol=x_reltol) && return true
        isfinite(f_reltol) && isapprox(f, candidate_f; rtol=f_reltol) && return true
        isfinite(g_abstol) && maximum(abs, g) ≤ g_abstol && return true
        isfinite(decrement_tol) && decr < decrement_tol && return true

        # TODO: Noise condition on Newton decrement:
        # noise ≈ (1/2)*​norm(p)*norm(r) + eps(T)*norm(g)*norm(p)
        # where r = g - H*p measures noise in linear solve.

        return false
    end

    # Newton direction p = H \ g
    p = cholesky!(Hermitian(H)) \ g
    if show_trace 
        println("Iter 0: decr=$(g'*p/2), |g|=$(norm(g)), f(x)=$f, x=$x")
    end
    has_converged(x, candidate_x, f, NaN, g, g'*p/2) && return x

    for k in 1:maxiters
        φ0  = dot(g, g)                         # |g|² at current x
        dφ0 = -2*φ0                             # slope of |g|² along -p at α=0
        σ = eps(T) * (φ0 + abs(armijo_c*dφ0)) # scale-aware noise floor

        # Start with full Newton step
        α = 1.0

        # Candidate updates to x and f.
        @. candidate_x = x - α * p
        candidate_f = fgh!(0.0, g, H, candidate_x)
        φ = dot(g, g)

        # Backtracking until both feasibility and Armijo on |g|² are satisfied.
        # Cannot apply usual Armijo on f because of numerical precision
        # limitations.
        while !isfinite(candidate_f) || φ > φ0 + armijo_c*α*dφ0 + σ
            has_converged(x, candidate_x, f, candidate_f, g, NaN) && return candidate_x
            α < armijo_α_min && error("Minimum step size reached (consider reducing armijo_α_min=$armijo_α_min)")

            α *= armijo_backoff
            @. candidate_x = x - α * p
            candidate_f = fgh!(0.0, g, H, candidate_x)
            φ = dot(g, g)
        end

        ldiv!(p, cholesky!(Hermitian(H)), g)
        if show_trace
            println("Iter $k: α=$α, decr=$(g'*p/2), |g|=$(sqrt(φ)), f(x)=$candidate_f, x=$candidate_x")
        end
        has_converged(x, candidate_x, f, candidate_f, g, g'*p/2) && return candidate_x

        # Accept candidate_x. Calculate p and decr to be carried into the next
        # iteration
        f = candidate_f
        x .= candidate_x
    end

    error("Failed to converge in maxiters=$maxiters iterations.")
end
