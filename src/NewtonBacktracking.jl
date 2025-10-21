function newton_with_backtracking(fgh!, x0; f_reltol=NaN, x_reltol=NaN, g_abstol=NaN, maxiters=40, armijo_c=1e-4, armijo_backoff=0.5, armijo_α_min=1e-12, show_trace=false)
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

        φ0  = dot(g, g)                    # |g₀|² at current x
        dφ0 = -2*φ0                        # slope of |g₀|² along -p at α=0

        # Start with full Newton step
        α = 1.0

        # Candidate updates to x and f.
        @. candidate_x = x - α * p
        candidate_f = fgh!(0.0, g, H, candidate_x)
        φ = dot(g, g)

        # Backtracking until both feasibility and Armijo on |g|² are satisfied.
        # Cannot apply usual Armijo on f because of numerical precision
        # limitations.
        σ = eps(eltype(x0))*(φ0 + armijo_c*α*abs(dφ0)) # noise floor
        while !isfinite(candidate_f) || φ > φ0 + armijo_c*α*dφ0 + σ
            has_converged(x, candidate_x, f, candidate_f, g) && return candidate_x
            α < armijo_α_min && error("Minimum step size reached (consider reducing armijo_α_min=$armijo_α_min)")

            α *= armijo_backoff
            @. candidate_x = x - α * p
            candidate_f = fgh!(0.0, g, H, candidate_x)
            φ = dot(g, g)
        end

        if show_trace
            println("Iter $k: α=$α, |g|=$(sqrt(φ)), f(x)=$candidate_f, x=$candidate_x")
        end
        has_converged(x, candidate_x, f, candidate_f, g) && return candidate_x

        # Accept candidate updates. Reuse latest g and H already computed at candidate_x.
        x .= candidate_x
        f = candidate_f
    end

    error("Failed to converge in maxiters=$maxiters iterations.")
end
