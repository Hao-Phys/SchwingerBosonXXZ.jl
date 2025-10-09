function optimize_μ0!(sbs::SchwingerBosonSystem, μ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=100))
    fg!(f, g, x) = fg_μ0!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), μ0, algorithm, options)
    @assert Optim.converged(ret) "Particle number not equal to 2S. Optimization failed."
    set_μ0!(sbs, ret.minimizer)
end

function optimize_mean_fields!(sbs::SchwingerBosonSystem, ϕ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_ϕ!(sbs, ϕ0)
    fg!(f, g, x) = fg_ϕ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm, options)
    set_ϕ!(sbs, ret.minimizer)
    return ret
end

function optimize_μ0_newton!(sbs::SchwingerBosonSystem, μ0; f_reltol=NaN, x_reltol=NaN, g_abstol=NaN, maxiters=20, armijo_c=1e-4, armijo_backoff=0.5, armijo_α_min=1e-4, show_trace=false)
    fgh!(f, g, h, x) = fgh_μ0!(sbs, f, g, h, x)
    ret = newton_with_backtracking(fgh!, μ0; f_reltol=f_reltol, x_reltol=x_reltol, g_abstol=g_abstol, maxiters=maxiters, armijo_c=armijo_c, armijo_backoff=armijo_backoff, armijo_α_min=armijo_α_min, show_trace=show_trace)
    set_μ0!(sbs, ret)
end