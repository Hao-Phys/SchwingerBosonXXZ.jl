function optimize_μ0!(sbs::SchwingerBosonSystem, μ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=100))
    fg!(f, g, x) = fg_μ0!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), μ0, algorithm, options)
    set_μ0!(sbs, ret.minimizer)
    return ret.g_residual
end

function optimize_mean_fields!(sbs::SchwingerBosonSystem, ϕ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_ϕ!(sbs, ϕ0)
    fg!(f, g, x) = fg_ϕ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm, options)
    set_ϕ!(sbs, ret.minimizer)
    return ret
end

function optimize_μ0_newton!(sbs::SchwingerBosonSystem, μ0; kwargs...)
    fgh!(f, g, h, x) = fgh_μ0!(sbs, f, g, h, x)
    ret = newton_with_backtracking(fgh!, μ0; kwargs...)
    set_μ0!(sbs, ret)
end
