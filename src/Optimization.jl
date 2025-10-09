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