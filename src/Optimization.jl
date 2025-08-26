function optimize_μ!(sbs::SchwingerBosonSystem, μ0)
    fg!(f, g, x) = fg_μ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), μ0, LBFGS())
    set_μ!(sbs, ret.minimizer)
end

function optimize_mean_fields!(sbs::SchwingerBosonSystem, ϕ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_ϕ!(sbs, ϕ0)
    fg!(f, g, x) = fg_ϕ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm, options)
    set_ϕ!(sbs, ret.minimizer)
end