function optimize_μ!(sbs::SchwingerBosonSystem, μ0)
    fg!(f, g, x) = fg_μ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), μ0, LBFGS())
    set_μ!(sbs, ret.minimizer)
end

function optimize_mean_fields!(sbs::SchwingerBosonSystem, x0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_x!(sbs, x0)
    ϕ0 = x0[1:24]
    fg!(f, g, x) = fg_ϕ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm, options)
    set_ϕ!(sbs, ret.minimizer)
end