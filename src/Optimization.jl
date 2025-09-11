function optimize_μ0!(sbs::SchwingerBosonSystem, μ0)
    fg!(f, g, x) = fg_μ0!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), μ0, LBFGS())
    set_μ0!(sbs, ret.minimizer)
end

function optimize_mean_fields!(sbs::SchwingerBosonSystem, ϕ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_ϕ!(sbs, ϕ0)
    fg!(f, g, x) = fg_ϕ!(sbs, f, g, x)
    ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm, options)
    set_ϕ!(sbs, ret.minimizer)
end

function optimize_mean_fields_gradient_free!(sbs::SchwingerBosonSystem, ϕ0; algorithm=Optim.LBFGS(), options = Optim.Options(show_trace=false, iterations=1000))
    set_ϕ!(sbs, ϕ0)
    f!(x) = fϕ!(sbs, x)
    ret = optimize(f!, ϕ0, algorithm, options)
    set_ϕ!(sbs, ret.minimizer)
end