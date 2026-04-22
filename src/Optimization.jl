function optimize_μ0!(sbs::SchwingerBosonSystem, μ0, aux::OptimAux; 
    options = Optim.Options(show_trace=false, iterations=1000), tol=1e-12, max_iters=500)
    f!(y) = f_y!(sbs, aux, y; tol, max_iters)
    ret = optimize(f!, μ0, Optim.NelderMead(), options)
    y_minimizer = ret.minimizer
    set_μ0!(sbs, y_minimizer)
    c_shift, conden_index = search_for_condensation_shift!(sbs, y_minimizer; tol, max_iters) 
    aux.c_shift = c_shift
    aux.conden_index = conden_index
    μ0_shifted = copy(y_minimizer) .- c_shift
    set_μ0!(sbs, μ0_shifted)
    return ret
end

# function optimize_mean_fields!(sbs::SchwingerBosonSystem, ϕ0; 
#     options_inner = Optim.Options(show_trace=false, iterations=1000),
#     options_outer = Optim.Options(show_trace=false, iterations=1000),
#     algorithm_outer = Optim.LBFGS(),
#     tol=1e-8, max_iters=100)

#     set_ϕ!(sbs, ϕ0)
#     fg!(f, g, x) = fg_ϕ!(sbs, f, g, x; options=options_inner, tol, max_iters)
#     ret = optimize(Optim.only_fg!(fg!), ϕ0, algorithm_outer, options_outer)
#     set_ϕ!(sbs, ret.minimizer)

#     return ret
# end