function self_consistent_mean_fields!(f, x, sbs::SchwingerBosonSystem)
    set_mean_fields_scf!(sbs, x)
    μ0 = real(sbs.mean_fields[13:15])
    optimize_μ!(sbs, μ0)
    expectations = expectation_values(sbs)
    f[1:12] = expectations[1:12] - x[1:12]
end

function solve_self_consistent_mean_fields!(sbs::SchwingerBosonSystem, x0::Vector{ComplexF64}; opts...)
    sce_eqn! = (f, x) -> self_consistent_mean_fields!(f, x, sbs)
    ret = nlsolve(sce_eqn!, x0; opts...)
    !converged(ret) && @warn "Self-consitent equations converged to a solution with residual $(ret.residual)"
    best_mean_fields = [ret.zero; sbs.mean_fields[13:15]]
    set_mean_fields!(sbs, best_mean_fields)
end