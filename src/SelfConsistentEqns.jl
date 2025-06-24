function self_consistent_mean_fields!(f, x, sbs::SchwingerBosonSystem)
    set_mean_fields!(sbs, x)
    expectations = expectation_values(sbs)
    f[1:12] = expectations[1:12] - x[1:12]
    f[13:15] = expectations[13:15] .- 2*sbs.S
end

function solve_self_consistent_mean_fields!(sbs::SchwingerBosonSystem, x0::Vector{ComplexF64}; opts...)
    sce_eqn! = (f, x) -> self_consistent_mean_fields!(f, x, sbs)
    ret = nlsolve(sce_eqn!, x0; opts...)
    !converged(ret) && @warn "Self-consitent equations converged to a solution with residual $(ret.residual)"
    set_mean_fields!(sbs, ret.zero)
end