function self_consistent_mean_fields!(f, x, sbs::SchwingerBosonSystem)
    (; L, T) = sbs
    set_mean_fields_scf!(sbs, x)
    # Buffers
    D = zeros(ComplexF64, 12, 12)

    # Maximize the mean-field free energy to find the optimal chemical potential
    # But we need a μ0 such that the dynamical matrix is positive definite
    eigvals_min = Float64[]
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        eigval_min = eigmin(D)
        push!(eigvals_min, eigval_min)
    end

    τ = max(0.0, -minimum(eigvals_min))
    μ0s = sbs.mean_fields[13:15] .- (τ + T)

    optimize_μ0!(sbs, μ0s)
    expectations = expectation_values(sbs)
    f[1:12] = expectations[1:12] - x[1:12]
end

function solve_self_consistent_mean_fields!(sbs::SchwingerBosonSystem, x0::Vector{ComplexF64}; opts...)
    sce_eqn! = (f, x) -> self_consistent_mean_fields!(f, x, sbs)
    ret = nlsolve(sce_eqn!, x0; opts...)
    !converged(ret) && @warn "Self-consitent equations converged to a solution with residual $(ret.residual_norm)"
    best_mean_fields = [ret.zero; sbs.mean_fields[13:15]]
    set_mean_fields!(sbs, best_mean_fields)
end