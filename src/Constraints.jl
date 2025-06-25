function constraints!(sbs::SchwingerBosonSystem, x, c)
    c .= 0.0

    set_mean_fields!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    ∂D∂X = zeros(ComplexF64, 12, 12)

    (; L, S) = sbs
    Nu = L^2

    for α in 1:3
        ∂D∂λ!(∂D∂X, α)
        for i in 1:L, j in 1:L
            q = Vec3([(i-1)/L, (j-1)/L, 0.0])
            single_particle_density_matrix!(P, D, V, tmp, sbs, q)
            mul!(tmp, P, Ĩ)
            copyto!(P, tmp)
            c[α] += real(tr(P * ∂D∂X)) / Nu
        end
        c[α] -= (2S+1)
    end

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        c[4] += real(logdet(D)) / Nu
    end
end