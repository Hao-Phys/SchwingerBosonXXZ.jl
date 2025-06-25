function single_particle_density_matrix!(P::Matrix{ComplexF64}, D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, tmp::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)
    P .= 0.0
    (; T) = sbs
    dynamical_matrix!(D, sbs, q_reshaped)
    try
        E = bogoliubov!(V, D)
        for i in eachindex(E)
            # Include the 1/2 factor here
            P[i, i] += coth(E[i] / (2T)) / 4
        end
        # P = V * diag( coth(E / (2T)) ) * inv(V) / 2
        mul!(tmp, V, P)
        mul!(P, tmp, inv(V))
    catch e
        P .= 0.0
    end
end


function grad_free_energy!(sbs::SchwingerBosonSystem, x, g)
    g .= 0.0
    set_x!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    (; L, J, Δ, mean_fields) = sbs
    Nu = L^2
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2

    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]

    ∂D∂X_re = zeros(ComplexF64, 12, 12)
    ∂D∂X_im = zeros(ComplexF64, 12, 12)

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)
        mul!(tmp, P, Ĩ)
        copyto!(P, tmp)

        for α in 1:3
            ∂D∂A!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            g[α] += real(tr(P * ∂D∂X_re))
            g[α+12] += real(tr(P * ∂D∂X_im))
            ∂D∂B!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            g[α+3] += real(tr(P * ∂D∂X_re))
            g[α+15] += real(tr(P * ∂D∂X_im))
            ∂D∂C!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            g[α+6] += real(tr(P * ∂D∂X_re))
            g[α+18] += real(tr(P * ∂D∂X_im))
            ∂D∂D!(∂D∂X_re, ∂D∂X_im, sbs, q, α)
            g[α+9] += real(tr(P * ∂D∂X_re))
            g[α+21] += real(tr(P * ∂D∂X_im))
            ∂D∂λ!(∂D∂X_re, α)
            g[α+24] += real(tr(P * ∂D∂X_re))
        end
    end

    @. g /= Nu
    
    for α in 1:3
        g[α]   += 3J₊ * real(As[α])
        g[α+3] -= 3J₊ * real(Bs[α])
        g[α+6] -= 3J₋ * real(Cs[α])
        g[α+9] += 3J₋ * real(Ds[α])
        g[α+12] += 3J₊ * imag(As[α])
        g[α+15] -= 3J₊ * imag(Bs[α])
        g[α+18] -= 3J₋ * imag(Cs[α])
        g[α+21] += 3J₋ * imag(Ds[α])
        g[α+24] -= 1
    end
end