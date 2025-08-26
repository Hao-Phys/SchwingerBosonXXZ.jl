function free_energy_boson(sbs::SchwingerBosonSystem)
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    (; T, L) = sbs
    Nu = L^2
    f = 0.0

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        try
            E = bogoliubov!(V, D)
            for n in 1:6
                f += E[n] / 2
                (T > 1e-8) && (f += real(T * log1p(-exp(-E[n]/T))))
            end
        catch e
            return Inf
        end
    end

    f /= Nu

    return f
end

function free_energy_mean_field(sbs::SchwingerBosonSystem)
    f = free_energy_boson(sbs)

    (; mean_fields, J, Δ, S) = sbs
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2

    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]
    μs = mean_fields[13:15]

    # Contribution from the "energy" term
    for α in 1:3
        f += -3 * (-J₊ * abs2(As[α]) + J₊ * abs2(Bs[α]) + J₋ * abs2(Cs[α]) - J₋ *abs2(Ds[α])) + real(μs[α]) + 2S*(real(μs[α]))
    end

    return f
end

function free_energy_variational(sbs::SchwingerBosonSystem)
    f = free_energy_boson(sbs)

    g = zeros(24)
    # Buffers
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    (; L, J, Δ, mean_fields, S) = sbs
    Nu = L^2
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2
    inv_J₊ = J₊ == 0 ? 0 : 1 / J₊
    inv_J₋ = J₋ == 0 ? 0 : 1 / J₋

    ϕ = zeros(24)
    for i in 1:12
        ϕ[i], ϕ[i+12] = reim(sbs.mean_fields[i])
    end

    # Buffers to hold the derivatives of the dynamical matrix
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 24)

    # The "interaction" strength
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2
    inv_J₊ = J₊ == 0 ? 0 : 1 / J₊
    inv_J₋ = J₋ == 0 ? 0 : 1 / J₋
    fα = zeros(24)
    inv_fα = zeros(24)
    for α in (1, 2, 3, 13, 14, 15)
        fα[α] = -J₊
        inv_fα[α] = -inv_J₊
    end
    for α in (4, 5, 6, 16, 17, 18)
        fα[α] = J₊
        inv_fα[α] = inv_J₊
    end
    for α in (7, 8, 9, 19, 20, 21)
        fα[α] = J₋
        inv_fα[α] = inv_J₋
    end
    for α in (10, 11, 12, 22, 23, 24)
        fα[α] = -J₋
        inv_fα[α] = -inv_J₋
    end


    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)

        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α], ∂ID∂ϕs[:, :, α+12], tmp, sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α+3], ∂ID∂ϕs[:, :, α+15], tmp, sbs, q, α)
            ∂ID∂C!(∂ID∂ϕs[:, :, α+6], ∂ID∂ϕs[:, :, α+18], tmp, sbs, q, α)
            ∂ID∂D!(∂ID∂ϕs[:, :, α+9], ∂ID∂ϕs[:, :, α+21], tmp, sbs, q, α)
        end

        @views for α in 1:24
            # Gradient from entropy term
            g[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
        end
    end

    # Now we add the contribution from the "correction" term L"⟨H - H_{MF}⟩_{MF}"
    for α in 1:24
        f += inv_fα[α] * g[α]^2 / 12 - g[α] * ϕ[α]
    end

    μs = sbs.mean_fields[13:15]
    for α in 1:3
        f += real(μs[α]) * (2S+1)
    end

    return f
end