function free_energy(sbs::SchwingerBosonSystem)
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    (; T, L, S) = sbs
    Nu = L^2
    f = 0.0

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        try
            E = bogoliubov!(V, D)
            for i in 1:6
                f += E[i] / 2
                (T > 1e-8) && (f += real(T * log1p(-exp(-E[i]/T))))
            end
        catch e
            return Inf
        end
    end

    f /= Nu

    (; mean_fields, J, Δ) = sbs
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