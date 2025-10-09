@inline log1mexp_modified(x) = (x < log(2)) ? log(-expm1(-x)) : log1p(-exp(-x))

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
                (T > 1e-8) && (f += real(T * log1mexp_modified(E[n]/T)))
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
    μ0s = mean_fields[13:15]

    # Contribution from the "energy" term
    for α in 1:3
        f += -3 * (-J₊ * abs2(As[α]) + J₊ * abs2(Bs[α]) + J₋ * abs2(Cs[α]) - J₋ *abs2(Ds[α])) + real(μ0s[α]) + 2S*(real(μ0s[α]))
    end

    return f
end