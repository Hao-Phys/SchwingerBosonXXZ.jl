function free_energy!(sbs::SchwingerBosonSystem, x)
    set_mean_fields!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    (; T, L) = sbs
    Nu = L^2
    f = 0.0

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)
        for i in 1:6
            f += real(2T * log(sinh(E[i]/(2T))))
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
    λs = mean_fields[13:15]

    # Contribution from the "energy" term
    for α in 1:3
        f += -1.5 * (-J₊ * abs2(As[α]) + J₊ * abs2(Bs[α]) + J₋ * abs2(Cs[α]) - J₋ *abs2(Ds[α])) - real(λs[α])
    end

    return f
end

function free_energy(sbs::SchwingerBosonSystem)
    x = zeros(27)
    for i in 1:12
        x[i], x[i+12] = reim(sbs.mean_fields[i])
    end
    for i in 1:3
        x[24+i] = real(sbs.mean_fields[12+i])
    end
    return free_energy!(sbs, x)
end

function free_energy_aux!(sbs::SchwingerBosonSystem, x)
    set_mean_fields!(sbs, x)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    (; T, L) = sbs
    Nu = L^2
    f = 0.0

    # Contribution from the "entropy" term
    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)
        Λ = Diagonal(sinh.(E/(2T)))
        f += T * tr(log(Λ * Ĩ))
    end

    f /= Nu

    (; mean_fields, J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2

    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]
    λs = mean_fields[13:15]

    # Contribution from the "energy" term
    for α in 1:3
        f += -1.5 * (-J₊ * abs2(As[α]) + J₊ * abs2(Bs[α]) + J₋ * abs2(Cs[α]) - J₋ *abs2(Ds[α])) - real(λs[α])
    end

    return f
end

function free_energy_aux(sbs::SchwingerBosonSystem)
    x = zeros(27)
    for i in 1:12
        x[i], x[i+12] = reim(sbs.mean_fields[i])
    end
    for i in 1:3
        x[24+i] = real(sbs.mean_fields[12+i])
    end
    return free_energy_aux!(sbs, x)
end