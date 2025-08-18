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

    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]
    μs = mean_fields[13:15]

    ∂D∂X_re = zeros(ComplexF64, 12, 12)
    ∂D∂X_im = zeros(ComplexF64, 12, 12)

    Πα_re = zeros(4, 3)
    Πα_im = zeros(4, 3)

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)

        for α in 1:3
            ∂ID∂A!(∂D∂X_re, ∂D∂X_im, tmp, sbs, q, α)
            # The factor of 1/2 is from the Writing derivative of complex number
            Πα_re[1, α] += -inv_J₊ * real(tr(P * ∂D∂X_re)) / (2Nu)
            Πα_im[1, α] += -inv_J₊ * real(tr(P * ∂D∂X_im)) / (2Nu)
            ∂ID∂B!(∂D∂X_re, ∂D∂X_im, tmp, sbs, q, α)
            Πα_re[2, α] +=  inv_J₊ * real(tr(P * ∂D∂X_re)) / (2Nu)
            Πα_im[2, α] +=  inv_J₊ * real(tr(P * ∂D∂X_im)) / (2Nu)
            ∂ID∂C!(∂D∂X_re, ∂D∂X_im, tmp, sbs, q, α)
            Πα_re[3, α] +=  inv_J₋ * real(tr(P * ∂D∂X_re)) / (2Nu)
            Πα_im[3, α] +=  inv_J₋ * real(tr(P * ∂D∂X_im)) / (2Nu)
            ∂ID∂D!(∂D∂X_re, ∂D∂X_im, tmp, sbs, q, α)
            Πα_re[4, α] += -inv_J₋ * real(tr(P * ∂D∂X_re)) / (2Nu)
            Πα_im[4, α] += -inv_J₋ * real(tr(P * ∂D∂X_im)) / (2Nu)
        end
    end

    for α in 1:3
        # The additional chemical potential term
        f += real(μs[α]) + 2S*(real(μs[α]))
        # The "corrections" minus the "constant" contributions to avoid add and substract
        f += -J₊ * ( 1/3 * (Πα_re[1, α]^2 + Πα_im[1, α]^2) - 2(real(As[α]) * Πα_re[1, α] + imag(As[α]) * Πα_im[1, α]) )
        f +=  J₊ * ( 1/3 * (Πα_re[2, α]^2 + Πα_im[2, α]^2) - 2(real(Bs[α]) * Πα_re[2, α] + imag(Bs[α]) * Πα_im[2, α]) )
        f +=  J₋ * ( 1/3 * (Πα_re[3, α]^2 + Πα_im[3, α]^2) - 2(real(Cs[α]) * Πα_re[3, α] + imag(Cs[α]) * Πα_im[3, α]) )
        f += -J₋ * ( 1/3 * (Πα_re[4, α]^2 + Πα_im[4, α]^2) - 2(real(Ds[α]) * Πα_re[4, α] + imag(Ds[α]) * Πα_im[4, α]) )
    end

    return f
end