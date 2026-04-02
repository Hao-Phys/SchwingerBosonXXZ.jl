# Below, we calculate the gradient of Ĩ * D with respect to the mean fields. 
#
# Note that ∂D/∂[Re(χ)] = ∂D/∂χ + ∂D/∂conj(χ), and ∂D/∂[Im(χ)] = 1im * (∂D/∂χ - ∂D/∂conj(χ)).
# `Ĩ * D` gives a minus sign for D21 and D22, and keeps D11 and D12 unchanged.

function ∂ID∂A!(out_re, out_im, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    out_re .= 0.0
    out_im .= 0.0

    (; J, Δ, α_dcoups) = sbs
    J₊ = J * (Δ + 1) / 2

    ∂ID12_re = view(out_re, 1:6, 7:12)
    ∂ID21_re = view(out_re, 7:12, 1:6)
    ∂ID12_im = view(out_im, 1:6, 7:12)
    ∂ID21_im = view(out_im, 7:12, 1:6)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        sign = σ == 1 ? 1 : -1
        i = (α-1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i+increment, 6)

        # P_link = -0.5 * (J₊*(1+α_dcoups[1])*σ*As[link] + ...)
        ∂ID12_re[i, j] -= 0.5 * J₊ * sign * phase * (1+α_dcoups[1])
        ∂ID12_re[j, i] -= 0.5 * J₊ * sign * conj(phase) * (1+α_dcoups[1])
        ∂ID21_re[i, j] += 0.5 * J₊ * sign * phase * (1+α_dcoups[1])
        ∂ID21_re[j, i] += 0.5 * J₊ * sign * conj(phase) * (1+α_dcoups[1])

        ∂ID12_im[i, j] -= 0.5 * J₊ * sign * ( 1im) * phase * (1+α_dcoups[1])
        ∂ID12_im[j, i] -= 0.5 * J₊ * sign * ( 1im) * conj(phase) * (1+α_dcoups[1])
        ∂ID21_im[i, j] += 0.5 * J₊ * sign * (-1im) * phase * (1+α_dcoups[1])
        ∂ID21_im[j, i] += 0.5 * J₊ * sign * (-1im) * conj(phase) * (1+α_dcoups[1])
    end
end

function ∂ID∂B!(out_re, out_im, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    out_re .= 0.0
    out_im .= 0.0

    (; J, Δ, α_dcoups) = sbs
    J₊ = J * (Δ + 1) / 2

    ∂ID11_re = view(out_re, 1:6, 1:6)
    ∂ID22_re = view(out_re, 7:12, 7:12)
    ∂ID11_im = view(out_im, 1:6, 1:6)
    ∂ID22_im = view(out_im, 7:12, 7:12)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        i = (α-1) * 2 + σ
        j = mod1(i+2, 6)

        # Q_link = 0.5 * (J₊*(1-α_dcoups[1])*Bs[link] + ...)
        ∂ID11_re[i, j] += 0.5 * J₊ * phase * (1-α_dcoups[1])
        ∂ID11_re[j, i] += 0.5 * J₊ * conj(phase) * (1-α_dcoups[1])
        ∂ID22_re[i, j] -= 0.5 * J₊ * phase * (1-α_dcoups[1])
        ∂ID22_re[j, i] -= 0.5 * J₊ * conj(phase) * (1-α_dcoups[1])

        ∂ID11_im[i, j] += 0.5 * J₊ * ( 1im) * phase * (1-α_dcoups[1])
        ∂ID11_im[j, i] += 0.5 * J₊ * (-1im) * conj(phase) * (1-α_dcoups[1])
        ∂ID22_im[i, j] -= 0.5 * J₊ * (-1im) * phase * (1-α_dcoups[1])
        ∂ID22_im[j, i] -= 0.5 * J₊ * ( 1im) * conj(phase) * (1-α_dcoups[1])
    end
end

function ∂ID∂C!(out_re, out_im, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    out_re .= 0.0
    out_im .= 0.0

    (; J, Δ, α_dcoups) = sbs
    J₋ = J * (Δ - 1) / 2

    ∂ID11_re = view(out_re, 1:6, 1:6)
    ∂ID22_re = view(out_re, 7:12, 7:12)
    ∂ID11_im = view(out_im, 1:6, 1:6)
    ∂ID22_im = view(out_im, 7:12, 7:12)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        sign = σ == 1 ? 1 : -1
        i = (α-1) * 2 + σ
        j = mod1(i+2, 6)

        # Q_link = 0.5 * (... + J₋*(1-α_dcoups[2])*Cs[link])
        ∂ID11_re[i, j] += 0.5 * J₋ * sign * phase * (1-α_dcoups[2])
        ∂ID11_re[j, i] += 0.5 * J₋ * sign * conj(phase) * (1-α_dcoups[2])
        ∂ID22_re[i, j] -= 0.5 * J₋ * sign * phase * (1-α_dcoups[2])
        ∂ID22_re[j, i] -= 0.5 * J₋ * sign * conj(phase) * (1-α_dcoups[2])

        ∂ID11_im[i, j] += 0.5 * J₋ * sign * ( 1im) * phase * (1-α_dcoups[2])
        ∂ID11_im[j, i] += 0.5 * J₋ * sign * (-1im) * conj(phase) * (1-α_dcoups[2])
        ∂ID22_im[i, j] -= 0.5 * J₋ * sign * (-1im) * phase * (1-α_dcoups[2])
        ∂ID22_im[j, i] -= 0.5 * J₋ * sign * ( 1im) * conj(phase) * (1-α_dcoups[2])
    end

end

function ∂ID∂D!(out_re, out_im, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    out_re .= 0.0
    out_im .= 0.0

    (; J, Δ, α_dcoups) = sbs
    J₋ = J * (Δ - 1) / 2

    ∂ID21_re = view(out_re, 7:12, 1:6)
    ∂ID12_re = view(out_re, 1:6, 7:12)
    ∂ID21_im = view(out_im, 7:12, 1:6)
    ∂ID12_im = view(out_im, 1:6, 7:12)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        i = (α-1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i+increment, 6)

        # P_link = -0.5 * (... + J₋*(1+α_dcoups[2])*Ds[link])
        ∂ID12_re[i, j] -= 0.5 * J₋ * phase * (1+α_dcoups[2])
        ∂ID12_re[j, i] -= 0.5 * J₋ * conj(phase) * (1+α_dcoups[2])
        ∂ID21_re[i, j] += 0.5 * J₋ * phase * (1+α_dcoups[2])
        ∂ID21_re[j, i] += 0.5 * J₋ * conj(phase) * (1+α_dcoups[2])

        ∂ID12_im[i, j] -= 0.5 * J₋ * ( 1im) * phase * (1+α_dcoups[2])
        ∂ID12_im[j, i] -= 0.5 * J₋ * ( 1im) * conj(phase) * (1+α_dcoups[2])
        ∂ID21_im[i, j] += 0.5 * J₋ * (-1im) * phase * (1+α_dcoups[2])
        ∂ID21_im[j, i] += 0.5 * J₋ * (-1im) * conj(phase) * (1+α_dcoups[2])
    end

end

# The derivative with respect to the chemical potentials:
function ∂ID∂μ0!(out, α::Int)
    out .= 0.0
    for σ in 1:2
        i = (α-1) * 2 + σ
        out[i, i] -= 1.0
        out[i+6, i+6] += 1.0
    end
end