# Below, we calculate the gradient of Ĩ * D with respect to the mean fields. 

# Note that ∂D/∂[Re(χ)] = ∂D/∂χ + ∂D/∂conj(χ), and ∂D/∂[Im(χ)] = 1im * (∂D/∂χ - ∂D/∂conj(χ)).
function ∂ID∂A!(∂D∂A_re, ∂D∂A_im, tmp, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂A_re .= 0.0
    ∂D∂A_im .= 0.0

    (; J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2

    ∂D12_re = view(∂D∂A_re, 1:6, 7:12)
    ∂D21_re = view(∂D∂A_re, 7:12, 1:6)
    ∂D12_im = view(∂D∂A_im, 1:6, 7:12)
    ∂D21_im = view(∂D∂A_im, 7:12, 1:6)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        sign = σ == 1 ? 1 : -1
        i = (α-1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i+increment, 6)

        ∂D12_re[i, j] += -0.5 * J₊ * sign * phase
        ∂D12_re[j, i] += -0.5 * J₊ * sign * conj(phase)
        ∂D21_re[i, j] += -0.5 * J₊ * sign * phase
        ∂D21_re[j, i] += -0.5 * J₊ * sign * conj(phase)

        ∂D21_im[i, j] += -0.5 * J₊ * sign * ( 1im) * phase
        ∂D21_im[j, i] += -0.5 * J₊ * sign * ( 1im) * conj(phase)
        ∂D12_im[i, j] += -0.5 * J₊ * sign * (-1im) * phase
        ∂D12_im[j, i] += -0.5 * J₊ * sign * (-1im) * conj(phase)
    end

    mul!(tmp, Ĩ, ∂D∂A_re)
    copyto!(∂D∂A_re, tmp)
    mul!(tmp, Ĩ, ∂D∂A_im)
    copyto!(∂D∂A_im, tmp)
end

function ∂ID∂B!(∂D∂B_re, ∂D∂B_im, tmp, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂B_re .= 0.0
    ∂D∂B_im .= 0.0

    (; J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2

    ∂D11_re = view(∂D∂B_re, 1:6, 1:6)
    ∂D22_re = view(∂D∂B_re, 7:12, 7:12)
    ∂D11_im = view(∂D∂B_im, 1:6, 1:6)
    ∂D22_im = view(∂D∂B_im, 7:12, 7:12)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        i = (α-1) * 2 + σ
        j = mod1(i+2, 6)

        # Below we follow the convention in Sunny to define the dynamical matrix
        # D11 and D22
        ∂D11_re[i, j] += 0.5 * J₊ * phase
        ∂D11_re[j, i] += 0.5 * J₊ * conj(phase)
        ∂D22_re[i, j] += 0.5 * J₊ * phase
        ∂D22_re[j, i] += 0.5 * J₊ * conj(phase)

        ∂D11_im[i, j] += 0.5 * J₊ * ( 1im) * phase
        ∂D11_im[j, i] += 0.5 * J₊ * (-1im) * conj(phase)
        ∂D22_im[i, j] += 0.5 * J₊ * (-1im) * phase
        ∂D22_im[j, i] += 0.5 * J₊ * ( 1im) * conj(phase)
    end

    mul!(tmp, Ĩ, ∂D∂B_re)
    copyto!(∂D∂B_re, tmp)
    mul!(tmp, Ĩ, ∂D∂B_im)
    copyto!(∂D∂B_im, tmp)
end

function ∂ID∂C!(∂D∂C_re, ∂D∂C_im, tmp, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂C_re .= 0.0
    ∂D∂C_im .= 0.0

    (; J, Δ) = sbs
    J₋ = J * (Δ - 1) / 2

    ∂D11_re = view(∂D∂C_re, 1:6, 1:6)
    ∂D22_re = view(∂D∂C_re, 7:12, 7:12)
    ∂D11_im = view(∂D∂C_im, 1:6, 1:6)
    ∂D22_im = view(∂D∂C_im, 7:12, 7:12)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        sign = σ == 1 ? 1 : -1
        i = (α-1) * 2 + σ
        j = mod1(i+2, 6)

        # Below we follow the convention in Sunny to define the dynamical matrix
        # D11 and D22
        ∂D11_re[i, j] += 0.5 * J₋ * sign * phase
        ∂D11_re[j, i] += 0.5 * J₋ * sign * conj(phase)
        ∂D22_re[i, j] += 0.5 * J₋ * sign * phase
        ∂D22_re[j, i] += 0.5 * J₋ * sign * conj(phase)

        ∂D11_im[i, j] += 0.5 * J₋ * sign * ( 1im) * phase
        ∂D11_im[j, i] += 0.5 * J₋ * sign * (-1im) * conj(phase)
        ∂D22_im[i, j] += 0.5 * J₋ * sign * (-1im) * phase
        ∂D22_im[j, i] += 0.5 * J₋ * sign * ( 1im) * conj(phase)
    end

    mul!(tmp, Ĩ, ∂D∂C_re)
    copyto!(∂D∂C_re, tmp)
    mul!(tmp, Ĩ, ∂D∂C_im)
    copyto!(∂D∂C_im, tmp)
end

function ∂ID∂D!(∂D∂D_re, ∂D∂D_im, tmp, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂D_re .= 0.0
    ∂D∂D_im .= 0.0

    (; J, Δ) = sbs
    J₋ = J * (Δ - 1) / 2

    ∂D21_re = view(∂D∂D_re, 7:12, 1:6)
    ∂D12_re = view(∂D∂D_re, 1:6, 7:12)
    ∂D21_im = view(∂D∂D_im, 7:12, 1:6)
    ∂D12_im = view(∂D∂D_im, 1:6, 7:12)

    phase = link_phase(α, q_reshaped)
    for σ in 1:2
        i = (α-1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i+increment, 6)

        ∂D12_re[i, j] += -0.5 * J₋ * phase
        ∂D12_re[j, i] += -0.5 * J₋ * conj(phase)
        ∂D21_re[i, j] += -0.5 * J₋ * phase
        ∂D21_re[j, i] += -0.5 * J₋ * conj(phase)

        ∂D21_im[i, j] += -0.5 * J₋ * ( 1im) * phase
        ∂D21_im[j, i] += -0.5 * J₋ * ( 1im) * conj(phase)
        ∂D12_im[i, j] += -0.5 * J₋ * (-1im) * phase
        ∂D12_im[j, i] += -0.5 * J₋ * (-1im) * conj(phase)
    end

    mul!(tmp, Ĩ, ∂D∂D_re)
    copyto!(∂D∂D_re, tmp)
    mul!(tmp, Ĩ, ∂D∂D_im)
    copyto!(∂D∂D_im, tmp)
end

function ∂ID∂μ!(∂D∂μ, tmp, α::Int)
    ∂D∂μ .= 0.0
    for σ in 1:2
        i = (α-1) * 2 + σ
        ∂D∂μ[i, i] -= 1.0
        ∂D∂μ[i+6, i+6] -= 1.0
    end
    mul!(tmp, Ĩ, ∂D∂μ)
    copyto!(∂D∂μ, tmp)
end