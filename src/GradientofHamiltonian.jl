# Note that ∂D/∂[Re(χ)] = ∂D/∂χ + ∂D/∂conj(χ), and ∂D/∂[Im(χ)] = -1im * (∂D/∂χ - ∂D/∂conj(χ)).
function ∂D∂A!(∂D∂A_re::Matrix{ComplexF64}, ∂D∂A_im::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂A_re .= 0.0
    ∂D∂A_im .= 0.0

    (; J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2

    ∂D12_re = view(∂D∂A_re, 1:6, 7:12)
    ∂D21_re = view(∂D∂A_re, 7:12, 1:6)
    ∂D12_im = view(∂D∂A_im, 1:6, 7:12)
    ∂D21_im = view(∂D∂A_im, 7:12, 1:6)

    for σ in 1:2
        sign = σ == 1 ? 1 : -1
        i = (α-1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i+increment, 6)

        ∂D21_re[i, j] += -0.5 * J₊ * sign * link_phase(α, q_reshaped)
        ∂D21_re[j, i] += -0.5 * J₊ * sign * conj(link_phase(α, q_reshaped))
        ∂D12_re[i, j] += -0.5 * J₊ * sign * link_phase(α, q_reshaped)
        ∂D12_re[j, i] += -0.5 * J₊ * sign * conj(link_phase(α, q_reshaped))

        ∂D21_im[i, j] += -0.5 * J₊ * sign * (-1im) * link_phase(α, q_reshaped)
        ∂D21_im[j, i] += -0.5 * J₊ * sign * (-1im) * conj(link_phase(α, q_reshaped))
        ∂D12_im[i, j] += -0.5 * J₊ * sign * (1im) * link_phase(α, q_reshaped)
        ∂D12_im[j, i] += -0.5 * J₊ * sign * (1im) * conj(link_phase(α, q_reshaped))
    end

end

function ∂D∂B!(∂D∂B_re::Matrix{ComplexF64}, ∂D∂B_im::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂B_re .= 0.0
    ∂D∂B_im .= 0.0

    (; J, Δ) = sbs
    J₊ = J * (Δ + 1) / 2

    ∂D11_re = view(∂D∂B_re, 1:6, 1:6)
    ∂D22_re = view(∂D∂B_re, 7:12, 7:12)
    ∂D11_im = view(∂D∂B_im, 1:6, 1:6)
    ∂D22_im = view(∂D∂B_im, 7:12, 7:12)

    for σ in 1:2
        i = (α-1) * 2 + σ
        j = mod1(i+2, 6)

        # Below we follow the convention in Sunny to define the dynamical matrix
        # D11 and D22
        ∂D11_re[i, j] += 0.5 * J₊ * link_phase(α, q_reshaped)
        ∂D11_re[j, i] += 0.5 * J₊ * conj(link_phase(α, q_reshaped))
        ∂D22_re[i, j] += 0.5 * J₊ * link_phase(α, q_reshaped)
        ∂D22_re[j, i] += 0.5 * J₊ * conj(link_phase(α, q_reshaped))

        ∂D11_im[i, j] += 0.5 * J₊ * (-1im) * link_phase(α, q_reshaped)
        ∂D11_im[j, i] += 0.5 * J₊ * ( 1im) * conj(link_phase(α, q_reshaped))
        ∂D22_im[i, j] += 0.5 * J₊ * ( 1im) * link_phase(α, q_reshaped)
        ∂D22_im[j, i] += 0.5 * J₊ * (-1im) * conj(link_phase(α, q_reshaped))
    end
end

function ∂D∂C!(∂D∂C_re::Matrix{ComplexF64}, ∂D∂C_im::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂C_re .= 0.0
    ∂D∂C_im .= 0.0

    (; J, Δ) = sbs
    J₋ = J * (Δ - 1) / 2

    ∂D11_re = view(∂D∂C_re, 1:6, 1:6)
    ∂D22_re = view(∂D∂C_re, 7:12, 7:12)
    ∂D11_im = view(∂D∂C_im, 1:6, 1:6)
    ∂D22_im = view(∂D∂C_im, 7:12, 7:12)

    for σ in 1:2
        sign = σ == 1 ? 1 : -1
        i = (α-1) * 2 + σ
        j = mod1(i+2, 6)

        # Below we follow the convention in Sunny to define the dynamical matrix
        # D11 and D22
        ∂D11_re[i, j] += 0.5 * J₋ * sign * link_phase(α, q_reshaped)
        ∂D11_re[j, i] += 0.5 * J₋ * sign * conj(link_phase(α, q_reshaped))
        ∂D22_re[i, j] += 0.5 * J₋ * sign * link_phase(α, q_reshaped)
        ∂D22_re[j, i] += 0.5 * J₋ * sign * conj(link_phase(α, q_reshaped))

        ∂D11_im[i, j] += 0.5 * J₋ * sign * (-1im) * link_phase(α, q_reshaped)
        ∂D11_im[j, i] += 0.5 * J₋ * sign * ( 1im) * conj(link_phase(α, q_reshaped))
        ∂D22_im[i, j] += 0.5 * J₋ * sign * ( 1im) * link_phase(α, q_reshaped)
        ∂D22_im[j, i] += 0.5 * J₋ * sign * (-1im) * conj(link_phase(α, q_reshaped))
    end
end

function ∂D∂D!(∂D∂D_re::Matrix{ComplexF64}, ∂D∂D_im::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3, α::Int)
    ∂D∂D_re .= 0.0
    ∂D∂D_im .= 0.0

    (; J, Δ) = sbs
    J₋ = J * (Δ - 1) / 2

    ∂D21_re = view(∂D∂D_re, 7:12, 1:6)
    ∂D12_re = view(∂D∂D_re, 1:6, 7:12)
    ∂D21_im = view(∂D∂D_im, 7:12, 1:6)
    ∂D12_im = view(∂D∂D_im, 1:6, 7:12)

    for σ in 1:2
        i = (α-1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i+increment, 6)

        ∂D21_re[i, j] += -0.5 * J₋ * link_phase(α, q_reshaped)
        ∂D21_re[j, i] += -0.5 * J₋ * conj(link_phase(α, q_reshaped))
        ∂D12_re[i, j] += -0.5 * J₋ * link_phase(α, q_reshaped)
        ∂D12_re[j, i] += -0.5 * J₋ * conj(link_phase(α, q_reshaped))

        ∂D21_im[i, j] += -0.5 * J₋ * (-1im) * link_phase(α, q_reshaped)
        ∂D21_im[j, i] += -0.5 * J₋ * (-1im) * conj(link_phase(α, q_reshaped))
        ∂D12_im[i, j] += -0.5 * J₋ * (1im) * link_phase(α, q_reshaped)
        ∂D12_im[j, i] += -0.5 * J₋ * (1im) * conj(link_phase(α, q_reshaped))
    end
end

function ∂D∂λ!(∂D∂λ::Matrix{ComplexF64}, α::Int)
    ∂D∂λ .= 0.0
    for σ in 1:2
        i = (α-1) * 2 + σ
        ∂D∂λ[i, i] += 1.0
        ∂D∂λ[i+6, i+6] += 1.0
    end
end