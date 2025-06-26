function link_phase(link::Int, q_reshaped::Vec3)
    if link == 1
        return 1 + exp(-2π*im * q_reshaped[2]) + exp(-2π*im * (q_reshaped[1]+q_reshaped[2]))
    elseif link == 2
        return 1 + exp( 2π*im * q_reshaped[2]) + exp(-2π*im * q_reshaped[1])
    elseif link == 3
        return 1 + exp( 2π*im * q_reshaped[1]) + exp( 2π*im * (q_reshaped[1]+q_reshaped[2]))
    else
        error("Invalid link index: $link")
    end
end

Q_link(link::Int, σ, J₊, J₋, Bs, Cs) =  0.5 * (J₊*Bs[link] + σ*J₋*Cs[link])
P_link(link::Int, σ, J₊, J₋, As, Ds) = -0.5 * (J₊*σ*As[link] + J₋*Ds[link])

function dynamical_matrix!(D::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)
    D .= 0.0
    (; J, Δ, mean_fields) = sbs
    J₊ = J * (Δ + 1) / 2
    J₋ = J * (Δ - 1) / 2

    # Extracting mean fields
    As = mean_fields[1:3]
    Bs = mean_fields[4:6]
    Cs = mean_fields[7:9]
    Ds = mean_fields[10:12]
    λs = mean_fields[13:15]

    # Block structure of the dynamical matrix
    D11 = view(D, 1:6, 1:6)
    D22 = view(D, 7:12, 7:12)
    D12 = view(D, 1:6, 7:12)
    D21 = view(D, 7:12, 1:6)

    for α in 1:3
        phase = link_phase(α, q_reshaped)
        for σ in 1:2
            sign = σ == 1 ? 1 : -1
            i = (α-1) * 2 + σ
            j = mod1(i+2, 6)

            # Below we follow the convention in Sunny to define the dynamical matrix
            # D11 and D22
            D11[i, j] += Q_link(α, sign, J₊, J₋, Bs, Cs) * phase
            D11[j, i] += conj(Q_link(α, sign, J₊, J₋, Bs, Cs)) * conj(phase)
            D22[i, j] += conj(Q_link(α, sign, J₊, J₋, Bs, Cs)) * phase
            D22[j, i] += Q_link(α, sign, J₊, J₋, Bs, Cs) * conj(phase)

            # D21 and D12
            increment = σ == 1 ? 3 : 1
            j = mod1(i+increment, 6)
            D12[i, j] += P_link(α, sign, J₊, J₋, As, Ds) * phase
            D12[j, i] += P_link(α, sign, J₊, J₋, As, Ds) * conj(phase)
            D21[i, j] += conj(P_link(α, sign, J₊, J₋, As, Ds)) * phase
            D21[j, i] += conj(P_link(α, sign, J₊, J₋, As, Ds)) * conj(phase)

            # Diagonal terms
            D11[i, i] += real(λs[α])
            D22[i, i] += real(λs[α])
        end
    end
end