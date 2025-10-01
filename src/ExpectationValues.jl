
@inline bose(E, T) = sign(E) * 1 / (exp(E / T) - 1)
@inline combine_index(α, σ) = (α-1) * 2 + σ

function expectation_values(sbs::SchwingerBosonSystem)
    (; L, T) = sbs
    Nu = L^2
    expectations = zeros(ComplexF64, 15)
    As_exp = view(expectations, 1:3)
    Bs_exp = view(expectations, 4:6)
    Cs_exp = view(expectations, 7:9)
    Ds_exp = view(expectations, 10:12)
    ns_exp = view(expectations, 13:15)

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)
        for α in 1:3, n in 1:6
            β = mod1(α+1, 3)
            As_exp[α] += link_phase(α, q)/(6Nu) * ( (conj(V[combine_index(α,1)+6, n]) * V[combine_index(β,2), n] - conj(V[combine_index(α,2)+6, n] * V[combine_index(β,1), n])) * bose(E[n], T)  + (conj(V[combine_index(α,1)+6, n+6]) * V[combine_index(β,2), n+6] - conj(V[combine_index(α,2)+6, n+6]) * V[combine_index(β,1), n+6]) * bose(E[n+6], T) )

            Bs_exp[α] += link_phase(α, -q)/(6Nu) * ( (conj(V[combine_index(β,1), n]) * V[combine_index(α,1), n] + conj(V[combine_index(β,2), n] * V[combine_index(α,2), n])) * bose(E[n], T)  + (conj(V[combine_index(β,1), n+6]) * V[combine_index(α,1), n+6] + conj(V[combine_index(β,2), n+6]) * V[combine_index(α,2), n+6]) * bose(E[n+6], T) )

            Cs_exp[α] += link_phase(α, -q)/(6Nu) * ( (conj(V[combine_index(β,1), n]) * V[combine_index(α,1), n] - conj(V[combine_index(β,2), n] * V[combine_index(α,2), n])) * bose(E[n], T)  + (conj(V[combine_index(β,1), n+6]) * V[combine_index(α,1), n+6] - conj(V[combine_index(β,2), n+6]) * V[combine_index(α,2), n+6]) * bose(E[n+6], T) )

            Ds_exp[α] += link_phase(α, q)/(6Nu) * ( (conj(V[combine_index(α,1)+6, n]) * V[combine_index(β,2), n] + conj(V[combine_index(α,2)+6, n] * V[combine_index(β,1), n])) * bose(E[n], T)  + (conj(V[combine_index(α,1)+6, n+6]) * V[combine_index(β,2), n+6] + conj(V[combine_index(α,2)+6, n+6]) * V[combine_index(β,1), n+6]) * bose(E[n+6], T) )

            ns_exp[α] += 1/Nu * ((abs2(V[combine_index(α,1), n]) + abs2(V[combine_index(α,2), n])) * bose(E[n], T) + (abs2(V[combine_index(α,1), n+6]) + abs2(V[combine_index(α,2), n+6])) * bose(E[n+6], T))
        end
    end

    return expectations
end

function spin_expectations(sbs::SchwingerBosonSystem)
    (; L, T, S) = sbs
    Nu = L^2

    S_exps = zeros(3, 3)
    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    for i in 1:L, j in 1:L
        q = Vec3([(i-1)/L, (j-1)/L, 0.0])
        dynamical_matrix!(D, sbs, q)
        E = bogoliubov!(V, D)
        for α in 1:3, n in 1:6
            spinor1 = [V[combine_index(α,1), n], V[combine_index(α,2), n]]
            spinor2 = [V[combine_index(α,1), n+6], V[combine_index(α,2), n+6]]
            for μ in 1:3
                S_exps[μ, α] += S/(Nu) * real(spinor1' * σs[μ] * spinor1 * bose(E[n], T) + spinor2' * σs[μ] * spinor2 * bose(E[n+6], T))
            end
        end
    end

    return S_exps
end