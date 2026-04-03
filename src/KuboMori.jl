@inline log1mexp_modified(x) = (x < log(2)) ? log(-expm1(-x)) : log1p(-exp(-x))

function single_particle_density_matrix_normal!(P::Matrix{ComplexF64}, 
    D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, 
    tmp::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)

    P .= 0.0
    (; T) = sbs
    dynamical_matrix!(D, sbs, q_reshaped)
    E = bogoliubov!(V, D)
    @inbounds for i in 1:6
        P[i, i] += coth(E[i] / (2T)) / 2
    end
    mul!(tmp, V, P)
    mul!(P, tmp, inv(V))

    return E
end

function single_particle_density_matrix_condensed!(P::Matrix{ComplexF64}, 
    D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, 
    tmp::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)

    E = single_particle_density_matrix_normal!(P, D, V, tmp, sbs, q_reshaped)

    (; condensation_ϵ, L, S) = sbs

    condensed_indices = findall(x -> isapprox(x, condensation_ϵ; atol=1e-8), E)

    num_cs = length(condensed_indices)

    if num_cs > 0
        cs_index1 = condensed_indices[1]  # Take the first one if there are multiple
        inv_V = inv(V)
        vr = copy(V[:, cs_index1])
        vl = transpose(inv_V[cs_index1, :])
        out = zeros(ComplexF64, 12, 12)
        ∂ID∂μ0!(out, 1)
        qc = - real(vl * out * vr)

        P_tmp = zeros(ComplexF64, 12, 12)
        D_tmp = zeros(ComplexF64, 12, 12)
        V_tmp = zeros(ComplexF64, 12, 12)

        N_normal = 0.0
        for i in 1:L, j in 1:L
            q_tmp = Vec3([(i-1)/L, (j-1)/L, 0.0])
            single_particle_density_matrix_normal!(P_tmp, D_tmp, V_tmp, tmp, sbs, q_tmp)
            N_normal += -real(tr(P_tmp * out)) / L^2
        end

        ξ = (2S + 1 - N_normal) / (num_cs * qc)

        # TODO: deal with the case ξ < 0
        @assert ξ ≥ 0 "Negative condensate fraction detected: ξ = $ξ. This indicates that the system is not in the condensed phase."

        for index in condensed_indices
            vr = copy(V[:, index])
            vl = transpose(inv_V[index, :])
            P .+= ξ * vr * vl * L^2
        end
    end

    return E
end

# Calculate the divided difference matrix as L"\mathbb{D}" defined in the note
function divided_difference!(sbs, Dmat, E; atol=1e-8)
    Dmat .= 0.0
    T = sbs.T
    for i in 1:6, j in 1:6
        if isapprox(E[i], E[j]; atol)
            Dmat[i, j] += T > 1e-8 ? -csch(E[i]/(2T))^2 / (4T) : 0.0
        else
            Dmat[i, j] += (coth(E[i]/2T) - coth(E[j]/2T)) / (2 * (E[i] - E[j]))
        end
    end
end

# Divided difference for the condensed phase.
# Extracts the effective occupation f_total[i] = (V⁻¹ P V)[i,i] directly from the
# condensed density matrix P (which already includes the ξ·L² condensate term).
# For non-condensate modes this reduces to coth(E[i]/2T)/2, recovering `divided_difference!`.
# The condensate fraction ξ is treated as constant w.r.t. E (its global constraint response
# is a non-local correction deferred to future work).
function divided_difference_condensed!(sbs, Dmat, E, P, V, inv_V; atol=1e-8)
    Dmat .= 0.0
    T = sbs.T
    # f_total[i] = diagonal of (V⁻¹ P V) for the positive-energy block i = 1:6
    PV = P * V   # 12×12, one allocation
    f = zeros(Float64, 6)
    for i in 1:6
        f[i] = real(dot(view(inv_V, i, :), view(PV, :, i)))
    end
    for i in 1:6, j in 1:6
        if isapprox(E[i], E[j]; atol)
            Dmat[i, j] += T > 1e-8 ? -csch(E[i]/(2T))^2 / (4T) : 0.0
        else
            Dmat[i, j] += (f[i] - f[j]) / (E[i] - E[j])
        end
    end
end

# The contents of `tmp` and `tmp2` are modified in-place. The remainings are fixed. The final result is stored in `tmp`.
function divided_aux!(tmp, tmp2, Dmat, ∂D∂X, V, inv_V)
    mul!(tmp, inv_V, ∂D∂X)
    mul!(tmp2, tmp, V)
    tmp .= Dmat .* tmp2
    mul!(tmp2, V, tmp)
    mul!(tmp, tmp2, inv_V)
end