@inline log1mexp_modified(x) = (x < log(2)) ? log(-expm1(-x)) : log1p(-exp(-x))

function single_particle_density_matrix!(P::Matrix{ComplexF64}, D::Matrix{ComplexF64}, V::Matrix{ComplexF64}, tmp::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q_reshaped::Vec3)

    P .= 0.0
    (; T) = sbs
    dynamical_matrix!(D, sbs, q_reshaped)
    try
        E = bogoliubov!(V, D)
        @inbounds for i in eachindex(E)
            P[i, i] += coth(E[i] / (2T)) / 4
        end
        mul!(tmp, V, P)
        mul!(P, tmp, inv(V))
        return E
    catch _
        P .= 0.0
        E = fill(NaN, 12)
        return E
    end
end

function condensation_results!(sbs::SchwingerBosonSystem, aux::CondensationAux)
    (; condensation_ϵ, L, S) = sbs
    conden_index = aux.conden_index
    if isnothing(conden_index)
        return nothing
    end

    Nu = L^2
    i = (conden_index - 1) ÷ L + 1
    j = (conden_index - 1) % L + 1
    q_condensed = Vec3([(i-1)/L, (j-1)/L, 0.0])

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)
    dynamical_matrix!(D, sbs, q_condensed)
    E = bogoliubov!(V, D)

    conden_band_indices = findall(x -> isapprox(x, condensation_ϵ; atol=1e-6), E)
    num_cs = length(conden_band_indices)
    @assert num_cs > 0 "No condensed band found at q = $q_condensed with energy ≈ $condensation_ϵ. This indicates that the system is not in the condensed phase."

    out = zeros(ComplexF64, 12, 12)
    ∂ID∂μ0!(out, 1)

    v1 = view(V, :, conden_band_indices[1])
    qc = - real(v1' * Ĩ * out * v1)

    P_tmp = zeros(ComplexF64, 12, 12)
    D_tmp = zeros(ComplexF64, 12, 12)
    V_tmp = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)

    N_normal = 0.0
    for k in 1:L, l in 1:L
        q_tmp = Vec3([(k-1)/L, (l-1)/L, 0.0])
        single_particle_density_matrix!(P_tmp, D_tmp, V_tmp, tmp, sbs, q_tmp)
        N_normal += -real(tr(P_tmp * out)) / Nu
    end

    ξ = (2S + 1 - N_normal) / (num_cs * qc)
    if ξ ≥ 0
        tmp .= 0.0
        for index in conden_band_indices
            v = view(V, :, index)
            tmp .+= ξ * Nu * (v * v' * Ĩ)
        end
        aux.ξ = ξ
        aux.num_conden = num_cs
        return tmp
    else
        @warn "Negative condensate fraction detected: ξ = $ξ. Reoptimizing μ₀ without the condensation shift to find the true minimum in the normal phase."
        aux.conden_index = nothing
        # Un-shift μ0 to remove the artificial shift introduced by the condensation search.
        μ0_unshifted = copy(sbs.mean_fields[13:15]) .+ aux.c_shift
        # Optimize μ0 again without the condensation shift to find the true minimum in the normal phase.
        optimize_μ0_legacy!(sbs, μ0_unshifted)
        return nothing
    end

end

# Calculate the divided difference matrix as L"\mathbb{D}" defined in the note
function divided_difference!(sbs, Dmat, E; atol=1e-8)
    Dmat .= 0.0
    T = sbs.T
    @inbounds for i in eachindex(E), j in eachindex(E)
        if isapprox(E[i], E[j]; atol)
            Dmat[i, j] += T > 1e-8 ? -csch(E[i]/(2T))^2 / (8T) : 0.0
        else
            Dmat[i, j] += (coth(E[i]/2T) - coth(E[j]/2T)) / (4 * (E[i] - E[j]))
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