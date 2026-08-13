"""
    enhanced_single_particle_density_matrix!(
        P,
        D,
        V,
        sbs,
        aux
    )

Add the enhanced-occupation contribution to the single-particle density
matrix `P` at the selected momentum.

Only the additional occupations in `aux.active_positive_weights` are
included. The ordinary unit-residue contribution must already be present in
`P`. The matrices `D` and `V` are used as scratch buffers.

Returns the updated matrix `P`.
"""
function enhanced_single_particle_density_matrix!(
    P::Matrix{ComplexF64},
    D::Matrix{ComplexF64},
    V::Matrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    aux::SpectralCondensationAux
)
    (; conden_band_indices, active_positive_weights) = aux

    all(iszero, active_positive_weights) && return P

    (; L) = sbs
    q_c = _spectral_condensation_momentum(aux, L)

    dynamical_matrix!(D, sbs, q_c)
    bogoliubov!(V, D)

    Nu = L^2

    for band in conden_band_indices
        ξ = active_positive_weights[band]
        iszero(ξ) && continue

        v = @view V[:, band]
        P .+= ξ * Nu * (v * v' * Ĩ)
    end

    return P
end


"""
    saddle_point_free_energy(sbs, aux)

Return the variational free energy per unit cell of an already solved
Schwinger-boson saddle point.

The auxiliary data `aux` must correspond to the current state of `sbs`.
Ordinary unit-residue contributions and the separate enhanced occupations
stored in `aux.active_positive_weights` are both included. This function
does not optimize or modify the saddle point.
"""
function saddle_point_free_energy(
    sbs::SchwingerBosonSystem,
    aux::SpectralCondensationAux
)
    (; S, mean_fields, L) = sbs

    D = zeros(ComplexF64, 12, 12)
    V = zeros(ComplexF64, 12, 12)

    f = bosonic_free_energy!(nothing, V, D, sbs)
    isfinite(f) || return f

    μ0s = @view mean_fields[13:15]
    f += (1 + 2S) * sum(real, μ0s)

    P = zeros(ComplexF64, 12, 12)
    tmp = zeros(ComplexF64, 12, 12)
    ∂ID∂ϕs = zeros(ComplexF64, 12, 12, 24)
    ∂F2α = zeros(24)

    inv_fα = inv_interaction_strengths(sbs)
    Nu = L^2

    for i in 1:L, j in 1:L
        q = Vec3([(i - 1) / L, (j - 1) / L, 0.0])
        single_particle_density_matrix!(P, D, V, tmp, sbs, q)

        q_index = (i - 1) * L + j

        if q_index == aux.conden_index
            enhanced_single_particle_density_matrix!(P, D, V, sbs, aux)
        end

        @views for α in 1:3
            ∂ID∂A!(∂ID∂ϕs[:, :, α], ∂ID∂ϕs[:, :, α + 12], sbs, q, α)
            ∂ID∂B!(∂ID∂ϕs[:, :, α + 3], ∂ID∂ϕs[:, :, α + 15], sbs, q, α)
            ∂ID∂C!(∂ID∂ϕs[:, :, α + 6], ∂ID∂ϕs[:, :, α + 18], sbs, q, α)
            ∂ID∂D!(∂ID∂ϕs[:, :, α + 9], ∂ID∂ϕs[:, :, α + 21], sbs, q, α)
        end

        @views for α in 1:24
            ∂F2α[α] += real(tr(P * ∂ID∂ϕs[:, :, α])) / Nu
        end
    end

    ϕ = zeros(24)

    for α in 1:12
        ϕ[α], ϕ[α + 12] = reim(mean_fields[α])
    end

    for α in 1:24
        f += inv_fα[α] * ∂F2α[α]^2 / 12
        f -= ∂F2α[α] * ϕ[α]
    end

    return f
end