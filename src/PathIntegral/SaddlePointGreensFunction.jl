
"""
    Green_SP_normal(sbs, q, z, aux=nothing)

Return the normal part of the saddle-point Green's function,

    G_SP^n(q, z) = sum_l C_l(q) / (ϵ_l(q) - z),

with the pinned condensate modes removed when `aux` is available.

Here `q` is assumed to already be in the reshaped reciprocal-lattice
coordinate used by `dynamical_matrix!`.
"""
function Green_SP_normal(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    z::Number,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    H = zeros(ComplexF64, 12, 12)
    V = similar(H)

    # Reuse the canonical BdG construction.
    dynamical_matrix!(H, sbs, q)

    # bogoliubov! may overwrite H, but H is not needed afterwards.
    ϵs = try
        bogoliubov!(V, H)
    catch
        error("BdG spectrum is unstable at q = $q.")
    end

    G = zero(H)

    for l in eachindex(ϵs)
        # In the canonical soft-minimum treatment, ξ is only the enhanced
        # occupation of the pinned ±ϵ condensate modes. Those pinned modes
        # still also carry the ordinary normal occupation.
        #
        # In the path-integral convention used here, Green_SP_condensed carries
        # the total pinned-mode weight. Therefore, when condensate information
        # is available, we remove the pinned ±ϵ modes from Green_SP_normal to
        # avoid double counting.
        if aux !== nothing && aux.conden_index !== nothing
            isapprox(abs(ϵs[l]), sbs.condensation_ϵ; atol = 1e-8) && continue
        end

        s = l <= 6 ? 1 : -1
        v = @view V[:, l]
        coeff = s / (ϵs[l] - z)

        # Equivalent to the less efficient direct expression
        #
        #     G .+= coeff .* (v * v')
        #
        # but written as an explicit loop to avoid allocating the 12×12
        # outer-product matrix for every BdG mode.
        @inbounds for j in axes(G, 2), i in axes(G, 1)
            G[i, j] += coeff * v[i] * conj(v[j])
        end
    end

    return G
end


"""
    Green_SP_condensed(sbs, q, z, aux=nothing)

Return the condensate part of the saddle-point Green's function,

    G_SP^c(q, z) = δ_{q,q_c} sum_i (ξ_i + 1) C_i^c / (ϵ_i^c - z).

Here `q_c` is the condensate momentum encoded by `aux.conden_index`.

If `aux` is `nothing`, or if no condensate is present, this returns the zero
matrix.
"""
function Green_SP_condensed(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    z::Number,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    H = zeros(ComplexF64, 12, 12)
    G = zero(H)

    aux === nothing && return G
    aux.conden_index === nothing && return G

    V = similar(H)

    # The condensate contribution is localized at q = q_c. We do not add a
    # separate norm(q) check here, because the relevant condensate momentum
    # sector is already encoded by aux.conden_index.
    dynamical_matrix!(H, sbs, q)

    ϵs = try
        bogoliubov!(V, H)
    catch
        error("BdG spectrum is unstable at q = $q.")
    end

    ξ = aux.ξ
    ϵ = sbs.condensation_ϵ

    for l in eachindex(ϵs)
        isapprox(abs(ϵs[l]), ϵ; atol = 1e-8) || continue

        s = l <= 6 ? 1 : -1
        v = @view V[:, l]

        # In the canonical soft-minimum treatment, ξ is only the enhanced
        # occupation of the pinned condensate mode. Since Green_SP_normal
        # removes the pinned ±ϵ modes entirely, the condensate Green's function
        # must carry the total pinned-mode weight, namely ξ + 1.
        coeff = (ξ + 1) * s / (ϵs[l] - z)

        # Equivalent to the less efficient direct expression
        #
        #     G .+= coeff .* (v * v')
        #
        # but written as an explicit loop to avoid allocating the 12×12
        # outer-product matrix for every BdG mode.
        @inbounds for j in axes(G, 2), i in axes(G, 1)
            G[i, j] += coeff * v[i] * conj(v[j])
        end
    end

    return G
end