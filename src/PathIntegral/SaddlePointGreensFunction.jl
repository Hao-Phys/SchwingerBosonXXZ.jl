# src/PathIntegral/SaddlePointGreensFunction.jl
"""
    Green_SP_normal_residues(sbs, q, aux=nothing)

Return the spectral data for the normal part of the saddle-point Green's
function.

The returned objects are `ϵs, V, weights` such that

    G_SP_normal(q, z)
      = sum_l weights[l] * Ĩ[l,l] * v_l v_l† / (ϵs[l] - z).

Here `q` must already be in the reshaped reciprocal-lattice coordinate used by
`dynamical_matrix!`.

If condensation data is available, the pinned ±ϵ modes are removed from the
normal sector by setting their weights to zero.
"""
function Green_SP_normal_residues(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    H = zeros(ComplexF64, 12, 12)
    V = similar(H)

    dynamical_matrix!(H, sbs, q)

    ϵs = try
        bogoliubov!(V, H)
    catch
        error("BdG spectrum is unstable at q = $q.")
    end

    weights = ones(Float64, length(ϵs))

    if aux !== nothing && aux.conden_index !== nothing
        for l in eachindex(ϵs)
            if isapprox(abs(ϵs[l]), sbs.condensation_ϵ; atol = 1e-8)
                weights[l] = 0.0
            end
        end
    end

    return ϵs, V, weights
end


"""
    Green_SP_condensed_residues(sbs, q, aux=nothing)

Return the spectral data for the condensate part of the saddle-point Green's
function.

The returned objects are `ϵs, V, weights` such that

    G_SP_condensed(q, z)
      = sum_l weights[l] * Ĩ[l,l] * v_l v_l† / (ϵs[l] - z).

The condensate contribution has support only in the condensate momentum sector
encoded by `aux.conden_index`.

In the canonical soft-minimum treatment, ξ is only the enhanced occupation of
the pinned condensate mode. Since `Green_SP_normal_residues` removes the pinned
±ϵ modes entirely, the condensate residue weight used here is `ξ + 1`.
"""
function Green_SP_condensed_residues(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    H = zeros(ComplexF64, 12, 12)
    V = similar(H)

    dynamical_matrix!(H, sbs, q)

    ϵs = try
        bogoliubov!(V, H)
    catch
        error("BdG spectrum is unstable at q = $q.")
    end

    weights = zeros(Float64, length(ϵs))

    if aux === nothing || aux.conden_index === nothing
        return ϵs, V, weights
    end

    ξtot = aux.ξ + 1
    ϵ = sbs.condensation_ϵ

    for l in eachindex(ϵs)
        if isapprox(abs(ϵs[l]), ϵ; atol = 1e-8)
            weights[l] = ξtot
        end
    end

    return ϵs, V, weights
end


"""
    Green_SP_from_residues(ϵs, V, weights, z)

Construct the Green's-function matrix from spectral pole data,

    G(q, z) = sum_l weights[l] * Ĩ[l,l] * v_l v_l† / (ϵs[l] - z).

The factor `Ĩ[l,l]` is the scalar metric sign appearing in the residue
`C_l = s_l v_l v_l†`. With the BdG ordering used here, `Ĩ[l,l]` is +1 for the
six positive-energy modes and -1 for the six negative-energy modes.
"""
function Green_SP_from_residues(
    ϵs::AbstractVector,
    V::AbstractMatrix,
    weights::AbstractVector,
    z::Number,
)
    G = zeros(ComplexF64, size(V, 1), size(V, 1))

    for l in eachindex(ϵs)
        iszero(weights[l]) && continue

        coeff = weights[l] * Ĩ[l, l] / (ϵs[l] - z)
        v = @view V[:, l]

        @inbounds for j in axes(G, 2), i in axes(G, 1)
            G[i, j] += coeff * v[i] * conj(v[j])
        end
    end

    return G
end


"""
    Green_SP_normal(sbs, q, z, aux=nothing)

Return the normal part of the saddle-point Green's function as a matrix.
"""
function Green_SP_normal(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    z::Number,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    ϵs, V, weights = Green_SP_normal_residues(sbs, q, aux)
    return Green_SP_from_residues(ϵs, V, weights, z)
end


"""
    Green_SP_condensed(sbs, q, z, aux=nothing)

Return the condensate part of the saddle-point Green's function as a matrix.
"""
function Green_SP_condensed(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    z::Number,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    ϵs, V, weights = Green_SP_condensed_residues(sbs, q, aux)
    return Green_SP_from_residues(ϵs, V, weights, z)
end


"""
    _residue_matrix_from_residues(V, weights)

Construct the equal-time residue matrix

    C = sum_l weights[l] * Ĩ[l,l] * v_l v_l†

from residue data.

This is the matrix version of the same residue convention used by
`Green_SP_from_residues`.
"""
function _residue_matrix_from_residues(
    V::AbstractMatrix,
    weights::AbstractVector,
)
    C = zeros(ComplexF64, size(V, 1), size(V, 1))

    for l in eachindex(weights)
        iszero(weights[l]) && continue

        coeff = weights[l] * Ĩ[l, l]
        v = @view V[:, l]

        @inbounds for j in axes(C, 2), i in axes(C, 1)
            C[i, j] += coeff * v[i] * conj(v[j])
        end
    end

    return C
end


"""
    _condensed_residue_matrix(sbs, qc, aux)

Construct the static condensate residue matrix from
`Green_SP_condensed_residues`.
"""
function _condensed_residue_matrix(
    sbs::SchwingerBosonSystem,
    qc::Vec3,
    aux::CondensationAux,
)
    _, Vc, weights_c = Green_SP_condensed_residues(sbs, qc, aux)
    return _residue_matrix_from_residues(Vc, weights_c)
end


"""
    _residue_vertex_trace(Vq, wq, n, A, Vk, wk, m, B)

Compute

    tr[C_{q,n} A C_{k,m} B]

using the rank-one residue form

    C_l = weights[l] * Ĩ[l,l] * v_l v_l†.

This avoids explicitly allocating the two residue matrices.
"""
function _residue_vertex_trace(
    Vq::AbstractMatrix,
    wq::AbstractVector,
    n::Int,
    A::AbstractMatrix,
    Vk::AbstractMatrix,
    wk::AbstractVector,
    m::Int,
    B::AbstractMatrix,
)
    vn = @view Vq[:, n]
    vm = @view Vk[:, m]

    coeff = wq[n] * Ĩ[n, n] * wk[m] * Ĩ[m, m]

    return coeff * dot(vn, A * vm) * dot(vm, B * vn)
end