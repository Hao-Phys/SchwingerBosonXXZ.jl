# src/PathIntegral/SaddlePointGreensFunction.jl

@inline function _same_momentum_mod1(
    q::Vec3,
    p::Vec3;
    atol::Float64 = 1e-8,
)
    return all(
        a -> abs(mod(q[a] - p[a] + 0.5, 1.0) - 0.5) <= atol,
        1:2,
    )
end

@inline function _spectral_condensation_momentum(
    aux::SpectralCondensationAux,
    L::Int,
)
    i = (aux.conden_index - 1) ÷ L + 1
    j = (aux.conden_index - 1) % L + 1

    return Vec3([(i - 1) / L, (j - 1) / L, 0.0])
end

"""
    Green_SP_normal_residues(sbs, q, aux)

Return the spectral data for the ordinary normal part of the saddle-point
Green function.

The returned objects are `ϵs, V, weights`, with

    G_SP_normal(q, z)
        = sum_l weights[l] * Ĩ[l,l] *
          v_l * v_l' / (ϵs[l] - z).

All ordinary BdG poles begin with unit residue. At the selected momentum, the
poles listed in `aux.conden_band_indices` are removed from this sector so that
they can be represented in `Green_SP_condensed_residues`.

No active soft-minimum weight is included in this Green function.
"""
function Green_SP_normal_residues(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    aux::SpectralCondensationAux,
)
    isempty(aux.conden_band_indices) &&
        error(
            "SpectralCondensationAux has no selected sector.",
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

    qc = _spectral_condensation_momentum(aux, sbs.L)

    if _same_momentum_mod1(q, qc)
        for l in aux.conden_band_indices
            weights[l] = 0.0
        end
    end

    return ϵs, V, weights
end

"""
    Green_SP_condensed_residues(sbs, q, aux)

Return the spectral data for the ordinary selected part of the saddle-point
Green function.

The returned objects are `ϵs, V, weights`, with

    G_SP_selected(q, z)
        = sum_l weights[l] * Ĩ[l,l] *
          v_l * v_l' / (ϵs[l] - z).

This contribution has support only at the selected momentum. Every selected
pole has ordinary unit BdG residue, independently of whether the soft-minimum
constraint is active.

The additional enhanced occupations `ξᵢ` are stored separately in
`aux.active_positive_weights` and must enter only through the active-constraint
curvature blocks.
"""
function Green_SP_condensed_residues(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    aux::SpectralCondensationAux,
)
    isempty(aux.conden_band_indices) &&
        error(
            "SpectralCondensationAux has no selected sector.",
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

    qc = _spectral_condensation_momentum(aux, sbs.L)

    if _same_momentum_mod1(q, qc)
        for l in aux.conden_band_indices
            weights[l] = 1.0
        end
    end

    return ϵs, V, weights
end

"""
    Green_SP_from_residues(ϵs, V, weights, z)

Construct the Green's-function matrix from spectral pole data,

    G(q, z)
        = sum_l weights[l] * Ĩ[l,l] * v_l v_l† / (ϵs[l] - z).

The factor `Ĩ[l,l]` is the scalar metric sign appearing in the residue

    C_l = s_l v_l v_l†.

With the BdG ordering used here, `Ĩ[l,l]` is +1 for the six positive-energy
modes and -1 for the six negative-energy modes.
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
    Green_SP_normal(sbs, q, z, aux)

Return the normal part of the saddle-point Green's function as a matrix.
"""
function Green_SP_normal(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux,
)
    ϵs, V, weights = Green_SP_normal_residues(sbs, q, aux)

    return Green_SP_from_residues(ϵs, V, weights, z)
end

"""
    Green_SP_condensed(sbs, q, z, aux)

Return the selected soft-mode part of the saddle-point Green's function as a
matrix.
"""
function Green_SP_condensed(
    sbs::SchwingerBosonSystem,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux,
)
    ϵs, V, weights = Green_SP_condensed_residues(sbs, q, aux)

    return Green_SP_from_residues(ϵs, V, weights, z)
end

"""
    _residue_matrix_from_residues(V, weights)

Construct the equal-time residue matrix

    C = sum_l weights[l] * Ĩ[l,l] * v_l v_l†

from residue data. This is the matrix version of the same residue convention
used by `Green_SP_from_residues`.
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

Construct the static selected soft-mode residue matrix from
`Green_SP_condensed_residues`.
"""
function _condensed_residue_matrix(
    sbs::SchwingerBosonSystem,
    qc::Vec3,
    aux::SpectralCondensationAux,
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