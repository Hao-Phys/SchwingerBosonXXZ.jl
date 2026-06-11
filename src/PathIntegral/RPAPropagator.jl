# src/PathIntegral/RPAPropagator.jl

"""
    InternalField

Label for one reduced internal auxiliary-field vertex.

Fields are

- `kind = :W` or `:Wbar` for Hubbard-Stratonovich fields,
- `kind = :λ` for constraint fields.

For HS fields, `channel ∈ (:A, :B, :C, :D)`, `a = 1,2,3`, and `δ = 1,2,3`.
For constraint fields, use `channel = :none` and `δ = 0`.
"""
struct InternalField
    kind::Symbol
    channel::Symbol
    a::Int
    δ::Int
end


"""
    internal_field_basis()

Return the ordered internal-field basis used for `Π0`, `Π`, and the RPA kernel.

The ordering is

1. all HS fields `W`, `Wbar` for `X = A,B,C,D`, `a = 1,2,3`, `δ = 1,2,3`;
2. the three constraint fields `λ_a`.

The total dimension is `4 * 3 * 3 * 2 + 3 = 75`.
"""
function internal_field_basis()
    fields = InternalField[]

    for X in (:A, :B, :C, :D)
        for a in 1:3
            for δ in 1:3
                push!(fields, InternalField(:W, X, a, δ))
                push!(fields, InternalField(:Wbar, X, a, δ))
            end
        end
    end

    for a in 1:3
        push!(fields, InternalField(:λ, :none, a, 0))
    end

    return fields
end


"""
    internal_vertices!(V, sbs, field::InternalField, k, p)

Fill `V` with the reduced internal vertex associated with `field`.

For `field.kind === :λ`, this dispatches to

    internal_vertices!(V, :λ, field.a)

For HS fields, this dispatches to

    internal_vertices!(V, sbs, field.kind, field.channel, field.a, field.δ, k, p)

The returned vertex is reduced: the Fourier normalization and the
momentum-frequency Kronecker delta are not included.
"""
function internal_vertices!(
    V::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    field::InternalField,
    k::Vec3,
    p::Vec3,
)
    size(V) == (12, 12) ||
        throw(DimensionMismatch("`V` must have size (12, 12)."))

    fill!(V, 0.0 + 0.0im)

    if field.kind === :λ
        return internal_vertices!(V, :λ, field.a)
    elseif field.kind === :W || field.kind === :Wbar
        return internal_vertices!(
            V,
            sbs,
            field.kind,
            field.channel,
            field.a,
            field.δ,
            k,
            p,
        )
    else
        throw(ArgumentError("Unknown internal-field kind `$(field.kind)`."))
    end
end


# ----------------------------------------------------------------------
# Bare auxiliary-field kernel Π0
# ----------------------------------------------------------------------

"""
    Pi0!(Π0, sbs, fields)

Fill `Π0` with the bare auxiliary-field kernel.

This implements

    Π0[W^X_{a,δ}, Wbar^X_{a,δ}] = κ^X_{a,δ} / 2
    Π0[Wbar^X_{a,δ}, W^X_{a,δ}] = κ^X_{a,δ} / 2

All entries involving `λ` are zero.
"""
function Pi0!(
    Π0::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
)
    nϕ = length(fields)

    size(Π0) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π0` must have size ($(nϕ), $(nϕ))."))

    fill!(Π0, 0.0 + 0.0im)

    for (i, field) in pairs(fields)
        field.kind === :W || continue

        partner = InternalField(:Wbar, field.channel, field.a, field.δ)
        j = _field_index(fields, partner)

        κ, _ = _κ_s(sbs, field.channel, field.δ)

        Π0[i, j] += κ / 2
        Π0[j, i] += κ / 2
    end

    return Π0
end


# ----------------------------------------------------------------------
# Retarded polarization operator
# ----------------------------------------------------------------------

"""
    polarization!(Π, sbs, fields, kgrid, q, ω; η, Nflavor=2, aux=nothing)

Fill `Π` with the retarded zero-temperature normal-normal polarization.

This branch is intentionally normal-only. Condensate-normal contributions are
implemented only in the Matsubara branch, which is used for fluctuation-matrix
stability and gauge-mode checks.
"""
function polarization!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    ω::Real;
    η::Real,
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    fill!(Π, 0.0 + 0.0im)

    return _polarization_normal_core!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        ComplexF64(ω, η);
        Nflavor = Nflavor,
        aux = aux,
    )
end


"""
    polarization_normal!(Π, sbs, fields, kgrid, q, ω; η, Nflavor=2, aux=nothing)

Add the retarded normal-normal contribution to `Π`.

This is a compatibility wrapper around `polarization!`.
"""
function polarization_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    ω::Real;
    η::Real,
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    return polarization!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        ω;
        η = η,
        Nflavor = Nflavor,
        aux = aux,
    )
end


# ----------------------------------------------------------------------
# Matsubara polarization operator for fluctuation-matrix checks
# ----------------------------------------------------------------------

"""
    polarization_matsubara!(
        Π, sbs, fields, kgrid, q, iωq;
        Nflavor = 2,
        aux = nothing,
    )

Fill `Π` with the Matsubara zero-temperature polarization operator.

This is intended for checking the Euclidean Gaussian fluctuation matrix

    K(q, iωq) = Π0 - Π(q, iωq)

before analytic continuation.

The normal-normal contribution is always included. If `aux` contains a
condensate, this function also adds the mixed condensate-normal terms

    Π_cn + Π_nc.

The purely elastic condensate-condensate contribution `Π_cc` is intentionally
omitted.
"""
function polarization_matsubara!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    iωq::Complex;
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    fill!(Π, 0.0 + 0.0im)

    polarization_matsubara_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        iωq;
        Nflavor = Nflavor,
        aux = aux,
    )

    polarization_matsubara_condensate_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        iωq;
        Nflavor = Nflavor,
        aux = aux,
    )

    return Π
end


"""
    polarization_matsubara_normal!(
        Π, sbs, fields, kgrid, q, iωq;
        Nflavor = 2,
        aux = nothing,
    )

Add the normal-normal Matsubara polarization contribution to `Π`.

This helper does not clear `Π`; it adds into the supplied matrix.
"""
function polarization_matsubara_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    iωq::Complex;
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    return _polarization_normal_core!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        iωq;
        Nflavor = Nflavor,
        aux = aux,
    )
end


"""
    _polarization_normal_core!(Π, sbs, fields, kgrid, q, zq; Nflavor=2, aux=nothing)

Add the zero-temperature normal-normal bubble to `Π`.

For the Matsubara branch, use `zq = iωq`.

For the retarded branch, use `zq = ω + iη`.

The expression is

    Π_{αβ}(q,zq)
      = 1/(2 Nflavor Nk) sum_k sum_mn
        tr[C_{k+q,n} V_α(k+q,k) C_{k,m} V_β(k,k+q)]
        [nB(E_{k,m}) - nB(E_{k+q,n})]
        / [zq + E_{k,m} - E_{k+q,n}].

At `T = 0`, the Bose factor is evaluated for BdG poles as

    nB(E) = 0   for E > 0
    nB(E) = -1  for E < 0.
"""
function _polarization_normal_core!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    zq::Complex;
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    nϕ = length(fields)

    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π` must have size ($(nϕ), $(nϕ))."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    Vα = zeros(ComplexF64, 12, 12)
    Vβ = zeros(ComplexF64, 12, 12)

    prefactor = 1 / (2 * Nflavor * Nk)

    for k in kgrid
        kq = k + q

        ϵs_k, Vk, weights_k =
            Green_SP_normal_residues(sbs, k, aux)

        ϵs_kq, Vkq, weights_kq =
            Green_SP_normal_residues(sbs, kq, aux)

        for (iα, α) in pairs(fields)
            internal_vertices!(Vα, sbs, α, kq, k)

            for (iβ, β) in pairs(fields)
                internal_vertices!(Vβ, sbs, β, k, kq)

                accum = 0.0 + 0.0im

                for m in eachindex(ϵs_k)
                    iszero(weights_k[m]) && continue

                    Em = ϵs_k[m]
                    nb_m = _nB_T0(Em)

                    for n in eachindex(ϵs_kq)
                        iszero(weights_kq[n]) && continue

                        En = ϵs_kq[n]
                        nb_n = _nB_T0(En)

                        occdiff = nb_m - nb_n
                        iszero(occdiff) && continue

                        denom = zq + Em - En

                        coherence = _residue_vertex_trace(
                            Vkq,
                            weights_kq,
                            n,
                            Vα,
                            Vk,
                            weights_k,
                            m,
                            Vβ,
                        )

                        accum += coherence * occdiff / denom
                    end
                end

                Π[iα, iβ] += prefactor * accum
            end
        end
    end

    return Π
end


"""
    polarization_matsubara_condensate_normal!(
        Π, sbs, fields, kgrid, q, iωq;
        Nflavor = 2,
        aux = nothing,
    )

Add the mixed condensate-normal Matsubara polarization contribution to `Π`.

This function adds only

    Π_cn + Π_nc,

where one Green's-function line is the static condensed contribution and the
other is the full normal saddle-point Green's function.

The function is a no-op unless

    aux !== nothing && aux.conden_index !== nothing.

Let

    qc = kgrid[aux.conden_index]

be the condensate momentum. The implemented expression is

    Π_cn =
        1/(2Nflavor) tr[
            Gn(qc+q, iωq) Vα(qc+q,qc)
            Cc(qc)        Vβ(qc,qc+q)
        ],

    Π_nc =
        1/(2Nflavor) tr[
            Cc(qc)          Vα(qc,qc-q)
            Gn(qc-q,-iωq)  Vβ(qc-q,qc)
        ].

The elastic condensate-condensate contribution `Π_cc` is intentionally omitted.
"""
function polarization_matsubara_condensate_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    iωq::Complex;
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    _has_condensate(aux) || return Π

    nϕ = length(fields)

    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π` must have size ($(nϕ), $(nϕ))."))

    qc = kgrid[aux.conden_index]

    prefactor = 1 / (2 * Nflavor)

    Cc = _condensed_residue_matrix(sbs, qc, aux)

    Vα = zeros(ComplexF64, 12, 12)
    Vβ = zeros(ComplexF64, 12, 12)

    # Π_cn:
    #
    #   k      = qc
    #   k + q  = qc + q
    #
    # The second line is condensed, so the normal line is evaluated at iωq.

    k_c = qc
    k_n = qc + q

    Gn_plus = _normal_green_from_residues(
        sbs,
        k_n,
        iωq;
        aux = aux,
    )

    for (iα, α) in pairs(fields)
        internal_vertices!(Vα, sbs, α, k_n, k_c)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vβ, sbs, β, k_c, k_n)

            Π[iα, iβ] += prefactor * _trace12(Gn_plus * Vα * Cc * Vβ)
        end
    end

    # Π_nc:
    #
    #   k      = qc - q
    #   k + q  = qc
    #
    # The first line is condensed, so the normal line is evaluated at -iωq.

    k_n = qc - q
    k_c = qc

    Gn_minus = _normal_green_from_residues(
        sbs,
        k_n,
        -iωq;
        aux = aux,
    )

    for (iα, α) in pairs(fields)
        internal_vertices!(Vα, sbs, α, k_c, k_n)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vβ, sbs, β, k_n, k_c)

            Π[iα, iβ] += prefactor * _trace12(Cc * Vα * Gn_minus * Vβ)
        end
    end

    return Π
end


# ----------------------------------------------------------------------
# RPA kernels
# ----------------------------------------------------------------------

"""
    rpa_kernel!(K, Π0, Π)

Fill `K` with the inverse RPA propagator kernel

    K = Π0 - Π.
"""
function rpa_kernel!(
    K::AbstractMatrix{ComplexF64},
    Π0::AbstractMatrix{ComplexF64},
    Π::AbstractMatrix{ComplexF64},
)
    size(K) == size(Π0) == size(Π) ||
        throw(DimensionMismatch("`K`, `Π0`, and `Π` must have the same size."))

    @. K = Π0 - Π

    return K
end


"""
    rpa_kernel!(K, sbs, fields, kgrid, q, ω; η, Nflavor=2, aux=nothing)

Compute the retarded normal-only inverse RPA propagator kernel

    K^R(q,ω) = Π0 - Π^R_nn(q,ω).
"""
function rpa_kernel!(
    K::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    ω::Real;
    η::Real,
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    nϕ = length(fields)

    size(K) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`K` must have size ($(nϕ), $(nϕ))."))

    Π0 = similar(K)
    Π = similar(K)

    Pi0!(Π0, sbs, fields)

    polarization!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        ω;
        η = η,
        Nflavor = Nflavor,
        aux = aux,
    )

    return rpa_kernel!(K, Π0, Π)
end


"""
    rpa_kernel_matsubara!(K, Π0, Π)

Fill `K` with the Matsubara inverse RPA propagator kernel

    K = Π0 - Π.
"""
function rpa_kernel_matsubara!(
    K::AbstractMatrix{ComplexF64},
    Π0::AbstractMatrix{ComplexF64},
    Π::AbstractMatrix{ComplexF64},
)
    return rpa_kernel!(K, Π0, Π)
end


"""
    rpa_kernel_matsubara!(
        K, sbs, fields, kgrid, q, iωq;
        Nflavor = 2,
        aux = nothing,
    )

Compute the Matsubara inverse RPA propagator kernel

    K(q, iωq) = Π0 - Π(q, iωq).

This is the kernel to use for positive-stability and gauge-mode checks before
analytic continuation.
"""
function rpa_kernel_matsubara!(
    K::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    iωq::Complex;
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    nϕ = length(fields)

    size(K) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`K` must have size ($(nϕ), $(nϕ))."))

    Π0 = similar(K)
    Π = similar(K)

    Pi0!(Π0, sbs, fields)

    polarization_matsubara!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        iωq;
        Nflavor = Nflavor,
        aux = aux,
    )

    return rpa_kernel_matsubara!(K, Π0, Π)
end


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@inline function _metric_sign(l::Int)
    return l <= 6 ? 1.0 : -1.0
end


"""
    _residue_vertex_trace(Vq, wq, n, A, Vk, wk, m, B)

Compute

    tr[C_{q,n} A C_{k,m} B]

using the rank-one residue form

    C_l = weight_l * s_l * v_l v_l†.

This avoids explicitly allocating residue matrices.
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

    sn = _metric_sign(n)
    sm = _metric_sign(m)

    coeff = wq[n] * sn * wk[m] * sm

    return coeff * dot(vn, A * vm) * dot(vm, B * vn)
end


"""
    _trace12(A)

Return the trace of a 12×12 matrix.
"""
@inline function _trace12(A::AbstractMatrix)
    size(A) == (12, 12) ||
        throw(DimensionMismatch("Expected a 12 × 12 matrix."))

    out = zero(eltype(A))

    @inbounds for i in 1:12
        out += A[i, i]
    end

    return out
end


"""
    _residue_matrix_from_residues(V, weights)

Construct the residue matrix

    C = sum_l weight_l * s_l * v_l v_l†

from a residue decomposition, where `s_l = +1` for positive BdG poles
`l <= 6` and `s_l = -1` for negative BdG poles `l > 6`.
"""
function _residue_matrix_from_residues(
    V::AbstractMatrix,
    weights::AbstractVector,
)
    size(V, 1) == 12 ||
        throw(DimensionMismatch("Residue eigenvector matrix must have 12 rows."))

    C = zeros(ComplexF64, 12, 12)

    for l in eachindex(weights)
        iszero(weights[l]) && continue

        v = @view V[:, l]
        s = _metric_sign(l)

        C .+= weights[l] * s * (v * v')
    end

    return C
end


"""
    _normal_green_from_residues(sbs, k, z; aux=nothing)

Construct the full normal saddle-point Green's function from the pole expansion,

    Gn(k,z) = sum_l C_l(k) / (z - E_l(k)),

where

    C_l = weight_l * s_l * v_l v_l†.

This includes both positive and negative BdG poles returned by
`Green_SP_normal_residues`.
"""
function _normal_green_from_residues(
    sbs::SchwingerBosonSystem,
    k,
    z::Complex;
    aux::Union{Nothing, CondensationAux} = nothing,
)
    ϵs, V, weights =
        Green_SP_normal_residues(sbs, k, aux)

    G = zeros(ComplexF64, 12, 12)

    for l in eachindex(ϵs)
        iszero(weights[l]) && continue

        v = @view V[:, l]
        s = _metric_sign(l)

        G .+= weights[l] * s * (v * v') / (z - ϵs[l])
    end

    return G
end


"""
    _condensed_residue_matrix(sbs, qc, aux)

Construct the static condensate residue matrix `Cc` from
`Green_SP_condensed_residues`.

All nonzero condensed residues returned by `Green_SP_condensed_residues` are
included.
"""
function _condensed_residue_matrix(
    sbs::SchwingerBosonSystem,
    qc,
    aux::CondensationAux,
)
    _, Vc, weights_c =
        Green_SP_condensed_residues(sbs, qc, aux)

    return _residue_matrix_from_residues(Vc, weights_c)
end


"""
    _nB_T0(E)

Zero-temperature Bose factor for BdG pole energies.

For positive poles, `nB(E) = 0`.
For negative poles, `nB(E) = -1`.

If a nonzero-weight pole is numerically at zero energy, the normal-only
polarization is ill-defined and the condensate treatment should be used.
"""
@inline function _nB_T0(E::Real)
    atol = 1e-12

    if E > atol
        return 0.0
    elseif E < -atol
        return -1.0
    else
        throw(ArgumentError(
            "Encountered a zero-energy pole in the normal bubble. " *
            "Pass condensation data through `aux` so pinned condensate modes are removed, " *
            "or treat the condensate contribution explicitly."
        ))
    end
end


function _field_index(fields::AbstractVector{InternalField}, target::InternalField)
    for (i, field) in pairs(fields)
        field == target && return i
    end

    throw(ArgumentError("Field `$target` was not found in the internal-field basis."))
end


@inline _has_condensate(aux::Nothing) = false

@inline function _has_condensate(aux::CondensationAux)
    return aux.conden_index !== nothing
end