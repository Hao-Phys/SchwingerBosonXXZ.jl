# src/PathIntegral/RPAPropagator.jl

"""
    InternalField

Label for one reduced internal auxiliary-field vertex.

Fields are

- `kind = :W` or `:Wbar` for Hubbard-Stratonovich fields,
- `kind = :λ` for constraint fields.

For HS fields, `channel ∈ (:A, :B, :C, :D)`, `a = 1,2,3`, and `δ = 1,2,3`.

For constraint fields, use `channel = :none` and `δ = 0`.

Sector convention:

- in sector `q`, `InternalField(:W, X, a, δ)` means the actual field
  `W^X_{a,δ}(q)`;
- in sector `q`, `InternalField(:Wbar, X, a, δ)` means the actual field
  `Wbar^X_{a,δ}(-q)`;
- in sector `q`, `InternalField(:λ, :none, a, 0)` means the actual field
  `λ_a(q)`.
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

The returned labels are sector labels. In sector `q`, the field `:Wbar`
represents the actual Fourier field `Wbar(-q)`, not `Wbar(q)`.
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

The momentum transfer of the reduced vertex is `k - p`. Therefore, when this
function is called as `internal_vertices!(V, sbs, α, k + q, k)`, the field
label `α` is interpreted as a sector-`q` field. For anomalous channels, the
called vertex may contain both the direct anomalous entry and its transposed
reversed Nambu-completion entry; both belong to the same sector momentum
`k - p`.
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

The matrix is stored in the same row/column convention as the RPA kernel:
rows belong to the sector `-q`, columns belong to the sector `q`. Since the
bare kernel is local and symmetric in the `W`, `Wbar` labels, this convention
does not change the entries, but it fixes how `Π0` combines with `Π`.
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
# Polarization operator
# ----------------------------------------------------------------------

"""
    polarization!(Π, sbs, fields, kgrid, q, z; Nflavor=2, aux=nothing)

Fill `Π` with the zero-temperature polarization operator at complex external
frequency `z`.

The normal-normal contribution is always included. If `aux` contains a
condensate, this function also adds the mixed condensate-normal terms
Π_cn + Π_nc.

The purely elastic condensate-condensate contribution `Π_cc` is intentionally
omitted.

Storage convention:

    Π[β, α](q)

where the row index `β` belongs to the sector `-q`, and the column index `α`
belongs to the sector `q`.
"""
function polarization!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing, CondensationAux} = nothing,
)
    fill!(Π, 0.0 + 0.0im)

    polarization_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z;
        Nflavor = Nflavor,
        aux = aux,
    )

    polarization_condensate_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z;
        Nflavor = Nflavor,
        aux = aux,
    )

    return Π
end


"""
    polarization_normal!(Π, sbs, fields, kgrid, q, z; Nflavor=2, aux=nothing)

Add the normal-normal polarization contribution to `Π`.

This helper does not clear `Π`; it adds into the supplied matrix.

The matrix is stored as `Π[β, α](q)`, where `α` labels the sector-`q` vertex
`V_α(k + q, k)`, and `β` labels the sector-`-q` vertex `V_β(k, k + q)`.
"""
function polarization_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
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

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iα, α) in pairs(fields)
            # Sector q: transfer (k + q) - k = q.
            internal_vertices!(Vα, sbs, α, kq, k)

            for (iβ, β) in pairs(fields)
                # Sector -q: transfer k - (k + q) = -q.
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

                        occdiff = nb_n - nb_m
                        iszero(occdiff) && continue

                        denom = z + Em - En

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

                # Store as Π[β, α](q): row sector -q, column sector q.
                Π[iβ, iα] += prefactor * accum
            end
        end
    end

    return Π
end


"""
    polarization_condensate_normal!(
        Π, sbs, fields, kgrid, q, z; Nflavor = 2, aux = nothing,
    )

Add the mixed condensate-normal polarization contribution to `Π`.

This function adds only Π_cn + Π_nc, where one Green's-function line is the
static condensed contribution and the other is the full normal saddle-point
Green's function. The function is a no-op unless

    aux !== nothing && aux.conden_index !== nothing.

Let `qc = kgrid[aux.conden_index]` be the condensate momentum.

The implemented expression is

    Π_cn = 1/(2Nflavor)
        tr[ G_normal(qc+q, z) Vα(qc+q,qc) Cc(qc) Vβ(qc,qc+q) ],

    Π_nc = 1/(2Nflavor)
        tr[ Cc(qc) Vα(qc,qc-q) G_normal(qc-q,-z) Vβ(qc-q,qc) ].

In both terms, `α` labels the sector-`q` vertex and `β` labels the sector-`-q`
vertex. The matrix is stored as `Π[β, α](q)`, with row sector `-q` and column
sector `q`.

The elastic condensate-condensate contribution `Π_cc` is intentionally omitted.
"""
function polarization_condensate_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
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
    #     k = qc
    #     k + q = qc + q
    #
    # The second line is condensed, so the normal line is evaluated at z.
    #
    # α is sector q; β is sector -q.
    k_c = qc
    k_n = qc + q
    Gn_plus = Green_SP_normal(sbs, k_n, z, aux)

    for (iα, α) in pairs(fields)
        internal_vertices!(Vα, sbs, α, k_n, k_c)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vβ, sbs, β, k_c, k_n)

            # Store as Π[β, α](q): row sector -q, column sector q.
            Π[iβ, iα] += prefactor * tr(Gn_plus * Vα * Cc * Vβ)
        end
    end

    # Π_nc:
    #
    #     k = qc - q
    #     k + q = qc
    #
    # The first line is condensed, so the normal line is evaluated at -z.
    #
    # α is sector q; β is sector -q.
    k_n = qc - q
    k_c = qc
    Gn_minus = Green_SP_normal(sbs, k_n, -z, aux)

    for (iα, α) in pairs(fields)
        internal_vertices!(Vα, sbs, α, k_c, k_n)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vβ, sbs, β, k_n, k_c)

            # Store as Π[β, α](q): row sector -q, column sector q.
            Π[iβ, iα] += prefactor * tr(Cc * Vα * Gn_minus * Vβ)
        end
    end

    return Π
end


# ----------------------------------------------------------------------
# RPA kernel
# ----------------------------------------------------------------------

"""
    rpa_kernel!(K, Π0, Π)

Fill `K` with the inverse RPA propagator kernel

    K = Π0 - Π.

Storage convention:

    K[β, α](q)

where the row index `β` belongs to the sector `-q`, and the column index `α`
belongs to the sector `q`.
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
    rpa_kernel!(K, sbs, fields, kgrid, q, z; Nflavor=2, aux=nothing)

Compute the inverse RPA propagator kernel

    K(q,z) = Π0 - Π(q,z).

The output is stored as `K[β, α](q)`, with row sector `-q` and column sector
`q`. Therefore solving `K \\ Sminus` gives the sector-`q` vector that contracts
with `Splus`.
"""
function rpa_kernel!(
    K::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
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
        z;
        Nflavor = Nflavor,
        aux = aux,
    )

    return rpa_kernel!(K, Π0, Π)
end


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

"""
    _nB_T0(E)

Zero-temperature Bose factor for BdG pole energies.

For positive poles, `nB(E) = 0`. For negative poles, `nB(E) = -1`.

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