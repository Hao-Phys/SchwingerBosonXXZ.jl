# src/PathIntegral/RPAPropagator.jl

"""
    InternalField

Label for one reduced internal auxiliary-field vertex in the fluctuation-sector
basis of the current note.

Column sector at four-momentum `q`:

    ϕ(q) = { W(q), Wbar(-q), λ(q) }.

Row sector paired with it:

    ϕ†(q) = { Wbar(q), W(-q), λ(-q) }.

The stored label is the column-sector label. For HS fields,

- `kind = :W` or `:Wbar`,
- `channel ∈ (:A, :B, :C, :D)`,
- `a = 1,2,3`,
- `δ = 1,2,3`.

For constraint fields, use `kind = :λ`, `channel = :none`, and `δ = 0`.
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

Important convention: in the sector `q`, the label `:Wbar` means the actual
Fourier field `Wbar(-q)`, not `Wbar(q)`.
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

Fill `V` with the reduced **column-side** internal vertex associated with
`field`.

For `field.kind === :λ`, this dispatches to

    internal_vertices!(V, :λ, field.a)

For HS fields, this dispatches to

    internal_vertices!(V, sbs, field.kind, field.channel, field.a, field.δ, k, p)

The returned vertex is reduced: the Fourier normalization and the
momentum-frequency Kronecker delta are not included.

This is the derivative with respect to the actual Fourier field contained in
the column sector variable. Thus `field.kind = :Wbar` in sector `q` denotes
the derivative with respect to `Wbar(-q)`.
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

"""
    row_internal_vertices!(V, sbs, field::InternalField, k, p)

Fill `V` with the reduced **row-side** internal vertex associated with
`field`.

The stored `field` label is still the column-sector label. The row partner is
mapped according to the complex-Gaussian sector convention:

- row label `:W` means the actual field `Wbar(q)`;
- row label `:Wbar` means the actual field `W(-q)`;
- row label `:λ` means the actual field `λ(-q)`.

For a row vertex appearing as `V†_α(k,p)`, the arguments are chosen so that
`k - p` is the row-side transfer. This function labels the row-side derivative
vertex. It does not take a Hermitian conjugate.
"""
function row_internal_vertices!(
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
    elseif field.kind === :W
        return internal_vertices!(
            V,
            sbs,
            :Wbar,
            field.channel,
            field.a,
            field.δ,
            k,
            p,
        )
    elseif field.kind === :Wbar
        return internal_vertices!(
            V,
            sbs,
            :W,
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

Fill `Π0` with the bare auxiliary-field kernel in the sector basis

    ϕ(q) = (W(q), Wbar(-q), λ(q)),
    ϕ†(q) = (Wbar(q), W(-q), λ(-q)).

With this row-column convention, the bare kernel is diagonal in the HS
integration variables:

    Π0[W,    W   ] = κ / 2,
    Π0[Wbar, Wbar] = κ / 2.

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
        if field.kind === :W || field.kind === :Wbar
            κ, _ = _κ_s(sbs, field.channel, field.δ)
            Π0[i, i] += κ / 2
        end
    end

    return Π0
end

# ----------------------------------------------------------------------
# Polarization operator
# ----------------------------------------------------------------------

"""
    polarization!(Π, sbs, fields, kgrid, q, z; Nflavor=2, aux=nothing,
                  force_T0_bose_factor=false)

Fill `Π` with the polarization operator at complex external frequency `z`.

The row-column convention is

    K_{αβ}(q,z) = Π0_{αβ} - Π_{αβ}(q,z),

where `α` is a row-sector label and `β` is a column-sector label. The column
vertex is `V_β(k+q,k)`, while the row vertex is `V†_α(k,k+q)`; the dagger is
only a row-side label and is not a Hermitian conjugate.

The normal-normal contribution is always included. If `aux` contains a
condensate, this function also adds the mixed condensate-normal terms. The
purely elastic condensate-condensate contribution is intentionally omitted.
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
    force_T0_bose_factor::Bool = false,
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
        force_T0_bose_factor = force_T0_bose_factor,
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
        force_T0_bose_factor = force_T0_bose_factor,
    )

    return Π
end

"""
    polarization_normal!(Π, sbs, fields, kgrid, q, z;
                         Nflavor=2, aux=nothing,
                         force_T0_bose_factor=false)

Add the normal-normal polarization contribution to `Π`.

This helper does not clear `Π`; it adds into the supplied matrix. The finite
grid represents the magnetic Brillouin-zone sum with the explicit prefactor
`1 / (2 * Nflavor * Nu)`, implemented as `1 / (2 * Nflavor * length(kgrid))`.
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
    force_T0_bose_factor::Bool = false,
)
    nϕ = length(fields)
    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π` must have size ($(nϕ), $(nϕ))."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    βtemp = _inverse_temperature(sbs)

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    prefactor = 1 / (2 * Nflavor * Nk)

    for k in kgrid
        kq = k + q

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iα, α) in pairs(fields)
            row_internal_vertices!(Vrow, sbs, α, k, kq)

            for (iβ, β) in pairs(fields)
                internal_vertices!(Vcol, sbs, β, kq, k)

                accum = 0.0 + 0.0im

                for m in eachindex(ϵs_k)
                    iszero(weights_k[m]) && continue
                    Em = ϵs_k[m]
                    nb_m = _pole_bose(Em, βtemp, force_T0_bose_factor)

                    for n in eachindex(ϵs_kq)
                        iszero(weights_kq[n]) && continue
                        En = ϵs_kq[n]
                        nb_n = _pole_bose(En, βtemp, force_T0_bose_factor)

                        occdiff = nb_n - nb_m
                        iszero(occdiff) && continue

                        denom = z + Em - En

                        coherence = _residue_vertex_trace(
                            Vkq, weights_kq, n, Vcol,
                            Vk, weights_k, m, Vrow,
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
    polarization_condensate_normal!(
        Π, sbs, fields, kgrid, q, z;
        Nflavor = 2,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Add the mixed condensate-normal contribution to `Π`.

This function adds the two mixed pieces in which exactly one Green's-function
line is the condensate contribution and the other is the normal saddle-point
Green's function. It is a no-op unless

    aux !== nothing && aux.conden_index !== nothing.

The row-column convention is the same as in `polarization_normal!`: the row
vertex is `V†_α(k,k+q)` and the column vertex is `V_β(k+q,k)`.

The purely elastic condensate-condensate contribution `Π_cc` is intentionally
omitted.
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
    force_T0_bose_factor::Bool = false,
)
    _has_condensate(aux) || return Π

    nϕ = length(fields)
    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π` must have size ($(nϕ), $(nϕ))."))

    Nu = length(kgrid)
    Nu > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    qc = kgrid[aux.conden_index]
    βtemp = _inverse_temperature(sbs)

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    prefactor = 1 / (2 * Nflavor * Nu)

    # Condensate on the k line, normal propagator on the k + q line.
    kc = qc
    kn = qc + q

    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)
    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kc, kn)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vcol, sbs, β, kn, kc)

            accum = 0.0 + 0.0im

            for m in eachindex(ϵs_c)
                iszero(weights_c[m]) && continue
                Em = ϵs_c[m]
                nb_m = _nB_T0(real(Em))

                for n in eachindex(ϵs_n)
                    iszero(weights_n[n]) && continue
                    En = ϵs_n[n]
                    nb_n = _pole_bose(En, βtemp, force_T0_bose_factor)

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vn, weights_n, n, Vcol,
                        Vc, weights_c, m, Vrow,
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Π[iα, iβ] += prefactor * accum
        end
    end

    # Normal propagator on the k line, condensate on the k + q line.
    kn = qc - q
    kc = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kn, kc)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vcol, sbs, β, kc, kn)

            accum = 0.0 + 0.0im

            for m in eachindex(ϵs_n)
                iszero(weights_n[m]) && continue
                Em = ϵs_n[m]
                nb_m = _pole_bose(Em, βtemp, force_T0_bose_factor)

                for n in eachindex(ϵs_c)
                    iszero(weights_c[n]) && continue
                    En = ϵs_c[n]
                    nb_n = _nB_T0(real(En))

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vc, weights_c, n, Vcol,
                        Vn, weights_n, m, Vrow,
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Π[iα, iβ] += prefactor * accum
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

Here `K[α, β]` has row index `α` and column index `β` in the sector convention
of the current note.
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
    rpa_kernel!(K, sbs, fields, kgrid, q, z;
                Nflavor=2,
                aux=nothing,
                force_T0_bose_factor=false)

Compute the inverse RPA propagator kernel

    K(q,z) = Π0 - Π(q,z).
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
    force_T0_bose_factor::Bool = false,
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
        force_T0_bose_factor = force_T0_bose_factor,
    )

    return rpa_kernel!(K, Π0, Π)
end

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

@inline function _inverse_temperature(sbs::SchwingerBosonSystem)
    T = sbs.T
    return iszero(T) ? Inf : inv(T)
end

"""
    _nB_T0(E)

Zero-temperature Bose factor for BdG pole energies.

For positive poles, `nB(E) = 0`. For negative poles, `nB(E) = -1`.
If a nonzero-weight pole is numerically at zero energy, the normal-only bubble
is ill-defined and the condensate treatment should be used.
"""
@inline function _nB_T0(E::Real)
    atol = 1e-12

    if E > atol
        return 0.0
    elseif E < -atol
        return -1.0
    else
        throw(ArgumentError(
            "Encountered a zero-energy pole in a normal bubble. " *
            "Pass condensation data through `aux` so pinned condensate modes " *
            "are removed, or treat the condensate contribution explicitly."
        ))
    end
end

"""
    _nB_BdG(E, β)

Finite-temperature Bose factor for a BdG pole energy `E`.

For very low temperature or very large `|βE|`, the result is evaluated by a
stable limiting expression. Negative BdG poles correctly approach `-1` as
`T -> 0`.
"""
@inline function _nB_BdG(E::Number, β::Real)
    Er = real(E)

    if !isfinite(β)
        return _nB_T0(Er)
    end

    x = β * Er

    if x > 700
        return 0.0
    elseif x < -700
        return -1.0
    elseif abs(x) < 1e-12
        throw(ArgumentError(
            "Encountered a zero-energy pole in a finite-temperature Bose factor."
        ))
    else
        return 1 / expm1(x)
    end
end

@inline function _pole_bose(E::Number, β::Real, force_T0_bose_factor::Bool)
    return force_T0_bose_factor ? _nB_T0(real(E)) : _nB_BdG(E, β)
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