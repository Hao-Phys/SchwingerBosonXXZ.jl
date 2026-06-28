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

Fill `V` with the reduced column-side internal vertex associated with `field`.
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

Fill `V` with the reduced row-side internal vertex associated with `field`.

The stored `field` label is still the column-sector label. The row partner is
mapped according to the complex-Gaussian sector convention:

- row label `:W` means the actual field `Wbar(q)`;
- row label `:Wbar` means the actual field `W(-q)`;
- row label `:λ` means the actual field `λ(-q)`.
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
    polarization!(Π, sbs, fields, kgrid, q, z; Nflavor=2, aux=nothing)

Fill `Π` with the polarization operator at complex external frequency `z`.

With the split

    G = G_normal + G_condensed,

the RPA kernel must use the complete bubble

    Π = Π_nn + Π_cn + Π_nc + Π_cc.

For `selection_kind === :finite_size_minimum`, this is only an algebraic split
of ordinary finite-size poles. The selected poles are reinserted with weight
one and with the same `1 / Nk` momentum normalization as the original bubble.

All BdG poles, normal and selected, use the same finite-temperature occupation
factor `_pole_bose(E, β)`.
"""
function polarization!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing, SpectralCondensationAux} = nothing,
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

    polarization_condensate_condensate!(
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
    polarization_normal!(Π, sbs, fields, kgrid, q, z;
                         Nflavor=2, aux=nothing)

Add the normal-normal polarization contribution to `Π`.
"""
function polarization_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing, SpectralCondensationAux} = nothing,
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
                    nb_m = _pole_bose(Em, βtemp)

                    for n in eachindex(ϵs_kq)
                        iszero(weights_kq[n]) && continue

                        En = ϵs_kq[n]
                        nb_n = _pole_bose(En, βtemp)

                        occdiff = nb_n - nb_m
                        iszero(occdiff) && continue

                        denom = z + Em - En

                        coherence = _residue_vertex_trace(
                            Vkq,
                            weights_kq,
                            n,
                            Vcol,
                            Vk,
                            weights_k,
                            m,
                            Vrow,
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
    )

Add the mixed selected-normal pieces to the polarization operator,

    G_c(k) G_n(k + q) + G_n(k) G_c(k + q).

The selected sector is specified by `SpectralCondensationAux` and is evaluated
with `Green_SP_condensed_residues`.

The selected pole uses the same finite-temperature BdG occupation as every
other pole. The selected sector differs only through its residue weight and the
branch-dependent collapsed momentum-sum factor.

The branch-dependent collapsed momentum-sum factor is

    finite_size_minimum: 1
    pinned:              L^2

implemented through `_condensate_sum_factor(aux, L)`. Therefore the prefactor is

    _condensate_sum_factor(aux, L) / (2 * Nflavor * Nk).

For `:finite_size_minimum`, this gives the same normalization as the original
finite-size momentum sum. For `:pinned`, this gives the macroscopic selected-line
normalization.
"""
function polarization_condensate_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing,SpectralCondensationAux} = nothing,
)
    aux === nothing && return Π
    isempty(aux.conden_band_indices) && return Π

    nϕ = length(fields)
    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π` must have size ($(nϕ), $(nϕ))."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    (; L) = sbs

    qc = _spectral_condensation_momentum(aux, L)
    βtemp = _inverse_temperature(sbs)

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    prefactor = _condensate_sum_factor(aux, L) / (2 * Nflavor * Nk)

    # ------------------------------------------------------------------
    # Selected pole on the k line, normal propagator on the k + q line.
    # ------------------------------------------------------------------

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
                nb_m = _pole_bose(Em, βtemp)

                for n in eachindex(ϵs_n)
                    iszero(weights_n[n]) && continue

                    En = ϵs_n[n]
                    nb_n = _pole_bose(En, βtemp)

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vn,
                        weights_n,
                        n,
                        Vcol,
                        Vc,
                        weights_c,
                        m,
                        Vrow,
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Π[iα, iβ] += prefactor * accum
        end
    end

    # ------------------------------------------------------------------
    # Normal propagator on the k line, selected pole on the k + q line.
    # ------------------------------------------------------------------

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
                nb_m = _pole_bose(Em, βtemp)

                for n in eachindex(ϵs_c)
                    iszero(weights_c[n]) && continue

                    En = ϵs_c[n]
                    nb_n = _pole_bose(En, βtemp)

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vc,
                        weights_c,
                        n,
                        Vcol,
                        Vn,
                        weights_n,
                        m,
                        Vrow,
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Π[iα, iβ] += prefactor * accum
        end
    end

    return Π
end

"""
    polarization_condensate_condensate!(
        Π, sbs, fields, kgrid, q, z;
        Nflavor = 2,
        aux = nothing,
    )

Add the selected-selected `G_c G_c` piece to the polarization operator.

For `:finite_size_minimum`, this piece is required for the split

    G_full = G_normal + G_condensed

to exactly reconstruct the original finite-size bubble.

For `:pinned`, this is the selected-sector counterpart of the macroscopic
condensate-condensate polarization.

All selected poles use the same finite-temperature BdG occupation as ordinary
poles. The selected-sector distinction is only the residue weight and collapsed
momentum-sum factor.

The branch-dependent prefactor is

    _condensate_sum_factor(aux, L)^2 / (2 * Nflavor * Nk).

Thus:

    finite_size_minimum: 1 / (2 * Nflavor * Nk)
    pinned:              Nk / (2 * Nflavor)

assuming the standard `Nk = L^2` grid.
"""
function polarization_condensate_condensate!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing,SpectralCondensationAux} = nothing,
)
    aux === nothing && return Π
    isempty(aux.conden_band_indices) && return Π

    nϕ = length(fields)
    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch("`Π` must have size ($(nϕ), $(nϕ))."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    (; L) = sbs

    βtemp = _inverse_temperature(sbs)

    qc = _spectral_condensation_momentum(aux, L)

    kc = qc
    kq = qc + q

    _same_momentum_mod1(kq, qc) || return Π

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    ϵs_k, Vk, weights_k = Green_SP_condensed_residues(sbs, kc, aux)
    ϵs_kq, Vkq, weights_kq = Green_SP_condensed_residues(sbs, kq, aux)

    condensate_sum_factor = _condensate_sum_factor(aux, L)
    prefactor = condensate_sum_factor^2 / (2 * Nflavor * Nk)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kc, kq)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vcol, sbs, β, kq, kc)

            accum = 0.0 + 0.0im

            for m in eachindex(ϵs_k)
                iszero(weights_k[m]) && continue

                Em = ϵs_k[m]
                nb_m = _pole_bose(Em, βtemp)

                for n in eachindex(ϵs_kq)
                    iszero(weights_kq[n]) && continue

                    En = ϵs_kq[n]
                    nb_n = _pole_bose(En, βtemp)

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vkq,
                        weights_kq,
                        n,
                        Vcol,
                        Vk,
                        weights_k,
                        m,
                        Vrow,
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
"""
function rpa_kernel!(
    K::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing, SpectralCondensationAux} = nothing,
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