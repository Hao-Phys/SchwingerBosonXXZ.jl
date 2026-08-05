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
            p
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
            p
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
            p
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
        throw(DimensionMismatch(
            "`Π0` must have size ($(nϕ), $(nϕ))."
        ))

    fill!(Π0, 0.0 + 0.0im)

    for (i, field) in pairs(fields)
        if field.kind === :W || field.kind === :Wbar
            κ, _ = _κ_s(sbs, field.channel, field.a)
            Π0[i, i] += κ / 2
        end
    end

    return Π0
end


# ----------------------------------------------------------------------
# Selected-sector helpers
# ----------------------------------------------------------------------

"""
    _full_sp_residues(sbs, q)

Return the complete unit-residue BdG spectral data at momentum `q`.
No selected bands are removed.
"""
function _full_sp_residues(
    sbs::SchwingerBosonSystem,
    q::Vec3,
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

    return ϵs, V, weights
end

# ----------------------------------------------------------------------
# Polarization operator
# ----------------------------------------------------------------------

"""
    polarization!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Fill `Π` with the ordinary unit-residue saddle-point polarization.

When a selected sector is present, the Green's function is split algebraically
as

    G_SP = G_normal + G_selected,

but every selected pole is reconstructed with unit residue. The enhanced
occupation `ξ` is not included in `Π`; it is added separately to the inverse
RPA kernel by `active_constraint_kernel!`.
"""
function polarization!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    fill!(Π, 0.0 + 0.0im)

    polarization_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = Nflavor,
    )

    polarization_condensate_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = Nflavor,
    )

    polarization_condensate_condensate!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = Nflavor,
    )

    return Π
end


"""
    polarization_normal!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Add the normal-normal ordinary saddle-point polarization.
"""
function polarization_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    nϕ = length(fields)

    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch(
            "`Π` must have size ($(nϕ), $(nϕ))."
        ))

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
                            Vrow
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
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Reconstruct the ordinary mixed selected-normal saddle-point polarization,

    G_selected G_normal + G_normal G_selected,

using unit selected-pole weights and the ordinary `1 / Nk` momentum
normalization.

No enhanced occupation `ξ` is included here.
"""
function polarization_condensate_normal!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    isempty(aux.conden_band_indices) && return Π

    nϕ = length(fields)

    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch(
            "`Π` must have size ($(nϕ), $(nϕ))."
        ))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    qc = _spectral_condensation_momentum(aux, sbs.L)
    βtemp = _inverse_temperature(sbs)

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    prefactor = 1 / (2 * Nflavor * Nk)

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
                        Vrow
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Π[iα, iβ] += prefactor * accum
        end
    end

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
                        Vrow
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
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Reconstruct the ordinary selected-selected saddle-point polarization using
unit selected-pole weights.

This contribution is nonzero only when the external momentum maps the
selected momentum back to itself. No enhanced occupation `ξ` is included.
"""
function polarization_condensate_condensate!(
    Π::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    isempty(aux.conden_band_indices) && return Π

    nϕ = length(fields)

    size(Π) == (nϕ, nϕ) ||
        throw(DimensionMismatch(
            "`Π` must have size ($(nϕ), $(nϕ))."
        ))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    qc = _spectral_condensation_momentum(aux, sbs.L)

    kc = qc
    kq = qc + q

    _same_momentum_mod1(kq, qc) || return Π

    βtemp = _inverse_temperature(sbs)

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    ϵs_k, Vk, weights_k = Green_SP_condensed_residues(sbs, kc, aux)
    ϵs_kq, Vkq, weights_kq = Green_SP_condensed_residues(sbs, kq, aux)

    prefactor = 1 / (2 * Nflavor * Nk)

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
                        Vrow
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
# Active-constraint curvature
# ----------------------------------------------------------------------

"""
    active_constraint_kernel!(
        K,
        sbs,
        fields,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Add the fixed-`ξ` soft-minimum contribution to the inverse RPA kernel.

The code-level spectral curvature accumulated below has the convention

    curvature = -Nflavor * Γξ,

so the kernel update is

    K += curvature / Nflavor.

Only active positive-energy selected modes occur in the outer sum. The
intermediate state runs over the complete BdG spectrum, except that the full
active positive-energy subspace is excluded whenever the intermediate momentum
coincides with the condensate momentum.

No Bose occupation factors occur in this contribution.
"""
function active_constraint_kernel!(
    K::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    aux.selection_kind === :pinned || return K
    isempty(aux.conden_band_indices) && return K

    nϕ = length(fields)

    size(K) == (nϕ, nϕ) ||
        throw(DimensionMismatch(
            "`K` must have size ($(nϕ), $(nϕ))."
        ))

    qc = _spectral_condensation_momentum(aux, sbs.L)

    Vrow = zeros(ComplexF64, 12, 12)
    Vcol = zeros(ComplexF64, 12, 12)

    kc = qc

    ϵs_c, Vc, _ = Green_SP_condensed_residues(sbs, kc, aux)

    active_weights = aux.active_positive_weights
    active_mask = active_weights .> 0.0

    unit_active_weights = zeros(Float64, length(ϵs_c))

    # ------------------------------------------------------------------
    # First ordering:
    #
    #     active mode at qc
    #         -> intermediate mode at qc + q
    #         -> active mode at qc
    #
    # Denominator:
    #
    #     Ei - En + z
    # ------------------------------------------------------------------

    kn = qc + q

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)

    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kc, kn)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vcol, sbs, β, kn, kc)

            curvature = 0.0 + 0.0im

            for i in eachindex(ϵs_c)
                ξi = active_weights[i]
                iszero(ξi) && continue

                Ei = ϵs_c[i]

                fill!(unit_active_weights, 0.0)
                unit_active_weights[i] = 1.0

                for n in eachindex(ϵs_n)
                    if exclude_active_intermediate &&
                       n <= length(active_mask) &&
                       active_mask[n]
                        continue
                    end

                    En = ϵs_n[n]
                    denom = Ei - En + z

                    coherence = _residue_vertex_trace(
                        Vn,
                        weights_n,
                        n,
                        Vcol,
                        Vc,
                        unit_active_weights,
                        i,
                        Vrow
                    )

                    curvature += ξi * coherence / denom
                end
            end

            K[iα, iβ] += curvature / Nflavor
        end
    end

    # ------------------------------------------------------------------
    # Second ordering:
    #
    #     active mode at qc
    #         -> intermediate mode at qc - q
    #         -> active mode at qc
    #
    # Denominator:
    #
    #     Ei - Em - z
    # ------------------------------------------------------------------

    kn = qc - q

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)

    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kn, kc)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vcol, sbs, β, kc, kn)

            curvature = 0.0 + 0.0im

            for i in eachindex(ϵs_c)
                ξi = active_weights[i]
                iszero(ξi) && continue

                Ei = ϵs_c[i]

                fill!(unit_active_weights, 0.0)
                unit_active_weights[i] = 1.0

                for m in eachindex(ϵs_n)
                    if exclude_active_intermediate &&
                       m <= length(active_mask) &&
                       active_mask[m]
                        continue
                    end

                    Em = ϵs_n[m]
                    denom = Ei - Em - z

                    coherence = _residue_vertex_trace(
                        Vc,
                        unit_active_weights,
                        i,
                        Vcol,
                        Vn,
                        weights_n,
                        m,
                        Vrow
                    )

                    curvature += ξi * coherence / denom
                end
            end

            K[iα, iβ] += curvature / Nflavor
        end
    end

    return K
end


# ----------------------------------------------------------------------
# RPA kernel
# ----------------------------------------------------------------------

"""
    rpa_kernel!(K, Π0, Π)

Fill `K` with

    K = Π0 - Π.
"""
function rpa_kernel!(
    K::AbstractMatrix{ComplexF64},
    Π0::AbstractMatrix{ComplexF64},
    Π::AbstractMatrix{ComplexF64},
)
    size(K) == size(Π0) == size(Π) ||
        throw(DimensionMismatch(
            "`K`, `Π0`, and `Π` must have the same size."
        ))

    @. K = Π0 - Π

    return K
end


"""
    rpa_kernel!(
        K,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Compute the fixed-`ξ` inverse RPA kernel

    K(q,z) = Π0 - Πordinary(q,z) + Γξ(q,z),

where `Πordinary` uses only unit BdG pole residues and `Γξ` is the
soft-minimum active-constraint contribution.
"""
function rpa_kernel!(
    K::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    nϕ = length(fields)

    size(K) == (nϕ, nϕ) ||
        throw(DimensionMismatch(
            "`K` must have size ($(nϕ), $(nϕ))."
        ))

    Π0 = similar(K)
    Π = similar(K)

    Pi0!(Π0, sbs, fields)

    polarization!(
        Π,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = Nflavor,
    )

    rpa_kernel!(K, Π0, Π)

    active_constraint_kernel!(
        K,
        sbs,
        fields,
        q,
        z,
        aux;
        Nflavor = Nflavor,
    )

    return K
end


# ----------------------------------------------------------------------
# Gauge-mode vectors
# ----------------------------------------------------------------------

@inline function _gauge_eta(X::Symbol)
    if X === :A || X === :D
        return 1.0
    elseif X === :B || X === :C
        return -1.0
    else
        throw(ArgumentError("Unknown HS channel `$X`."))
    end
end


@inline function _canonical_mean_field_value(
    sbs::SchwingerBosonSystem,
    X::Symbol,
    a::Int,
)
    @boundscheck @assert 1 <= a <= 3

    if X === :A
        return sbs.mean_fields[a]
    elseif X === :B
        return sbs.mean_fields[a + 3]
    elseif X === :C
        return sbs.mean_fields[a + 6]
    elseif X === :D
        return sbs.mean_fields[a + 9]
    else
        throw(ArgumentError("Unknown HS channel `$X`."))
    end
end


@inline function _hs_saddle_value_for_gauge_mode(
    sbs::SchwingerBosonSystem,
    kind::Symbol,
    X::Symbol,
    a::Int,
)
    mf = _canonical_mean_field_value(sbs, X, a)

    if kind === :W
        if X === :A || X === :D
            return mf
        elseif X === :B || X === :C
            return -mf
        else
            throw(ArgumentError("Unknown HS channel `$X`."))
        end
    elseif kind === :Wbar
        return conj(mf)
    else
        throw(ArgumentError(
            "Expected `:W` or `:Wbar`; got `$kind`."
        ))
    end
end


function gauge_mode_vector!(
    φD::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q::Vec3,
    z::Number,
    θ::AbstractVector,
)
    nϕ = length(fields)

    length(φD) == nϕ ||
        throw(DimensionMismatch(
            "`φD` must have length $(nϕ)."
        ))

    length(θ) == 3 ||
        throw(DimensionMismatch(
            "`θ` must have length 3."
        ))

    fill!(φD, 0.0 + 0.0im)

    for (i, field) in pairs(fields)
        if field.kind === :λ
            φD[i] = ComplexF64(z * θ[field.a])
        elseif field.kind === :W || field.kind === :Wbar
            X = field.channel
            a = field.a
            ap = mod1(a + 1, 3)

            ηX = _gauge_eta(X)
            phase = _bond_phase(a, field.δ, q)
            θbond = θ[a] + ηX * phase * θ[ap]
            Wsp = _hs_saddle_value_for_gauge_mode(
                sbs,
                field.kind,
                X,
                a
            )

            if field.kind === :W
                φD[i] = im * Wsp * θbond
            else
                φD[i] = -im * Wsp * θbond
            end
        else
            throw(ArgumentError(
                "Unknown internal-field kind `$(field.kind)`."
            ))
        end
    end

    return φD
end


"""
    gauge_mode_vectors(sbs, fields, q, z)

Return a matrix whose three columns are the gauge tangent vectors generated by

    θ = (1,0,0),
    θ = (0,1,0),
    θ = (0,0,1).
"""
function gauge_mode_vectors(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q::Vec3,
    z::Number,
)
    nϕ = length(fields)

    ΦD = zeros(ComplexF64, nϕ, 3)
    θ = zeros(ComplexF64, 3)

    for aθ in 1:3
        fill!(θ, 0.0 + 0.0im)
        θ[aθ] = 1.0 + 0.0im

        gauge_mode_vector!(
            view(ΦD, :, aθ),
            sbs,
            fields,
            q,
            z,
            θ
        )
    end

    return ΦD
end


"""
    apply_rpa_kernel_to_gauge_modes(
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = 2,
    )

Construct the RPA kernel and the three gauge tangent vectors, and return

    K, ΦD, R,

where

    R = K * ΦD.
"""
function apply_rpa_kernel_to_gauge_modes(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    nϕ = length(fields)
    K = zeros(ComplexF64, nϕ, nϕ)

    rpa_kernel!(
        K,
        sbs,
        fields,
        kgrid,
        q,
        z,
        aux;
        Nflavor = Nflavor)

    ΦD = gauge_mode_vectors(sbs, fields, q, z)
    R = K * ΦD

    return K, ΦD, R
end


"""
    gauge_mode_residuals(K, ΦD, R)

Return the absolute residual, relative residual, and gauge-vector norm for
each of the three gauge directions.
"""
function gauge_mode_residuals(
    K::AbstractMatrix{ComplexF64},
    ΦD::AbstractMatrix{ComplexF64},
    R::AbstractMatrix{ComplexF64},
)
    size(ΦD, 2) == 3 ||
        throw(DimensionMismatch(
            "`ΦD` must have three columns."
        ))

    size(R, 2) == 3 ||
        throw(DimensionMismatch(
            "`R` must have three columns."
        ))

    size(K, 2) == size(ΦD, 1) ||
        throw(DimensionMismatch(
            "`K` and `ΦD` have incompatible dimensions."
        ))

    size(K, 1) == size(R, 1) ||
        throw(DimensionMismatch(
            "`K` and `R` have incompatible dimensions."
        ))

    abs_res = zeros(Float64, 3)
    rel_res = zeros(Float64, 3)
    φ_norms = zeros(Float64, 3)

    K_norm = norm(K)

    for aθ in 1:3
        φ = view(ΦD, :, aθ)
        r = view(R, :, aθ)

        φ_norms[aθ] = norm(φ)
        abs_res[aθ] = norm(r)
        rel_res[aθ] =
            abs_res[aθ] /
            max(K_norm * φ_norms[aθ], eps(Float64))
    end

    return abs_res, rel_res, φ_norms
end