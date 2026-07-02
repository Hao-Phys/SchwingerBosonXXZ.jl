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

@inline function _selected_line_sum_factor_for_rpa(
    aux::SpectralCondensationAux,
    L::Int,
)
    raw_factor = _condensate_sum_factor(aux, L)

    aux.selection_kind === :pinned || return raw_factor

    selected_weights = aux.condensate_weights[aux.conden_band_indices]
    total_weight = maximum(selected_weights)

    total_weight > 0 || return raw_factor

    xi = max(total_weight - 1.0, 0.0)

    return (1.0 + raw_factor * xi) / total_weight
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

    # prefactor = _condensate_sum_factor(aux, L) / (2 * Nflavor * Nk)
    prefactor = _selected_line_sum_factor_for_rpa(aux, L) / (2 * Nflavor * Nk)

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

    # condensate_sum_factor = _condensate_sum_factor(aux, L)
    # prefactor = condensate_sum_factor^2 / (2 * Nflavor * Nk)
    selected_line_sum_factor = _selected_line_sum_factor_for_rpa(aux, L)
    prefactor = selected_line_sum_factor^2 / (2 * Nflavor * Nk)

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
        # Internal-vertex HS convention:
        #
        #   W^A =  A,   W^D =  D,
        #   W^B = -B,   W^C = -C.
        #
        # The minus signs for B and C are required because their HS couplings
        # have negative sign, while the canonical BdG matrix is written in
        # terms of the positive mean-field variables B and C.
        if X === :A || X === :D
            return mf
        elseif X === :B || X === :C
            return -mf
        else
            throw(ArgumentError("Unknown HS channel `$X`."))
        end

    elseif kind === :Wbar
        # The barred saddle field is the actual Wbar field appearing in the
        # row/partner blocks. In the current canonical BdG construction this
        # corresponds to conj(A), conj(B), conj(C), conj(D), without the extra
        # B/C minus sign used for the unbarred W field.
        return conj(mf)

    else
        throw(ArgumentError("Expected `:W` or `:Wbar`; got `$kind`."))
    end
end

"""
    gauge_mode_vector!(φD, sbs, fields, q, z, θ)

Fill `φD` with the gauge tangent vector in the column-sector basis

    ϕ(q) = { W(q), Wbar(-q), λ(q) }.

`θ` is a length-3 vector of sublattice gauge angles. The frequency `z`
must be the same external frequency used in the RPA kernel. In Matsubara
notation, `z = im * ωq`; after analytic continuation, use
`z = ω + im * η`.
"""
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
        throw(DimensionMismatch("`φD` must have length $(nϕ)."))

    length(θ) == 3 ||
        throw(DimensionMismatch("`θ` must have length 3."))

    fill!(φD, 0.0 + 0.0im)

    for (i, field) in pairs(fields)
        if field.kind === :λ
            # Gauge transformation:
            #
            #     λ_a(q) -> λ_a(q) + δλ_a(q),
            #     δλ_a(q) = iω_q θ_a(q).
            #
            # Since the RPA kernel is evaluated at external frequency `z`,
            # use the same frequency variable here.
            φD[i] = ComplexF64(z * θ[field.a])

        elseif field.kind === :W || field.kind === :Wbar
            X = field.channel
            a = field.a
            ap = mod1(a + 1, 3)

            ηX = _gauge_eta(X)
            phase = _bond_phase(a, field.δ, q)

            θbond = θ[a] + ηX * phase * θ[ap]
            Wsp = _hs_saddle_value_for_gauge_mode(sbs, field.kind, X, a)

            if field.kind === :W
                φD[i] = im * Wsp * θbond
            else
                # The sector component labeled :Wbar is Wbar(-q).
                φD[i] = -im * Wsp * θbond
            end

        else
            throw(ArgumentError("Unknown internal-field kind `$(field.kind)`."))
        end
    end

    return φD
end

"""
    gauge_mode_vectors(sbs, fields, q, z)

Return a matrix `ΦD` whose three columns are the three gauge tangent vectors
generated by

    θ = (1,0,0), (0,1,0), (0,0,1).
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
            θ,
        )
    end

    return ΦD
end

"""
    apply_rpa_kernel_to_gauge_modes(
        sbs, fields, kgrid, q, z; Nflavor=2, aux=nothing
    )

Construct the RPA kernel

    K(q,z) = Π0 - Π(q,z),

construct the three gauge vectors, and return

    K, ΦD, R

where

    R = K * ΦD.
"""
function apply_rpa_kernel_to_gauge_modes(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q::Vec3,
    z::Number;
    Nflavor::Real = 2,
    aux::Union{Nothing,SpectralCondensationAux} = nothing,
)
    nϕ = length(fields)

    K = zeros(ComplexF64, nϕ, nϕ)

    rpa_kernel!(
        K,
        sbs,
        fields,
        kgrid,
        q,
        z;
        Nflavor = Nflavor,
        aux = aux,
    )

    ΦD = gauge_mode_vectors(sbs, fields, q, z)
    R = K * ΦD

    return K, ΦD, R
end

"""
    gauge_mode_residuals(K, ΦD, R)

Return absolute and relative residuals for the three right gauge vectors.
"""
function gauge_mode_residuals(
    K::AbstractMatrix{ComplexF64},
    ΦD::AbstractMatrix{ComplexF64},
    R::AbstractMatrix{ComplexF64},
)
    size(ΦD, 2) == 3 ||
        throw(DimensionMismatch("`ΦD` must have three columns."))

    size(R, 2) == 3 ||
        throw(DimensionMismatch("`R` must have three columns."))

    abs_res = zeros(Float64, 3)
    rel_res = zeros(Float64, 3)
    φ_norms = zeros(Float64, 3)

    K_norm = norm(K)

    for aθ in 1:3
        φ = view(ΦD, :, aθ)
        r = view(R, :, aθ)

        φ_norms[aθ] = norm(φ)
        abs_res[aθ] = norm(r)
        rel_res[aθ] = norm(r) / max(K_norm * φ_norms[aθ], eps(Float64))
    end

    return abs_res, rel_res, φ_norms
end