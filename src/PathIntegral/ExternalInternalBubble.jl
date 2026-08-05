# ----------------------------------------------------------------------
# External-internal bubbles S^{1+1}
# ----------------------------------------------------------------------

function external_internal_bubble!(
    Sβ::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux,
    Nflavor::Real = 2
)
    length(Sβ) == length(fields) ||
        throw(DimensionMismatch(
            "`Sβ` must have length $(length(fields))."
        ))

    fill!(Sβ, 0.0 + 0.0im)

    external_internal_bubble_normal!(
        Sβ,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux
    )

    external_internal_bubble_condensate!(
        Sβ,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux
    )

    external_internal_bubble_active_constraint!(
        Sβ,
        sbs,
        fields,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux,
        Nflavor = Nflavor
    )

    return Sβ
end


function external_internal_bubble_row!(
    Sα::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux,
    Nflavor::Real = 2
)
    length(Sα) == length(fields) ||
        throw(DimensionMismatch(
            "`Sα` must have length $(length(fields))."
        ))

    fill!(Sα, 0.0 + 0.0im)

    external_internal_bubble_row_normal!(
        Sα,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux
    )

    external_internal_bubble_row_condensate!(
        Sα,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux
    )

    external_internal_bubble_row_active_constraint!(
        Sα,
        sbs,
        fields,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux,
        Nflavor = Nflavor
    )

    return Sα
end

"""
    external_internal_bubble_normal!(
        Sβ,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η,
        aux
    )

Add the ordinary normal-normal column external-internal bubble from `S_eff`.
"""
function external_internal_bubble_normal!(
    Sβ::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux
)
    nϕ = length(fields)

    length(Sβ) == nϕ ||
        throw(DimensionMismatch("`Sβ` must have length $(nϕ)."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    @boundscheck @assert 1 <= μ <= 3

    (; L) = sbs

    Nu = L^2
    Ns = 3Nu
    βtemp = _inverse_temperature(sbs)
    z = ω + im * η

    Uq = external_vertex(μ, q_ext)
    Vβ = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    for k in kgrid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vβ, sbs, β, kq, k)

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
                        Vβ,
                        Vk,
                        weights_k,
                        m,
                        Uq
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Sβ[iβ] += prefactor * accum
        end
    end

    return Sβ
end


"""
    external_internal_bubble_row_normal!(
        Sα,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η,
        aux
    )

Add the ordinary normal-normal row external-internal bubble from `S_eff`.
"""
function external_internal_bubble_row_normal!(
    Sα::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux
)
    nϕ = length(fields)

    length(Sα) == nϕ ||
        throw(DimensionMismatch("`Sα` must have length $(nϕ)."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    @boundscheck @assert 1 <= μ <= 3

    (; L) = sbs

    Nu = L^2
    Ns = 3Nu
    βtemp = _inverse_temperature(sbs)
    z = ω + im * η

    Umq = external_vertex(μ, -q_ext)
    Vrow = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    for k in kgrid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iα, α) in pairs(fields)
            row_internal_vertices!(Vrow, sbs, α, k, kq)

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
                        Vk,
                        weights_k,
                        m,
                        Vrow,
                        Vkq,
                        weights_kq,
                        n,
                        Umq
                    )

                    accum += coherence * occdiff / denom
                end
            end

            Sα[iα] += prefactor * accum
        end
    end

    return Sα
end


"""
    external_internal_bubble_condensate!(
        Sβ,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η,
        aux
    )

Add the ordinary mixed selected-normal column external-internal bubble from
`S_eff`.

The selected poles carry unit BdG residues. No enhanced occupation `ξ` is
inserted here. The selected-selected elastic block is omitted.
"""
function external_internal_bubble_condensate!(
    Sβ::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux
)
    isempty(aux.conden_band_indices) && return Sβ

    nϕ = length(fields)

    length(Sβ) == nϕ ||
        throw(DimensionMismatch("`Sβ` must have length $(nϕ)."))

    Nu = length(kgrid)
    Nu > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    (; L) = sbs

    Ns = 3L^2
    βtemp = _inverse_temperature(sbs)
    z = ω + im * η

    qc = _spectral_condensation_momentum(aux, L)

    Uq = external_vertex(μ, q_ext)
    Vβ = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    # Selected pole on the k line, normal propagator on the k + q line.
    kc = qc
    kn = qc + q_reshaped

    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)
    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kn, kc)

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
                    Vβ,
                    Vc,
                    weights_c,
                    m,
                    Uq
                )

                accum += coherence * occdiff / denom
            end
        end

        Sβ[iβ] += prefactor * accum
    end

    # Normal propagator on the k line, selected pole on the k + q line.
    kn = qc - q_reshaped
    kc = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kc, kn)

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
                    Vβ,
                    Vn,
                    weights_n,
                    m,
                    Uq
                )

                accum += coherence * occdiff / denom
            end
        end

        Sβ[iβ] += prefactor * accum
    end

    return Sβ
end


"""
    external_internal_bubble_row_condensate!(
        Sα,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η,
        aux
    )

Add the ordinary mixed selected-normal row external-internal bubble from
`S_eff`.

The selected poles carry unit BdG residues. No enhanced occupation `ξ` is
inserted here. The selected-selected elastic block is omitted.
"""
function external_internal_bubble_row_condensate!(
    Sα::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux
)
    isempty(aux.conden_band_indices) && return Sα

    nϕ = length(fields)

    length(Sα) == nϕ ||
        throw(DimensionMismatch("`Sα` must have length $(nϕ)."))

    Nu = length(kgrid)
    Nu > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    (; L) = sbs

    Ns = 3L^2
    βtemp = _inverse_temperature(sbs)
    z = ω + im * η

    qc = _spectral_condensation_momentum(aux, L)

    Umq = external_vertex(μ, -q_ext)
    Vrow = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    # Selected pole on the k line, normal propagator on the k + q line.
    kc = qc
    kn = qc + q_reshaped

    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)
    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kc, kn)

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
                    Vc,
                    weights_c,
                    m,
                    Vrow,
                    Vn,
                    weights_n,
                    n,
                    Umq
                )

                accum += coherence * occdiff / denom
            end
        end

        Sα[iα] += prefactor * accum
    end

    # Normal propagator on the k line, selected pole on the k + q line.
    kn = qc - q_reshaped
    kc = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kn, kc)

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
                    Vn,
                    weights_n,
                    m,
                    Vrow,
                    Vc,
                    weights_c,
                    n,
                    Umq
                )

                accum += coherence * occdiff / denom
            end
        end

        Sα[iα] += prefactor * accum
    end

    return Sα
end

function external_internal_bubble_active_constraint!(
    Sβ::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux,
    Nflavor::Real = 2
)
    aux.selection_kind === :pinned || return Sβ
    isempty(aux.conden_band_indices) && return Sβ

    nϕ = length(fields)

    length(Sβ) == nϕ ||
        throw(DimensionMismatch("`Sβ` must have length $(nϕ)."))

    @boundscheck @assert 1 <= μ <= 3

    qc = _spectral_condensation_momentum(aux, sbs.L)
    z = ω + im * η

    Uq = external_vertex(μ, q_ext)
    Vβ = zeros(ComplexF64, 12, 12)

    kc = qc

    ϵs_c, Vc, _ = Green_SP_condensed_residues(sbs, kc, aux)

    active_weights = aux.active_positive_weights
    active_mask = active_weights .> 0.0
    unit_active_weights = zeros(Float64, length(ϵs_c))

    prefactor = Nflavor

    kn = qc + q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kn, kc)

        accum = 0.0 + 0.0im

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
                    Vβ,
                    Vc,
                    unit_active_weights,
                    i,
                    Uq
                )

                accum += ξi * coherence / denom
            end
        end

        Sβ[iβ] += prefactor * accum
    end

    kn = qc - q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kc, kn)

        accum = 0.0 + 0.0im

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
                    Vβ,
                    Vn,
                    weights_n,
                    m,
                    Uq
                )

                accum += ξi * coherence / denom
            end
        end

        Sβ[iβ] += prefactor * accum
    end

    return Sβ
end

function external_internal_bubble_row_active_constraint!(
    Sα::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int;
    η::Real,
    aux::SpectralCondensationAux,
    Nflavor::Real = 2
)
    aux.selection_kind === :pinned || return Sα
    isempty(aux.conden_band_indices) && return Sα

    nϕ = length(fields)

    length(Sα) == nϕ ||
        throw(DimensionMismatch("`Sα` must have length $(nϕ)."))

    @boundscheck @assert 1 <= μ <= 3

    qc = _spectral_condensation_momentum(aux, sbs.L)
    z = ω + im * η

    Umq = external_vertex(μ, -q_ext)
    Vrow = zeros(ComplexF64, 12, 12)

    kc = qc

    ϵs_c, Vc, _ = Green_SP_condensed_residues(sbs, kc, aux)

    active_weights = aux.active_positive_weights
    active_mask = active_weights .> 0.0
    unit_active_weights = zeros(Float64, length(ϵs_c))

    prefactor = Nflavor

    kn = qc + q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kc, kn)

        accum = 0.0 + 0.0im

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
                    Vc,
                    unit_active_weights,
                    i,
                    Vrow,
                    Vn,
                    weights_n,
                    n,
                    Umq
                )

                accum += ξi * coherence / denom
            end
        end

        Sα[iα] += prefactor * accum
    end

    kn = qc - q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kn, kc)

        accum = 0.0 + 0.0im

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
                    Vn,
                    weights_n,
                    m,
                    Vrow,
                    Vc,
                    unit_active_weights,
                    i,
                    Umq
                )

                accum += ξi * coherence / denom
            end
        end

        Sα[iα] += prefactor * accum
    end

    return Sα
end

"""
    external_internal_bubble_pair!(
        Splus,
        Srow,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ,
        ν;
        η,
        aux
    )

Compute the two external-internal bubbles needed for Fig. 1(b):

    Splus[β] = S^{1+1;μ,R}_{β}(q,ω),
    Srow[α]  = S^{†,1+1;ν,R}_{α}(q,ω).
"""
function external_internal_bubble_pair!(
    Splus::AbstractVector{ComplexF64},
    Srow::AbstractVector{ComplexF64},
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    ω::Real,
    μ::Int,
    ν::Int;
    η::Real,
    aux::SpectralCondensationAux
)
    external_internal_bubble!(
        Splus,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        μ;
        η = η,
        aux = aux
    )

    external_internal_bubble_row!(
        Srow,
        sbs,
        fields,
        kgrid,
        q_ext,
        q_reshaped,
        ω,
        ν;
        η = η,
        aux = aux
    )

    return Splus, Srow
end