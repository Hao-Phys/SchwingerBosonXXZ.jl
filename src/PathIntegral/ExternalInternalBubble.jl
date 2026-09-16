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
    build_external_internal_bubble_normal_cache(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)

Build a `VectorChannelCache` for `external_internal_bubble_normal!` at fixed
`(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)`. Mirrors its loop nest
exactly, deferring only the frequency-dependent `denom`. Note this cache is
specific to `μ`, since `Uq = external_vertex(μ, q_ext)` enters the
coherence trace.
"""
function build_external_internal_bubble_normal_cache(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    μ::Int,
    aux::SpectralCondensationAux,
)
    nϕ = length(fields)

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    @boundscheck @assert 1 <= μ <= 3

    (; L) = sbs

    Nu = L^2
    Ns = 3Nu
    βtemp = _inverse_temperature(sbs)

    Uq = external_vertex(μ, q_ext)
    Vβ = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    ΔEs = Float64[]
    residues = [ComplexF64[] for _ in 1:nϕ]
    _reserve_channel_cache!(ΔEs, residues, Nk * 144)

    for k in kgrid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iβ, β) in pairs(fields)
            internal_vertices!(Vβ, sbs, β, kq, k)

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

                    iβ == 1 && push!(ΔEs, real(Em - En))
                    push!(residues[iβ], prefactor * coherence * occdiff)
                end
            end
        end
    end

    return VectorChannelCache(ΔEs, residues)
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
    build_external_internal_bubble_row_normal_cache(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)

Build a `VectorChannelCache` for `external_internal_bubble_row_normal!` at
fixed `(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)`.
"""
function build_external_internal_bubble_row_normal_cache(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    μ::Int,
    aux::SpectralCondensationAux,
)
    nϕ = length(fields)

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    @boundscheck @assert 1 <= μ <= 3

    (; L) = sbs

    Nu = L^2
    Ns = 3Nu
    βtemp = _inverse_temperature(sbs)

    Umq = external_vertex(μ, -q_ext)
    Vrow = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    ΔEs = Float64[]
    residues = [ComplexF64[] for _ in 1:nϕ]
    _reserve_channel_cache!(ΔEs, residues, Nk * 144)

    for k in kgrid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iα, α) in pairs(fields)
            row_internal_vertices!(Vrow, sbs, α, k, kq)

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

                    iα == 1 && push!(ΔEs, real(Em - En))
                    push!(residues[iα], prefactor * coherence * occdiff)
                end
            end
        end
    end

    return VectorChannelCache(ΔEs, residues)
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
    build_external_internal_bubble_condensate_cache(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)

Build a `VectorChannelCache` for `external_internal_bubble_condensate!` at
fixed `(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)`.
"""
function build_external_internal_bubble_condensate_cache(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    μ::Int,
    aux::SpectralCondensationAux,
)
    nϕ = length(fields)
    ΔEs = Float64[]
    residues = [ComplexF64[] for _ in 1:nϕ]
    _reserve_channel_cache!(ΔEs, residues, 2 * 144)  # 2 fixed-momentum orderings, 12x12 pole pairs each

    isempty(aux.conden_band_indices) && return VectorChannelCache(ΔEs, residues)

    Nu = length(kgrid)
    Nu > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    (; L) = sbs

    Ns = 3L^2
    βtemp = _inverse_temperature(sbs)

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

                iβ == 1 && push!(ΔEs, real(Em - En))
                push!(residues[iβ], prefactor * coherence * occdiff)
            end
        end
    end

    # Normal propagator on the k line, selected pole on the k + q line.
    kn = qc - q_reshaped
    kc = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kc, kn)

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

                iβ == 1 && push!(ΔEs, real(Em - En))
                push!(residues[iβ], prefactor * coherence * occdiff)
            end
        end
    end

    return VectorChannelCache(ΔEs, residues)
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

"""
    build_external_internal_bubble_row_condensate_cache(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)

Build a `VectorChannelCache` for `external_internal_bubble_row_condensate!`
at fixed `(sbs, fields, kgrid, q_ext, q_reshaped, μ, aux)`.
"""
function build_external_internal_bubble_row_condensate_cache(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    kgrid,
    q_ext::Vec3,
    q_reshaped::Vec3,
    μ::Int,
    aux::SpectralCondensationAux,
)
    nϕ = length(fields)
    ΔEs = Float64[]
    residues = [ComplexF64[] for _ in 1:nϕ]
    _reserve_channel_cache!(ΔEs, residues, 2 * 144)  # 2 fixed-momentum orderings, 12x12 pole pairs each

    isempty(aux.conden_band_indices) && return VectorChannelCache(ΔEs, residues)

    Nu = length(kgrid)
    Nu > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    (; L) = sbs

    Ns = 3L^2
    βtemp = _inverse_temperature(sbs)

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

                iα == 1 && push!(ΔEs, real(Em - En))
                push!(residues[iα], prefactor * coherence * occdiff)
            end
        end
    end

    # Normal propagator on the k line, selected pole on the k + q line.
    kn = qc - q_reshaped
    kc = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kn, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kc, aux)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kn, kc)

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

                iα == 1 && push!(ΔEs, real(Em - En))
                push!(residues[iα], prefactor * coherence * occdiff)
            end
        end
    end

    return VectorChannelCache(ΔEs, residues)
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

    # Normalization of the enhanced source-auxiliary block, S_β += +N Γ_ξ^{jϕ}.
    #
    #   Γ_ξ^{jϕ} = ξ_note ω_min,jϕ,   ξ_note = β Nu ξ / N,
    #
    # with ξ = `aux.active_positive_weights` a DENSITY (the constrained solve
    # in SpectralCondensation.jl imposes `N_normal + ξ qcsum = 2S + 1`, with
    # `N_normal` averaged over the Nu Brillouin-zone points, so the pinned mode
    # holds N₀ = Nu ξ bosons) and 1/N converting the physical multiplier to the
    # per-flavor action convention Z = ∫ D[ϕ] exp(-N S_eff).
    #
    # ω_min,jϕ is the mixed second-order curvature of the pinned BdG
    # eigenvalue. One insertion is the FULL external vertex u^μ, carrying
    # 1/(2 sqrt(Ns β)) by Eq. (E7) of the note -- the 2 is OUTSIDE the radical
    # -- and the other is the FULL internal vertex v_β, carrying 1/sqrt(Nu β)
    # by Eq. (H43). Relative to the reduced vertices used in the loops below,
    #
    #   ω_min,jϕ = [1/(2 β sqrt(Ns Nu))] Σ (reduced coherence)/(denominator),
    #
    # so S_β picks up N ξ_note/(2 β sqrt(Ns Nu)) = ξ sqrt(Nu/(4 Ns)). Both β
    # and the flavor count N cancel identically, leaving a pure geometric
    # ratio; `Nflavor` therefore does not enter this prefactor.
    #
    # The overall scale is confirmed independently by `dssf_mean_field`, which
    # reaches the same response through the canonical Lehmann representation
    # and shares none of these vertex conventions.
    Ns = 3 * sbs.L^2
    Nu = sbs.L^2
    prefactor = sqrt(Nu / (4 * Ns))

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

"""
    build_external_internal_bubble_active_constraint_cache(sbs, fields, q_ext, q_reshaped, μ, aux; Nflavor = 2)

Build a `VectorChannelCache` for `external_internal_bubble_active_constraint!`
at fixed `(sbs, fields, q_ext, q_reshaped, μ, aux)`. Same `denom` sign
convention as `build_active_constraint_kernel_cache`: the first ordering's
`denom = Ei - En + z` gives ΔE `Ei - En`; the second ordering's `denom =
Ei - Em - z = -(z + (Em - Ei))` gives ΔE `Em - Ei` with a sign-flipped
residue.
"""
function build_external_internal_bubble_active_constraint_cache(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q_ext::Vec3,
    q_reshaped::Vec3,
    μ::Int,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    # Prefactor sqrt(Nu/(4 Ns)); see the derivation comment in
    # `external_internal_bubble_active_constraint!`. It is independent of
    # `Nflavor` because the N of Eq. (I21)'s +N Γ_ξ cancels the 1/N carried by
    # ξ_note = β Nu ξ / N.
    prefactor = sqrt(sbs.L^2 / (4 * 3 * sbs.L^2))

    nϕ = length(fields)
    ΔEs = Float64[]
    residues = [ComplexF64[] for _ in 1:nϕ]
    _reserve_channel_cache!(ΔEs, residues, 2 * 144)  # 2 orderings, up to 12x12 (i, n) pairs each

    aux.selection_kind === :pinned || return VectorChannelCache(ΔEs, residues)
    isempty(aux.conden_band_indices) && return VectorChannelCache(ΔEs, residues)

    @boundscheck @assert 1 <= μ <= 3

    qc = _spectral_condensation_momentum(aux, sbs.L)

    Uq = external_vertex(μ, q_ext)
    Vβ = zeros(ComplexF64, 12, 12)

    kc = qc

    ϵs_c, Vc, _ = Green_SP_condensed_residues(sbs, kc, aux)

    active_weights = aux.active_positive_weights
    active_mask = active_weights .> 0.0
    unit_active_weights = zeros(Float64, length(ϵs_c))

    kn = qc + q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kn, kc)

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

                iβ == 1 && push!(ΔEs, real(Ei - En))
                push!(residues[iβ], prefactor * ξi * coherence)
            end
        end
    end

    kn = qc - q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iβ, β) in pairs(fields)
        internal_vertices!(Vβ, sbs, β, kc, kn)

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

                iβ == 1 && push!(ΔEs, real(Em - Ei))
                push!(residues[iβ], -prefactor * ξi * coherence)
            end
        end
    end

    return VectorChannelCache(ΔEs, residues)
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

    # Row partner of the enhanced source-auxiliary block, S_α += +N Γ_ξ^{ϕj}.
    # Same normalization as the column block; see the derivation comment in
    # `external_internal_bubble_active_constraint!`.
    Ns = 3 * sbs.L^2
    Nu = sbs.L^2
    prefactor = sqrt(Nu / (4 * Ns))

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
    build_external_internal_bubble_row_active_constraint_cache(sbs, fields, q_ext, q_reshaped, μ, aux; Nflavor = 2)

Build a `VectorChannelCache` for
`external_internal_bubble_row_active_constraint!` at fixed `(sbs, fields,
q_ext, q_reshaped, μ, aux)`. Same `denom` sign convention as
`build_external_internal_bubble_active_constraint_cache`.
"""
function build_external_internal_bubble_row_active_constraint_cache(
    sbs::SchwingerBosonSystem,
    fields::AbstractVector{InternalField},
    q_ext::Vec3,
    q_reshaped::Vec3,
    μ::Int,
    aux::SpectralCondensationAux;
    Nflavor::Real = 2,
)
    # Same prefactor as the column cache; see the derivation comment in
    # `external_internal_bubble_active_constraint!`.
    prefactor = sqrt(sbs.L^2 / (4 * 3 * sbs.L^2))

    nϕ = length(fields)
    ΔEs = Float64[]
    residues = [ComplexF64[] for _ in 1:nϕ]
    _reserve_channel_cache!(ΔEs, residues, 2 * 144)  # 2 orderings, up to 12x12 (i, n) pairs each

    aux.selection_kind === :pinned || return VectorChannelCache(ΔEs, residues)
    isempty(aux.conden_band_indices) && return VectorChannelCache(ΔEs, residues)

    @boundscheck @assert 1 <= μ <= 3

    qc = _spectral_condensation_momentum(aux, sbs.L)

    Umq = external_vertex(μ, -q_ext)
    Vrow = zeros(ComplexF64, 12, 12)

    kc = qc

    ϵs_c, Vc, _ = Green_SP_condensed_residues(sbs, kc, aux)

    active_weights = aux.active_positive_weights
    active_mask = active_weights .> 0.0
    unit_active_weights = zeros(Float64, length(ϵs_c))

    kn = qc + q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kc, kn)

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

                iα == 1 && push!(ΔEs, real(Ei - En))
                push!(residues[iα], prefactor * ξi * coherence)
            end
        end
    end

    kn = qc - q_reshaped

    ϵs_n, Vn, weights_n = _full_sp_residues(sbs, kn)
    exclude_active_intermediate = _same_momentum_mod1(kn, qc)

    for (iα, α) in pairs(fields)
        row_internal_vertices!(Vrow, sbs, α, kn, kc)

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

                iα == 1 && push!(ΔEs, real(Em - Ei))
                push!(residues[iα], -prefactor * ξi * coherence)
            end
        end
    end

    return VectorChannelCache(ΔEs, residues)
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