# ----------------------------------------------------------------------
# External-internal bubbles S^{1+1}
# ----------------------------------------------------------------------

"""
    external_internal_bubble!(
        Sβ, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
        include_condensate = true,
    )

Fill `Sβ` with the retarded column external-internal bubble

    S^{1+1;μ,R}_{β}(q,ω)

for all column internal fields `β`.

Momentum conventions:

- `q_ext` is the external momentum in the original reciprocal-lattice
  coordinate. It is used only in `external_vertex(μ, q_ext)`.
- `q_reshaped` is the folded/reshaped magnetic-Brillouin-zone momentum. It is
  used in the internal Green's functions and internal vertices.

The column bubble uses the column internal vertex `V_β(k+q,k)` and the
phase-dressed external vertex `U^μ_q = external_vertex(μ, q_ext)`.

If `include_condensate=true` and `aux` contains a condensate, the mixed
normal-condensate and condensate-normal pieces are included. The purely
elastic condensate-condensate piece is omitted.
"""
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
    include_condensate::Bool = true,
)
    length(Sβ) == length(fields) ||
        throw(DimensionMismatch(
            "`Sβ` must have length $(length(fields))."
        ))

    Sβ .= 0.0 + 0.0im

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
        aux = aux,
        force_T0_bose_factor = force_T0_bose_factor,
    )

    if include_condensate
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
            aux = aux,
            force_T0_bose_factor = force_T0_bose_factor,
        )
    end

    return Sβ
end

"""
    external_internal_bubble_row!(
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
        include_condensate = true,
    )

Fill `Sα` with the retarded row external-internal bubble

    S^{†,1+1;μ,R}_{α}(q,ω)

for all row labels `α`.

The stored field labels are the same labels used for the column basis. The
row-side vertex is obtained by the row-column mapping

    row :W    -> actual Wbar(q),
    row :Wbar -> actual W(-q),
    row :λ    -> actual λ(-q).

The dagger in the notation labels the row-side derivative vertex. It does not
mean Hermitian conjugation.

If `include_condensate=true` and `aux` contains a condensate, the mixed
normal-condensate and condensate-normal pieces are included. The purely
elastic condensate-condensate piece is omitted.
"""
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
    include_condensate::Bool = true,
)
    length(Sα) == length(fields) ||
        throw(DimensionMismatch(
            "`Sα` must have length $(length(fields))."
        ))

    Sα .= 0.0 + 0.0im

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
        aux = aux,
        force_T0_bose_factor = force_T0_bose_factor,
    )

    if include_condensate
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
            aux = aux,
            force_T0_bose_factor = force_T0_bose_factor,
        )
    end

    return Sα
end

"""
    external_internal_bubble_normal!(
        Sβ, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Add the normal-normal contribution to the column bubble `Sβ`.

This implements the Matsubara-summed expression

    S^{1+1;μ}_{β,nn}(q,z)
        = -1 / (4 sqrt(Ns Nu))
          sum_k sum_{m,n}
          tr[C_{k+q,n} V_β(k+q,k) C_{k,m} U^μ_q]
          [nB(E_{k+q,n}) - nB(E_{k,m})]
          / [z + E_{k,m} - E_{k+q,n}].

The finite grid is the explicit momentum sum in this formula; no additional
division by `length(kgrid)` is applied.
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
)
    nϕ = length(fields)
    length(Sβ) == nϕ ||
        throw(DimensionMismatch("`Sβ` must have length $(nϕ)."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    @boundscheck begin
        @assert 1 <= μ <= 3
    end

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
                nb_m = _pole_bose(Em, βtemp, force_T0_bose_factor)

                for n in eachindex(ϵs_kq)
                    iszero(weights_kq[n]) && continue
                    En = ϵs_kq[n]
                    nb_n = _pole_bose(En, βtemp, force_T0_bose_factor)

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vkq, weights_kq, n, Vβ,
                        Vk, weights_k, m, Uq,
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
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Add the normal-normal contribution to the row bubble `Sα`.

This is the row-side partner of `external_internal_bubble_normal!`:

    S^{†,1+1;μ}_{α,nn}(q,z)
        = -1 / (4 sqrt(Ns Nu))
          sum_k sum_{m,n}
          tr[C_{k,m} V†_α(k,k+q) C_{k+q,n} U^μ_{-q}]
          [nB(E_{k+q,n}) - nB(E_{k,m})]
          / [z + E_{k,m} - E_{k+q,n}].

The row-side vertex is not a Hermitian conjugate.
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
)
    nϕ = length(fields)
    length(Sα) == nϕ ||
        throw(DimensionMismatch("`Sα` must have length $(nϕ)."))

    Nk = length(kgrid)
    Nk > 0 || throw(ArgumentError("`kgrid` must not be empty."))

    @boundscheck begin
        @assert 1 <= μ <= 3
    end

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
                nb_m = _pole_bose(Em, βtemp, force_T0_bose_factor)

                for n in eachindex(ϵs_kq)
                    iszero(weights_kq[n]) && continue
                    En = ϵs_kq[n]
                    nb_n = _pole_bose(En, βtemp, force_T0_bose_factor)

                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    denom = z + Em - En

                    coherence = _residue_vertex_trace(
                        Vk, weights_k, m, Vrow,
                        Vkq, weights_kq, n, Umq,
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
        Sβ, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Add the mixed condensate-normal pieces to the column external-internal bubble.

The function is a no-op unless `aux` contains a condensate. The elastic
condensate-condensate contribution is omitted.
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
)
    _has_condensate(aux) || return Sβ

    nϕ = length(fields)
    length(Sβ) == nϕ ||
        throw(DimensionMismatch("`Sβ` must have length $(nϕ)."))

    (; L) = sbs
    Nu = L^2
    Ns = 3Nu

    βtemp = _inverse_temperature(sbs)
    z = ω + im * η

    qc = kgrid[aux.conden_index]

    Uq = external_vertex(μ, q_ext)
    Vβ = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    # Condensate on the k line, normal on the k + q line.
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
            nb_m = _nB_T0(real(Em))

            for n in eachindex(ϵs_n)
                iszero(weights_n[n]) && continue
                En = ϵs_n[n]
                nb_n = _pole_bose(En, βtemp, force_T0_bose_factor)

                occdiff = nb_n - nb_m
                iszero(occdiff) && continue

                denom = z + Em - En

                coherence = _residue_vertex_trace(
                    Vn, weights_n, n, Vβ,
                    Vc, weights_c, m, Uq,
                )

                accum += coherence * occdiff / denom
            end
        end

        Sβ[iβ] += prefactor * accum
    end

    # Normal on the k line, condensate on the k + q line.
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
            nb_m = _pole_bose(Em, βtemp, force_T0_bose_factor)

            for n in eachindex(ϵs_c)
                iszero(weights_c[n]) && continue
                En = ϵs_c[n]
                nb_n = _nB_T0(real(En))

                occdiff = nb_n - nb_m
                iszero(occdiff) && continue

                denom = z + Em - En

                coherence = _residue_vertex_trace(
                    Vc, weights_c, n, Vβ,
                    Vn, weights_n, m, Uq,
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
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Add the mixed condensate-normal pieces to the row external-internal bubble.

The function is a no-op unless `aux` contains a condensate. The elastic
condensate-condensate contribution is omitted.
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
)
    _has_condensate(aux) || return Sα

    nϕ = length(fields)
    length(Sα) == nϕ ||
        throw(DimensionMismatch("`Sα` must have length $(nϕ)."))

    (; L) = sbs
    Nu = L^2
    Ns = 3Nu

    βtemp = _inverse_temperature(sbs)
    z = ω + im * η

    qc = kgrid[aux.conden_index]

    Umq = external_vertex(μ, -q_ext)
    Vrow = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    # Condensate on the k line, normal on the k + q line.
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
            nb_m = _nB_T0(real(Em))

            for n in eachindex(ϵs_n)
                iszero(weights_n[n]) && continue
                En = ϵs_n[n]
                nb_n = _pole_bose(En, βtemp, force_T0_bose_factor)

                occdiff = nb_n - nb_m
                iszero(occdiff) && continue

                denom = z + Em - En

                coherence = _residue_vertex_trace(
                    Vc, weights_c, m, Vrow,
                    Vn, weights_n, n, Umq,
                )

                accum += coherence * occdiff / denom
            end
        end

        Sα[iα] += prefactor * accum
    end

    # Normal on the k line, condensate on the k + q line.
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
            nb_m = _pole_bose(Em, βtemp, force_T0_bose_factor)

            for n in eachindex(ϵs_c)
                iszero(weights_c[n]) && continue
                En = ϵs_c[n]
                nb_n = _nB_T0(real(En))

                occdiff = nb_n - nb_m
                iszero(occdiff) && continue

                denom = z + Em - En

                coherence = _residue_vertex_trace(
                    Vn, weights_n, m, Vrow,
                    Vc, weights_c, n, Umq,
                )

                accum += coherence * occdiff / denom
            end
        end

        Sα[iα] += prefactor * accum
    end

    return Sα
end

"""
    external_internal_bubble_pair!(
        Splus, Srow, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ, ν;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
        include_condensate = true,
    )

Compute the two external-internal bubbles needed for Fig. 1(b):

    Splus[β] = S^{1+1;μ,R}_{β}(q,ω),
    Srow[α]  = S^{†,1+1;ν,R}_{α}(q,ω).

The second object is the row-side bubble in the same external sector `q`; it
is not obtained by Hermitian conjugation.

If `include_condensate=true` and `aux` contains a condensate, the mixed
normal-condensate and condensate-normal pieces are included. The purely
elastic condensate-condensate piece is omitted.
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
    aux::Union{Nothing, CondensationAux} = nothing,
    force_T0_bose_factor::Bool = false,
    include_condensate::Bool = true,
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
        aux = aux,
        force_T0_bose_factor = force_T0_bose_factor,
        include_condensate = include_condensate,
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
        aux = aux,
        force_T0_bose_factor = force_T0_bose_factor,
        include_condensate = include_condensate,
    )

    return Splus, Srow
end