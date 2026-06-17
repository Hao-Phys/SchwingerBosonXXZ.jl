# ----------------------------------------------------------------------
# External-internal bubbles S^{1+1}
# ----------------------------------------------------------------------

"""
    external_internal_bubble!(
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Fill `Sα` with the retarded external-internal bubble
`S^{1+1;μ,R}_{α}(q,ω)` for all internal fields `α`.

This currently includes only the normal-normal contribution.

Momentum conventions:

- `q_ext` is the external momentum in the original reciprocal-lattice
  coordinate. It is used only in `external_vertex(μ, q_ext)`.
- `q_reshaped` is the folded/reshaped magnetic-Brillouin-zone momentum. It is
  used in the internal Green's functions and internal vertices.

By default, the normal-normal contribution uses finite-temperature Bose
factors in the Matsubara-summed expression. Setting
`force_T0_bose_factor = true` restores the legacy zero-temperature
negative-pole to positive-pole contribution.
"""
function external_internal_bubble!(
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
    length(Sα) == length(fields) ||
        throw(DimensionMismatch(
            "`Sα` must have length $(length(fields))."
        ))

    Sα .= 0.0 + 0.0im

    external_internal_bubble_normal!(
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

    return Sα
end


"""
    external_internal_bubble_normal!(
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Add the normal-normal contribution to `Sα`.

This helper is separated from `external_internal_bubble!` so that later
condensate-normal and normal-condensate contributions can be added without
changing the public API.
"""
function external_internal_bubble_normal!(
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
    βtemp = 1 / sbs.T

    # `external_vertex(μ, q_ext)` already includes the phase factor
    # sum_ρ exp(-i q⋅dρ) U^μ_{ρ,0}.
    Uq = external_vertex(μ, q_ext)

    Vα = zeros(ComplexF64, 12, 12)

    prefactor = -1 / (4 * sqrt(Ns * Nu))

    for k in kgrid
        kq = k + q_reshaped

        ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
        ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

        for (iα, α) in pairs(fields)
            internal_vertices!(Vα, sbs, α, kq, k)

            accum = 0.0 + 0.0im

            if force_T0_bose_factor
                # Legacy zero-temperature expression: C^- on the k line and
                # C^+ on the k+q line.
                for a in 1:6
                    lneg = 6 + a
                    iszero(weights_k[lneg]) && continue

                    ω1 = -ϵs_k[lneg]

                    for b in 1:6
                        lpos = b
                        iszero(weights_kq[lpos]) && continue

                        ω2 = ϵs_kq[lpos]

                        denom = ω + im * η - ω1 - ω2

                        coherence = _residue_vertex_trace(
                            Vkq,
                            weights_kq,
                            lpos,
                            Vα,
                            Vk,
                            weights_k,
                            lneg,
                            Uq,
                        )

                        accum += coherence / denom
                    end
                end
            else
                # Finite-temperature Matsubara-summed expression.
                for m in eachindex(ϵs_k)
                    iszero(weights_k[m]) && continue

                    Em = ϵs_k[m]
                    nb_m = _nB_BdG(Em, βtemp)

                    for n in eachindex(ϵs_kq)
                        iszero(weights_kq[n]) && continue

                        En = ϵs_kq[n]
                        nb_n = _nB_BdG(En, βtemp)

                        occdiff = nb_n - nb_m
                        iszero(occdiff) && continue

                        denom = ω + im * η + Em - En

                        coherence = _residue_vertex_trace(
                            Vkq,
                            weights_kq,
                            n,
                            Vα,
                            Vk,
                            weights_k,
                            m,
                            Uq,
                        )

                        accum += coherence * occdiff / denom
                    end
                end
            end

            # The k-grid represents the magnetic-Brillouin-zone average.
            Sα[iα] += prefactor * accum / Nk
        end
    end

    return Sα
end


"""
    external_internal_bubble_pair!(
        Splus, Sminus, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ, ν;
        η,
        aux = nothing,
        force_T0_bose_factor = false,
    )

Compute the two normal-normal external-internal bubbles needed for Fig. 1(b):

    Splus[α]  = S^{1+1;μ,R}_{α}( q,  ω)
    Sminus[α] = S^{1+1;ν,R}_{α}(-q, -ω)

Both are normal-normal only at this stage.
"""
function external_internal_bubble_pair!(
    Splus::AbstractVector{ComplexF64},
    Sminus::AbstractVector{ComplexF64},
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
    )

    external_internal_bubble!(
        Sminus,
        sbs,
        fields,
        kgrid,
        -q_ext,
        -q_reshaped,
        -ω,
        ν;
        η = -η,
        aux = aux,
        force_T0_bose_factor = force_T0_bose_factor,
    )

    return Splus, Sminus
end