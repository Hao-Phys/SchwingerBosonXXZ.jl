# ----------------------------------------------------------------------
# External-internal bubbles S^{1+1}
# ----------------------------------------------------------------------

"""
    external_internal_bubble!(
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η, aux=nothing
    )

Fill `Sα` with the retarded zero-temperature external-internal bubble

    S^{1+1;μ,R}_{α}(q, ω)

for all internal fields `α`.

This currently includes only the normal-normal contribution. Condensate-normal
and normal-condensate pieces are intentionally left for later.

Momentum conventions:

- `q_ext` is the external momentum in the original reciprocal-lattice coordinate.
  It is used only in `external_vertex(μ, q_ext)`.
- `q_reshaped` is the folded/reshaped magnetic-Brillouin-zone momentum.
  It is used in the internal Green's functions and internal vertices.

The implemented normal-normal contribution follows Eq. (171):

    S^{1+1;μ,R}_{α,nn}(q,ω)
      = 1 / (4 sqrt(Ns Nu))
        sum_k sum_{a,b}
        tr[C^+_{k+q,b} V_α(k+q,k) C^-_{k,a} U^μ_q]
        / [ω + iη - ω_{-k,a} - ω_{k+q,b}]

where `U^μ_q` is already phase dressed.
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
    )

    return Sα
end


"""
    external_internal_bubble_normal!(
        Sα, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ;
        η, aux=nothing
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

            # Eq. (171): C^-_{k,a} on the k line and C^+_{k+q,b}
            # on the k+q line.
            for a in 1:6
                lneg = 6 + a
                iszero(weights_k[lneg]) && continue

                ω1 = -ϵs_k[lneg]

                for b in 1:6
                    lpos = b
                    iszero(weights_kq[lpos]) && continue

                    ω2 = ϵs_kq[lpos]

                    denom = ω + im * η - ω1 - ω2

                    # _residue_vertex_trace is definined in InternalVertices.jl
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

            # The k-grid represents the magnetic-Brillouin-zone average.
            # Eq. (171) has an explicit sum over k, so the finite-grid
            # implementation uses the average normalization here.
            Sα[iα] += prefactor * accum / Nk
        end
    end

    return Sα
end


"""
    external_internal_bubble_pair!(
        Splus, Sminus, sbs, fields, kgrid, q_ext, q_reshaped, ω, μ, ν;
        η, aux=nothing
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
    )

    return Splus, Sminus
end