# src/PathIntegral/ExternalVertices.jl

"""
    nambu_index(η, α, σ)

Return the matrix index in the Nambu ⊗ sublattice ⊗ spin basis.

Conventions:

- `η = 1`: particle sector
- `η = 2`: hole sector
- `α = 1, 2, 3`: magnetic sublattice
- `σ = 1, 2`: spin index

The corresponding ordering is

    (b1↑, b1↓, b2↑, b2↓, b3↑, b3↓,
     b̄1↑, b̄1↓, b̄2↑, b̄2↓, b̄3↑, b̄3↓).
"""
@inline function nambu_index(η::Int, α::Int, σ::Int)
    @boundscheck begin
        @assert 1 <= η <= 2
        @assert 1 <= α <= 3
        @assert 1 <= σ <= 2
    end

    return σ + 2 * (α - 1) + 6 * (η - 1)
end


"""
    external_vertex(μ, q)

Return the phase-dressed reduced external spin vertex

    U^μ_q = sum_α exp(-i q ⋅ d_α) U^μ_{α,0},

where

    U^μ_{α,0}
        = P+ ⊗ Pα ⊗ σ^μ
        + P- ⊗ Pα ⊗ (σ^μ)^T.

This is the convergence-factor-free vertex used after Matsubara summation.

Here `q` is the external wave vector in the original reciprocal-lattice
coordinate. The real-space phase is evaluated using

    q_global = recipvecs_origin * q
    d_α = global_position(α).

The returned vertex is already phase dressed, so later bubble formulas should
use `external_vertex(μ, q)` directly rather than adding separate sublattice
phase factors.
"""
function external_vertex(μ::Int, q)
    @assert 1 <= μ <= 3

    U = zeros(ComplexF64, 12, 12)

    σμ = σs[μ]
    q_global = recipvecs_origin * q

    for α in 1:3
        rα = global_position(α)

        # Phase-dressing factor exp(-i q ⋅ d_α).
        # This absorbs the sublattice-position phase into the external vertex.
        phase = exp(-im * dot(q_global, rα))

        for σ in 1:2, σ′ in 1:2
            # Particle block:
            #
            #     P+ ⊗ Pα ⊗ σ^μ
            i = nambu_index(1, α, σ)
            j = nambu_index(1, α, σ′)
            U[i, j] += phase * σμ[σ, σ′]

            # Hole block:
            #
            #     P- ⊗ Pα ⊗ (σ^μ)^T
            i = nambu_index(2, α, σ)
            j = nambu_index(2, α, σ′)
            U[i, j] += phase * σμ[σ′, σ]
        end
    end

    return U
end