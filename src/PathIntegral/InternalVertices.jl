"""
    internal_vertices!(V, :λ, a)
    internal_vertices!(V, :lambda, a)

Fill `V` with the reduced internal vertex for the constraint field `λ_a`.

This is the reduced version of the constraint-field vertex in Appendix H of
the path-integral note. The common Fourier normalization
`1 / sqrt(Nu * β)` and the corresponding momentum-frequency Kronecker delta
have been stripped off.

The reduced vertex is independent of `k`, `p`, and `sbs`. It is local in
sublattice space and diagonal in Nambu and spin space. In the convergence
factor-free reduced vertex used after the Matsubara summation, both particle
and hole diagonal blocks carry the coefficient `im`.

The factor `im` comes directly from the constraint coupling in the action.
"""
function internal_vertices!(
    V::AbstractMatrix{ComplexF64},
    field::Symbol,
    a::Int,
)
    size(V) == (12, 12) ||
        throw(DimensionMismatch("`V` must have size (12, 12)."))

    _canonical_lambda_field(field)

    @boundscheck begin
        @assert 1 <= a <= 3
    end

    V .= 0.0 + 0.0im

    for σ in 1:2
        V[nambu_index(1, a, σ), nambu_index(1, a, σ)] += im
        V[nambu_index(2, a, σ), nambu_index(2, a, σ)] += im
    end

    return V
end

"""
    internal_vertices!(V, sbs, kind, X, a, δ, k, p)

Fill `V` with the reduced **column-side** internal vertex for one independent
auxiliary field.

Arguments:

- `kind = :W` or `:Wbar`;
- `X = :A, :B, :C, :D`;
- `a = 1,2,3` labels the oriented sublattice pair `a -> a + 1`;
- `δ = 1,2,3` labels one of the three independent bonds in `δ_{a -> a+1}`;
- `k`, `p` are the two bosonic momenta appearing in `V_α(k,p)`.

Sector convention:

- in sector `q`, `kind = :W` means the actual field `W(q)`;
- in sector `q`, `kind = :Wbar` means the actual field `Wbar(-q)`;
- the reduced vertex is called with transfer `k - p = q`.

The common normalization `1 / sqrt(Nu * β)` and the momentum-frequency
Kronecker delta are not included.

This implements the Appendix-H reduced vertices of the current note. All
bond operators carry the operator-level factor `1/2`, so every HS vertex
contains the coefficient `-κ / 2` or `-κ s / 2`. For normal channels B and C,
the vertex has both 11 and 22 Nambu blocks. For anomalous channels A and D,
the vertex contains both the direct anomalous block and the transposed
reversed block required by the corrected Nambu completion.
"""
function internal_vertices!(
    V::AbstractMatrix{ComplexF64},
    sbs::SchwingerBosonSystem,
    kind::Symbol,
    X::Symbol,
    a::Int,
    δ::Int,
    k::Vec3,
    p::Vec3,
)
    size(V) == (12, 12) ||
        throw(DimensionMismatch("`V` must have size (12, 12)."))

    kind = _canonical_internal_kind(kind)
    X = _canonical_internal_channel(X)

    @boundscheck begin
        @assert 1 <= a <= 3
        @assert 1 <= δ <= 3
    end

    V .= 0.0 + 0.0im

    κ, s = _κ_s(sbs, X, δ)

    if X === :B
        _internal_B!(V, kind, κ, s, a, δ, k, p)
    elseif X === :C
        _internal_C!(V, kind, κ, s, a, δ, k, p)
    elseif X === :A
        _internal_A!(V, kind, κ, s, a, δ, k, p)
    elseif X === :D
        _internal_D!(V, kind, κ, s, a, δ, k, p)
    else
        error("Unreachable internal channel: $X.")
    end

    return V
end

# ----------------------------------------------------------------------
# Couplings from Eqs. (25), (26), and (44)
# ----------------------------------------------------------------------

@inline function _κ_s(sbs::SchwingerBosonSystem, X::Symbol, δ::Int)
    (; J, Δ, α_dcoups) = sbs

    Jp = J * (Δ + 1) / 2
    Jm = J * (Δ - 1) / 2

    α1 = α_dcoups[1, δ]
    α2 = α_dcoups[2, δ]

    g = if X === :A
        (1 + α1) * Jp
    elseif X === :B
        -(1 - α1) * Jp
    elseif X === :C
        -(1 - α2) * Jm
    elseif X === :D
        (1 + α2) * Jm
    else
        error("Unreachable internal channel: $X.")
    end

    return abs(g), sign(g)
end

# ----------------------------------------------------------------------
# Bond phases e^{i q⋅δ}
# ----------------------------------------------------------------------

"""
    _bond_phase(a, δ, q)

Return `exp(i q ⋅ δ)` for the oriented bond `δ ∈ δ_{a -> a+1}`.

The input `q` is assumed to be in the same reciprocal-coordinate convention as
the existing BdG/path-integral code.

Implementation note: `cis(x) = cos(x) + im * sin(x) = exp(im * x)`, so
`cis(2π * x)` is the unit-modulus phase `exp(2π * im * x)`.
"""
@inline function _bond_phase(a::Int, δ::Int, q::Vec3)
    x = if a == 1
        if δ == 1
            0.0
        elseif δ == 2
            -q[2]
        else
            -q[1] - q[2]
        end
    elseif a == 2
        if δ == 1
            0.0
        elseif δ == 2
            q[2]
        else
            -q[1]
        end
    else
        if δ == 1
            0.0
        elseif δ == 2
            q[1]
        else
            q[1] + q[2]
        end
    end

    return cis(2π * x)
end

# ----------------------------------------------------------------------
# B channel
# ----------------------------------------------------------------------

function _internal_B!(
    V::AbstractMatrix{ComplexF64},
    kind::Symbol,
    κ::Real,
    s::Real,
    a::Int,
    δ::Int,
    k::Vec3,
    p::Vec3,
)
    ap = mod1(a + 1, 3)

    if kind === :W
        # 11 block:
        # -(κ/2) e^{i p⋅δ} P_{a,a+1} ⊗ σ0
        _add_spin_matrix!(
            V, 1, 1, a, ap,
            -κ / 2 * _bond_phase(a, δ, p),
            :σ0,
        )

        # 22 block from M22(k,p) = M11(-p,-k)^T:
        # -(κ/2) e^{-i k⋅δ} P_{a+1,a} ⊗ σ0
        _add_spin_matrix!(
            V, 2, 2, ap, a,
            -κ / 2 * conj(_bond_phase(a, δ, k)),
            :σ0,
        )
    elseif kind === :Wbar
        # 11 block:
        # -(κ s/2) e^{-i k⋅δ} P_{a+1,a} ⊗ σ0
        _add_spin_matrix!(
            V, 1, 1, ap, a,
            -κ * s / 2 * conj(_bond_phase(a, δ, k)),
            :σ0,
        )

        # 22 block from M22(k,p) = M11(-p,-k)^T:
        # -(κ s/2) e^{i p⋅δ} P_{a,a+1} ⊗ σ0
        _add_spin_matrix!(
            V, 2, 2, a, ap,
            -κ * s / 2 * _bond_phase(a, δ, p),
            :σ0,
        )
    else
        error("Unreachable internal kind: $kind.")
    end

    return V
end

# ----------------------------------------------------------------------
# C channel
# ----------------------------------------------------------------------

function _internal_C!(
    V::AbstractMatrix{ComplexF64},
    kind::Symbol,
    κ::Real,
    s::Real,
    a::Int,
    δ::Int,
    k::Vec3,
    p::Vec3,
)
    ap = mod1(a + 1, 3)

    if kind === :W
        # 11 block:
        # -(κ/2) e^{i p⋅δ} P_{a,a+1} ⊗ σz
        _add_spin_matrix!(
            V, 1, 1, a, ap,
            -κ / 2 * _bond_phase(a, δ, p),
            :σz,
        )

        # 22 block:
        # -(κ/2) e^{-i k⋅δ} P_{a+1,a} ⊗ σz
        _add_spin_matrix!(
            V, 2, 2, ap, a,
            -κ / 2 * conj(_bond_phase(a, δ, k)),
            :σz,
        )
    elseif kind === :Wbar
        # 11 block:
        # -(κ s/2) e^{-i k⋅δ} P_{a+1,a} ⊗ σz
        _add_spin_matrix!(
            V, 1, 1, ap, a,
            -κ * s / 2 * conj(_bond_phase(a, δ, k)),
            :σz,
        )

        # 22 block:
        # -(κ s/2) e^{i p⋅δ} P_{a,a+1} ⊗ σz
        _add_spin_matrix!(
            V, 2, 2, a, ap,
            -κ * s / 2 * _bond_phase(a, δ, p),
            :σz,
        )
    else
        error("Unreachable internal kind: $kind.")
    end

    return V
end

# ----------------------------------------------------------------------
# A channel
# ----------------------------------------------------------------------

function _internal_A!(
    V::AbstractMatrix{ComplexF64},
    kind::Symbol,
    κ::Real,
    s::Real,
    a::Int,
    δ::Int,
    k::Vec3,
    p::Vec3,
)
    ap = mod1(a + 1, 3)

    if kind === :W
        # Direct 12 block:
        # -(κ/2) e^{i p⋅δ} P_{a,a+1} ⊗ iσy
        _add_spin_matrix!(
            V, 1, 2, a, ap,
            -κ / 2 * _bond_phase(a, δ, p),
            :iσy,
        )

        # Transposed reversed 12 block:
        # -(κ/2) e^{-i k⋅δ} P_{a+1,a} ⊗ (iσy)^T.
        # Since (iσy)^T = -iσy, this is +(κ/2) e^{-i k⋅δ}
        # multiplying the iσy matrix.
        _add_spin_matrix!(
            V, 1, 2, ap, a,
            κ / 2 * conj(_bond_phase(a, δ, k)),
            :iσy,
        )
    elseif kind === :Wbar
        # Direct 21 block:
        # -(κ s/2) e^{i p⋅δ} P_{a,a+1} ⊗ iσy
        _add_spin_matrix!(
            V, 2, 1, a, ap,
            -κ * s / 2 * _bond_phase(a, δ, p),
            :iσy,
        )

        # Transposed reversed 21 block:
        # -(κ s/2) e^{-i k⋅δ} P_{a+1,a} ⊗ (iσy)^T.
        _add_spin_matrix!(
            V, 2, 1, ap, a,
            κ * s / 2 * conj(_bond_phase(a, δ, k)),
            :iσy,
        )
    else
        error("Unreachable internal kind: $kind.")
    end

    return V
end

# ----------------------------------------------------------------------
# D channel
# ----------------------------------------------------------------------

function _internal_D!(
    V::AbstractMatrix{ComplexF64},
    kind::Symbol,
    κ::Real,
    s::Real,
    a::Int,
    δ::Int,
    k::Vec3,
    p::Vec3,
)
    ap = mod1(a + 1, 3)

    if kind === :W
        # Direct 12 block:
        # -(κ/2) e^{i p⋅δ} P_{a,a+1} ⊗ σx
        _add_spin_matrix!(
            V, 1, 2, a, ap,
            -κ / 2 * _bond_phase(a, δ, p),
            :σx,
        )

        # Transposed reversed 12 block:
        # -(κ/2) e^{-i k⋅δ} P_{a+1,a} ⊗ (σx)^T.
        # Since (σx)^T = σx, the sign is unchanged.
        _add_spin_matrix!(
            V, 1, 2, ap, a,
            -κ / 2 * conj(_bond_phase(a, δ, k)),
            :σx,
        )
    elseif kind === :Wbar
        # Direct 21 block:
        # -(κ s/2) e^{i p⋅δ} P_{a,a+1} ⊗ σx
        _add_spin_matrix!(
            V, 2, 1, a, ap,
            -κ * s / 2 * _bond_phase(a, δ, p),
            :σx,
        )

        # Transposed reversed 21 block:
        # -(κ s/2) e^{-i k⋅δ} P_{a+1,a} ⊗ (σx)^T.
        _add_spin_matrix!(
            V, 2, 1, ap, a,
            -κ * s / 2 * conj(_bond_phase(a, δ, k)),
            :σx,
        )
    else
        error("Unreachable internal kind: $kind.")
    end

    return V
end

# ----------------------------------------------------------------------
# Matrix-entry helper
# ----------------------------------------------------------------------

function _add_spin_matrix!(
    V::AbstractMatrix{ComplexF64},
    ηrow::Int,
    ηcol::Int,
    arow::Int,
    acol::Int,
    coeff,
    spin_matrix::Symbol,
)
    if spin_matrix === :σ0
        V[nambu_index(ηrow, arow, 1), nambu_index(ηcol, acol, 1)] += coeff
        V[nambu_index(ηrow, arow, 2), nambu_index(ηcol, acol, 2)] += coeff
    elseif spin_matrix === :σz
        V[nambu_index(ηrow, arow, 1), nambu_index(ηcol, acol, 1)] += coeff
        V[nambu_index(ηrow, arow, 2), nambu_index(ηcol, acol, 2)] -= coeff
    elseif spin_matrix === :iσy
        # iσy = [0  1
        #        -1 0]
        V[nambu_index(ηrow, arow, 1), nambu_index(ηcol, acol, 2)] += coeff
        V[nambu_index(ηrow, arow, 2), nambu_index(ηcol, acol, 1)] -= coeff
    elseif spin_matrix === :σx
        # σx = [0 1
        #       1 0]
        V[nambu_index(ηrow, arow, 1), nambu_index(ηcol, acol, 2)] += coeff
        V[nambu_index(ηrow, arow, 2), nambu_index(ηcol, acol, 1)] += coeff
    else
        throw(ArgumentError("Unknown spin matrix label `$spin_matrix`."))
    end

    return V
end

# ----------------------------------------------------------------------
# Symbol canonicalization
# ----------------------------------------------------------------------

@inline function _canonical_internal_kind(kind::Symbol)
    kind === :W && return :W
    kind === :Wbar && return :Wbar
    throw(ArgumentError("`kind` must be `:W` or `:Wbar`; got `$kind`."))
end

@inline function _canonical_internal_channel(X::Symbol)
    X in (:A, :B, :C, :D) && return X
    throw(ArgumentError("`X` must be one of `:A`, `:B`, `:C`, `:D`; got `$X`."))
end

@inline function _canonical_lambda_field(field::Symbol)
    field === :λ && return :λ
    field === :lambda && return :λ
    throw(ArgumentError("`field` must be `:λ` or `:lambda`; got `$field`."))
end