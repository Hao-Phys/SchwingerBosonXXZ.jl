# ----------------------------------------------------------------------
# Pole occupation and DSSF transition factors
# ----------------------------------------------------------------------

@inline function _inverse_temperature(sbs::SchwingerBosonSystem)
    T = sbs.T
    return iszero(T) ? Inf : inv(T)
end

"""
    _pole_bose(E, β)

Finite-temperature Bose factor for a bosonic BdG pole energy `E`.

For positive poles, this is the usual Bose factor. For negative BdG poles,

    nB(-E) = -1 - nB(E),

and the result approaches `-1` as `T -> 0`.

This function is used for every ordinary unit-residue BdG pole, including
poles in the algebraically selected sector. Active soft-minimum occupations
are not Bose factors and are added separately through fixed-`ξ` curvature
blocks.
"""
@inline function _pole_bose(E::Number, β::Real)
    Er = real(E)
    atol = 1e-12

    if !isfinite(β)
        if Er > atol
            return 0.0
        elseif Er < -atol
            return -1.0
        else
            throw(ArgumentError(
                "Encountered a zero-energy BdG pole. " *
                "Pass condensation data through `aux` and treat selected modes explicitly.",
            ))
        end
    end

    x = β * Er

    if x > 700
        return 0.0
    elseif x < -700
        return -1.0
    elseif abs(x) < 1e-12
        throw(ArgumentError(
            "Encountered a zero-energy pole in a finite-temperature Bose factor.",
        ))
    else
        return 1 / expm1(x)
    end
end

"""
    _dssf_fluctuation_dissipation_factor(ΔE, β)

Return the positive-frequency fluctuation-dissipation factor

    1 / (1 - exp(-β ΔE)).

The zero-frequency tolerance and large-exponent branches match the conventions
used by the DSSF transition factors below.
"""
@inline function _dssf_fluctuation_dissipation_factor(
    ΔE::Real,
    β::Real,
)
    ΔE <= 1e-12 && return 0.0
    !isfinite(β) && return 1.0

    x = β * ΔE

    if abs(x) < 1e-10
        return 0.0
    elseif x > 700
        return 1.0
    elseif x < -700
        return 0.0
    else
        return inv(-expm1(-x))
    end
end

"""
    _dssf_transition_factor(Em, En, β)

Finite-temperature transition factor for an ordinary unit-residue DSSF
bubble. Both `Em` and `En` are signed BdG pole energies.

The same function applies to normal-normal and mixed selected-normal
transitions. The selected ordinary poles retain unit residue. Active
soft-minimum occupations are not included here.
"""
@inline function _dssf_factor_from_occupations(
    ΔE::Real,
    nb_m::Real,
    nb_n::Real,
    β::Real,
)
    ΔE <= 1e-12 && return 0.0

    occdiff = nb_n - nb_m
    iszero(occdiff) && return 0.0

    return occdiff *
        _dssf_fluctuation_dissipation_factor(ΔE, β)
end

"""
    _dssf_transition_factor(Em, En, β)

Finite-temperature transition factor for a DSSF bubble.

Both `Em` and `En` are BdG pole energies. This same function is used for
normal-normal and selected-normal transitions. Selected poles are distinguished
by their residue weights and collapsed momentum-sum factors, not by a different
thermal occupation.
"""
@inline function _dssf_transition_factor(
    Em::Number,
    En::Number,
    β::Real,
)
    nb_m = _pole_bose(Em, β)
    nb_n = _pole_bose(En, β)

    return _dssf_factor_from_occupations(
        real(En - Em),
        nb_m,
        nb_n,
        β,
    )
end
