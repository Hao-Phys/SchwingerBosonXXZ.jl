# ----------------------------------------------------------------------
# Pole occupation and selected-mode helper functions
# ----------------------------------------------------------------------

@inline function _inverse_temperature(sbs::SchwingerBosonSystem)
    T = sbs.T
    return iszero(T) ? Inf : inv(T)
end

"""
    _nB_T0(E)

Zero-temperature Bose factor for BdG pole energies.

For positive poles, `nB(E) = 0`. For negative poles, `nB(E) = -1`.
"""
@inline function _nB_T0(E::Real)
    atol = 1e-12

    if E > atol
        return 0.0
    elseif E < -atol
        return -1.0
    else
        throw(ArgumentError(
            "Encountered a zero-energy pole in a normal bubble. " *
            "Pass condensation data through `aux` so pinned condensate modes " *
            "are removed, or treat the condensate contribution explicitly.",
        ))
    end
end

"""
    _nB_BdG(E, β)

Finite-temperature Bose factor for a BdG pole energy `E`.

Negative BdG poles correctly approach `-1` as `T -> 0`.
"""
@inline function _nB_BdG(E::Number, β::Real)
    Er = real(E)

    if !isfinite(β)
        return _nB_T0(Er)
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

@inline function _pole_bose(
    E::Number,
    β::Real,
    force_T0_bose_factor::Bool,
)
    return force_T0_bose_factor ? _nB_T0(real(E)) : _nB_BdG(E, β)
end

"""
    _condensate_sum_factor(aux, L)

Collapsed momentum-sum factor for the selected soft-mode sector.

For `:finite_size_minimum`, the selected pole is only an ordinary finite-size
BdG pole split out from the normal sector, so no macroscopic enhancement is
applied.

For `:pinned`, the selected pole represents the active soft-min condensate
sector, so the collapsed finite-size momentum sum contributes `L^2`.
"""
@inline function _condensate_sum_factor(
    aux::SpectralCondensationAux,
    L::Int,
)
    return aux.selection_kind === :pinned ? Float64(L^2) : 1.0
end

"""
    _dssf_transition_factor(Em, En, β, force_T0_bose_factor)

Finite-temperature transition factor for the normal-normal DSSF bubble.

The pole on the incoming line has BdG energy `Em`; the pole on the outgoing
line has BdG energy `En`. The physical external energy is `En - Em`.

When `force_T0_bose_factor = true`, this reduces to the legacy zero-temperature
rule: only negative-pole to positive-pole transitions contribute.
"""
@inline function _dssf_transition_factor(
    Em::Number,
    En::Number,
    β::Real,
    force_T0_bose_factor::Bool,
)
    ΔE = real(En - Em)

    if force_T0_bose_factor
        nb_m = _nB_T0(real(Em))
        nb_n = _nB_T0(real(En))
        occdiff = nb_n - nb_m

        return ΔE > 1e-12 ? occdiff : 0.0
    end

    nb_m = _nB_BdG(Em, β)
    nb_n = _nB_BdG(En, β)
    occdiff = nb_n - nb_m

    iszero(occdiff) && return 0.0

    x = β * ΔE

    if abs(x) < 1e-10
        nb_mid = _nB_BdG((Em + En) / 2, β)
        return -nb_mid * (1 + nb_mid)
    elseif x > 700
        return occdiff
    elseif x < -700
        return 0.0
    else
        return occdiff / (1 - exp(-x))
    end
end

"""
    _dssf_condensate_transition_factor(ΔE, β, force_T0_bose_factor)

Transition factor for a mixed selected-normal DSSF contribution, where the
selected line is treated as the condensed/soft line and the normal line carries
positive physical energy `ΔE`.
"""
@inline function _dssf_condensate_transition_factor(
    ΔE::Real,
    β::Real,
    force_T0_bose_factor::Bool,
)
    if force_T0_bose_factor
        return ΔE > 1e-12 ? 1.0 : 0.0
    end

    x = β * ΔE

    if abs(x) < 1e-10
        return 0.0
    elseif x > 700
        return 1.0
    elseif x < -700
        return 0.0
    else
        return 1 / (1 - exp(-x))
    end
end