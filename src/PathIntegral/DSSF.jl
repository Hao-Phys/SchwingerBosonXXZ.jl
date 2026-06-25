# src/PathIntegral/DSSF.jl

"""
    dssf_SP(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ;
        aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
        force_T0_bose_factor::Bool = false,
    )

Compute the saddle-point dynamical spin structure factor using the
path-integral Green-function trace formula.

Returns `ret_normal, ret_condensate`, where both arrays have size
`3 × length(energies)`.

The normal-normal part uses `Green_SP_normal_residues`, so the selected
condensed modes stored in `aux` are removed from the normal sector. The
condensate-normal part uses `Green_SP_condensed_residues`, so the same selected
modes and the same condensate weights are inserted into the condensate sector.

By default, the normal-normal part uses finite-temperature Bose factors and
the fluctuation-dissipation prefactor

    1 / (1 - exp(-βω)).

Setting `force_T0_bose_factor = true` restores the legacy zero-temperature
negative-pole to positive-pole contribution.
"""
function dssf_SP(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
    force_T0_bose_factor::Bool = false,
)
    num_energies = length(energies)

    ret_normal = zeros(Float64, 3, num_energies)
    ret_condensate = zeros(Float64, 3, num_energies)

    (; L) = sbs

    Ns = 3L^2
    βtemp = _inverse_temperature(sbs)

    q_ext = Vec3(q[1], q[2], q[3])
    q_reshaped = to_reshaped_rlu(q_ext)

    Uq = [external_vertex(μ, q_ext) for μ in 1:3]
    Umq = [external_vertex(μ, -q_ext) for μ in 1:3]

    k_grid = Vec3[]

    for i in 1:L, j in 1:L
        push!(k_grid, Vec3([(i - 1) / L, (j - 1) / L, 0.0]))
    end

    # ------------------------------------------------------------------
    # Normal-normal contribution.
    #
    # The normal residue provider removes the selected condensed poles at
    # aux.conden_index. This keeps the normal sector and condensate sector
    # disjoint by construction.
    # ------------------------------------------------------------------
    for k in k_grid
        kq = k + q_reshaped

        if force_T0_bose_factor
            ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
            ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

            for a in 1:6
                lneg = 6 + a
                iszero(weights_k[lneg]) && continue

                ω1 = -ϵs_k[lneg]

                for b in 1:6
                    lpos = b
                    iszero(weights_kq[lpos]) && continue

                    ω2 = ϵs_kq[lpos]
                    ΔE = ω1 + ω2

                    for μ in 1:3
                        trace_weight = _residue_vertex_trace(
                            Vkq, weights_kq, lpos, Umq[μ],
                            Vk, weights_k, lneg, Uq[μ],
                        )

                        weight = -real(trace_weight) / (8Ns)

                        for (ie, energy) in enumerate(energies)
                            ret_normal[μ, ie] += weight * lorentzian(energy - ΔE, Γ)
                        end
                    end
                end
            end
        else
            ϵs_k, Vk, weights_k = Green_SP_normal_residues(sbs, k, aux)
            ϵs_kq, Vkq, weights_kq = Green_SP_normal_residues(sbs, kq, aux)

            for m in eachindex(ϵs_k)
                iszero(weights_k[m]) && continue

                Em = ϵs_k[m]
                nb_m = _nB_BdG(Em, βtemp)

                for n in eachindex(ϵs_kq)
                    iszero(weights_kq[n]) && continue

                    En = ϵs_kq[n]
                    nb_n = _nB_BdG(En, βtemp)

                    ΔE = En - Em
                    occdiff = nb_n - nb_m
                    iszero(occdiff) && continue

                    x = βtemp * real(ΔE)

                    transition_factor = if abs(x) < 1e-10
                        nb_mid = _nB_BdG((Em + En) / 2, βtemp)
                        -nb_mid * (1 + nb_mid)
                    elseif x > 700
                        occdiff
                    elseif x < -700
                        0.0
                    else
                        occdiff / (1 - exp(-x))
                    end

                    iszero(transition_factor) && continue

                    for μ in 1:3
                        trace_weight = _residue_vertex_trace(
                            Vkq, weights_kq, n, Umq[μ],
                            Vk, weights_k, m, Uq[μ],
                        )

                        weight = -real(trace_weight) * transition_factor / (8Ns)

                        for (ie, energy) in enumerate(energies)
                            ret_normal[μ, ie] += weight * lorentzian(energy - ΔE, Γ)
                        end
                    end
                end
            end
        end
    end

    # ------------------------------------------------------------------
    # Condensate-normal contribution.
    #
    # Green-function split:
    #
    #     G G = G_n G_n
    #         + G_c G_n
    #         + G_n G_c
    #         + G_c G_c.
    #
    # The normal-normal part was already computed above using
    # Green_SP_normal_residues on both lines. Here we add both mixed
    # condensate-normal orientations and omit the elastic G_c G_c piece.
    #
    # The extra L^2 is the collapsed finite-size momentum sum.
    # ------------------------------------------------------------------
    i = (aux.conden_index - 1) ÷ L + 1
    j = (aux.conden_index - 1) % L + 1

    qc = Vec3([(i - 1) / L, (j - 1) / L, 0.0])

    # --------------------------------------------------------------
    # Orientation 1:
    #
    #     G_n(k + q) G_c(k),
    #
    # with k = qc. This is the old V2 / -k condensed-line contribution.
    # The condensed pole is a negative BdG pole on the k line.
    # --------------------------------------------------------------
    k = qc
    kq = qc + q_reshaped

    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, k, aux)
    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, kq, aux)

    for a in 1:6
        lcond_neg = 6 + a
        iszero(weights_c[lcond_neg]) && continue

        for b in 1:6
            lpos = b
            iszero(weights_n[lpos]) && continue

            ΔE = ϵs_n[lpos]

            thermal_factor = if force_T0_bose_factor
                1.0
            else
                x = βtemp * real(ΔE)

                if abs(x) < 1e-10
                    continue
                elseif x > 700
                    1.0
                elseif x < -700
                    0.0
                else
                    1 / (1 - exp(-x))
                end
            end

            for μ in 1:3
                trace_weight = _residue_vertex_trace(
                    Vn, weights_n, lpos, Umq[μ],
                    Vc, weights_c, lcond_neg, Uq[μ],
                )

                weight = thermal_factor * (-L^2 * real(trace_weight) / (8Ns))

                for (ie, energy) in enumerate(energies)
                    ret_condensate[μ, ie] += weight * lorentzian(energy - ΔE, Γ)
                end
            end
        end
    end

    # --------------------------------------------------------------
    # Orientation 2:
    #
    #     G_c(k + q) G_n(k),
    #
    # with k + q = qc, i.e. k = qc - q. The condensed pole is now a
    # positive BdG pole on the k + q line.
    # --------------------------------------------------------------
    k = qc - q_reshaped
    kq = qc

    ϵs_n, Vn, weights_n = Green_SP_normal_residues(sbs, k, aux)
    ϵs_c, Vc, weights_c = Green_SP_condensed_residues(sbs, kq, aux)

    for a in 1:6
        lneg = 6 + a
        iszero(weights_n[lneg]) && continue

        for b in 1:6
            lcond_pos = b
            iszero(weights_c[lcond_pos]) && continue

            ΔE = -ϵs_n[lneg]

            thermal_factor = if force_T0_bose_factor
                1.0
            else
                x = βtemp * real(ΔE)

                if abs(x) < 1e-10
                    continue
                elseif x > 700
                    1.0
                elseif x < -700
                    0.0
                else
                    1 / (1 - exp(-x))
                end
            end

            for μ in 1:3
                trace_weight = _residue_vertex_trace(
                    Vc, weights_c, lcond_pos, Umq[μ],
                    Vn, weights_n, lneg, Uq[μ],
                )

                weight = thermal_factor * (-L^2 * real(trace_weight) / (8Ns))

                for (ie, energy) in enumerate(energies)
                    ret_condensate[μ, ie] += weight * lorentzian(energy - ΔE, Γ)
                end
            end
        end
    end

    return ret_normal, ret_condensate
end

"""
    dssf_FL(
        sbs::SchwingerBosonSystem,
        q,
        energies,
        Γ;
        aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
        Nflavor::Real = 2,
        force_T0_bose_factor::Bool = false,
        κtol::Real = 1e-12,
    )

Compute the Gaussian-fluctuation counter-diagram contribution to the dynamical
spin structure factor.

This implements the Fig. 1(b) contribution in the row-column sector convention
of the current note:

    χ_FL^{μμ}(q,ω)
        = (1/N) S^{1+1;μ}_{β}(q,ω)
                D_{βα}(q,ω)
                S^{†,1+1;μ}_{α}(q,ω),

with

    D(q,z) = [Π0(q) - Π(q,z)]^{-1}.

The first external bubble is a column bubble, while the second is the row-side
bubble in the same external sector. The row-side dagger labels the Gaussian
row partner and does not denote Hermitian conjugation.

Returns `ret_FL_normal, ret_FL_condensate`, where both arrays have size
`3 × length(energies)`.

The RPA propagator is always built from the full polarization, including the
normal-normal and mixed condensate-normal pieces. The returned split is made
only at the level of the external-internal bubbles:

    ret_FL_normal:
        uses the normal parts of both external-internal bubbles.

    ret_FL_condensate:
        is the difference between the full external-bubble result and the
        normal external-bubble result.

The internal-field basis is restricted to active fields. This removes exactly
inactive channels whose bare kernel and vertices vanish, preventing artificial
singular blocks in the RPA kernel.
"""
function dssf_FL(
    sbs::SchwingerBosonSystem,
    q,
    energies,
    Γ;
    aux::SpectralCondensationAux = spectral_condensation_aux(sbs),
    Nflavor::Real = 2,
    force_T0_bose_factor::Bool = false,
    κtol::Real = 1e-12,
)
    num_energies = length(energies)

    ret_FL_normal = zeros(Float64, 3, num_energies)
    ret_FL_condensate = zeros(Float64, 3, num_energies)

    (; L) = sbs

    q_ext = Vec3(q[1], q[2], q[3])
    q_reshaped = to_reshaped_rlu(q_ext)

    k_grid = Vec3[]

    for i in 1:L, j in 1:L
        push!(k_grid, Vec3([(i - 1) / L, (j - 1) / L, 0.0]))
    end

    fields_all = internal_field_basis()
    nϕ_all = length(fields_all)

    Π0_all = zeros(ComplexF64, nϕ_all, nϕ_all)
    Pi0!(Π0_all, sbs, fields_all)

    active_indices = Int[]

    for i in eachindex(fields_all)
        field = fields_all[i]

        if field.kind === :λ || abs(Π0_all[i, i]) > κtol
            push!(active_indices, i)
        end
    end

    fields = fields_all[active_indices]
    nϕ = length(fields)

    Π0 = zeros(ComplexF64, nϕ, nϕ)
    Π = zeros(ComplexF64, nϕ, nϕ)
    K = zeros(ComplexF64, nϕ, nϕ)

    Scol_normal = zeros(ComplexF64, nϕ)
    Srow_normal = zeros(ComplexF64, nϕ)

    Scol_full = zeros(ComplexF64, nϕ)
    Srow_full = zeros(ComplexF64, nϕ)

    Pi0!(Π0, sbs, fields)

    for (ie, energy) in enumerate(energies)
        z = energy + im * Γ

        fill!(Π, 0.0 + 0.0im)
        fill!(K, 0.0 + 0.0im)

        polarization!(
            Π,
            sbs,
            fields,
            k_grid,
            q_reshaped,
            z;
            Nflavor = Nflavor,
            aux = aux,
            force_T0_bose_factor = force_T0_bose_factor,
        )

        rpa_kernel!(K, Π0, Π)

        for μ in 1:3
            external_internal_bubble!(
                Scol_normal,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = Γ,
                aux = aux,
                force_T0_bose_factor = force_T0_bose_factor,
                include_condensate = false,
            )

            external_internal_bubble_row!(
                Srow_normal,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = Γ,
                aux = aux,
                force_T0_bose_factor = force_T0_bose_factor,
                include_condensate = false,
            )

            external_internal_bubble!(
                Scol_full,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = Γ,
                aux = aux,
                force_T0_bose_factor = force_T0_bose_factor,
                include_condensate = true,
            )

            external_internal_bubble_row!(
                Srow_full,
                sbs,
                fields,
                k_grid,
                q_ext,
                q_reshaped,
                energy,
                μ;
                η = Γ,
                aux = aux,
                force_T0_bose_factor = force_T0_bose_factor,
                include_condensate = true,
            )

            χ_FL_normal =
                (1 / Nflavor) * (transpose(Scol_normal) * (K \ Srow_normal))

            χ_FL_full =
                (1 / Nflavor) * (transpose(Scol_full) * (K \ Srow_full))

            ret_FL_normal[μ, ie] = imag(χ_FL_normal) / π
            ret_FL_condensate[μ, ie] =
                imag(χ_FL_full - χ_FL_normal) / π
        end
    end

    return ret_FL_normal, ret_FL_condensate
end