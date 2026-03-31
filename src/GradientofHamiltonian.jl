# ── Conjugate Wirtinger derivatives ∂(ĨD)/∂conj(O_α) ────────────────────────
#
# For optimization over complex mean fields O ∈ {A, B, C, D}, the relevant
# gradient for a real-valued objective F is the conjugate Wirtinger derivative:
#
#   ∂F/∂conj(O) = Tr[P_H * ∂(ĨD)/∂conj(O)]   .
#
# Therefore the steepest-descent update of O uses
#
#   ΔO ∝ - ∂F/∂conj(O).
#
# Here Ĩ = diagm(vcat(ones(6), -ones(6))) multiplies D from the left, so all
# rows in the lower block (7:12) pick up an extra minus sign.
#
# Block structure implied by dynamical_matrix!:
#   A,D : O enters D12, but conj(O) enters D21
#   B,C : O enters D11[i,j] and D22[j+6,i+6]
#         conj(O) enters D11[j,i] and D22[i+6,j+6]
function ∂ID∂conjA_w!(out::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q::Vec3, α::Int)
    out .= 0.0
    (; J, Δ, α_dcoups) = sbs
    J₊ = J * (Δ + 1) / 2
    phase = link_phase(α, q)

    for σ in 1:2
        sign_σ = σ == 1 ? 1 : -1
        i = (α - 1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i + increment, 6)

        # conj(P_link) = -0.5 * J₊ * sign_σ * (1 + α_dcoups[1]) * conj(A[α]) + ...
        c = -0.5 * J₊ * sign_σ * (1 + α_dcoups[1])

        # D21 block before multiplying by Ĩ:
        #   D21[i,j] += conj(P_link) * phase
        #   D21[j,i] += conj(P_link) * conj(phase)
        #
        # Since rows 7:12 carry a minus sign under Ĩ, these contributions pick up -1.
        out[i+6, j] -= c * phase
        out[j+6, i] -= c * conj(phase)
    end
end

function ∂ID∂conjB_w!(out::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q::Vec3, α::Int)
    out .= 0.0
    (; J, Δ, α_dcoups) = sbs
    J₊ = J * (Δ + 1) / 2
    phase = link_phase(α, q)

    for σ in 1:2
        i = (α - 1) * 2 + σ
        j = mod1(i + 2, 6)

        c = 0.5 * J₊ * (1 - α_dcoups[1])

        # conj(Q_link) = 0.5 * J₊ * (1 - α_dcoups[1]) * conj(B[α]) + ...
        # conj(Q_link) enters:
        #   D11[j,i] += conj(Q_link) * conj(phase)
        #   D22[i,j] += conj(Q_link) * phase
        #
        # D11 rows are in the + block, D22 rows are in the - block.
        out[j, i] += c * conj(phase)
        out[i+6, j+6] -= c * phase
    end
end

function ∂ID∂conjC_w!(out::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q::Vec3, α::Int)
    out .= 0.0
    (; J, Δ, α_dcoups) = sbs
    J₋ = J * (Δ - 1) / 2
    phase = link_phase(α, q)

    for σ in 1:2
        sign_σ = σ == 1 ? 1 : -1
        i = (α - 1) * 2 + σ
        j = mod1(i + 2, 6)

        # conj(Q_link) = 0.5 * J₋ * sign_σ * (1 - α_dcoups[2]) * conj(C[α]) + ...
        c = 0.5 * J₋ * sign_σ * (1 - α_dcoups[2])
        # conj(Q_link) contribution, same slot structure as B
        out[j, i] += c * conj(phase)
        out[i+6, j+6] -= c * phase
    end
end

function ∂ID∂conjD_w!(out::Matrix{ComplexF64}, sbs::SchwingerBosonSystem, q::Vec3, α::Int)
    out .= 0.0
    (; J, Δ, α_dcoups) = sbs
    J₋ = J * (Δ - 1) / 2
    phase = link_phase(α, q)

    for σ in 1:2
        i = (α - 1) * 2 + σ
        increment = σ == 1 ? 3 : 1
        j = mod1(i + increment, 6)

        # conj(P_link) = -0.5 * J₋ * (1 + α_dcoups[2]) * conj(D[α]) + ...
        c = -0.5 * J₋ * (1 + α_dcoups[2])

        # conj(P_link) contribution in D21, with lower-block minus sign from Ĩ
        out[i+6, j] -= c * phase
        out[j+6, i] -= c * conj(phase)
    end
end

# Chemical potential μ₀ (real) hits the diagonal of D11 and D22, with a negative sign by Ĩ:
function ∂ID∂μ0!(out, α::Int)
    out .= 0.0
    for σ in 1:2
        i = (α-1) * 2 + σ
        out[i, i] -= 1.0
        out[i+6, i+6] += 1.0
    end
end