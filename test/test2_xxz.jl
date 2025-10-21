using SchwingerBosonXXZ
using LinearAlgebra

S = 0.5
T = 1e-4
Δ = 10.0
L = 12
h_SB = 0.0
θ_ref = acos(Δ/(1+Δ))
θs_classical = [π+θ_ref, 0, π-θ_ref]
sbs = SchwingerBosonSystem(1.0, Δ, S, T, L, h_SB, θs_classical)

SchwingerBosonXXZ.set_ϕ!(sbs, [0.48850421045919723, 0.48850421045919723, -0.20829889522526548, 0.10660035817780524, 0.10660035817780524, -0.45454545454545453, 0.10660035817780524, 0.10660035817780524, 0.5, -0.48850421045919723, 0.48850421045919723, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0])
μ0s = [-50.0001, -50.0001, -50.0001]
SchwingerBosonXXZ.set_μ0!(sbs, μ0s)

@time SchwingerBosonXXZ.optimize_μ0_newton!(sbs, μ0s; decrement_tol=1e-20, maxiters=40, show_trace=true)
