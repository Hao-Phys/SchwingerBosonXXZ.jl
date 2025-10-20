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
SchwingerBosonXXZ.set_μ0!(sbs, [-100.0, -100.0, -100.0])

μ0s = [-100.0, -100.0, -100.0]
SchwingerBosonXXZ.optimize_μ0_newton!(sbs, μ0s; g_abstol=1e-6, maxiters=100, armijo_c=1e-4, armijo_backoff=0.5, armijo_α_min=1e-12, show_trace=true)
