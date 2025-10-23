using SchwingerBosonXXZ
using LinearAlgebra

S = 0.5
T = 1e-3
Δ = 10.0
L = 6
h_SB = 0.0
θ_ref = acos(Δ/(1+Δ))
θs_classical = [π+θ_ref, 0, π-θ_ref]
sbs = SchwingerBosonSystem(1.0, Δ, S, T, L, h_SB, θs_classical)

SchwingerBosonXXZ.set_ϕ!(sbs, [8.750379094826187, 8.750379094826187, -5.641195568904617, -5.070418856079793, -5.070418854217148, 16.912012829021975, -2.461780953194117, -2.4617809569194073, -11.103228360414505, -6.776314275311435, 6.776314297663177, -8.257351857832163e-9, 1.2731075117956328e-10, -1.2731080875876042e-10, 1.9367477606791106e-16, -2.7781489058382375e-11, -2.778163921238198e-11, -1.3030682404844625e-10, -2.273036410435694e-11, -2.2730420298840783e-11, 9.692240420805887e-11, -1.0416333649883396e-10, -1.0416341987166549e-10, 4.4415515843147333e-11])
μ0s = [-322.50568428873504, -322.50568439207433, -322.50568428873504]
SchwingerBosonXXZ.set_μ0!(sbs, μ0s)

# @time SchwingerBosonXXZ.optimize_μ0_newton!(sbs, μ0s; decrement_tol=1e-20, armijo_α_min=1e-12, maxiters=200, show_trace=true, verbose=false)

# SchwingerBosonXXZ.optimize_μ0_newton!(sbs, μ0s; g_abstol=1e-6, maxiters=100, armijo_c=1e-4, armijo_backoff=0.5, armijo_α_min=1e-12, show_trace=true)


#####

using Optim

algorithm = Optim.ConjugateGradient()
g_abstol = 1e-8
options = Optim.Options(; show_trace=true, iterations=1000, g_abstol, x_abstol=NaN, x_reltol=NaN, f_reltol=NaN, f_abstol=NaN)
SchwingerBosonXXZ.optimize_μ0!(sbs, μ0s; algorithm, options)
