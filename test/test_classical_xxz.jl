using SchwingerBosonXXZ
using LinearAlgebra

S = 0.5
T = 1e-4
Δ = 10.0
h_SB = 0.0
θ_ref = acos(Δ/(1+Δ))
θs_classical = [π+θ_ref, 0, π-θ_ref]
sbs = SchwingerBosonSystem(1.0, Δ, S, T, 12, h_SB, θs_classical)

ψA = 0.0
ψB = 0.0
ψC = 0.0
A_AB =  2 * exp(1im * (ψA + ψB)) * S^2 * cos(θ_ref/2)
A_BC =  2 * exp(1im * (ψB + ψC)) * S^2 * cos(θ_ref/2)
A_CA = -2 * exp(1im * (ψC + ψA)) * S^2 * sin(θ_ref)
B_AB =  2 * exp(1im * (ψA - ψB)) * S^2 * sin(θ_ref/2)
B_BC =  2 * exp(1im * (ψB - ψC)) * S^2 * sin(θ_ref/2)
B_CA = -2 * exp(1im * (ψC - ψA)) * S^2 * cos(θ_ref)
C_AB =  2 * exp(1im * (ψA - ψB)) * S^2 * sin(θ_ref/2)
C_BC =  2 * exp(1im * (ψB - ψC)) * S^2 * sin(θ_ref/2)
C_CA =  2 * exp(1im * (ψC - ψA)) * S^2
D_AB = -2 * exp(1im * (ψA + ψB)) * S^2 * cos(θ_ref/2)
D_BC =  2 * exp(1im * (ψB + ψC)) * S^2 * cos(θ_ref/2)
D_CA =  0.0
mean_fields = [A_AB, A_BC, A_CA, B_AB, B_BC, B_CA, C_AB, C_BC, C_CA, D_AB, D_BC, D_CA, -50.0, -50.0, -50.0]

set_mean_fields!(sbs, mean_fields)

μ0s = zeros(3)
for i in 1:3
    μ0s[i] = real(sbs.mean_fields[i+12])
end

# SchwingerBosonXXZ.optimize_μ0!(sbs, μ0s; algorithm=Optim.GradientDescent(), options = Optim.Options(show_trace=true, iterations=100, extended_trace=true))
SchwingerBosonXXZ.optimize_μ0_newton!(sbs, μ0s; g_abstol=1e-6, maxiters=100, armijo_α_min=1e-12, show_trace=true)


############

x = [-1.5000054912524656, -1.5003065230744614, -1.5000054912524659]
x_good = [-1.5000005912524657, -1.5000005230744615, -1.5000005912524659]

(; f, g, h) = SchwingerBosonXXZ.fgh_μ0(sbs, x)
f
norm(g)
eigvals(h)

(; f, g, h) = SchwingerBosonXXZ.fgh_μ0(sbs, x_good)
f
norm(g)
eigvals(h)


############

using GLMakie

cs = collect(range(0.98, 1.0026; length=400))
fs = Float64[]
dfs = Float64[]
d²fs = Float64[]

for c in cs
    μ0 = (1-c) * x + c * x_good # x + c Δx   (Δx = x_good - x)
    (; f, g, h) = SchwingerBosonXXZ.fgh_μ0(sbs, μ0)

    Δx = x_good - x
    # df/dc = df/dμ₀ ⋅ dμ₀/dc = g ⋅ Δx
    df = g' * Δx

    # d²f/dc² = (d/dμ₀)df/dc ⋅ dμ₀/dc = (d/dμ₀) (g ⋅ Δx) ⋅ Δx
    #         = Δxᵀ h Δx
    d²f = Δx' * h * Δx

    push!(fs, f)
    push!(dfs, df)
    push!(d²fs, d²f)
end

# df/dc using finite difference
dc = cs[2] - cs[1]
cs′ = (cs[2:end] + cs[1:end-1]) / 2
dfs′ = (fs[2:end] - fs[1:end-1]) / dc

# d²f/dc² using finite difference
cs″ = cs[2:end-1]
d²fs″ = (fs[1:end-2] - 2fs[2:end-1] + fs[3:end]) / dc^2


fig = Figure()
lines(fig[1, 1], cs, fs; label="f")
lines(fig[2, 1], cs, dfs; label="Analytic df/dc")
lines!(fig[2, 1], cs′, dfs′; label="FD df/dc")
lines(fig[3, 1], cs, d²fs; label="Analytic d²f/dc²")
lines!(fig[3, 1], cs″, d²fs″; label="FD d²f/dc²")
fig
