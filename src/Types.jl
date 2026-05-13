mutable struct SchwingerBosonSystem
    J :: Float64 # Nearest-neighbor exchange interaction (units of energy)
    Δ :: Float64 # XXZ anisotropy
    S :: Float64 # Spin magnitude
    T :: Float64 # Temperature (in units of J)
    L :: Int # Linear size of the system, Nu = L²
    h_SB :: Float64 # Small symmetry-breaking field
    θs :: Vector{Float64} # Angle of the symmetry-breaking field
    mean_fields :: Vector{ComplexF64} # Dynamic variables to store mean fields
    Δμs :: Vector{Float64} # The difference between μ (the physical chemical potential) and μ₀ (the mean-field (variational) chemical potential)
    α_dcoups :: Matrix{Float64} # Continuous parameter for the mean-field decoupling scheme, with dimensions (2, 3) for the two types of decouplings and three links
    condensation_ϵ :: Float64 # Energy threshold for identifying condensation (in units of J), default 0.0
end

# Constructor with default values for mean fields, Δμs, and α_dcoups
SchwingerBosonSystem(
    J::Float64,
    Δ::Float64,
    S::Float64,
    T::Float64,
    L::Int;
    h_SB::Float64 = 0.0,
    θs::Vector{Float64} = zeros(3),
    α_dcoups::Matrix{Float64} = zeros(2, 3),
    condensation_ϵ::Float64 = 0.0,
) = SchwingerBosonSystem(
    J,
    Δ,
    S,
    T,
    L,
    h_SB,
    θs,
    zeros(ComplexF64, 15),
    zeros(3),
    α_dcoups,
    condensation_ϵ,
)

function Base.show(io::IO, ::MIME"text/plain", sbs::SchwingerBosonSystem)
    (; J, Δ, S, T, L, mean_fields, α_dcoups) = sbs
    printstyled(io, "SchwingerBosonSystem", "\n"; bold=true, color=:underline)
    println(io, "J = ", J, " Δ = ", Δ, " S = ", S, " T = ", T, " L = ", L, " α_dcoups = ", α_dcoups)
    println(io, "Mean field values: ")
    for i in 1:3
        println("A[$i]= ", mean_fields[i], " B[$i]= ", mean_fields[i+3])
        println("C[$i]= ", mean_fields[i+6], " D[$i]= ", mean_fields[i+9])
        println("μ0[$i]= ", mean_fields[i+12])
    end
end

mutable struct CondensationAux
    positive_shift::Float64
    c_shift::Float64
    conden_index::Union{Int, Nothing}
    ξ::Float64
    num_conden::Int
end

function set_mean_fields!(sbs::SchwingerBosonSystem, mean_fields::Vector{ComplexF64})
    if length(mean_fields) ≠ 15
        throw(ArgumentError("Mean fields vector must have length 15."))
    end
    for i in 1:15
        sbs.mean_fields[i] = mean_fields[i]
    end
end

function set_ϕ!(sbs::SchwingerBosonSystem, ϕ)
    if length(ϕ) ≠ 24
        throw(ArgumentError("Mean field vector must have length 24."))
    end
    for i in 1:12
        sbs.mean_fields[i] = ϕ[i] + 1im * ϕ[i+12]
    end
end

function set_μ0!(sbs::SchwingerBosonSystem, μ0)
    if length(μ0) ≠ 3
        throw(ArgumentError("Mean-field chemical potential vector must have length 3."))
    end
    for i in 1:3
        sbs.mean_fields[12+i] = μ0[i]
    end
end

const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

# The z-component does not matter here, as we are only interested in the 2D model
# Our choice corresponds to c=10a
const recipvecs_reduce = Mat3([4π/3 2π/3 0; 0 2π/√3 0; 0 0 2π/10])
const recipvecs_origin = Mat3([2π 0 0; 2π/√3 4π/√3 0; 0 0 2π/10])

const Ĩ = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

const σs = [SMatrix{2, 2}([0.0 1.0; 1.0 0.0]), SMatrix{2, 2}([0.0 -1im; 1im 0.0]), SMatrix{2, 2}([1.0 0.0; 0.0 -1.0])]

function set_classical_mean_fields!(sbs::SchwingerBosonSystem, μ0s, θA, θB, θC, ϕA, ϕB, ϕC)
    (; S) = sbs
    @inline bup(θ, ϕ) = √2S * cos(θ/2) * exp(-im * ϕ/2)
    @inline bdw(θ, ϕ) = √2S * sin(θ/2) * exp( im * ϕ/2)

    A_AB = (bup(θA, ϕA) * bdw(θB, ϕB) - bdw(θA, ϕA) * bup(θB, ϕB)) / 2
    A_BC = (bup(θB, ϕB) * bdw(θC, ϕC) - bdw(θB, ϕB) * bup(θC, ϕC)) / 2
    A_CA = (bup(θC, ϕC) * bdw(θA, ϕA) - bdw(θC, ϕC) * bup(θA, ϕA)) / 2

    B_AB = conj(conj(bup(θA, ϕA)) * bup(θB, ϕB) + conj(bdw(θA, ϕA)) * bdw(θB, ϕB)) / 2
    B_BC = conj(conj(bup(θB, ϕB)) * bup(θC, ϕC) + conj(bdw(θB, ϕB)) * bdw(θC, ϕC)) / 2
    B_CA = conj(conj(bup(θC, ϕC)) * bup(θA, ϕA) + conj(bdw(θC, ϕC)) * bdw(θA, ϕA)) / 2

    C_AB = conj(conj(bup(θA, ϕA)) * bup(θB, ϕB) - conj(bdw(θA, ϕA)) * bdw(θB, ϕB)) / 2
    C_BC = conj(conj(bup(θB, ϕB)) * bup(θC, ϕC) - conj(bdw(θB, ϕB)) * bdw(θC, ϕC)) / 2
    C_CA = conj(conj(bup(θC, ϕC)) * bup(θA, ϕA) - conj(bdw(θC, ϕC)) * bdw(θA, ϕA)) / 2

    D_AB = (bup(θA, ϕA) * bdw(θB, ϕB) + bdw(θA, ϕA) * bup(θB, ϕB)) / 2
    D_BC = (bup(θB, ϕB) * bdw(θC, ϕC) + bdw(θB, ϕB) * bup(θC, ϕC)) / 2
    D_CA = (bup(θC, ϕC) * bdw(θA, ϕA) + bdw(θC, ϕC) * bup(θA, ϕA)) / 2

    set_mean_fields!(sbs, [A_AB, A_BC, A_CA, B_AB, B_BC, B_CA, C_AB, C_BC, C_CA, D_AB, D_BC, D_CA, μ0s...])
end