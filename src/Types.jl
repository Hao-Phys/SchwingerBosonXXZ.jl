mutable struct SchwingerBosonSystem
    J :: Float64 # Nearest-neighbor exchange interaction (units of energy)
    Δ :: Float64 # XXZ anisotropy
    S :: Float64 # Spin magnitude
    T :: Float64 # Temperature (in units of J)
    L :: Int # Linear size of the system, Nu = L²
    mean_fields :: Vector{ComplexF64} # Dynamic variables to store mean fields
    μs :: Vector{Float64} # The chemical potentials (not the mean-field variables)
end

SchwingerBosonSystem(J::Float64, Δ::Float64, S::Float64, T::Float64, L::Int) = SchwingerBosonSystem(J, Δ, S, T, L, zeros(ComplexF64, 15), zeros(3))

function Base.show(io::IO, ::MIME"text/plain", sbs::SchwingerBosonSystem)
    (; J, Δ, S, T, L, mean_fields) = sbs
    printstyled(io, "SchwingerBosonSystem", "\n"; bold=true, color=:underline)
    println(io, "J = ", J, " Δ = ", Δ, " S = ", S, " T = ", T, " L = ", L)
    println("Mean field values: ")
    for i in 1:3
        println("A[$i]= ", mean_fields[i], " B[$i]= ", mean_fields[i+3])
        println("C[$i]= ", mean_fields[i+6], " D[$i]= ", mean_fields[i+9])
        println("μ0[$i]= ", mean_fields[i+12])
    end
end


function set_mean_fields!(sbs::SchwingerBosonSystem, mean_fields::Vector{ComplexF64})
    if length(mean_fields) ≠ 15
        throw(ArgumentError("Mean fields vector must have length 15."))
    end
    for i in 1:15
        sbs.mean_fields[i] = mean_fields[i]
    end
end

function set_mean_fields_scf!(sbs::SchwingerBosonSystem, mean_fields::Vector{ComplexF64})
    if length(mean_fields) ≠ 12
        throw(ArgumentError("Mean fields vector must have length 12."))
    end
    for i in 1:12
        sbs.mean_fields[i] = mean_fields[i]
    end
end

function set_x!(sbs::SchwingerBosonSystem, x)
    if length(x) ≠ 27
        throw(ArgumentError("Input vector must have length 27."))
    end
    for i in 1:12
        sbs.mean_fields[i] = x[i] + 1im * x[i+12]
    end
    for i in 1:3
        sbs.mean_fields[12+i] = x[24+i]
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