mutable struct SchwingerBosonSystem
    J :: Float64 # Nearest-neighbor exchange interaction (units of energy)
    Δ :: Float64 # XXZ anisotropy
    S :: Float64 # Spin magnitude
    T :: Float64 # Temperature (units of energy)
    L :: Int # Linear size of the system, Nu = L²
    mean_fields :: Vector{ComplexF64} # Dynamic variables to store mean fields
end

SchwingerBosonSystem(J::Float64, Δ::Float64, S::Float64, T::Float64, L::Int) = SchwingerBosonSystem(J, Δ, S, T, L, zeros(ComplexF64, 15))

function set_mean_fields!(sbs::SchwingerBosonSystem, mean_fields::Vector{ComplexF64})
    if length(mean_fields) ≠ 15
        throw(ArgumentError("Mean fields vector must have length 15."))
    end
    sbs.mean_fields = mean_fields
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

const Vec3 = SVector{3, Float64}
const Mat3 = SMatrix{3, 3, Float64, 9}

# The z-component does not matter here, as we are only interested in the 2D model
# Our choice corresponds to c=10a
const recipvecs_reduce = Mat3([4π/3 2π/3 0; 0 2π/√3 0; 0 0 2π/10])
const recipvecs_origin = Mat3([2π 0 0; 2π/√3 4π/√3 0; 0 0 2π/10])

const Ĩ = Diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])