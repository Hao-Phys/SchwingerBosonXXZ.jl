module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!
include("HamiltonianMeanField.jl")
include("DispersionAndIntensities.jl")
export excitations, dispersion
include("FreeEnergy.jl")
include("GradientofHamiltonian.jl")
include("GradientofFreeEnergy.jl")

end
