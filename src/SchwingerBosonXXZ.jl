module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix
using NLsolve
using Optim

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!, set_μ0!, set_ϕ!
include("HamiltonianMeanField.jl")
include("DispersionAndIntensities.jl")
export excitations, dispersion
include("GradientofHamiltonian.jl")

include("KuboMori.jl")
include("ObjectiveFunctions.jl")

include("ExpectationValues.jl")
export expectation_values

include("Optimization.jl")
export optimize_mean_fields!, optimize_μ0!

# For legacy code that we want to keep for reference but not export.
include("Legacy.jl")

end
