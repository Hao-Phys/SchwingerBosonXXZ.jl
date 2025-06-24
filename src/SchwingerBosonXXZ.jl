module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix
using NLsolve

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!
include("HamiltonianMeanField.jl")
include("DispersionAndIntensities.jl")
export excitations, dispersion
include("FreeEnergy.jl")
include("GradientofHamiltonian.jl")
include("GradientofFreeEnergy.jl")

include("Constraints.jl")
include("GradientofConstraints.jl")

include("ExpectationValues.jl")
export expectation_values
include("SelfConsistentEqns.jl")
export solve_self_consistent_mean_fields!

end
