module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix
using NLsolve
using Optim

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!, set_x!
include("HamiltonianMeanField.jl")
include("DispersionAndIntensities.jl")
export excitations, dispersion
include("FreeEnergy.jl")
include("GradientofHamiltonian.jl")
include("ObjectiveFunctions.jl")

include("ExpectationValues.jl")
export expectation_values, spin_expectations
include("SelfConsistentEqns.jl")
export solve_self_consistent_mean_fields!

include("Optimization.jl")
export optimize_mean_fields!, optimize_μ!

include("NewtonBacktracking.jl")
include("CondensedSectorTreatment.jl")

end
