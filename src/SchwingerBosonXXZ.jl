module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix, setindex
using NLsolve
using Optim
using HCubature

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!, set_μ0!, set_ϕ!, set_classical_mean_fields!

include("FiniteSizeTools.jl")
export q_space_path_sbs

include("HamiltonianMeanField.jl")
include("DispersionAndIntensities.jl")
export excitations, dispersion, dssf_mean_field
include("GradientofHamiltonian.jl")

include("KuboMori.jl")
include("ObjectiveFunctions.jl")
export variational_free_energy

include("ExpectationValues.jl")
export expectation_values

include("Optimization.jl")
export optimize_mean_fields!, optimize_μ0!

# For legacy code that we want to keep for reference but not export.
include("Legacy.jl")

include("SelfConsistentEqns.jl")
export solve_self_consistent_mean_fields_condensed!

end
