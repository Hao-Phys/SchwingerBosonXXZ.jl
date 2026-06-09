module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix, setindex
using NLsolve
using Optim
using HCubature

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!, set_μ0!, set_ϕ!, set_classical_mean_fields!

include("CanonicalMeanField/FiniteSizeTools.jl")
export q_space_path_sbs

include("CanonicalMeanField/HamiltonianMeanField.jl")
include("CanonicalMeanField/DispersionAndIntensities.jl")
export excitations, dispersion, dssf_mean_field
include("CanonicalMeanField/GradientofHamiltonian.jl")

include("CanonicalMeanField/KuboMori.jl")
include("CanonicalMeanField/ObjectiveFunctions.jl")
export variational_free_energy

include("CanonicalMeanField/ExpectationValues.jl")
export expectation_values

include("CanonicalMeanField/Optimization.jl")
export optimize_mean_fields!, optimize_μ0!

# For legacy code that we want to keep for reference but not export.
include("CanonicalMeanField/Legacy.jl")

include("CanonicalMeanField/SelfConsistentEqns.jl")
export solve_self_consistent_mean_fields_condensed!

include("PathIntegral/SaddlePointGreensFunction.jl")
include("PathIntegral/ExternalVertices.jl")
include("PathIntegral/DSSF_SP.jl")
export dssf_SP

end
