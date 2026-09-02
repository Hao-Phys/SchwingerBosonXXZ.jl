module SchwingerBosonXXZ

using LinearAlgebra
import StaticArrays: SVector, SMatrix, setindex
using NLsolve
using Optim
using HCubature

include("Types.jl")
export SchwingerBosonSystem, set_mean_fields!, set_μ0!, set_ϕ!, set_classical_mean_fields!
export set_external_field!
export SpectralCondensationAux, easy_axis_angles, easy_plane_angles

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

include("SpectralCondensation.jl")
export spectral_condensation_aux

# For legacy code that we want to keep for reference but not export.
include("CanonicalMeanField/Legacy.jl")

include("CanonicalMeanField/SelfConsistentEqns.jl")
export solve_self_consistent_mean_fields_condensed!

include("PoleOccupationFactors.jl")

include("PathIntegral/SaddlePointGreensFunction.jl")
include("PathIntegral/ExternalVertices.jl")

include("PathIntegral/InternalVertices.jl")
include("PathIntegral/RPAPropagator.jl")
include("PathIntegral/ExternalInternalBubble.jl")

include("PathIntegral/DSSF.jl")
export dssf_SP, dssf_FL

include("SaddlePointFreeEnergy.jl")
export saddle_point_free_energy

end
