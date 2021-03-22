module GenParticleFilters

using Gen, Distributions
using Gen: ParticleFilterState

export ParticleFilterState, ParticleFilterSubState, ParticleFilterView

include("view.jl")
include("utils.jl")
include("initialize.jl")
include("update.jl")
include("resample.jl")
include("rejuvenate.jl")

end
