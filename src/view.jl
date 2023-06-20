@doc """
    ParticleFilterState

A data structure that represents the current state of a particle filter.
"""
ParticleFilterState

"""
    ParticleFilterSubState

A data structure that represents a view into a subset of traces in a
particle filter. If `state` is a [`ParticleFilterState`](@ref), then
`state[idxs]` or `view(state, idxs)` can be used to construct a
`ParticleFilterSubState` which only contains the traces at the specified `idxs`.
"""
struct ParticleFilterSubState{U,S,I,L}
    source::S
    traces::SubArray{U,1,Vector{U},I,L}
    new_traces::SubArray{U,1,Vector{U},I,L}
    log_weights::SubArray{Float64,1,Vector{Float64},I,L}
    parents::SubArray{Int,1,Vector{Int},I,L}
end

# Unclear how one could copy a substate without copying the whole state.
Base.copy(::ParticleFilterSubState) =
    error("Cannot copy a particle filter substate. If needed, copy the whole particle filter state.")

Gen.get_traces(state::ParticleFilterSubState) = state.traces
Gen.get_log_weights(state::ParticleFilterSubState) = state.log_weights

"""
    ParticleFilterView

Union of `ParticleFilterState` and `ParticleFilterSubState`.
"""
const ParticleFilterView{U} =
    Union{ParticleFilterState{U}, ParticleFilterSubState{U}} where {U}

function Base.view(state::ParticleFilterState{U},
                   indices::AbstractVector) where {U}
    L = Base.viewindexing((indices,)) == IndexLinear()
    return ParticleFilterSubState{U,typeof(state),Tuple{typeof(indices)},L}(
        state,
        view(state.traces, indices),
        view(state.new_traces, indices),
        view(state.log_weights, indices),
        view(state.parents, indices)
    )
end

Base.getindex(state::ParticleFilterState, indices) =
    Base.view(state, indices)

Base.firstindex(state::ParticleFilterState) = 1
Base.lastindex(state::ParticleFilterState) = length(state.traces)
