struct ParticleFilterSubState{U,I,L}
    traces::SubArray{U,1,Vector{U},I,L}
    new_traces::SubArray{U,1,Vector{U},I,L}
    log_weights::SubArray{Float64,1,Vector{Float64},I,L}
    parents::SubArray{Int,1,Vector{Int},I,L}
end

Gen.get_traces(state::ParticleFilterSubState) = state.traces
Gen.get_log_weights(state::ParticleFilterSubState) = state.log_weights

const ParticleFilterView{U} =
    Union{ParticleFilterState{U}, ParticleFilterSubState{U}} where {U}

function Base.view(state::ParticleFilterState{U},
                   indices::AbstractVector) where {U}
    L = Base.viewindexing((indices,)) == IndexLinear()
    return ParticleFilterSubState{U,Tuple{typeof(indices)},L}(
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
