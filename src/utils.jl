## Various utility functions ##
export get_log_norm_weights, get_norm_weights
export effective_sample_size, get_ess
export mean, var

using Gen: effective_sample_size
using Statistics

@inline function update_refs!(state::ParticleFilterState)
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end

@inline function update_refs!(state::ParticleFilterSubState)
    state.traces[:] = state.new_traces
end

lognorm(v::AbstractVector) = v .- logsumexp(v)

"""
    get_log_norm_weights(state::ParticleFilterState)

Return the vector of normalized log weights for the current state,
one for each particle.
"""
get_log_norm_weights(state::ParticleFilterView) = lognorm(state.log_weights)

"""
    get_norm_weights(state::ParticleFilterState)

Return the vector of normalized weights for the current state,
one for each particle.
"""
get_norm_weights(state::ParticleFilterView) = exp.(get_log_norm_weights(state))

"""
    effective_sample_size(state::ParticleFilterState)

Computes the effective sample size of the particles in the filter.
"""
Gen.effective_sample_size(state::ParticleFilterView) =
    Gen.effective_sample_size(get_log_norm_weights(state))

"""
    get_ess(state::ParticleFilterState)

Alias for `effective_sample_size`(@ref). Computes the effective sample size.
"""
get_ess(state::ParticleFilterView) = Gen.effective_sample_size(state)

"""
    mean(state::ParticleFilterState[, addr])

Returns the weighted empirical mean for a particular trace address `addr`.
If `addr` is not provided, returns the empirical mean of the return value.
"""
Statistics.mean(state::ParticleFilterView, addr) =
    sum(get_norm_weights(state) .* getindex.(state.traces, addr))

Statistics.mean(state::ParticleFilterView) =
    sum(get_norm_weights(state) .* get_retval.(state.traces))

"""
    var(state::ParticleFilterState[, addr])

Returns the empirical variance for a particular trace address `addr`.
If `addr` is not provided, returns the empirical variance of the return value.
"""
Statistics.var(state::ParticleFilterView, addr) =
    sum(get_norm_weights(state) .*
        (getindex.(state.traces, addr) .- mean(state, addr)).^2)

Statistics.var(state::ParticleFilterView) =
    sum(get_norm_weights(state) .*
        (get_retval.(state.traces) .- mean(state)).^2)
