## Various utility functions ##
export get_log_norm_weights, get_norm_weights
export effective_sample_size, get_ess
export log_ml_estimate, get_lml_est

using Gen: effective_sample_size, log_ml_estimate

"Replace traces with newly updated traces."
@inline function update_refs!(state::ParticleFilterState)
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end

@inline function update_refs!(state::ParticleFilterSubState)
    # Perform assignment
    state.traces[:] = state.new_traces
end

lognorm(vs::AbstractVector) = vs .- logsumexp(vs)

function softmax(vs::AbstractVector{T}) where {T <: Real}
    if isempty(vs) return T[] end
    ws = exp.(vs .- maximum(vs))
    return ws ./ sum(ws)
end

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
get_norm_weights(state::ParticleFilterView) = softmax(state.log_weights)

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

function Gen.log_ml_estimate(state::ParticleFilterSubState)
    n_particles = length(state.traces)
    source_lml_est = state.source.log_ml_est
    return source_lml_est + logsumexp(state.log_weights) - log(n_particles)
end

"""
    get_lml_est(state::ParticleFilterState)

Alias for `log_ml_estimate`(@ref). Returns the particle filter's current 
estimate of the log marginal likelihood.
"""
get_lml_est(state::ParticleFilterView) = Gen.log_ml_estimate(state)
