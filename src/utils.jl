## Various utility functions ##

export get_log_norm_weights, effective_sample_size, get_ess

"""
    log_norm_weights = get_log_norm_weights(state::ParticleFilterState)
Return the vector of normalized log weights for the current state,
one for each particle.
"""
get_log_norm_weights(state::ParticleFilterState) =
    state.log_weights .- logsumexp(state.log_weights)

"""
    ess = effective_sample_size(state::ParticleFilterState)
Computes the effective sample size of the particles in the filter.
"""
Gen.effective_sample_size(state::ParticleFilterState) =
    Gen.effective_sample_size(get_log_norm_weights(state))

"""
    ess = get_ess(state::ParticleFilterState)
Alias for `effective_sample_size`(@ref). Computes the effective sample size.
"""
get_ess(state::ParticleFilterState) = effective_sample_size(state)
