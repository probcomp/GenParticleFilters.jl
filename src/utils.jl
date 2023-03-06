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

"""
    stratified_map!(f, n_total, strata, args...; layout=:contiguous)

Maps the function `f` in a stratified fashion given `strata`. The function
`f` should have the form `f(i, stratum, args...)`, and specify how the `i`th
index should be handled given the `stratum`.
"""
function stratified_map!(f::Function, n_total::Int, strata, args...;
                         layout=:contiguous)
    n_strata = length(strata)
    block_size = n_total ÷ n_strata
    # Update strata in a contiguous or interleaved manner
    for (k, stratum) in enumerate(strata)
        if layout == :contiguous
            idxs = ((k-1)*block_size+1):k*block_size
        else # layout == :interleaved
            idxs = k:n_strata:n_strata*block_size
        end
        for i in idxs
            f(i, stratum, args...)
        end
    end
    # Allocate remaining indices to random strata
    n_remaining = n_total - n_strata * block_size
    if n_remaining > 0
        strata = strata isa Vector ? strata : collect(strata)
        remainder = sample(strata, n_remaining)
        for (k, stratum) in enumerate(remainder)
            i = n_total - n_remaining + k
            for i in idxs
                f(i, stratum, args...)
            end
        end
    end
    return nothing
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
