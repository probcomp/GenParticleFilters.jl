## Various utility functions ##
export choiceproduct
export get_log_norm_weights, get_norm_weights
export effective_sample_size, get_ess
export log_ml_estimate, get_lml_est

using Gen: effective_sample_size, log_ml_estimate, sample_unweighted_traces

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
            f(i, stratum, args...)
        end
    end
    return nothing
end

"""
    choiceproduct((addr, vals))
    choiceproduct((addr, vals), choices::Tuple...)
    choiceproduct(choices::Dict)

Returns an iterator over `ChoiceMap`s given a tuple or sequence of tuples of
the form  `(addr, vals)`, where `addr` specifies a choice address, and
`vals` specifies a list of values for that address.
    
If multiple tuples are provided, the iterator will be a Cartesian product over
the `(addr, vals)` pairs, where each resulting `ChoiceMap` contains all
specified addresses. Instead of specifying multiple tuples, a dictionary mapping
addresses to values can also be provided.

# Examples

This function can be used to conveniently generate `ChoiceMap`s for stratified
sampling. For example, we can use `choiceproduct` instead of manually
constructing a list of strata:

```julia
# Manual construction
strata = [choicemap((:a,  1), (:b,  3)), choicemap((:a,  2), (:b,  3))]
# Using choiceproduct
strata = choiceproduct((:a, [1, 2]), (:b, [3]))
```
"""
function choiceproduct(choices::Tuple...)
    prod_iter = Iterators.product((((addr, v) for v in vals) for
                                  (addr, vals) in choices)...)
    return (choicemap(cs...) for cs in prod_iter)
end

function choiceproduct(choices::Dict)
    prod_iter = Iterators.product((((addr, v) for v in vals) for
                                  (addr, vals) in choices)...)
    return (choicemap(cs...) for cs in prod_iter)
end

function choiceproduct((addr, vals)::Tuple)
    return (choicemap((addr, v)) for v in vals)
end

lognorm(vs::AbstractVector) = vs .- logsumexp(vs)

"Computes the softmax of a vector of (unnormalized) log probabilities."
function softmax(vs::AbstractVector{T}) where {T <: Real}
    isempty(vs) && return T[]
    ws = exp.(vs .- maximum(vs))
    return ws ./ sum(ws)
end

"""
    probs, invalid = safe_softmax(vs; warn::Bool=true)

Returns the softmax of a vector of (unnormalized) log probabilities, and
a boolean indicating whether the result is invalid. Invalid outputs can occur if 
`vs` contains any `NaN` values, or if all weights sum to zero. Warning messages
are printed if `warn` is `true`.
"""
function safe_softmax(vs::AbstractVector{T}; warn::Bool=true) where {T <: Real}
    isempty(vs) && return T[]
    if any(isnan, vs)
        warn && @warn("NaN found in input values. Returning NaN weights.")
        ws = fill(convert(float(T), NaN), length(vs))
        return (ws, true)
    elseif all(==(-Inf), vs)
        warn && @warn("All input values are -Inf. Returning uniform weights.")
        ws = ones(float(T), length(vs)) ./ length(vs)
        return (ws, true)
    end
    ws = exp.(vs .- maximum(vs))
    total_w = sum(ws)
    if iszero(total_w)
        warn && @warn("All weights are zero. Returning uniform weights.")
        ws = ones(float(T), length(vs)) ./ length(vs)
        return (ws, true)
    elseif isnan(total_w)
        warn && @warn("Total weight is NaN. Returning NaN weights.")
        ws = fill(convert(float(T), NaN), length(vs))
        return (ws, true)
    end
    return (ws ./ sum(ws), false)
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

Alias for [`effective_sample_size`](@ref). Computes the effective sample size.
"""
get_ess(state::ParticleFilterView) = Gen.effective_sample_size(state)

# Extend to support sub-states
function Gen.log_ml_estimate(state::ParticleFilterSubState)
    n_particles = length(state.traces)
    source_lml_est = state.source.log_ml_est
    return source_lml_est + logsumexp(state.log_weights) - log(n_particles)
end

"""
    get_lml_est(state::ParticleFilterState)

Alias for [`log_ml_estimate`](@ref). Returns the particle filter's current 
estimate of the log marginal likelihood.
"""
get_lml_est(state::ParticleFilterView) = Gen.log_ml_estimate(state)

# Extend to support sub-states
function Gen.sample_unweighted_traces(state::ParticleFilterSubState,
                                      n_samples::Int)
    weights = get_norm_weights(state)
    traces = sample(state.traces, Weights(weights), n_samples)
    return traces
end
