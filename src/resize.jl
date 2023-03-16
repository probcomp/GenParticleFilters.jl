## Functions for resizing particle filters ##
export pf_resize!, pf_multinomial_resize!, pf_residual_resize!
export pf_replicate!, pf_dereplicate!

"""
    pf_resize!(state::ParticleFilterState, n_particles::Int,
               method=:multinomial; kwargs...)

Resizes a particle filter by resampling existing particles until a total of 
`n_particles` have been sampled. The resampling method can optionally be
specified: `:multinomial` (default) or `:residual`.

A `priority_fn` can also be specified as a keyword argument, which maps log
particle weights to custom log priority scores for the purpose of resampling
(e.g. `w -> w/2` for less aggressive pruning).
"""
function pf_resize!(state::ParticleFilterState, n_particles::Int,
                    method::Symbol=:multinomial; kwargs...)
    if method == :multinomial
        return pf_multinomial_resize!(state, n_particles; kwargs...)
    elseif method == :residual
        return pf_residual_resize!(state, n_particles; kwargs...)
    else
        error("Resampling method $method not recognized.")
    end
end

"""
    pf_multinomial_resize!(state::ParticleFilterState, n_particles::Int;
                           kwargs...)

Resizes a particle filter through multinomial resampling (i.e. simple random
resampling) of existing particles until `n_particles` are sampled. Each trace
(i.e. particle) is resampled with probability proportional to its weight.

A `priority_fn` can be specified as a keyword argument, which maps log particle
weights to custom log priority scores for the purpose of resampling
(e.g. `w -> w/2` for less aggressive pruning).
"""
function pf_multinomial_resize!(state::ParticleFilterState, n_particles::Int;
                                priority_fn=nothing)
    # Update estimate of log marginal likelihood
    update_lml_est!(state)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn === nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Resize arrays
    resize!(state.parents, n_particles)
    resize!(state.new_traces, n_particles)
    # Resample new traces according to current normalized weights
    weights = softmax(log_priorities)
    rand!(Categorical(weights), state.parents)
    state.new_traces .= view(state.traces, state.parents)
    # Reweight particles and update trace references
    update_weights!(state, n_particles, log_priorities)
    update_refs!(state, n_particles)
    return state
end

"""
    pf_residual_resize!(state::ParticleFilterState, n_particles::Int; kwargs...)

Resizes a particle filter through residual resampling of existing particles.
For each particle with normalized weight ``w_i``, ``⌊n w_i⌋`` copies are
resampled, w here ``n`` is `n_particles`. The remainder are sampled with
probability proportional to ``n w_i - ⌊n w_i⌋`` for each particle ``i``.

A `priority_fn` can be specified as a keyword argument, which maps log particle
weights to custom log priority scores for the purpose of resampling
(e.g. `w -> w/2` for less aggressive pruning).
"""
function pf_residual_resize!(state::ParticleFilterState, n_particles::Int;
                             priority_fn=nothing)
    # Update estimate of log marginal likelihood
    update_lml_est!(state)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn === nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Resize arrays
    resize!(state.parents, n_particles)
    resize!(state.new_traces, n_particles)
    # Deterministically copy previous particles according to their weights
    n_resampled = 0
    weights = softmax(log_priorities)
    for (i, w) in enumerate(weights)
        n_copies = floor(Int, n_particles * w)
        if n_copies == 0 continue end
        state.parents[n_resampled+1:n_resampled+n_copies] .= i
        for j in 1:n_copies
            state.new_traces[n_resampled+j] = state.traces[i]
        end
        n_resampled += n_copies
    end
    # Sample remainder according to residual weights
    if n_resampled < n_particles
        r_weights = n_particles .* weights .- floor.(n_particles .* weights)
        r_weights = r_weights / sum(r_weights)
        r_idxs = n_resampled+1:n_particles
        r_parents = view(state.parents, r_idxs)
        rand!(Categorical(r_weights), r_parents)
        state.new_traces[r_idxs] .= view(state.traces, r_parents)
    end
    # Reweight particles and update trace references
    update_weights!(state, n_particles, log_priorities)
    update_refs!(state, n_particles)
    return state
end

"""
    pf_replicate!(state::ParticleFilterState, n_replicates::Int;
                  layout=:contiguous)

Expands a particle filter by replicating each particle `n_replicates` times. 

If `layout` is `:contiguous`, each particle's replicates will be arranged in a
continguous block (e.g., replicates of the first particle will have indices
`1:n_replicates`).

Otherwise, if `layout` is `:interleaved`, each particle's replicates will be
interleaved with the replicates of the other particles (e.g., replicates of the
first particle will have indices `1:n_replicates:N*n_replicates`, where `N` is
the original number of particles).
"""
function pf_replicate!(state::ParticleFilterState, n_replicates::Int;
                       layout::Symbol=:contiguous)
    _repeat(x, k) = layout == :contiguous ? repeat(x; inner=k) : repeat(x, k)
    state.parents = _repeat(eachindex(state.traces), n_replicates)
    state.new_traces = _repeat(state.traces, n_replicates) 
    state.log_weights = _repeat(state.log_weights, n_replicates)
    update_refs!(state, length(state.new_traces))
    return state
end

"""
    pf_dereplicate!(state::ParticleFilterState, n_replicates::Int;
                    layout=:contiguous, method=:keepfirst)

Shrinks a particle filter by retaining one out of every `n_replicates`
particles. The total number of particles must be a multiple of `n_replicates`.

If `layout` is `:contiguous`, each set of replicates is assumed to be arranged
in a contiguous block (e.g., the first particle will be selected out of the
indices `1:n_replicates`).

Otherwise, if `layout` is `:interleaved`, each set of replicates is assumed to
be interleaved with others (e.g., the first particle will be selected out of
the indices `1:n_replicates:N`, where `N` is the total number of particles).

Retained particles can be selected by different methods. If `method` is
`:keepfirst`, only the first particle of each set of replicates is retained,
exactly reversing the effect of [`pf_replicate!`](@ref). If `method` is 
`:sample`, the retained particle is sampled according to its normalized weight
among the replicates.
"""
function pf_dereplicate!(state::ParticleFilterState, n_replicates::Int;
                         layout::Symbol=:contiguous, method::Symbol=:keepfirst)
    n_old = length(state.traces)
    @assert n_old % n_replicates == 0
    n_new = n_old ÷ n_replicates
    if method == :keepfirst
        # Select first particle in each block of replicates
        idxs = layout == :contiguous ? (1:n_replicates:n_old) :  1:n_new
        # Retain original weight of selected particles
        state.log_weights = state.log_weights[idxs]
    else # if method == :sample
        block_iter = layout == :contiguous ?
            (((k-1)*n_replicates+1):k*n_replicates for k in 1:n_new) :
            (k:n_new:n_old for k in 1:n_new)
        # Sample particle from each block of replicates
        idxs = map(block_iter) do block_idxs
            weights = softmax(view(state.log_weights, block_idxs))
            return block_idxs[rand(Categorical(weights))]
        end
        # Set new weight to the average particle weight of each block
        log_weights = map(block_iter) do block_idxs
            log_total_weight = logsumexp(view(state.log_weights, block_idxs))
            return log_total_weight - log(n_replicates)
        end
        state.log_weights = log_weights
    end
    state.parents = collect(idxs)
    state.new_traces = state.traces[idxs]
    update_refs!(state, n_new)
    return state
end

"Update particle weights after a resizing step."
function update_weights!(state::ParticleFilterState, n_particles::Int,
                         log_priorities)
    # Ensure logsumexp(log_weights) == log(n_particles) after resampling
    if log_priorities === state.log_weights
        # If priorities aren't customized, set all log weights to 0
        resize!(state.log_weights, n_particles)
        state.log_weights .= 0.0
    else
        # Otherwise, set new weights to the ratio of weights over priorities
        log_ws = state.log_weights[state.parents] .- log_priorities[state.parents]
        # Adjust new weights such that they sum to the number of particles
        resize!(state.log_weights, n_particles)
        state.log_weights .= log_ws .+ (log(n_particles) - logsumexp(log_ws))
    end
end

"Replace traces with newly updated traces after a resizing step."
@inline function update_refs!(state::ParticleFilterState, n_particles::Int)
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    # Resize to ensure dimensions match
    @assert length(state.traces) == n_particles
    resize!(state.new_traces, n_particles)
end
