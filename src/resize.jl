## Functions for resizing particle filters ##
export pf_resize!
export pf_multinomial_resize!, pf_residual_resize!, pf_optimal_resize!
export pf_replicate!, pf_dereplicate!
export pf_coalesce!
export pf_introduce!

"""
    pf_resize!(state::ParticleFilterState, n_particles::Int,
               method=:multinomial; kwargs...)

Resizes a particle filter by resampling existing particles until a total of 
`n_particles` have been sampled. The resampling method can optionally be
specified: `:multinomial` (default), `:residual` or `:optimal`.

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
    elseif method == :optimal
        return pf_optimal_resize!(state, n_particles; kwargs...)
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
resampled, where ``n`` is `n_particles`. The remainder are sampled with
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
    pf_optimal_resize!(state::ParticleFilterState, n_particles::Int;
                       kwargs...)

Resizes a particle filter through the optimal resampling algorithm for 
Fearnhead and Clifford [1]. This guarantees that each resampled particle is 
unique as long as all of the original particles are unique, while minimizing 
the variance of the resulting weight distribution with respect to the original
weight distribution. Note that `n_particles` should not be greater than the
current number of particles.

[1] Paul Fearnhead , Peter Clifford, On-Line Inference for Hidden Markov Models
via Particle Filters, Journal of the Royal Statistical Society Series B:
Statistical Methodology, Volume 65, Issue 4, November 2003, Pages 887–899,
https://doi.org/10.1111/1467-9868.00421
"""
function pf_optimal_resize!(state::ParticleFilterState, n_particles::Int;
                            kwargs...)
    # Resize arrays
    n_old = length(state.traces)
    @assert n_particles <= n_old
    resize!(state.parents, n_particles)
    resize!(state.new_traces, n_particles)
    # Normalize weights and compute inverse weight threshold
    weights = softmax(state.log_weights)
    inv_w_thresh = find_inv_w_threshold(weights, n_particles)
    # Find particles to keep deterministically vs. resample with stratification
    keep_idxs = (inv_w_thresh .* weights .>= 1)
    strat_idxs = .!(keep_idxs)
    # Keep selected indices
    keep_idxs = findall(keep_idxs)
    n_keep = length(keep_idxs)
    state.parents[1:n_keep] .= keep_idxs
    # Perform stratified resampling on remaining indices
    n_resample = n_particles - n_keep
    resample_idxs = Int[]
    strat_idxs = findall(strat_idxs)
    n_strat = length(strat_idxs)
    norm_strat_weights = softmax(state.log_weights[strat_idxs])
    # Compute resampled indices
    step_size = 1 / n_resample
    u = rand() * step_size
    for i in 1:n_strat
        u = u - norm_strat_weights[i]
        if u < 0
            push!(resample_idxs, strat_idxs[i])
            u += step_size
        end
    end
    # Keep resampled indices
    @assert length(resample_idxs) == n_resample
    state.parents[n_keep+1:n_particles] .= resample_idxs
    # Update weights
    log_n_ratio = log(n_particles) - log(n_old)
    log_tot_weight = logsumexp(state.log_weights)
    keep_log_weights = state.log_weights[keep_idxs]
    resample_log_weight = log_tot_weight - log(inv_w_thresh)
    resize!(state.log_weights, n_particles)
    state.log_weights[1:n_keep] .= keep_log_weights .+ log_n_ratio
    state.log_weights[n_keep+1:n_particles] .= resample_log_weight .+ log_n_ratio
    # Update trace references
    state.new_traces .= view(state.traces, state.parents)
    update_refs!(state, n_particles)
    return state
end

"Finds inverse weight threshold for optimal particle filter resizing."
function find_inv_w_threshold(weights, n_particles::Int)
    weights = sort(weights)
    # Find threshold weight κ
    A = length(weights) # Number of weights greater than κ so far
    B = 0.0 # Sum of weights less than or equal to κ so far
    for κ in weights
        A -= 1
        B += κ
        # Check that κ meets the threshold condition
        n_check = B / κ + A 
        if n_check <= n_particles + eps(n_check)
            # Return inverse weight c such that B * c + A = N exactly
            return (n_particles - A) / B
        end
    end
    return float(n_particles)
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

"""
    pf_coalesce!(state::ParticleFilterState; by=get_choices)

Coalesces traces that are equivalent according to `by` (defaulting to
`get_choices`). Each set of equivalent traces is replaced by a single trace with 
weight equal to the sum of the original weights, multiplied by the ratio of 
the number of coalesced traces to the number of original traces.

To ensure that coalesced traces are *identical*, set `by` to `identity`.
"""
function pf_coalesce!(state::ParticleFilterState; by=get_choices)
    if isempty(state.traces) return state end
    # Apply transformation to traces
    vals = by === identity ? state.traces : (by(tr) for tr in state.traces)
    # Combine weights and extract indices for equivalent traces
    coalesced_idxs = Dict{eltype(vals), Int}()
    coalesced_weights = zeros(length(state.traces))
    for (idx, (v, w)) in enumerate(zip(vals, state.log_weights))
        idx = get!(coalesced_idxs, v, idx)
        coalesced_weights[idx] += exp(w)
    end
    # Update weights and new traces
    n_old = length(state.traces)
    n_particles = length(coalesced_idxs)
    log_n_ratio = log(n_particles) - log(n_old)
    for (new_idx, old_idx) in enumerate(values(coalesced_idxs))
        state.parents[new_idx] = old_idx
        state.new_traces[new_idx] = state.traces[old_idx]
        state.log_weights[new_idx] = log(coalesced_weights[old_idx]) + log_n_ratio
    end
    resize!(state.log_weights, n_particles)
    resize!(state.new_traces, n_particles)
    resize!(state.parents, n_particles)
    update_refs!(state, n_particles)
    return state
end

"""
    pf_introduce!(state::ParticleFilterState,
                  [model::GenerativeFunction, model_args::Tuple,]
                  observations::ChoiceMap, n_particles::Int)

    pf_introduce!(state::ParticleFilterState,
                  [model::GenerativeFunction, model_args::Tuple,]
                  observations::ChoiceMap, proposal::GenerativeFunction,
                  proposal_args::Tuple, n_particles::Int)

Introduce `n_particles` new traces into a particle filter, constrained to the
provided `observations`. If `model` and `model_args` are omitted, then
this function will use the same model and arguments as the first trace in the
particle filter. A custom `proposal` can be used to propose choices.
"""
function pf_introduce!(
    state::ParticleFilterState,
    model::Union{Nothing, GenerativeFunction},
    model_args::Union{Nothing, Tuple},
    observations::ChoiceMap,
    n_particles::Int;
) 
    model = isnothing(model) ? get_gen_fn(state.traces[1]) : model
    model_args = isnothing(model_args) ? get_args(state.traces[1]) : model_args
    n_old = length(state.traces)
    # Adjust weights of existing particles
    if state.log_ml_est != 0.0
        state.log_weights .+= state.log_ml_est
        state.log_ml_est = 0.0
    end
    # Resize arrays
    resize!(state.traces, n_old + n_particles)
    resize!(state.log_weights, n_old + n_particles)
    resize!(state.parents, n_old + n_particles)
    resize!(state.new_traces, n_old + n_particles)
    # Generate new traces
    for i=1:n_particles
        state.traces[n_old+i], state.log_weights[n_old+i] =
            generate(model, model_args, observations)
    end
    return state
end

function pf_introduce!(state::ParticleFilterState, observations::ChoiceMap,
                       n_particles::Int)
    return pf_introduce!(state, nothing, nothing, observations, n_particles)
end

function pf_introduce!(
    state::ParticleFilterState,
    model::Union{Nothing, GenerativeFunction},
    model_args::Union{Nothing, Tuple},
    observations::ChoiceMap,
    proposal::GenerativeFunction,
    proposal_args::Tuple,
    n_particles::Int;
) 
    model = isnothing(model) ? get_gen_fn(state.traces[1]) : model
    model_args = isnothing(model_args) ? get_args(state.traces[1]) : model_args
    n_old = length(state.traces)
    # Adjust weights of existing particles
    if state.log_ml_est != 0.0
        state.log_weights .+= state.log_ml_est
        state.log_ml_est = 0.0
    end
    # Resize arrays
    resize!(state.traces, n_old + n_particles)
    resize!(state.log_weights, n_old + n_particles)
    resize!(state.parents, n_old + n_particles)
    resize!(state.new_traces, n_old + n_particles)
    # Generate new traces
    for i=1:n_particles
        (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
        (state.traces[n_old+i], model_weight) =
            generate(model, model_args, merge(observations, prop_choices))
        state.log_weights[n_old+i] = model_weight - prop_weight
    end
    return state
end

function pf_introduce!(state::ParticleFilterState, observations::ChoiceMap,
                       proposal::GenerativeFunction, proposal_args::Tuple,
                       n_particles::Int) 
    return pf_introduce!(state, nothing, nothing, observations,
                         proposal, proposal_args, n_particles)
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
