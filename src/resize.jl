## Functions for resizing particle filters ##
export pf_replicate!, pf_dereplicate!

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
    state.traces = copy(state.new_traces)
    state.log_weights = _repeat(state.log_weights, n_replicates)
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
    state.traces = copy(state.new_traces)
    return state
end