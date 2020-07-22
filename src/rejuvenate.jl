## Rejuvenation moves for SMC algorithms ##

"Rejuvenate particles by repeated application of a Metropolis-Hastings kernel."
function pf_move_accept!(state::ParticleFilterState,
                         kern, kern_args::Tuple=(), n_iters::Int=1)
    # Potentially rejuvenate each trace
    for (i, trace) in enumerate(state.traces)
        for k = 1:n_iters
            trace, accept = kern(trace, kern_args...)
            @debug "Accepted: $accept"
        end
        state.new_traces[i] = trace
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end

"Rejuvenate particles via repeated move-reweight steps."
function pf_move_reweight!(state::ParticleFilterState,
                           kern, kern_args::Tuple=(), n_iters::Int=1)
    # Move and reweight each trace
    for (i, trace) in enumerate(state.traces)
        weight = 0
        for k = 1:n_iters
            trace, rel_weight = kern(trace, kern_args...)
            weight += rel_weight
            @debug "Rel. Weight: $rel_weight"
        end
        state.new_traces[i] = trace
        state.log_weights[i] += weight
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
end
