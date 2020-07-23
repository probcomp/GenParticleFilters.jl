## Rejuvenation moves for SMC algorithms ##
export pf_rejuvenate!, pf_move_accept!, pf_move_reweight!

"""
    pf_rejuvenate!(state::ParticleFilterState, kern, kern_args::Tuple=(),
                   n_iters::Int=1; method=:move)

Rejuvenates particles by repeated application of a kernel `kern`. `kern`
should be a callable which takes a trace as its first argument, and returns
a tuple with a trace as the first return value. `method` specifies the
rejuvenation method: `:move` for MCMC moves without a reweighting step,
and `:reweight` for rejuvenation with a reweighting step.
"""
function pf_rejuvenate!(state::ParticleFilterState, kern, kern_args::Tuple=(),
                        n_iters::Int=1; method::Symbol=:move)
    if method == :move
        return pf_move_accept!(state, kern, kern_args, n_iters)
    elseif method == :reweight
        return pf_move_reweight!(state, kern, kern_args, n_iters)
    else
        error("Method $method not recognized.")
    end
end

"""
    pf_move_accept!(state::ParticleFilterState, kern,
                    kern_args::Tuple=(), n_iters::Int=1)

Rejuvenates particles by repeated application of a MCMC kernel `kern`. `kern`
should be a callable which takes a trace as its first argument, and returns
a tuple `(trace, accept)`, where `trace` is the (potentially) new trace, and
`accept` is true if the MCMC move was accepted. Subsequent arguments to `kern`
can be supplied with `kern_args`. The kernel is repeatedly applied to each trace
for `n_iters`.
"""
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
    return state
end

"""
    pf_move_reweight!(state::ParticleFilterState, kern,
                      kern_args::Tuple=(), n_iters::Int=1)

Rejuvenates and reweights particles by repeated application of a reweighting
kernel `kern`, as described in [1]. `kern` should be a callable which takes a
trace as its first argument, and returns a tuple `(trace, rel_weight)`,
where `trace` is the new trace, and `rel_weight` is the relative log-importance
weight. Subsequent arguments to `kern` can be supplied with `kern_args`.
The kernel is repeatedly applied to each trace for `n_iters`, and the weights
accumulated accordingly.

[1] R. A. G. Marques and G. Storvik, "Particle move-reweighting strategies for
online inference," Preprint series. Statistical Research Report, 2013.
"""
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
    return state
end
