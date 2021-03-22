## Rejuvenation moves for SMC algorithms ##
export pf_rejuvenate!, pf_move_accept!, pf_move_reweight!
export move_reweight

"""
    pf_rejuvenate!(state::ParticleFilterState, kern, kern_args::Tuple=(),
                   n_iters::Int=1; method=:move)

Rejuvenates particles by repeated application of a kernel `kern`. `kern`
should be a callable which takes a trace as its first argument, and returns
a tuple with a trace as the first return value. `method` specifies the
rejuvenation method: `:move` for MCMC moves without a reweighting step,
and `:reweight` for rejuvenation with a reweighting step.
"""
function pf_rejuvenate!(state::ParticleFilterView, kern, kern_args::Tuple=(),
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
function pf_move_accept!(state::ParticleFilterView,
                         kern, kern_args::Tuple=(), n_iters::Int=1)
    # Potentially rejuvenate each trace
    for (i, trace) in enumerate(state.traces)
        for k = 1:n_iters
            trace, accept = kern(trace, kern_args...)
            @debug "Accepted: $accept"
        end
        state.new_traces[i] = trace
    end
    update_refs!(state)
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
function pf_move_reweight!(state::ParticleFilterView,
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
    update_refs!(state)
    return state
end

"""
    move_reweight(trace, selection)
    move_reweight(trace, proposal, proposal_args)
    move_reweight(trace, proposal, proposal_args, involution)
    move_reweight(trace, proposal_fwd, args_fwd,
                  proposal_bwd, args_bwd, involution)

Move-reweight MCMC kernel, which takes in a `trace` and returns a new trace
along with a relative importance weight. This can be used for rejuvenation
within a particle filter, as described in [1].

Several variants of `move_reweight` exist, differing in the complexity
involved in proposing and re-weighting random choices:

1. The `selection` variant regenerates the addresses specified by `selection`
   using the model's default proposal.
2. A `proposal` generative function and arguments can be provided, to propose
   new random choices using a custom proposal distribution. `proposal` must
   take the original trace as its first input argument.
3. An `involution` can also be provided, to handle more complex proposal
   distributions which add/remove trace addresses.
4. Separate forward and backward proposal distributions, `proposal_fwd` and
   `proposal_bwd`, can be provided, as long they share the same support. This
   adjusts the computation of the relative importance weight by scoring
   the backward choices under the backward proposal.

[1] R. A. G. Marques and G. Storvik, "Particle move-reweighting strategies for
online inference," Preprint series. Statistical Research Report, 2013.
"""
function move_reweight(trace::Trace, selection::Selection)
    args = get_args(trace)
    argdiffs = map((_) -> NoChange(), args)
    new_trace, rel_weight = regenerate(trace, args, argdiffs, selection)
    return new_trace, rel_weight
end

function move_reweight(trace::Trace, proposal::GenerativeFunction,
                       proposal_args::Tuple)
    model_args = Gen.get_args(trace)
    argdiffs = map((_) -> NoChange(), model_args)
    fwd_choices, fwd_score, fwd_ret =
        propose(proposal, (trace, proposal_args...,))
    new_trace, weight, _, discard =
        update(trace, model_args, argdiffs, fwd_choices)
    bwd_score, bwd_ret =
        assess(proposal, (new_trace, proposal_args...), discard)
    rel_weight = weight - fwd_score + bwd_score
    return new_trace, rel_weight
end

function move_reweight(trace::Trace, proposal::GenerativeFunction,
                       proposal_args::Tuple, involution)
    fwd_choices, fwd_score, fwd_ret =
        propose(proposal, (trace, proposal_args...,))
    new_trace, bwd_choices, weight =
        involution(trace, fwd_choices, fwd_ret, proposal_args)
    bwd_score, bwd_ret =
        assess(proposal, (new_trace, proposal_args...), bwd_choices)
    rel_weight = weight - fwd_score + bwd_score
    return new_trace, rel_weight
end

function move_reweight(trace::Trace, proposal_fwd::GenerativeFunction,
                       args_fwd::Tuple, proposal_bwd::GenerativeFunction,
                       args_bwd::Tuple, involution)
    fwd_choices, fwd_score, fwd_ret =
        propose(proposal_fwd, (trace, args_fwd...,))
    new_trace, bwd_choices, weight =
        involution(trace, fwd_choices, fwd_ret, args_fwd)
    bwd_score, bwd_ret =
        assess(proposal_bwd, (new_trace, args_bwd...), bwd_choices)
    rel_weight = weight - fwd_score + bwd_score
    return new_trace, rel_weight
end
