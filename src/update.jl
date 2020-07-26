## Particle filter update ##
export pf_update!

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap)
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap,
               proposal::GenerativeFunction, proposal_args::Tuple)

Perform a particle filter update, where the model arguments are adjusted and
new observations are conditioned upon. If a `proposal` is not provided, new
latent choices are sampled from the model's default proposal.

If a custom `proposal` is provided, then it will be used in combination with
the model's default proposal.  That is, for each particle:
* `proposal` is evaluated with arguments `(t_old, proposal_args...)`,
   where `t_old` is the old model trace, and produces its own trace `t_prop`.
* The old model trace is replaced by a new model trace `t_new`, constructed
  by merging the choices in `t_old` and `t_prop`, and sampling any remaining
  choices from the model's default proposal.

The choicemap of `t_new` satisfies the following conditions:
1. `get_choices(t_old)` is a subset of `get_choices(t_new)`;
2. `observations` is a subset of `get_choices(t_new)`;
3. `get_choices(proposal_trace)` is a subset of `get_choices(t_new)`.
where one choicemap `a` is a "subset" of another choicemap `b`, when all keys
that occur in `a` also occur in `b`, and the values at those addresses are
equal. It is an error if no trace `t_new` satisfying the above conditions
exists in the support of the model (with the new arguments).
"""
function pf_update!(state::ParticleFilterState, new_args::Tuple,
                    argdiffs::Tuple, observations::ChoiceMap)
    n_particles = length(state.traces)
    for i=1:n_particles
        state.new_traces[i], increment, _, discard =
            update(state.traces[i], new_args, argdiffs, observations)
        if !isempty(discard)
            error("Choices were updated or deleted: $discard")
        end
        state.log_weights[i] += increment
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end

function pf_update!(state::ParticleFilterState, new_args::Tuple,
                    argdiffs::Tuple, observations::ChoiceMap,
                    proposal::GenerativeFunction, proposal_args::Tuple)
    n_particles = length(state.traces)
    for i=1:n_particles
        (prop_choices, prop_weight, _) =
            propose(proposal, (state.traces[i], proposal_args...))
        constraints = merge(observations, prop_choices)
        state.new_traces[i], up_weight, _, discard =
            update(state.traces[i], new_args, argdiffs, constraints)
        if !isempty(discard)
            error("Choices were updated or deleted: $discard")
        end
        state.log_weights[i] += up_weight - prop_weight
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end
