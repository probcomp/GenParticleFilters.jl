## Particle filter update ##
export pf_update!

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap)

Perform a particle filter update, where the model arguments are adjusted and
new observations are conditioned upon. New latent choices are sampled from
the model's default proposal.
"""
function pf_update!(state::ParticleFilterView, new_args::Tuple,
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
    update_refs!(state)
    return state
end

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap,
               proposal::GenerativeFunction, proposal_args::Tuple)

Perform a particle filter update, where the model arguments are adjusted and
new observations are conditioned upon. New latent choices are sampled from a
custom `proposal` distribution in conjuction with the model's default proposal.
For each particle:
* `proposal` is evaluated with arguments `(t_old, proposal_args...)`,
   where `t_old` is the old model trace, and produces its own trace `t_prop`.
* The old model trace is replaced by a new model trace `t_new`, constructed
  by merging the choices in `t_old` and `t_prop`, and sampling any remaining
  choices from the model's default proposal.

The choicemap of `t_new` satisfies the following conditions:
1. `observations` is a subset of `get_choices(t_new)`;
2. `get_choices(t_old)` is a subset of `get_choices(t_new)`;
3. `get_choices(t_prop)` is a subset of `get_choices(t_new)`.
where one choicemap `a` is a "subset" of another choicemap `b`, when all keys
that occur in `a` also occur in `b`, and the values at those addresses are
equal. It is an error if no trace `t_new` satisfying the above conditions
exists in the support of the model (with the new arguments).
"""
function pf_update!(state::ParticleFilterView, new_args::Tuple,
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
    update_refs!(state)
    return state
end

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap,
               fwd_proposal::GenerativeFunction, fwd_args::Tuple,
               bwd_proposal::GenerativeFunction, bwd_args::Tuple)

Perform a particle filter update, with a custom forward and backward kernel.
New latent choices are sampled from 'fwd_proposal', and any discarded choices
are evaluated under `bwd_proposal`. For each particle:
* `fwd_proposal` is evaluated with arguments `(t_old, fwd_args...)`,
   where `t_old` is the old model trace, and produces its own trace `t_fwd`.
* The old model trace is replaced by a new model trace `t_new`, constructed
  by merging the choices in `t_old` and `t_fwd`, sampling any remaining
  choices from the model's default proposal, and discarding `disc_choices`,
  the choices in `t_old` inconsistent with those in `t_fwd`.
* The probability of `disc_choices` is assessed under `bwd_proposal` with the
  arguments `(t_new, bwd_args)`, giving a backward weight. This weight is used
  as a correction within the importance weight update, ensuring that the
  particle filter remains a valid approximation of the posterior.

The choicemap of `t_new` and `disc_choices` satisfy the following conditions:
1. `observations` is a subset of `get_choices(t_new)`;
2. `get_choices(t_old) ∖ disc_choices` is a subset of `get_choices(t_new)`;
3. `get_choices(t_fwd)` is a subset of `get_choices(t_new)`.
4. `disc_choices` is  within the support of `bwd_proposal`

For valid posterior inference conditioned on prior observations, note that
`fwd_proposal` should not cause any of those observations to be discarded,
(i.e., `disc_choices` should not contain any `observations` given in previous
calls to `pf_update!`).

Similar functionality is provided by [`move_reweight`](@ref), except that
`pf_update!` also allows model arguments to be updated.
"""
function pf_update!(state::ParticleFilterView, new_args::Tuple,
                    argdiffs::Tuple, observations::ChoiceMap,
                    fwd_proposal::GenerativeFunction, fwd_args::Tuple,
                    bwd_proposal::GenerativeFunction, bwd_args::Tuple)
    n_particles = length(state.traces)
    for i=1:n_particles
        (fwd_choices, fwd_weight, _) =
            propose(fwd_proposal, (state.traces[i], fwd_args...))
        constraints = merge(observations, fwd_choices)
        state.new_traces[i], up_weight, _, discard =
            update(state.traces[i], new_args, argdiffs, constraints)
        bwd_weight, _ = isempty(discard) ? (0.0, nothing) :
            assess(bwd_proposal, (state.new_traces[i], bwd_args...), discard)
        state.log_weights[i] += up_weight - fwd_weight + bwd_weight
    end
    update_refs!(state)
    return state
end
