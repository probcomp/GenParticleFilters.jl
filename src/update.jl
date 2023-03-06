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
    pf_update!(state::ParticleFilterState, translator; translator_args...)

Perform a particle filter update using an arbitrary trace `translator`, which
takes in each previous trace and returns a new trace and incremental log.
importance weight. `translator_args` are additional keyword arguments which
are passed to the `translator` when it is called.
"""
function pf_update!(state::ParticleFilterView, translator; translator_args...)
    n_particles = length(state.traces)
    for i=1:n_particles
        state.new_traces[i], log_weight =
            translator(state.traces[i]; translator_args...)
        state.log_weights[i] += log_weight
    end
    update_refs!(state)
    return state
end

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap,
               proposal::GenerativeFunction, proposal_args::Tuple,
               [transform::TraceTransformDSLProgram];
               check::Bool=true)

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

If a deterministic trace `transform` is also provided, `t_prop` is transformed
by a deterministic function before its choices are merged with `t_old`.

By default, a `check` is performed to ensure that no choices are 
discarded as a result of updating the trace. Setting `check = false` allows
for updates where previous observations are replaced with new ones.
"""
function pf_update!(
    state::ParticleFilterView,
    new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap,
    proposal::GenerativeFunction, proposal_args::Tuple,
    transform::Union{TraceTransformDSLProgram,Nothing} = nothing;
    check::Bool=true
)
    translator = ExtendingTraceTranslator(
        p_new_args=new_args,
        p_argdiffs=argdiffs,
        new_observations=observations,
        q_forward=proposal,
        q_forward_args=proposal_args,
        transform=transform
    )
    return pf_update!(state, translator; check=check)
end

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap,
               fwd_proposal::GenerativeFunction, fwd_args::Tuple,
               bwd_proposal::GenerativeFunction, bwd_args::Tuple,
               [transform::TraceTransformDSLProgram];
               check::Bool=false, prev_observations=EmptyChoiceMap())

Perform a particle filter update, with a custom forward and backward kernel.
New latent choices are sampled from `fwd_proposal`, and any discarded choices
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

If a trace `transform` is also provided, then more general forward and backward
kernels can be used: `t_new` (the model's new trace) and `t_bwd` (the trace
for the backward kernel) are constructed as a function of `t_old`, `t_fwd`,
and any new `observations`. The `check` and `prev_observations` keyword
arguments can also be set to `true` to check for correctness.
(See [`UpdatingTraceTranslator`](@ref) for more details.)

Similar functionality is provided by [`move_reweight`](@ref), except that
`pf_update!` also allows model arguments to be updated.
"""
function pf_update!(
    state::ParticleFilterView,
    new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap,
    fwd_proposal::GenerativeFunction, fwd_args::Tuple,
    bwd_proposal::GenerativeFunction, bwd_args::Tuple,
    transform::Union{TraceTransformDSLProgram,Nothing} = nothing;
    kwargs...
)
    translator = UpdatingTraceTranslator(
        p_new_args=new_args,
        p_argdiffs=argdiffs,
        new_observations=observations,
        q_forward=fwd_proposal,
        q_forward_args=fwd_args,
        q_backward=bwd_proposal,
        q_backward_args=bwd_args,
        transform=transform
    )
    return pf_update!(state, translator; kwargs...)
end

"""
    pf_update!(state::ParticleFilterState, new_args::Tuple,
               argdiffs::Tuple, observations::ChoiceMap, strata,
               [fwd_proposal::GenerativeFunction, fwd_args::Tuple,
                bwd_proposal::GenerativeFunction, bwd_args::Tuple,
                transform::TraceTransformDSLProgram];
               layout=:interleaved, kwargs...)

    pf_update!(state::ParticleFilterState, translator::TraceTranslator, strata;
               layout=:interleaved, translator_args...)

Perform a *stratified* particle filter update,  given a set of `strata`
specified as an iterator over choicemaps. Each generated trace is
constrained to both the provided `observations` and the choicemap for the
stratum it belongs to. Forward proposals, backward proposals, trace transforms,
or trace translators can also be specified, as in the non-stratified versions of
`pf_update!`.

For a filter with `N` particles and `K` strata, each stratum is assigned at
least `B = ⌊N / K⌋` particles. If `layout` is `:contiguous`, these particles
will be assigned in continguous blocks (e.g., the particles for the
first stratum will have indices `1:B`). If `layout` is `:interleaved`, then 
particles from each stratum will have interleaved indices (e.g., the first
stratum will have indices `1:K:B*K`). The remaining `R` particles are
distributed at random among the strata, and allocated the indices `N-R:N`.

By default the `layout` is `:interleaved`, as this allows for convenient
sub-stratification of the contiguous blocks allocated to each stratum when 
performing stratified initialization with [`pf_initialize`](@ref).
"""
function pf_update!(state::ParticleFilterView, new_args::Tuple,
                    argdiffs::Tuple, observations::ChoiceMap, strata;
                    layout=:interleaved)
    # Update traces in a stratified manner
    n_particles = length(state.traces)
    stratified_map!(n_particles, strata; layout=layout) do i, stratum
        constraints = merge(stratum, observations)
        state.new_traces[i], increment, _, discard =
            update(state.traces[i], new_args, argdiffs, constraints)
        if !isempty(discard)
            error("Choices were updated or deleted: $discard")
        end
        state.log_weights[i] += increment    
    end
    update_refs!(state)
    return state
end

function pf_update!(state::ParticleFilterView, translator::TraceTranslator,
                    strata; layout=:interleaved, translator_args...)
    # Make copy of translator to ensure thread-safe mutation
    translator = copy(translator)
    observations = translator.new_observations
    # Update traces in a stratified manner
    n_particles = length(state.traces)
    stratified_map!(n_particles, strata; layout=layout) do i, stratum
        translator.new_observations = merge(stratum, observations)
        state.new_traces[i], log_weight =
            translator(state.traces[i]; translator_args...)
        state.log_weights[i] += log_weight
    end
    update_refs!(state)
    return state
end

function pf_update!(
    state::ParticleFilterView,
    new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap, strata,
    proposal::GenerativeFunction, proposal_args::Tuple,
    transform::Union{TraceTransformDSLProgram,Nothing} = nothing;
    layout=:interleaved, check::Bool=true
)
    translator = ExtendingTraceTranslator(
        p_new_args=new_args,
        p_argdiffs=argdiffs,
        new_observations=observations,
        q_forward=proposal,
        q_forward_args=proposal_args,
        transform=transform
    )
    return pf_update!(state, translator, strata; layout=layout, check=check)
end

function pf_update!(
    state::ParticleFilterView,
    new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap, strata,
    fwd_proposal::GenerativeFunction, fwd_args::Tuple,
    bwd_proposal::GenerativeFunction, bwd_args::Tuple,
    transform::Union{TraceTransformDSLProgram,Nothing} = nothing;
    kwargs...
)
    translator = UpdatingTraceTranslator(
        p_new_args=new_args,
        p_argdiffs=argdiffs,
        new_observations=observations,
        q_forward=fwd_proposal,
        q_forward_args=fwd_args,
        q_backward=bwd_proposal,
        q_backward_args=bwd_args,
        transform=transform
    )
    return pf_update!(state, translator, strata; kwargs...)
end
