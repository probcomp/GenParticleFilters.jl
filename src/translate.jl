using Gen: jacobian_correction, check_round_trip

# Monkey patch for bug in Gen.run_first_pass
function run_first_pass(transform::TraceTransformDSLProgram, model_trace)
    state = Gen.FirstPassState(model_trace, model_trace)
    transform.fn!(state)
    return state.results
end

##################################
# ExtendingTraceTranslator #
##################################

"""
    translator = ExtendingTraceTranslator(;
        p_new_args::Tuple = (),
        p_argdiffs::Tuple = (),
        new_observations::ChoiceMap = EmptyChoiceMap(),
        q_forward::GenerativeFunction,
        q_forward_args::Tuple = (),
        f::Union{TraceTransformDSLProgram,Nothing} = nothing)
Constructor for a extending trace translator.
Run the translator with:
    (output_trace, log_weight) = translator(input_trace)
"""
@with_kw mutable struct ExtendingTraceTranslator
    p_new_args::Tuple = ()
    p_argdiffs::Tuple = ()
    new_observations::ChoiceMap = EmptyChoiceMap()
    q_forward::GenerativeFunction
    q_forward_args::Tuple = ()
    f::Union{TraceTransformDSLProgram,Nothing} = nothing # a bijection
end

function (translator::ExtendingTraceTranslator)(prev_model_trace::Trace)

    # simulate from auxiliary program
    forward_proposal_trace = simulate(translator.q_forward,
        (prev_model_trace, translator.q_forward_args...,))
    forward_proposal_score = get_score(forward_proposal_trace)

    # transform forward proposal
    if translator.f === nothing
        constraints = get_choices(forward_proposal_trace)
    else
        first_pass_results =
            run_first_pass(translator.f, forward_proposal_trace)
        # TODO: adjust after fixing bugs in Gen
        log_abs_determinant =
            jacobian_correction(translator.f, forward_proposal_trace,
                forward_proposal_trace, first_pass_results, nothing)
        constraints = first_pass_results.constraints
    end

    # computing the new trace via update
    constraints = merge(constraints, translator.new_observations)
    (new_model_trace, log_model_weight, _, discard) = update(
        prev_model_trace, translator.p_new_args,
        translator.p_argdiffs, constraints)

    if !isempty(discard)
        @error("Can only extend with new choices, not remove existing choices")
        error("Invalid ExtendingTraceTranslator")
    end

    log_weight = log_model_weight - forward_proposal_score + log_abs_determinant
    return (new_model_trace, log_weight)
end

##################################
# UpdatingTraceTranslator #
##################################

"""
    translator = UpdatingTraceTranslator(;
        p_new_args::Tuple = (),
        p_argdiffs::Tuple = (),
        new_observations::ChoiceMap = EmptyChoiceMap(),
        q_forward::GenerativeFunction,
        q_forward_args::Tuple  = (),
        q_backward::GenerativeFunction,
        q_backward_args::Tuple  = (),
        f::TraceTransformDSLProgram)

Constructor for a updating trace translator, which updates the trace of model
given a forward kernel, backward kernel, new arguments for that model, and
a trace transform that specifies

Run the translator with:
```
    (output_trace, log_weight) =
        translator(input_trace; check=false, prev_observations=EmptyChoiceMap())
```
Use `check` to enable a bijection check (this requires that the transform `f`
has been paired with its inverse using `pair_bijections! or `is_involution!`).
If `check` is enabled, then `prev_observations` is a choice map containing
the observed random choices in the previous trace.
"""
@with_kw mutable struct UpdatingTraceTranslator
    p_new_args::Tuple = ()
    p_argdiffs::Tuple = ()
    new_observations::ChoiceMap = EmptyChoiceMap()
    q_forward::GenerativeFunction
    q_forward_args::Tuple  = ()
    q_backward::GenerativeFunction
    q_backward_args::Tuple  = ()
    f::TraceTransformDSLProgram
end

function updating_trace_translator_run_transform(
        f::TraceTransformDSLProgram, new_observations::ChoiceMap,
        prev_model_trace::Trace, forward_proposal_trace::Trace,
        p_new_args::Tuple, p_argdiffs::Tuple,
        q_backward::GenerativeFunction, q_backward_args::Tuple)
    first_pass_results =
        Gen.run_first_pass(f, prev_model_trace, forward_proposal_trace)
    constraints = merge(first_pass_results.constraints, new_observations)
    (new_model_trace, _, _, discard) = update(
        prev_model_trace, p_new_args, p_argdiffs, constraints)
    log_abs_determinant = jacobian_correction(f, prev_model_trace,
        forward_proposal_trace, first_pass_results, discard)
    backward_proposal_trace, = generate(q_backward,
        (new_model_trace, q_backward_args...), first_pass_results.u_back)
    return (new_model_trace, backward_proposal_trace, log_abs_determinant)
end

function (translator::UpdatingTraceTranslator)(
        prev_model_trace::Trace; check=false, prev_observations=EmptyChoiceMap())

    # sample auxiliary trace
    forward_proposal_trace = simulate(translator.q_forward,
        (prev_model_trace, translator.q_forward_args...,))

    # apply trace transform
    (new_model_trace, backward_proposal_trace, log_abs_determinant) =
        updating_trace_translator_run_transform(
            translator.f, translator.new_observations,
            prev_model_trace, forward_proposal_trace,
            translator.p_new_args, translator.p_argdiffs,
            translator.q_backward, translator.q_backward_args)

    # compute log weight
    prev_model_score = get_score(prev_model_trace)
    new_model_score = get_score(new_model_trace)
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = new_model_score - prev_model_score +
        backward_proposal_score + forward_proposal_score + log_abs_determinant

    if check
        forward_proposal_choices = get_choices(forward_proposal_trace)
        f_inv = inverse(translator.f)
        argdiffs = map((_) -> UnknownChange(), get_args(prev_model_trace))
        (prev_model_trace_rt, forward_proposal_trace_rt, _) =
            updating_trace_translator_run_transform(
                inverse(translator.f), prev_observations,
                new_model_trace, backward_proposal_trace,
                get_args(prev_model_trace), argdiffs,
                translator.q_forward, translator.q_forward_args)
        check_round_trip(
            prev_model_trace, prev_model_trace_rt,
            forward_proposal_trace, forward_proposal_trace_rt)
    end

    return (new_model_trace, log_weight)
end
