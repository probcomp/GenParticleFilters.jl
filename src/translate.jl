## Trace translators for SMC / SMCP^3 ##
export ExtendingTraceTranslator, UpdatingTraceTranslator

using Gen: run_first_pass, jacobian_correction, check_round_trip, run_transform

function Base.copy(translator::T) where {T <: TraceTranslator}
    return T((getfield(translator, name) for name in fieldnames(T))...)
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
        transform::Union{TraceTransformDSLProgram,Nothing} = nothing
    )

Constructor for an extending trace translator, which extends a trace by 
sampling new random choices from a forward proposal `q_forward`, optionally
applying a trace `transform` to those choices, then updating the trace with
the proposed (and transformed) choices, along with `new_observations`.
"""
@with_kw mutable struct ExtendingTraceTranslator{
    T <: Union{Nothing, TraceTransformDSLProgram},
    PA <: Tuple, PD <: Tuple, Q <: GenerativeFunction, QA <: Tuple
} <: TraceTranslator
    p_new_args::PA = ()
    p_argdiffs::PD = map((_) -> UnknownChange(), p_new_args)
    new_observations::ChoiceMap = EmptyChoiceMap()
    q_forward::Q
    q_forward_args::QA = ()
    transform::T = nothing # a bijection
end

"""
    (output_trace, log_weight) = translator(input_trace; check=true)

Run an [`ExtendingTraceTranslator`](@ref) on an input trace, returning the
output trace and the (incremental) importance weight.

If `check` is enabled, a check is performed to ensure that no choices are 
discarded as a result of updating the trace. This is recommended by default,
but can be disabled to allow trace updates that replace previously constrained
observations with different values.
"""
function (translator::ExtendingTraceTranslator)(
    prev_model_trace::Trace;
    check::Bool=true
)
    @unpack q_forward, q_forward_args = translator
    @unpack p_new_args, p_argdiffs, transform = translator
    # Simulate forward kernel
    fwd_proposal_trace =
        simulate(q_forward, (prev_model_trace, q_forward_args...))
    fwd_proposal_score = get_score(fwd_proposal_trace)
    # Transform forward proposal choices
    first_pass_results = run_first_pass(transform, fwd_proposal_trace, nothing)
    log_abs_determinant =
        jacobian_correction(transform, fwd_proposal_trace,
                            nothing, first_pass_results, nothing)
    constraints = first_pass_results.constraints
    # Compute the new trace via update
    constraints = merge(constraints, translator.new_observations)
    (new_model_trace, model_score_diff, _, discard) =
        update(prev_model_trace, p_new_args, p_argdiffs, constraints)
    if check && !isempty(discard)
        error("Choices were updated or deleted: $discard")
    end 
    # Compute the incremental importance weight
    log_weight = model_score_diff - fwd_proposal_score + log_abs_determinant
    return (new_model_trace, log_weight)
end

# Specialized implementation for the case where no transform is used
function (translator::ExtendingTraceTranslator{Nothing})(
    prev_model_trace::Trace;
    check::Bool=true
)
    @unpack q_forward_args, p_new_args, p_argdiffs = translator
    # Simulate proposal
    fwd_choices, fwd_proposal_score, _ =
        propose(translator.q_forward, (prev_model_trace, q_forward_args...))
    # Merge proposed choices with new observations
    constraints = merge(fwd_choices, translator.new_observations)
    # Compute the new trace via update
    (new_model_trace, model_score_diff, _, discard) =
        update(prev_model_trace, p_new_args, p_argdiffs, constraints)
    if check && !isempty(discard)
        error("Choices were updated or deleted: $discard")
    end
    # Compute the incremental importance weight
    log_weight = model_score_diff - fwd_proposal_score
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
        transform::Union{TraceTransformDSLProgram,Nothing} = nothing
    )

Constructor for a updating trace translator, which updates the trace of model
given a forward kernel `q_forward`, backward kernel `q_backward`,
new arguments for that model `p_new_args`, and `new_observations`.
    
Optionally, a trace `transform` can be provided, specifying how the input
model trace and forward kernel trace get mapped to the output model trace and 
backward kernel trace.
"""
@with_kw mutable struct UpdatingTraceTranslator{
    T <: Union{Nothing, TraceTransformDSLProgram},
    PA <: Tuple, PD <: Tuple,
    K <: GenerativeFunction, KA <: Tuple,
    L <: GenerativeFunction, LA <: Tuple
} <: TraceTranslator
    p_new_args::PA = ()
    p_argdiffs::PD = map((_) -> UnknownChange(), p_new_args)
    new_observations::ChoiceMap = EmptyChoiceMap()
    q_forward::K
    q_forward_args::KA = ()
    q_backward::L
    q_backward_args::LA = ()
    transform::T
end

function Gen.inverse(
    translator::UpdatingTraceTranslator,
    prev_model_trace::Trace,
    prev_observations::ChoiceMap=EmptyChoiceMap()
)
    return UpdatingTraceTranslator(
        get_args(prev_model_trace),
        map((_)->UnknownChange(), get_args(prev_model_trace)),
        prev_observations,
        translator.q_backward, translator.q_backward_args,
        translator.q_forward, translator.q_forward_args,
        inverse(translator.transform)
    )
end

function Gen.run_transform(
    translator::UpdatingTraceTranslator,
    prev_model_trace::Trace,
    forward_proposal_trace::Trace
)
    @unpack transform, new_observations = translator
    @unpack p_new_args, p_argdiffs, q_backward, q_backward_args = translator
    first_pass_results =
        run_first_pass(transform, prev_model_trace, forward_proposal_trace)
    constraints = merge(first_pass_results.constraints, new_observations)
    new_model_trace, model_score_diff, _, discard =
        update(prev_model_trace, p_new_args, p_argdiffs, constraints)
    log_abs_determinant =
        jacobian_correction(transform, prev_model_trace, forward_proposal_trace,
                            first_pass_results, discard)
    backward_proposal_trace, _ =
        generate(q_backward, (new_model_trace, q_backward_args...),
                 first_pass_results.u_back)
    return (new_model_trace, backward_proposal_trace,
            log_abs_determinant, model_score_diff)
end

# Specialized implementation for the case where no transform is used
function Gen.run_transform(
    translator::UpdatingTraceTranslator{Nothing},
    prev_model_trace::Trace,
    forward_proposal_trace::Trace
)
    @unpack transform, new_observations = translator
    @unpack p_new_args, p_argdiffs, q_backward, q_backward_args = translator
    constraints = merge(get_choices(forward_proposal_trace), new_observations)
    new_model_trace, model_score_diff, _, discard =
        update(prev_model_trace, p_new_args, p_argdiffs, constraints)
    log_abs_determinant = 0.0
    backward_proposal_trace, _ =
        generate(q_backward, (new_model_trace, q_backward_args...), discard)
    return (new_model_trace, backward_proposal_trace,
            log_abs_determinant, model_score_diff)
end

"""
    (output_trace, log_weight) = translator(input_trace; kwargs...)

Run an [`UpdatingTraceTranslator`](@ref) on an input trace, returning the
output trace and the (incremental) importance weight.

# Keyword Arguments

Set `check = true` to enable a bijection check (this requires that the transform
has been paired with its inverse using `pair_bijections! or `is_involution!`).

If `check` is enabled, then `prev_observations` should be provided as a
choicemap containing the random choices in the input trace that are replaced
by `new_observations`, or deleted due to a change in the arguments. Typically
no observations are replaced or deleted, so `prev_observations` is an
`EmptyChoiceMap` by default.
"""
function (translator::UpdatingTraceTranslator)(
    prev_model_trace::Trace;
    check=false, prev_observations=EmptyChoiceMap()
)
    @unpack q_forward, q_forward_args = translator
    @unpack p_new_args, p_argdiffs, transform = translator
    # Simulate forward proposal
    forward_proposal_trace =
        simulate(q_forward, (prev_model_trace, q_forward_args...))
    # Apply trace transform
    (new_model_trace, backward_proposal_trace, log_abs_det, model_score_diff) =
        run_transform(translator, prev_model_trace, forward_proposal_trace)
    # Compute incremental importance weight
    forward_proposal_score = get_score(forward_proposal_trace)
    backward_proposal_score = get_score(backward_proposal_trace)
    log_weight = model_score_diff + log_abs_det -
                 forward_proposal_score + backward_proposal_score
    # Perform bijection check
    if check
        inverter = inverse(translator, prev_model_trace, prev_observations)
        (prev_model_trace_rt, forward_proposal_trace_rt, _) =
            run_transform(inverter, new_model_trace, backward_proposal_trace)
        check_round_trip(prev_model_trace, prev_model_trace_rt,
                         forward_proposal_trace, forward_proposal_trace_rt)
    end
    return (new_model_trace, log_weight)
end
