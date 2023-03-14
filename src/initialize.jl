## Particle filter initialization ##
export pf_initialize

ParticleFilterState(trs::Vector{T}) where {T <: Trace} =
    ParticleFilterState{T}(trs, Vector{T}(undef, length(trs)),
                           fill(0., length(trs)), 0., collect(1:length(trs)))

ParticleFilterState(trs::Vector{T}, ws::Vector{Float64}) where {T <: Trace} =
    ParticleFilterState{T}(trs, Vector{T}(undef, length(trs)),
                           ws, 0., collect(1:length(trs)))

"""
    pf_initialize(model::GenerativeFunction, model_args::Tuple,
                  observations::ChoiceMap, n_particles::Int;
                  dynamic=false)

    pf_initialize(model::GenerativeFunction, model_args::Tuple,
                  observations::ChoiceMap, proposal::GenerativeFunction,
                  proposal_args::Tuple, n_particles::Int;
                  dynamic=false)

Initialize a particle filter, generating traces (i.e. particles)
from the `model` constrained to the provided `observations`. A custom `proposal`
can optionally be used to propose choices for the initial set of traces. Returns
a [`ParticleFilterState`](@ref).

If `dynamic` is `true`, the particle filter will not be specialized to a fixed
trace type, allowing for sequential Monte Carlo inference over a sequence
of models with differing structure, at the expense of more memory usage.
"""
function pf_initialize(
    model::GenerativeFunction{T,U}, model_args::Tuple,
    observations::ChoiceMap, n_particles::Int;
    dynamic::Bool=false
) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    for i=1:n_particles
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end

function pf_initialize(
    model::GenerativeFunction{T,U}, model_args::Tuple, observations::ChoiceMap,
    proposal::GenerativeFunction, proposal_args::Tuple, n_particles::Int;
    dynamic::Bool=false
) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    for i=1:n_particles
        (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
        (traces[i], model_weight) =
            generate(model, model_args, merge(observations, prop_choices))
        log_weights[i] = model_weight - prop_weight
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end

"""
    pf_initialize(model::GenerativeFunction, model_args::Tuple, 
                  observations::ChoiceMap, strata, n_particles::Int;
                  layout=:contiguous, dynamic=false)

    pf_initialize(model::GenerativeFunction, model_args::Tuple,
                  observations::ChoiceMap, strata,
                  proposal::GenerativeFunction, proposal_args::Tuple,
                  n_particles::Int; layout=:contiguous, dynamic=false)

Perform *stratified* initialization of a particle filter, given a set of
`strata` specified as an iterator over choicemaps. Each generated trace is
constrained to both the provided `observations` and the choicemap for the
stratum it belongs to. A custom `proposal` can optionally be used to propose
unconstrained choices.

For a filter with `N` particles and `K` strata, each stratum is first
allocated `B = ⌊N / K⌋` particles. If `layout` is `:contiguous`, these
particles will be allocated in contiguous blocks (e.g., the particles for the
first stratum will have indices `1:B`). If `layout` is `:interleaved`, then 
particles from each stratum will have interleaved indices (e.g., the first
stratum will have indices `1:K:B*K`). The remaining `R` particles are
distributed at random among the strata, and allocated the indices `N-R:N`.

If `dynamic` is `true`, the particle filter will not be specialized to a fixed
trace type, allowing for sequential Monte Carlo inference over a sequence
of models with differing structure, at the expense of more memory usage.
"""
function pf_initialize(
    model::GenerativeFunction{T,U}, model_args::Tuple,
    observations::ChoiceMap, strata, n_particles::Int;
    layout::Symbol=:contiguous, dynamic::Bool=false
) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    # Generate traces in a stratified manner
    stratified_map!(n_particles, strata; layout=layout) do i, stratum
        constraints = merge(stratum, observations)
        (traces[i], log_weights[i]) = generate(model, model_args, constraints)
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end

function pf_initialize(
    model::GenerativeFunction{T,U}, model_args::Tuple,
    observations::ChoiceMap, strata,
    proposal::GenerativeFunction, proposal_args::Tuple, n_particles::Int;
    layout::Symbol=:contiguous, dynamic::Bool=false
) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    stratified_map!(n_particles, strata; layout=layout) do i, stratum
        (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
        constraints = merge(stratum, observations, prop_choices)
        (traces[i], model_weight) = generate(model, model_args, constraints)
        log_weights[i] = model_weight - prop_weight
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end
