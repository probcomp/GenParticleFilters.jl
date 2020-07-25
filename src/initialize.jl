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
                  observations::ChoiceMap, n_particles::Int, dynamic::Bool=false)
    pf_initialize(model::GenerativeFunction, model_args::Tuple,
                  observations::ChoiceMap, proposal::GenerativeFunction,
                  proposal_args::Tuple, n_particles::Int, dynamic::Bool=false)

Initialize the state of a particle filter, generating traces (i.e. particles)
from the `model` constrained to the provided `observations`. A custom `proposal`
can optionally be used to propose choices for the initial set of traces.

If `dynamic` is `true`, the particle filter will not be specialized to a fixed
trace type, allowing for sequential Monte Carlo inference over a sequence
of models with differing structure, at the expense of more memory usage.
"""
function pf_initialize(model::GenerativeFunction{T,U}, model_args::Tuple,
                       observations::ChoiceMap, n_particles::Int,
                       dynamic::Bool=false) where {T,U}
    traces = Vector{Any}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    for i=1:n_particles
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
    end
    return dynamic ?
        ParticleFilterState{Trace}(traces, Vector{Trace}(undef, n_particles),
                               log_weights, 0., collect(1:n_particles)) :
        ParticleFilterState{U}(traces, Vector{U}(undef, n_particles),
                               log_weights, 0., collect(1:n_particles))
end

function pf_initialize(model::GenerativeFunction{T,U}, model_args::Tuple,
                       observations::ChoiceMap, proposal::GenerativeFunction,
                       proposal_args::Tuple, n_particles::Int,
                       dynamic::Bool=false) where {T,U}
    traces = Vector{Any}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    for i=1:n_particles
        (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
        (traces[i], model_weight) =
            generate(model, model_args, merge(observations, prop_choices))
        log_weights[i] = model_weight - prop_weight
    end
    return dynamic ?
        ParticleFilterState{Trace}(traces, Vector{Trace}(undef, n_particles),
                               log_weights, 0., collect(1:n_particles)) :
        ParticleFilterState{U}(traces, Vector{U}(undef, n_particles),
                               log_weights, 0., collect(1:n_particles))
end
