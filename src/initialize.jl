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
                  dynamic::Bool=false)

    pf_initialize(model::GenerativeFunction, model_args::Tuple,
                  observations::ChoiceMap, proposal::GenerativeFunction,
                  proposal_args::Tuple, n_particles::Int;
                  dynamic::Bool=false)

Initialize a particle filter, generating traces (i.e. particles)
from the `model` constrained to the provided `observations`. A custom `proposal`
can optionally be used to propose choices for the initial set of traces. Returns
a [`ParticleFilterState`](@ref).

If `dynamic` is `true`, the particle filter will not be specialized to a fixed
trace type, allowing for sequential Monte Carlo inference over a sequence
of models with differing structure, at the expense of more memory usage.
"""
function pf_initialize(model::GenerativeFunction{T,U}, model_args::Tuple,
                       observations::ChoiceMap, n_particles::Int;
                       dynamic::Bool=false) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    for i=1:n_particles
        (traces[i], log_weights[i]) = generate(model, model_args, observations)
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end

function pf_initialize(model::GenerativeFunction{T,U}, model_args::Tuple,
                       observations::ChoiceMap, proposal::GenerativeFunction,
                       proposal_args::Tuple, n_particles::Int;
                       dynamic::Bool=false) where {T,U}
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
    pf_initialize(model::GenerativeFunction, model_args::Tuple, strata,
                  observations::ChoiceMap, n_particles::Int;
                  layout::Symbol=:contiguous, dynamic::Bool=false)

    pf_initialize(model::GenerativeFunction, model_args::Tuple, strata,
                  observations::ChoiceMap, proposal::GenerativeFunction,
                  proposal_args::Tuple, n_particles::Int;
                  layout::Symbol=:contiguous, dynamic::Bool=false)

Perform *stratified* initialization of a particle filter, given a set of
`strata` specified as an iterator over choicemaps. Each generated trace is
constrained to both the provided `observations` and the choicemap for the
stratum it belongs to. A custom `proposal` can optionally be used to propose
unconstrained choices.

For a filter with `N` particles and `K` strata, each stratum is first
allocated `B = ⌊N / K⌋` particles. If `layout` is `:contiguous`, these
particles will be allocated in continguous blocks (e.g., the particles for the
first stratum will have indices `1:B`). If `layout` is `:interleaved`, then 
particles from each stratum will have interleaved indices (e.g., the first
stratum will have indices `1:K:B*K`). The remaining `R` particles are
distributed at random among the strata, and allocated the indices `N-R:N`.

If `dynamic` is `true`, the particle filter will not be specialized to a fixed
trace type, allowing for sequential Monte Carlo inference over a sequence
of models with differing structure, at the expense of more memory usage.
"""
function pf_initialize(model::GenerativeFunction{T,U}, model_args::Tuple,
                       strata, observations::ChoiceMap, n_particles::Int;
                       layout::Symbol=:contiguous, dynamic::Bool=false) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    # Generate traces for each stratum in a contiguous or interleaved manner
    n_strata = length(strata)
    block_size = n_particles ÷ n_strata
    for (k, stratum) in enumerate(strata)
        if layout == :contiguous
            idxs = ((k-1)*block_size+1):k*block_size
        else # layout == :interleaved
            idxs = k:n_strata:n_strata*block_size
        end
        for i in idxs
            constraints = merge(stratum, observations)
            (traces[i], log_weights[i]) = generate(model, model_args, constraints)
        end
    end
    # Allocate remaining traces to random strata
    n_remaining = n_particles - n_strata * block_size
    if n_remaining > 0
        strata = strata isa Vector ? strata : collect(strata)
        remainder = sample(strata, n_remaining)
        for (k, stratum) in enumerate(remainder)
            i = n_particles - n_remaining + k
            constraints = merge(stratum, observations)
            (traces[i], log_weights[i]) = generate(model, model_args, constraints)
        end
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end

function pf_initialize(model::GenerativeFunction{T,U}, model_args::Tuple, strata,
                       observations::ChoiceMap, proposal::GenerativeFunction,
                       proposal_args::Tuple, n_particles::Int;
                       layout::Symbol=:contiguous, dynamic::Bool=false) where {T,U}
    V = dynamic ? Trace : U # Determine trace type for particle filter
    traces = Vector{V}(undef, n_particles)
    log_weights = Vector{Float64}(undef, n_particles)
    # Generate traces for each stratum in a contiguous or interleaved manner
    n_strata = length(strata)
    block_size = n_particles ÷ n_strata
    for (k, stratum) in enumerate(strata)
        if layout == :contiguous
            idxs = ((k-1)*block_size+1):k*block_size
        else # layout == :interleaved
            idxs = k:n_strata:n_strata*block_size
        end
        for i in idxs
            (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
            constraints = merge(stratum, observations, prop_choices)
            (traces[i], model_weight) = generate(model, model_args, constraints)
            log_weights[i] = model_weight - prop_weight
        end
    end
    # Allocate remaining traces to random strata
    n_remaining = n_particles - n_strata * block_size
    if n_remaining > 0
        strata = strata isa Vector ? strata : collect(strata)
        remainder = sample(strata, n_remaining)
        for (k, stratum) in enumerate(remainder)
            i = n_particles - n_remaining + k
            (prop_choices, prop_weight, _) = propose(proposal, proposal_args)
            constraints = merge(stratum, observations, prop_choices)
            (traces[i], model_weight) = generate(model, model_args, constraints)
            log_weights[i] = model_weight - prop_weight
        end
    end
    return ParticleFilterState{V}(traces, Vector{V}(undef, n_particles),
                                  log_weights, 0., collect(1:n_particles))
end
