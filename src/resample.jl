## Resampling methods for SMC algorithms ##
export pf_resample!
export pf_residual_resample!, pf_multinomial_resample!, pf_stratified_resample!

using Distributions: rand!

"""
    pf_resample!(state::ParticleFilterState, method=:multinomial; kwargs...)

Resamples particles in the filter, stochastically pruning low-weight particles.
The resampling method can optionally be specified: `:multinomial` (default),
`:residual`, or `:stratified`. See [1] for a survey of resampling methods
and their variance properties. A `priority_fn` can also be specified as
keyword argument, which maps log particle weights to custom log priority scores
for the purpose of resampling (e.g. `w -> w/2` for less aggressive pruning).

[1] R. Douc and O. Cappé, "Comparison of resampling schemes
for particle filtering," in ISPA 2005. Proceedings of the 4th International
Symposium on Image and Signal Processing and Analysis, 2005., 2005, pp. 64–69.
"""
function pf_resample!(state::ParticleFilterState,
                      method::Symbol=:multinomial; kwargs...)
    if method == :multinomial
        return pf_multinomial_resample!(state; kwargs...)
    elseif method == :residual
        return pf_residual_resample!(state; kwargs...)
    elseif method == :stratified
        return pf_stratified_resample!(state; kwargs...)
    else
        error("Resampling method $method not recognized.")
    end
end

"""
    pf_multnomial_resample!(state::ParticleFilterState; kwargs...)

Performs multinomial resampling (i.e. simple random resampling) of the
particles in the filter. Each trace (i.e. particle) is resampled with
probability proportional to its weight.

A `priority_fn` can be specified as keyword argument, which maps log particle
weights to custom log priority scores for the purpose of resampling
(e.g. `w -> w/2` for less aggressive pruning).
"""
function pf_multinomial_resample!(state::ParticleFilterState;
                                  priority_fn=nothing)
    # Update estimate of log marginal likelihood
    n_particles = length(state.traces)
    state.log_ml_est += logsumexp(state.log_weights) - log(n_particles)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn == nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Resample new traces according to current normalized weights
    rand!(Categorical(exp.(lognorm(log_priorities))), state.parents)
    state.new_traces[1:end] = state.traces[state.parents]
    # Reweight particles
    if priority_fn == nothing
        state.log_weights .= 0.0
    else
        ws = state.log_weights[state.parents] .- log_priorities[state.parents]
        state.log_weights = ws .+ (log(n_particles) - logsumexp(ws))
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end

"""
    pf_residual_resample!(state::ParticleFilterState; kwargs...)

Performs residual resampling of the particles in the filter, which reduces
variance relative to multinomial sampling. For each particle with
normalized weight ``w_i``, ``⌊n w_i⌋`` copies are resampled, where ``n`` is the
total number of particles. The remainder are sampled with probability
proportional to ``n w_i - ⌊n w_i⌋`` for each particle ``i``.

A `priority_fn` can be specified as keyword argument, which maps log particle
weights to custom log priority scores for the purpose of resampling
(e.g. `w -> w/2` for less aggressive pruning).
"""
function pf_residual_resample!(state::ParticleFilterState;
                               priority_fn=nothing)
    # Update estimate of log marginal likelihood
    n_particles = length(state.traces)
    state.log_ml_est += logsumexp(state.log_weights) - log(n_particles)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn == nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Deterministically copy previous particles according to their weights
    n_resampled = 0
    weights = exp.(lognorm(log_priorities))
    for (i, w) in enumerate(weights)
        n_copies = floor(Int, n_particles * w)
        if n_copies == 0 continue end
        state.parents[n_resampled+1:n_resampled+n_copies] .= i
        for j in 1:n_copies
            state.new_traces[n_resampled+j] = state.traces[i]
        end
        n_resampled += n_copies
    end
    # Sample remainder according to residual weights
    if n_resampled < n_particles
        r_weights = n_particles .* weights .- floor.(n_particles .* weights)
        r_weights = r_weights / sum(r_weights)
        rand!(Categorical(r_weights), state.parents[n_resampled+1:end])
        state.new_traces[n_resampled+1:end] =
            state.traces[state.parents[n_resampled+1:end]]
    end
    # Reweight particles
    if priority_fn == nothing
        state.log_weights .= 0.0
    else
        ws = state.log_weights[state.parents] .- log_priorities[state.parents]
        state.log_weights = ws .+ (log(n_particles) - logsumexp(ws))
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end

"""
    pf_stratified_resample!(state::ParticleFilterState; kwargs...)

Performs stratified resampling of the particles in the filter, which reduces
variance relative to multinomial sampling. First, uniform random samples
``u_1, ..., u_n`` are drawn within the strata ``[0, 1/n)``, ..., ``[n-1/n, 1)``,
where ``n`` is the number of particles. Then, given the cumulative normalized
weights ``W_k = Σ_{j=1}^{k} w_j ``, sample the ``k``th particle for each ``u_i``
where ``W_{k-1} ≤ u_i < W_k``.

A `priority_fn` can be specified as keyword argument, which maps log particle
weights to custom log priority scores for the purpose of resampling
(e.g. `w -> w/2` for less aggressive pruning). The `sort_particles` keyword
argument controls whether particles are sorted by weight before stratification
(default: true).
"""
function pf_stratified_resample!(state::ParticleFilterState;
                                 priority_fn=nothing, sort_particles::Bool=true)
    # Update estimate of log marginal likelihood
    n_particles = length(state.traces)
    state.log_ml_est += logsumexp(state.log_weights) - log(n_particles)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn == nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    weights = exp.(lognorm(log_priorities))
    # Optionally sort particles by weight before resampling
    order = sort_particles ? sortperm(log_priorities) : collect(1:n_particles)
    # Sample particles within each weight stratum [i, i+1/n)
    i_old, accum_weight = 0, 0.0
    for (i_new, lower) in enumerate((0:n_particles-1)/n_particles)
        u = rand() * (1/n_particles) + lower
        while accum_weight < u
            accum_weight += weights[order[i_old+1]]
            i_old += 1
        end
        state.parents[i_new] = order[i_old]
        state.new_traces[i_new] = state.traces[order[i_old]]
    end
    # Reweight particles
    if priority_fn == nothing
        state.log_weights .= 0.0
    else
        ws = state.log_weights[state.parents] .- log_priorities[state.parents]
        state.log_weights = ws .+ (log(n_particles) - logsumexp(ws))
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end
