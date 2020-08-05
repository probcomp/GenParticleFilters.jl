## Resampling methods for SMC algorithms ##
export pf_resample!
export pf_residual_resample!, pf_multinomial_resample!, pf_stratified_resample!

"""
    pf_resample!(state::ParticleFilterState, method=:multinomial; kwargs...)

Resamples particles in the filter, stochastically pruning low-weight particles.
The resampling method can optionally be specified: `:multinomial` (default),
`:residual`, or `:stratified`. See [1] for a survey of resampling methods
 and their variance properties.

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
    pf_multnomial_resample!(state::ParticleFilterState)

Performs multinomial resampling (i.e. simple random resampling) of the
particles in the filter. Each trace (i.e. particle) is resampled with
probability proportional to its weight.
"""
function pf_multinomial_resample!(state::ParticleFilterState)
    n_particles = length(state.traces)
    log_total_weight, log_weights = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_weights)
    state.log_ml_est += log_total_weight - log(n_particles)
    # Resample new traces according to current normalized weights
    Distributions.rand!(Categorical(weights), state.parents)
    for i=1:n_particles
        state.new_traces[i] = state.traces[state.parents[i]]
        state.log_weights[i] = 0.
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end

"""
    pf_residual_resample!(state::ParticleFilterState)

Performs residual resampling of the particles in the filter, which reduces
variance relative to multinomial sampling. For each particle with
normalized weight ``w_i``, ``⌊n w_i⌋`` copies are resampled, where ``n`` is the
total number of particles. The remainder are sampled with probability
proportional to ``n w_i - ⌊n w_i⌋`` for each particle ``i``.
"""
function pf_residual_resample!(state::ParticleFilterState)
    n_particles = length(state.traces)
    log_total_weight, log_weights = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_weights)
    state.log_ml_est += log_total_weight - log(n_particles)
    # Deterministically copy previous particles according to their weights
    n_resampled = 0
    for (i, w) in enumerate(weights)
        n_copies = floor(Int, n_particles * w)
        if n_copies == 0 continue end
        state.parents[n_resampled+1:n_resampled+n_copies] .= i
        for j in 1:n_copies
            state.new_traces[n_resampled+j] = state.traces[i]
        end
        n_resampled += n_copies
    end
    # Sample remainder according to weight remainders
    if n_resampled < n_particles
        r_weights = n_particles .* weights .- floor.(n_particles .* weights)
        r_weights = r_weights ./ sum(r_weights)
        Distributions.rand!(Categorical(r_weights), state.parents[n_resampled+1:end])
        state.new_traces[n_resampled+1:end] =
            state.traces[state.parents[n_resampled+1:end]]
    end
    state.log_weights .= 0.0
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end

"""
    pf_stratified_resample!(state::ParticleFilterState)

Performs stratified resampling of the particles in the filter, which reduces
variance relative to multinomial sampling. First, uniform random samples
``u_1, ..., u_n`` are drawn within the strata ``[0, 1/n)``, ..., ``[n-1/n, 1)``,
where ``n`` is the number of particles. Then, given the cumulative normalized
weights ``W_k = Σ_{j=1}^{k} w_j ``, sample the ``k``th particle for each ``u_i``
where ``W_{k-1} ≤ u_i < W_k``.
"""
function pf_stratified_resample!(state::ParticleFilterState;
                                 sort_particles::Bool=true)
    # Optionally sort particles by weight before resampling
    if sort_particles
        order = sortperm(state.log_weights)
        state.log_weights = state.log_weights[order]
        state.traces = state.traces[order]
    end
    n_particles = length(state.traces)
    log_total_weight, log_weights = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_weights)
    state.log_ml_est += log_total_weight - log(n_particles)
    # Sample particles within each weight stratum [i, i+1/n)
    i_old, accum_weight = 0, 0.0
    for (i_new, lower) in enumerate((0:n_particles-1)/n_particles)
        u = rand() * (1/n_particles) + lower
        while accum_weight < u
            accum_weight += weights[i_old+1]
            i_old += 1
        end
        state.parents[i_new] = sort_particles ? order[i_old] : i_old
        state.new_traces[i_new] = state.traces[i_old]
        state.log_weights[i_new] = 0.0
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end
