## Resampling methods for SMC algorithms ##
export pf_resample!, pf_multinomial_resample!, pf_stratified_resample!

"Resamples particles in the filter."
function pf_resample!(state::ParticleFilterState,
                      method::Symbol=:multinomial; kwargs...)
    if method == :multinomial
        return pf_multinomial_resample!(state)
    elseif method == :stratified
        return pf_stratified_resample!(state; kwargs...)
    else
        error("Resampling method $method not recognized.")
    end
end

"Perform multinomial resampling (i.e. simple random resampling) of the particles."
function pf_multinomial_resample!(state::ParticleFilterState)
    n_particles = length(state.traces)
    log_total_weight, log_weights = Gen.normalize_weights(state.log_weights)
    weights = exp.(log_weights)
    state.log_ml_est += log_total_weight - log(n_particles)
    # Resample new traces according to current normalized weights
    rand!(Categorical(weights / sum(weights)), state.parents)
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

"Perform stratified resampling of the particles in the filter."
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
        state.new_traces[i_new] = state.traces[i_old]
        state.log_weights[i_new] = 0.0
    end
    # Swap references
    tmp = state.traces
    state.traces = state.new_traces
    state.new_traces = tmp
    return state
end
