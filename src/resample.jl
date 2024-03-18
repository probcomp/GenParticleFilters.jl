## Resampling methods for SMC algorithms ##
export pf_resample!
export pf_residual_resample!, pf_multinomial_resample!, pf_stratified_resample!

using Distributions: rand!

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
function pf_resample!(state::ParticleFilterView,
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
    pf_multinomial_resample!(state::ParticleFilterState; kwargs...)

Performs multinomial resampling (i.e. simple random resampling) of the
particles in the filter. Each trace (i.e. particle) is resampled with
probability proportional to its weight.

# Keyword Arguments

- `priority_fn = nothing`: An optional function that maps particle weights to
  custom log priority scores (e.g. `w -> w/2` for less aggressive pruning).
- `check = :warn`: Set to `true` to throw an error for invalid normalized
   weights (all NaNs or zeros), `:warn` to issue warnings, or `false` to
   suppress checks. In the latter two cases, zero weights will be renormalized
   to uniform weights for resampling.
"""
function pf_multinomial_resample!(state::ParticleFilterView;
                                  priority_fn=nothing, check=:warn)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn === nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Normalize weights and check their validity
    weights, invalid = safe_softmax(log_priorities, warn = (check != false))
    check == true && invalid && error("Invalid weights.")
    # Update estimate of log marginal likelihood
    update_lml_est!(state)
    # Resample new traces according to current normalized weights
    rand!(Categorical(weights), state.parents)
    state.new_traces .= view(state.traces, state.parents)
    # Reweight particles and update trace references
    update_weights!(state, log_priorities)
    update_refs!(state)
    return state
end

"""
    pf_residual_resample!(state::ParticleFilterState; kwargs...)

Performs residual resampling of the particles in the filter, which reduces
variance relative to multinomial sampling. For each particle with
normalized weight ``w_i``, ``⌊n w_i⌋`` copies are resampled, where ``n`` is the
total number of particles. The remainder are sampled with probability
proportional to ``n w_i - ⌊n w_i⌋`` for each particle ``i``.

# Keyword Arguments

- `priority_fn = nothing`: An optional function that maps particle weights to
  custom log priority scores (e.g. `w -> w/2` for less aggressive pruning).
- `check = :warn`: Set to `true` to throw an error for invalid normalized
   weights (all NaNs or zeros), `:warn` to issue warnings, or `false` to
   suppress checks. In the latter two cases, zero weights will be renormalized
   to uniform weights for resampling.
"""
function pf_residual_resample!(state::ParticleFilterView;
                               priority_fn=nothing, check=:warn)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn === nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Normalize weights and check their validity
    weights, invalid = safe_softmax(log_priorities, warn = (check != false))
    check == true && invalid && error("Invalid weights.")
    # Update estimate of log marginal likelihood
    update_lml_est!(state)
    # Deterministically copy previous particles according to their weights
    n_resampled = 0
    n_particles = length(state.traces)
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
        r_idxs = n_resampled+1:n_particles
        r_parents = view(state.parents, r_idxs)
        rand!(Categorical(r_weights), r_parents)
        state.new_traces[r_idxs] .= view(state.traces, r_parents)
    end
    # Reweight particles and update trace references
    update_weights!(state, log_priorities)
    update_refs!(state)
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

# Keyword Arguments

- `priority_fn = nothing`: An optional function that maps particle weights to
  custom log priority scores (e.g. `w -> w/2` for less aggressive pruning).
- `check = :warn`: Set to `true` to throw an error for invalid normalized
   weights (all NaNs or zeros), `:warn` to issue warnings, or `false` to
   suppress checks. In the latter two cases, zero weights will be renormalized
   to uniform weights for resampling.
- `sort_particles = true`: Set to `true` to sort particles by weight before
   stratification.
"""
function pf_stratified_resample!(state::ParticleFilterView;
                                 priority_fn=nothing, check=:warn,
                                 sort_particles::Bool=true)
    # Compute priority scores if priority function is provided
    log_priorities = priority_fn === nothing ?
        state.log_weights : priority_fn.(state.log_weights)
    # Normalize weights and check their validity
    weights, invalid = safe_softmax(log_priorities, warn = (check != false))
    check == true && invalid && error("Invalid weights.")
    # Update estimate of log marginal likelihood
    update_lml_est!(state)
    # Optionally sort particles by weight before resampling
    n_particles = length(state.traces)
    order = sort_particles ?
        sortperm(log_priorities, rev=true) : collect(1:n_particles)
    # Sample particles within each weight stratum [i, i+1/n)
    i_old, weight_step, accum_weight = 0, 1/n_particles, 0.0
    for (i_new, lower) in enumerate(0.0:weight_step:1.0-weight_step)
        if lower + weight_step > accum_weight
            u = rand() * weight_step + lower
            while accum_weight < u
                accum_weight += weights[order[i_old+1]]
                i_old += 1
            end
        end
        state.parents[i_new] = order[i_old]
        state.new_traces[i_new] = state.traces[order[i_old]]
    end
    # Reweight particles and update trace references
    update_weights!(state, log_priorities)
    update_refs!(state)
    return state
end

"Update log marginal likelihood estimate before a resampling step."
function update_lml_est!(state::ParticleFilterState)
    n_particles = length(state.traces)
    state.log_ml_est += logsumexp(state.log_weights) - log(n_particles)
    return nothing
end

# Do nothing for substates, since they don't separately track the estimate
function update_lml_est!(state::ParticleFilterSubState)
    return nothing
end

"Update particle weights after a resampling step."
function update_weights!(state::ParticleFilterState, log_priorities)
    n_particles = length(state.traces)
    # Ensure logsumexp(log_weights) == log(n_particles) after resampling
    if log_priorities === state.log_weights
        # If priorities aren't customized, set all log weights to 0
        state.log_weights .= 0.0
    else
        # Otherwise, set new weights to the ratio of weights over priorities
        log_ws = state.log_weights[state.parents] .- log_priorities[state.parents]
        # Adjust new weights such that they sum to the number of particles
        state.log_weights .= log_ws .+ (log(n_particles) - logsumexp(log_ws))
    end
end

# Handle sub-state resampling differently since we can't update log_ml_est
function update_weights!(state::ParticleFilterSubState, log_priorities)
    n_particles = length(state.traces)
    # Ensure logsumexp(log_weights) remains equal after resampling
    if log_priorities === state.log_weights
        # If priorities aren't customized, set all weights to average weight
        state.log_weights .= logsumexp(state.log_weights) - log(n_particles)
    else
        # Otherwise, set new weights to the ratio of weights over priorities
        log_ws = state.log_weights[state.parents] .- log_priorities[state.parents]
        # Adjust new weights such that they sum to the original total weight
        log_total_weight = logsumexp(state.log_weights)
        state.log_weights .= log_ws .+ (log_total_weight - logsumexp(log_ws))
    end
end
