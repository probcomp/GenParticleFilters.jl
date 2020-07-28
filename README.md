# GenParticleFilters.jl

Building blocks for simple and advanced particle filtering in Gen.

## Installation

Press `]` at the Julia REPL to enter the package manager, then run:
```julia
add https://github.com/probcomp/GenParticleFilters.jl.git
```

## Example

Suppose we are trying to infer the position `y` of an object that is either staying still or moving sinusoidally, given noisy measurements `y_obs`. We can write a model of this object's motion as an `@gen` function:

```julia
@gen function object_motion(T::Int)
    y, moving = 0, false
    y_obs_all = Float64[]
    for t=1:T
        moving = {t => :moving} ~ bernoulli(moving ? 0.75 : 0.25)
        vel_y = moving ? sin(t) : 0.0
        y = {t => :y} ~ normal(y + vel_y, 0.01)
        y_obs = {t => :y_obs} ~ normal(y, 0.25)
        push!(y_obs_all, y_obs)
    end
    return y_obs_all
end
```

We can then construct a particle filter with resampling and rejuvenation moves, in order to infer both the object's position `y` and whether the object was `moving` at each timestep.

```julia
function particle_filter(observations, n_particles, ess_thresh=0.5)
    # Initialize particle filter with first observation
    n_obs = length(observations)
    obs_choices = [choicemap((t => :y_obs, observations[t])) for t=1:n_obs]
    state = pf_initialize(object_motion, (1,), obs_choices[1], n_particles)
    # Iterate across timesteps
    for t=2:n_obs
        # Resample and rejuvenate if the effective sample size is too low
        if effective_sample_size(state) < ess_thresh * n_particles
            # Perform residual resampling, pruning low-weight particles
            pf_resample!(state, :residual)
            # Perform a Gibbs rejuvenation move on past choices
            rejuv_sel = select(t-1=>:moving, t-1=>:y, t=>:moving, t=>:y)
            pf_rejuvenate!(state, mh, (rejuv_sel,))
        end
        # Update filter state with new observation at timestep t
        pf_update!(state, (t,), (UnknownChange(),), obs_choices[t])
    end
    return state
end
```

We can then run the particle filter on a sequence of observations, e.g., of the
object staying still for 5 timesteps then oscillating for 5 timesteps:

```julia
# Generate synthetic dataset of object motion
constraints = choicemap([(t => :moving, t > 5) for t in 1:10]...)
trace, _ = generate(object_motion, (10,), constraints)
observations = get_retval(trace)
# Run particle filter with 100 particles
state = particle_filter(observations, 100)
```

We can then use `mean` and `var` to compute the empirical posterior mean
and variance for variables of interest:
```julia
julia> mean(state, 5=>:moving) |> x->round(x, digits=2) # Prob. motion at t=5
0.07
julia> var(state, 5=>:moving) |> x->round(x, digits=2) # Variance at t=5
0.07
julia> mean(state, 6=>:moving) |> x->round(x, digits=2) # Prob. motion at t=6
0.95
julia> var(state, 6=>:moving) |> x->round(x, digits=2) # Variance at t=6
0.05
```

We see that the filter accurately infers a change in motion from `t=5` to `t=6`.
