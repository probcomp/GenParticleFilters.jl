# GenParticleFilters.jl

A library for simple and advanced particle filtering in [Gen](https://www.gen.dev/), a general-purpose probabilistic programming system.

## Installation

Press `]` at the Julia REPL to enter the package manager, then run:
```julia
add GenParticleFilters
```

To install the development version, run:
```julia
add https://github.com/probcomp/GenParticleFilters.jl.git
```

## Features

In addition to basic particle filtering functionality (i.e., initializing a particle filter and updating it with new observations), this package provides support for:

- Particle updates that allow discarding of old choices, provided that backward kernels are specified [[1]](#1)
- Multiple resampling methods, including variance-reducing methods such as residual resampling [[2]](#2)
- Custom priority weights for resampling, to control the aggressiveness of pruning [[3]](#3)
- Metropolis-Hasting (i.e. move-accept) rejuvenation moves, to increase particle diversity [[4]](#4)
- Move-reweight rejuvenation, which increases particle diversity while reweighting particles [[5]](#5)
- Sequential Monte Carlo over a series of distinct models, via [trace translators](https://www.gen.dev/stable/ref/trace_translators/) [[6]](#6)
- SMCP³, a method which generalizes [[1]](#1), [[5]](#5) and [[6]](#6) through particle updates that support auxiliary randomness and deterministic transformations  [[7]](#7)
- Particle filter resizing methods, which can be used in online adaptation of the total number of particles [[8]](#8)
- Utility functions to compute distributional statistics (e.g. mean and variance) for the inferred latent variables

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
            # Perform a rejuvenation move on past choices
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

## References

```@raw html
<p>
<a id="1">[1]</a> P. D. Moral, A. Doucet, and A. Jasra, "Sequential Monte Carlo samplers," Journal of the Royal Statistical Society: Series B (Statistical Methodology), vol. 68, no. 3, pp. 411–436, 2006.
</p>
<p>
<a id="2">[2]</a> R. Douc and O. Cappé, "Comparison of resampling schemes for particle filtering," in ISPA 2005. Proceedings of the 4th International Symposium on Image and Signal Processing and Analysis, 2005., 2005, pp. 64-69.
</p>
<p>
<a id="3">[3]</a> R. Chen, "Sequential Monte Carlo methods and their applications," in Markov Chain Monte Carlo, vol. Volume 7, 0 vols., Singapore University Press, 2005, pp. 147–182.
</p>
<p>
<a id="4">[4]</a> N. Chopin, “A sequential particle filter method for static models,” Biometrika 89.3, 2000, pp. 539-552.
</p>
<p>
<a id="5">[5]</a> R. A. G. Marques and G. Storvik, "Particle move-reweighting strategies for online inference," Preprint series. Statistical Research Report, 2013.
</p>
<p>
<a id="6">[6]</a> M. Cusumano-Towner, B. Bichsel, T. Gehr, M. Vechev, and V. K. Mansinghka, “Incremental inference for probabilistic programs,” in Proceedings of the 39th ACM SIGPLAN Conference on Programming Language Design and Implementation, Philadelphia PA USA, Jun. 2018, pp. 571–585.
</p>
<p>
<a id="7">[7]</a> Lew, A. K., Matheos, G., Zhi-Xuan, T., Ghavamizadeh, M., Gothoskar, N., Russell, S., and Mansinghka, V. K. "SMCP3: Sequential Monte Carlo with Probabilistic Program Proposals." AISTATS, 2023.
</p>
<p>
<a id="8">[8]</a> V. Elvira, J. Míguez and P. M. Djurić, "Adapting the Number of Particles in Sequential Monte Carlo Methods Through an Online Scheme for Convergence Assessment," in IEEE Transactions on Signal Processing, vol. 65, no. 7, pp. 1781-1794, 1 April 2017, doi: 10.1109/TSP.2016.2637324.
</p>
```