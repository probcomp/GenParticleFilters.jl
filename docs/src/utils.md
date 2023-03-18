# Data Structures and Utilities

```@meta
CurrentModule = GenParticleFilters
```

The state of a particle filter is represented by a [`ParticleFilterState`](@ref), which stores a set of traces (representing particles) and their associated (unnormalized) weights:

```@docs
ParticleFilterState
```

The following functions can be used to access the traces and weights:

```@docs
get_traces
get_log_weights
get_log_norm_weights
get_norm_weights
```

Users can also sample traces from the particle filter, with probability proportional to their weights:

```@docs
sample_unweighted_traces
```

To enable parallelism and block-wise operations, users can index into a `ParticleFilterState` to obtain a `ParticleFilterSubState`:

```@docs
ParticleFilterSubState
```

Aside from resizing, all particle filtering operations that can be applied to a [`ParticleFilterState`](@ref) can also be applied to a [`ParticleFilterSubState`](@ref).

## Diagnostics

Several functions are provided to compute particle filter diagnostics.

```@docs
effective_sample_size
get_ess
log_ml_estimate
get_lml_est
```

## Statistics

The following methods for computing summary statistics are also provided.

```@docs
mean(::ParticleFilterView, addr)
mean(f::Function, ::ParticleFilterView, addr, addrs...)
var(::ParticleFilterView, addr)
var(f::Function, ::ParticleFilterView, addr, addrs...)
proportionmap(::ParticleFilterView, addr)
proportionmap(f::Function, ::ParticleFilterView, addr, addrs...)
```

## Stratification

To support the use of stratified sampling, the [`choiceproduct`](@ref) method can be used to conveniently generate a list of choicemap strata:

```@docs
choiceproduct
```