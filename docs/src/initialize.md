# Particle Initialization

```@meta
CurrentModule = GenParticleFilters
```

To initialize a particle filter, GenParticleFilters.jl supports both regular initialization via simple random sampling and stratified initialization.

## Random initialization

By default, particles are sampled randomly according to the internal proposal of the provided `model`, or a custem `proposal` if one is specified.

```@docs
pf_initialize(model::GenerativeFunction, model_args::Tuple,
              observations::ChoiceMap, n_particles::Int)
```

## Stratified initialization

To reduce variance, particle filters can also be initialized via stratified sampling, given a list of provided `strata`.

```@docs
pf_initialize(model::GenerativeFunction, model_args::Tuple,
              observations::ChoiceMap, strata, n_particles::Int)
```
