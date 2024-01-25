# Particle Filter Resizing

```@meta
CurrentModule = GenParticleFilters
```

Some particle filtering algorithms may adaptively resize the number of particles in order to trade-off time and accuracy, or temporarily replicate existing particles in order to better explore the state space. GenParticleFilters.jl supports these algorithms by providing methods for resizing or replicating particles.

```@docs
pf_resize!
```

## Resizing via multinomial resampling

```@docs
pf_multinomial_resize!
```

## Resizing via residual resampling

```@docs
pf_residual_resize!
```

## Resizing via optimal resampling

```@docs
pf_optimal_resize!
```

## Particle replication

```@docs
pf_replicate!
```

## Particle dereplication

```@docs
pf_dereplicate!
```

## Particle coalescing

```@docs
pf_coalesce!
```

## Particle introduction

```@docs
pf_introduce!
```
