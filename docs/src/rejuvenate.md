# Particle Rejuvenation

```@meta
CurrentModule = GenParticleFilters
```

To increase particle diversity (e.g. after a resampling step), GenParticleFilters.jl provides support for both MCMC and move-reweight rejuvenation kernels.

```@docs
pf_rejuvenate!
```

## MCMC rejuvenation

MCMC rejuvenation can be used to diversify the set of existing particles by applying an invariant MCMC kernel (e.g. a Metropolis-Hastings kernel) to each particle. Since the kernel is expected to be invariant, kernel application does not adjust the particle weights.

```@docs
pf_move_accept!
```

## Move-reweight rejuvenation

Move-reweight rejuvenation can be used to simultaneously diversify the set of existing particles while updating their weights.

```@docs
pf_move_reweight!
```

To define a move-reweight kernel, the following methods are provided:

```@docs
move_reweight
```

Care must be taken in choosing forward and backward kernels to reduce variance in the weight updates. A helpful rule-of-thumb when specifying backward kernels is to try and make them *locally optimal*: If $$\pi(x)$$ is the initial distribution of particles prior to the rejuvenation move, and $$K(x'; x)$$ is the forward kernel that will be applied to perturb each particle, then the backward kernel $$L(x; x')$$ should be designed to approximate the local posterior $$P(x | x') \propto \pi(x) K(x'; x)$$.