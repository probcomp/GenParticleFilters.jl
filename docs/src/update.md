# Particle Updating

```@meta
CurrentModule = GenParticleFilters
```

GenParticleFilters.jl provides numerous methods for updating particles with new observations or input arguments, ranging from simply conditioning on the new observations, to complex SMC moves that allow for the use of auxiliary randomness and deterministic transformations.

## Updating with the default proposal

```@docs
pf_update!(state::ParticleFilterView, new_args::Tuple,
           argdiffs::Tuple, observations::ChoiceMap)
```

## Updating with a custom proposal

```@docs
pf_update!(
    state::ParticleFilterView,
    new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap,
    proposal::GenerativeFunction, proposal_args::Tuple,
    transform::Union{TraceTransformDSLProgram,Nothing}
)
```

## Updating with custom forward and backward proposals

```@docs
pf_update!(
    state::ParticleFilterView,
    new_args::Tuple, argdiffs::Tuple,
    observations::ChoiceMap,
    fwd_proposal::GenerativeFunction, fwd_args::Tuple,
    bwd_proposal::GenerativeFunction, bwd_args::Tuple,
    transform::Union{TraceTransformDSLProgram,Nothing}
)
```

## Updating with a trace translator

Trace translators can be used to update particles in a highly general fashion, including the translation of traces from one generative function into traces of a different generative function.

```@docs
pf_update!(state::ParticleFilterView, translator)
```

## Updating with stratified sampling

All of the above methods can also be combined with stratified sampling. This can be used to ensure deterministic coverage of the sample space of newly introduced random variables, thereby reducing variance.

```@docs
pf_update!(state::ParticleFilterView, new_args::Tuple, argdiffs::Tuple,
           observations::ChoiceMap, strata)
```

For convenience, strata can be generated using the [`choiceproduct`](@ref) function.