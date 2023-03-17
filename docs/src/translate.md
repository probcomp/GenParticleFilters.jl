# Trace Translators

```@meta
CurrentModule = GenParticleFilters
```

GenParticleFilters.jl provides two additional types of [trace translators](https://www.gen.dev/docs/stable/ref/trace_translators/) for use in particle filtering and sequential Monte Carlo algorithms.

## Extending Trace Translator

This trace translator can be used to sample new latent variables from a custom proposal, then optionally apply a deterministic transform to those latent variables before they are used to update the model trace.

```@docs
ExtendingTraceTranslator
ExtendingTraceTranslator(::Trace)
```

## Updating Trace Translator

This trace translator supports forward and backward proposals (which may include auxiliary randomness), as well as deterministic transformations between proposed choices and model choices.

```@docs
UpdatingTraceTranslator
UpdatingTraceTranslator(::Trace)
```
