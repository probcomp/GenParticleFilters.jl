## Functions for calculating empirical statistics ##
export mean, var, proportionmap

using Statistics, StatsBase

"""
    mean(state::ParticleFilterState[, addr])

Returns the weighted empirical mean for a particular trace address `addr`.
If `addr` is not provided, returns the empirical mean of the return value
associated with each trace in the particle filter.
"""
Statistics.mean(state::ParticleFilterView, addr) =
    sum(get_norm_weights(state) .* getindex.(state.traces, addr))

Statistics.mean(state::ParticleFilterView) =
    sum(get_norm_weights(state) .* get_retval.(state.traces))

"""
    mean(f::Function, state::ParticleFilterState[, addrs...])

Returns the weighted empirical mean of a function `f` applied to the values
at one or more trace addresses `addrs`, where `f` takes in a number of
arguments equal to the number of addresses.

If no addresses are provided, `f` is applied to the return value of each trace.
"""
function Statistics.mean(f::Function, state::ParticleFilterView, addr, addrs...)
    argvals = (getindex.(state.traces, a) for a in addrs)
    fvals = broadcast(f, getindex.(state.traces, addr), argvals...)
    return sum(get_norm_weights(state) .* fvals)
end

Statistics.mean(f::Function, state::ParticleFilterView, addr) =
    sum(get_norm_weights(state) .* f.(getindex.(state.traces, addr)))

Statistics.mean(f::Function, state::ParticleFilterView) =
    sum(get_norm_weights(state) .* f.(get_retval.(state.traces)))


"""
    var(state::ParticleFilterState[, addr])

Returns the empirical variance for a particular trace address `addr`.
If `addr` is not provided, returns the empirical variance of the return value
associated with each trace in the particle filter.
"""
Statistics.var(state::ParticleFilterView, addr) =
    sum(get_norm_weights(state) .*
        (getindex.(state.traces, addr) .- mean(state, addr)).^2)

Statistics.var(state::ParticleFilterView) =
    sum(get_norm_weights(state) .*
        (get_retval.(state.traces) .- mean(state)).^2)

"""
    var(f::Function, state::ParticleFilterState[, addrs...])

Returns the empirical variance of a function `f` applied to the values
at one or more trace addresses `addrs`, where `f` takes in a number of
arguments equal to the number of addresses.

If no addresses are provided, `f` is applied to the return value of each trace.
"""
function Statistics.var(f::Function, state::ParticleFilterView, addr, addrs...)
    argvals = (getindex.(state.traces, a) for a in addrs)
    fvals = broadcast(f, getindex.(state.traces, addr), argvals...)
    ws = Weights(get_norm_weights(state))
    return var(fvals, ws, corrected=false)
end

function Statistics.var(f::Function, state::ParticleFilterView, addr)
    vs = f.(getindex.(state.traces, addr))
    ws = Weights(get_norm_weights(state))
    return var(vs, ws, corrected=false)
end

function Statistics.var(f::Function, state::ParticleFilterView)
    vs = f.(get_retval.(state.traces))
    ws = Weights(get_norm_weights(state))
    return var(vs, ws, corrected=false)
end

"""
    proportionmap(state::ParticleFilterState[, addr])

Returns a dictionary mapping each unique value at a trace address `addr` to
its proportion (i.e. sum of normalized weights) in the particle filter. If
`addr` is not provided, returns the proportions of each possible return value.
"""
function StatsBase.proportionmap(state::ParticleFilterView, addr)
    vs = getindex.(state.traces, addr)
    ws = Weights(get_norm_weights(state))
    return countmap(vs, ws)
end

function StatsBase.proportionmap(state::ParticleFilterView)
    vs = get_retval.(state.traces)
    ws = Weights(get_norm_weights(state))
    return countmap(vs, ws)
end

"""
    proportionmap(f::Function, state::ParticleFilterState[, addrs...])

Applies `f` to the values at one or more trace addresses `addrs`, then returns
a dictionary mapping each unique value to its proportion (i.e. sum of
normalized weights) in the particle filter. 

If no addresses are provided, `f` is applied to the return value of each trace.
"""
function StatsBase.proportionmap(f::Function, state::ParticleFilterView,
                                 addr, addrs...)
    argvals = (getindex.(state.traces, a) for a in addrs)
    fvals = broadcast(f, getindex.(state.traces, addr), argvals...)
    ws = Weights(get_norm_weights(state))
    return countmap(fvals, ws)
end

function StatsBase.proportionmap(f::Function, state::ParticleFilterView, addr)
    vs = f.(getindex.(state.traces, addr))
    ws = Weights(get_norm_weights(state))
    return countmap(vs, ws)
end

function StatsBase.proportionmap(f::Function, state::ParticleFilterView)
    vs = f.(get_retval.(state.traces))
    ws = Weights(get_norm_weights(state))
    return countmap(vs, ws)
end
