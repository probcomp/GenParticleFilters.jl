using GenParticleFilters, Gen, Test, Logging

@gen (static) function line_step(t::Int, x::Float64, slope::Float64)
    x = x + 1
    outlier ~ bernoulli(0.1)
    y ~ normal(x * slope, outlier ? 10.0 : 1.0)
    return x
end

line_unfold = Unfold(line_step)

@gen (static) function line_model(n::Int)
    slope ~ uniform_discrete(-2, 2)
    line ~ line_unfold(n, 0, slope)
    return line
end

@load_generated_functions()

line_choicemap(n::Int, slope::Float64=0.) =
    choicemap([(:line => i => :y, i*slope) for i in 1:n]...)

include("utils.jl")
include("statistics.jl")
include("initialize.jl")
include("update.jl")
include("resample.jl")
include("rejuvenate.jl")
