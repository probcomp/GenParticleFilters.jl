using GenParticleFilters, Gen, Test

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

generate_line(n::Int, slope::Float64=0.) =
    choicemap([(:line => i => :y, i*slope) for i in 1:n]...)

@testset "Particle filter initialization" begin

@testset "Initialize with default proposal" begin
state = pf_initialize(line_model, (0,), choicemap(), 100)
@test all([-2 <= tr[:slope] <= 2 for tr in get_traces(state)])
state = pf_initialize(line_model, (1,), generate_line(1), 100)
@test all([tr[:line => 1 => :y] == 0 for tr in get_traces(state)])
state = pf_initialize(line_model, (10,), generate_line(10), 100)
@test all([tr[:line => 10 => :y] == 0 for tr in get_traces(state)])
end

@gen line_propose(s) =
    slope ~ uniform_discrete(0, 0)
@gen outlier_propose(idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]

@testset "Initialize with custom proposal" begin
state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 100)
@test all([tr[:slope] == 0 for tr in get_traces(state)])
state = pf_initialize(line_model, (1,), choicemap(), outlier_propose, ([1],), 100)
@test all([tr[:line => 1 => :outlier] == false for tr in get_traces(state)])
state = pf_initialize(line_model, (10,), choicemap(), outlier_propose, ([10],), 100)
@test all([tr[:line => 10 => :outlier] == false for tr in get_traces(state)])
end

@testset "Initialize for dynamic model structure" begin
state = pf_initialize(line_model, (0,), choicemap(), 10, true)
@test typeof(state) == ParticleFilterState{Trace}
state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 10, true)
@test typeof(state) == ParticleFilterState{Trace}
end

end

@testset "Particle filter update" begin

@testset "Update with default proposal" begin
state = pf_initialize(line_model, (0,), choicemap(), 100)
state = pf_update!(state, (10,), (UnknownChange(),), generate_line(10))
@test all([tr[:line => 10 => :y] == 0 for tr in get_traces(state)])
@test all([w != 0 for w in get_log_weights(state)])
end

@gen outlier_propose(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]

@testset "Update with custom proposal" begin
state = pf_initialize(line_model, (0,), choicemap(), 100)
state = pf_update!(state, (10,), (UnknownChange(),), generate_line(10),
                   outlier_propose, (10,))
@test all([tr[:line => 10 => :y] == 0 for tr in get_traces(state)])
@test all([tr[:line => 10 => :outlier] == false for tr in get_traces(state)])
@test all([w != 0 for w in get_log_weights(state)])
end


end
