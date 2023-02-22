@testset "Particle initialization" begin

@testset "Initialize with default proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    @test all(-2 <= tr[:slope] <= 2 for tr in get_traces(state))
    state = pf_initialize(line_model, (1,), generate_line(1), 100)
    @test all(tr[:line => 1 => :y] == 0 for tr in get_traces(state))
    state = pf_initialize(line_model, (10,), generate_line(10), 100)
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
end

@gen line_propose(s) =
    slope ~ uniform_discrete(0, 0)
@gen outlier_propose(idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]

@testset "Initialize with custom proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 100)
    @test all(tr[:slope] == 0 for tr in get_traces(state))
    state = pf_initialize(line_model, (1,), choicemap(), outlier_propose, ([1],), 100)
    @test all(tr[:line => 1 => :outlier] == false for tr in get_traces(state))
    state = pf_initialize(line_model, (10,), choicemap(), outlier_propose, ([10],), 100)
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
end

@testset "Initialize for dynamic model structure" begin
    state = pf_initialize(line_model, (0,), choicemap(), 10, true)
    @test typeof(state) == ParticleFilterState{Trace}
    state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 10, true)
    @test typeof(state) == ParticleFilterState{Trace}
end
    
end