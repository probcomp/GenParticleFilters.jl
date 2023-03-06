@testset "Particle initialization" begin

@testset "Initialize with default proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    @test all(-2 <= tr[:slope] <= 2 for tr in get_traces(state))
    state = pf_initialize(line_model, (1,), line_choicemap(1), 100)
    @test all(tr[:line => 1 => :y] == 0 for tr in get_traces(state))
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
end

@gen line_propose(s) =
    slope ~ uniform_discrete(0, 0)
@gen outlier_propose(idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]

@testset "Initialize with custom proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 100)
    @test all(tr[:slope] == 0 for tr in get_traces(state))
    state = pf_initialize(line_model, (1,), line_choicemap(1),
                          outlier_propose, ([1],), 100)
    @test all(tr[:line => 1 => :outlier] == false for tr in get_traces(state))
    @test all(tr[:line => 1 => :y] == 0 for tr in get_traces(state))
    state = pf_initialize(line_model, (10,), line_choicemap(10),
                          outlier_propose, ([10],), 100)
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
end

@testset "Initialize for dynamic model structure" begin
    state = pf_initialize(line_model, (0,), choicemap(), 10; dynamic=true)
    @test typeof(state) == ParticleFilterState{Trace}
    state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 10; dynamic=true)
    @test typeof(state) == ParticleFilterState{Trace}
end

@testset "Initialize with stratification" begin
    slope_strata = (slope_choicemap(s) for s in -2:1:2)
    observations = line_choicemap(1)
    # Test contiguous stratification
    state = pf_initialize(line_model, (1,), observations,
                          slope_strata, 100; layout=:contiguous)
    for (k, slope) in zip([20, 40, 60, 80, 100], -2:1:2)
        traces = get_traces(state[(k-20+1):k])
        @test all(tr[:slope] == slope for tr in traces)
        @test all(tr[:line => 1 => :y] == 0 for tr in traces)
    end
    # Test interleaved stratification
    state = pf_initialize(line_model, (1,), observations,
                          slope_strata, 100; layout=:interleaved)
    for (k, slope) in zip([1, 2, 3, 4, 5], -2:1:2)
        traces = get_traces(state[k:20:100])
        @test all(tr[:slope] == slope for tr in traces)
        @test all(tr[:line => 1 => :y] == 0 for tr in traces)
    end
end

@testset "Initialize with stratification and custom proposal" begin
    slope_strata = (slope_choicemap(s) for s in -2:1:2)
    observations = line_choicemap(1)
    # Test contiguous stratification
    state = pf_initialize(line_model, (1,), observations, slope_strata,
                          outlier_propose, ([1],), 100; layout=:contiguous)
    for (k, slope) in zip([20, 40, 60, 80, 100], -2:1:2)
        traces = get_traces(state[(k-20+1):k])
        @test all(tr[:slope] == slope for tr in traces)
        @test all(tr[:line => 1 => :outlier] == false for tr in traces)
        @test all(tr[:line => 1 => :y] == 0 for tr in traces)
    end
    # Test interleaved stratification
    state = pf_initialize(line_model, (1,), observations, slope_strata,
                          outlier_propose, ([1],), 100; layout=:interleaved)
    for (k, slope) in zip([1, 2, 3, 4, 5], -2:1:2)
        traces = get_traces(state[k:20:100])
        @test all(tr[:slope] == slope for tr in traces)
        @test all(tr[:line => 1 => :outlier] == false for tr in traces)
        @test all(tr[:line => 1 => :y] == 0 for tr in traces)
    end
end

end