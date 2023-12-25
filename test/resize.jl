@testset "Particle filter resizing" begin

@testset "Multinomial resizing" begin
    # Test multinomial resampling of all particles
    for n_particles in [50, 150]
        state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
        old_traces = copy(get_traces(state))
        old_lml_est = get_lml_est(state)
        state = pf_resize!(state, n_particles, :multinomial)
        new_traces = get_traces(state)
        new_lml_est = get_lml_est(state)
        @test length(new_traces) == n_particles
        @test new_traces == old_traces[state.parents]
        @test new_lml_est ≈ old_lml_est
    end

    # Same as above, with a custom priority function
    p_fn = w -> w / 2
    for n_particles in [50, 150]
        state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
        old_traces = copy(get_traces(state))
        old_lml_est = get_lml_est(state)
        state = pf_resize!(state, n_particles, :multinomial; priority_fn=p_fn)
        new_traces = get_traces(state)
        new_lml_est = get_lml_est(state)
        @test length(new_traces) == n_particles
        @test new_traces == old_traces[state.parents]
        @test new_lml_est ≈ old_lml_est
    end
end

@testset "Residual resizing" begin
    # Test that at least the minimum number of copies are resampled
    for n_particles in [50, 150]
        state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
        old_traces = copy(get_traces(state))
        old_lml_est = get_lml_est(state)
        weights = get_norm_weights(state)
        min_copies = floor.(Int, weights * n_particles)
        state = pf_resize!(state, n_particles, :residual)
        new_traces = get_traces(state)
        @test length(new_traces) == n_particles
        copies = [sum([t1 == t2 for t1 in new_traces]) for t2 in old_traces]
        @test new_traces == old_traces[state.parents]
        @test all(copies .>= min_copies)
        new_lml_est = get_lml_est(state)
        @test new_lml_est ≈ old_lml_est
    end

    # Same test but with a custom priority function
    p_fn = w -> w / 2
    for n_particles in [50, 150]
        state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
        old_traces = copy(get_traces(state))
        old_lml_est = get_lml_est(state)
        log_priorities = p_fn.(get_log_weights(state))
        weights = exp.(log_priorities .- logsumexp(log_priorities))
        min_copies = floor.(Int, weights * n_particles)
        state = pf_resize!(state, n_particles, :residual; priority_fn=p_fn)
        new_traces = get_traces(state)
        @test length(new_traces) == n_particles
        copies = [sum([t1 == t2 for t1 in new_traces]) for t2 in old_traces]
        @test new_traces == old_traces[state.parents]
        @test all(copies .>= min_copies)
        new_lml_est = get_lml_est(state)
        @test new_lml_est ≈ old_lml_est
    end
end

@testset "Particle replication" begin
    slope_strata = (slope_choicemap(s) for s in -2:1:2)
    observations = line_choicemap(1)
    # Test contiguous replication
    state = pf_initialize(line_model, (1,), observations, slope_strata, 5)
    old_lml_est = get_lml_est(state)
    state = pf_replicate!(state, 20; layout=:contiguous)
    new_lml_est = get_lml_est(state)
    for (k, slope) in zip([20, 40, 60, 80, 100], -2:1:2)
        traces = get_traces(state[(k-20+1):k])
        weights = get_log_weights(state[(k-20+1):k])
        @test all(tr[:slope] == slope for tr in traces)
        @test all(tr === traces[1] for tr in traces)
        @test all(w === weights[1] for w in weights)
    end
    @test new_lml_est ≈ old_lml_est
    # Test interleaved replication
    state = pf_initialize(line_model, (1,), observations, slope_strata, 5)
    old_lml_est = get_lml_est(state)
    state = pf_replicate!(state, 20; layout=:interleaved)
    new_lml_est = get_lml_est(state)
    for (k, slope) in zip([1, 2, 3, 4, 5], -2:1:2)
        traces = get_traces(state[k:5:100])
        weights = get_log_weights(state[k:5:100])
        @test all(tr[:slope] == slope for tr in traces)
        @test all(tr === traces[1] for tr in traces)
        @test all(w === weights[1] for w in weights)
    end
    @test new_lml_est ≈ old_lml_est
end

@testset "Particle dereplication (keep first)" begin
    slope_strata = (slope_choicemap(s) for s in -2:1:2)
    observations = line_choicemap(1)
    # Test contiguous dereplication
    state = pf_initialize(line_model, (1,), observations, slope_strata, 5)
    old_traces = copy(get_traces(state))
    old_log_weights = copy(get_log_weights(state))
    old_lml_est = get_lml_est(state)
    state = pf_replicate!(state, 20; layout=:contiguous)
    state = pf_dereplicate!(state, 20; layout=:contiguous, method=:keepfirst)
    new_traces = get_traces(state)
    new_log_weights = get_log_weights(state)
    new_lml_est = get_lml_est(state)
    for (k, slope) in enumerate(-2:1:2)
        @test new_traces[k][:slope] == slope
        @test new_traces[k] === old_traces[k]
        @test new_log_weights[k] == old_log_weights[k]
    end
    @test new_lml_est ≈ old_lml_est
    # Test interleaved replication
    state = pf_initialize(line_model, (1,), observations, slope_strata, 5)
    old_traces = copy(get_traces(state))
    old_log_weights = copy(get_log_weights(state))
    old_lml_est = get_lml_est(state)
    state = pf_replicate!(state, 20; layout=:interleaved)
    state = pf_dereplicate!(state, 20; layout=:interleaved, method=:keepfirst)
    new_traces = get_traces(state)
    new_log_weights = get_log_weights(state)
    new_lml_est = get_lml_est(state)
    for (k, slope) in enumerate(-2:1:2)
        @test new_traces[k][:slope] == slope
        @test new_traces[k] === old_traces[k]
        @test new_log_weights[k] == old_log_weights[k]
    end
    @test new_lml_est ≈ old_lml_est
end

@testset "Particle dereplication (sample)" begin
    slope_strata = (slope_choicemap(s) for s in -2:1:2)
    observations = line_choicemap(1)
    # Test contiguous dereplication
    state = pf_initialize(line_model, (0,), choicemap(), slope_strata, 5)
    state = pf_replicate!(state, 20; layout=:contiguous)
    state = pf_update!(state, (1,), (UnknownChange(),), observations)
    old_traces = copy(get_traces(state))
    old_log_weights = copy(get_log_weights(state))
    old_lml_est = get_lml_est(state)
    state = pf_dereplicate!(state, 20; layout=:contiguous, method=:sample)
    new_traces = get_traces(state)
    new_log_weights = get_log_weights(state)
    new_lml_est = get_lml_est(state)
    for (i, (k, slope)) in enumerate(zip([20, 40, 60, 80, 100], -2:1:2))
        traces = old_traces[(k-20+1):k]
        weights = old_log_weights[(k-20+1):k]
        @test new_traces[i][:slope] == slope
        @test new_traces[i] in traces
        @test new_log_weights[i] ≈ logsumexp(weights) - log(20)
    end
    @test new_lml_est ≈ old_lml_est
    # Test interleaved replication
    state = pf_initialize(line_model, (0,), choicemap(), slope_strata, 5)
    state = pf_replicate!(state, 20; layout=:interleaved)
    state = pf_update!(state, (1,), (UnknownChange(),), observations)
    old_traces = copy(get_traces(state))
    old_log_weights = copy(get_log_weights(state))
    old_lml_est = get_lml_est(state)
    state = pf_dereplicate!(state, 20; layout=:interleaved, method=:sample)
    new_traces = get_traces(state)
    new_log_weights = get_log_weights(state)
    new_lml_est = get_lml_est(state)
    for (i, (k, slope)) in enumerate(zip([1, 2, 3, 4, 5], -2:1:2))
        traces = old_traces[k:5:100]
        weights = old_log_weights[k:5:100]
        @test new_traces[i][:slope] == slope
        @test new_traces[i] in traces
        @test new_log_weights[i] ≈ logsumexp(weights) - log(20)
    end
    @test new_lml_est ≈ old_lml_est
end

@testset "Particle introduction (default proposal)" begin
    # Test explicit specification of model and arguments
    state = pf_initialize(line_model, (0,), choicemap(), 50)
    state = pf_introduce!(state, line_model, (0,), choicemap(), 50)
    @test length(get_traces(state)) == 100
    @test all(-2 <= tr[:slope] <= 2 for tr in get_traces(state))
    @test all(w ≈ 0 for w in get_log_weights(state))

    state = pf_initialize(line_model, (10,), line_choicemap(10), 50)
    state = pf_introduce!(state, line_model, (10,), line_choicemap(10), 50)
    @test length(get_traces(state)) == 100
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))

    # Test implicit specification of model and arguments
    state = pf_initialize(line_model, (0,), choicemap(), 50)
    state = pf_introduce!(state, choicemap(), 50)
    @test length(get_traces(state)) == 100
    @test all(get_gen_fn(tr) === line_model for tr in get_traces(state))
    @test all(get_args(tr) == (0,) for tr in get_traces(state))
    @test all(-2 <= tr[:slope] <= 2 for tr in get_traces(state))
    @test all(w ≈ 0 for w in get_log_weights(state))

    state = pf_initialize(line_model, (10,), line_choicemap(10), 50)
    state = pf_introduce!(state, line_choicemap(10), 50)
    @test length(get_traces(state)) == 100
    @test all(get_gen_fn(tr) === line_model for tr in get_traces(state))
    @test all(get_args(tr) == (10,) for tr in get_traces(state))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
end

@testset "Particle introduction (custom proposal)" begin
    @gen line_propose(s) =
        slope ~ uniform_discrete(0, 0)
    @gen outlier_propose(idxs) =
        [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]

    # Test explicit specification of model and arguments
    state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 50)
    state = pf_introduce!(state, line_model, (0,), choicemap(),
                          line_propose, (0,), 50)
    @test all(tr[:slope] == 0 for tr in get_traces(state))
    @test all(w ≈ log(1/5) for w in get_log_weights(state))

    state = pf_initialize(line_model, (1,), line_choicemap(1),
                          outlier_propose, ([1],), 50)
    state = pf_introduce!(state, line_model, (1,), line_choicemap(1),
                          outlier_propose, ([1],), 50)
    @test all(tr[:line => 1 => :outlier] == false for tr in get_traces(state))
    @test all(tr[:line => 1 => :y] == 0 for tr in get_traces(state))

    state = pf_initialize(line_model, (10,), line_choicemap(10),
                          outlier_propose, ([10],), 50)
    state = pf_introduce!(state, line_model, (10,), line_choicemap(10),
                          outlier_propose, ([10],), 50)
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))

    # Test implicit specification of model and arguments
    state = pf_initialize(line_model, (0,), choicemap(), line_propose, (0,), 50)
    state = pf_introduce!(state, choicemap(), line_propose, (0,), 50)
    @test length(get_traces(state)) == 100
    @test all(get_gen_fn(tr) === line_model for tr in get_traces(state))
    @test all(get_args(tr) == (0,) for tr in get_traces(state))
    @test all(tr[:slope] == 0 for tr in get_traces(state))
    @test all(w ≈ log(1/5) for w in get_log_weights(state))

    state = pf_initialize(line_model, (1,), line_choicemap(1),
                          outlier_propose, ([1],), 50)
    state = pf_introduce!(state, line_choicemap(1), outlier_propose, ([1],), 50)
    @test length(get_traces(state)) == 100
    @test all(get_gen_fn(tr) === line_model for tr in get_traces(state))
    @test all(get_args(tr) == (1,) for tr in get_traces(state))
    @test all(tr[:line => 1 => :outlier] == false for tr in get_traces(state))
    @test all(tr[:line => 1 => :y] == 0 for tr in get_traces(state))

    state = pf_initialize(line_model, (10,), line_choicemap(10),
                          outlier_propose, ([10],), 50)
    state = pf_introduce!(state, line_choicemap(10), outlier_propose, ([10],), 50)
    @test length(get_traces(state)) == 100
    @test all(get_gen_fn(tr) === line_model for tr in get_traces(state))
    @test all(get_args(tr) == (10,) for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
end

end