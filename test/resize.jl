@testset "Particle filter resizing" begin

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

end