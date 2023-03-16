@testset "Particle resampling" begin

@testset "Multinomial resampling" begin
    # Test multinomial resampling of all particles
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    old_traces = get_traces(state)
    old_lml_est = logsumexp(get_log_weights(state)) - log(100)
    state = pf_multinomial_resample!(state)
    new_traces = get_traces(state)
    new_lml_est = get_lml_est(state)
    @test new_traces == old_traces[state.parents]
    @test new_lml_est ≈ old_lml_est

    # Same as above, with a custom priority function
    p_fn = w -> w / 2
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    old_traces = get_traces(state)
    old_lml_est = logsumexp(get_log_weights(state)) - log(100)
    state = pf_multinomial_resample!(state; priority_fn=p_fn)
    new_traces = get_traces(state)
    new_lml_est = get_lml_est(state)
    @test new_traces == old_traces[state.parents]
    @test new_lml_est ≈ old_lml_est
end

@testset "Residual resampling" begin
    # Test that no resampling occurs if all weights are equal
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    old_traces = get_traces(state)
    state = pf_residual_resample!(state)
    new_traces = get_traces(state)
    @test new_traces == old_traces

    # Test that at least the minimum number of copies are resampled
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    old_traces = get_traces(state)
    old_lml_est = logsumexp(get_log_weights(state)) - log(100)
    weights = get_norm_weights(state)
    min_copies = floor.(Int, weights * 100)
    state = pf_residual_resample!(state)
    new_traces = get_traces(state)
    copies = [sum([t1 == t2 for t1 in new_traces]) for t2 in old_traces]
    @test new_traces == old_traces[state.parents]
    @test all(copies .>= min_copies)
    new_lml_est = get_lml_est(state)
    @test new_lml_est ≈ old_lml_est

    # Same test but with a custom priority function
    p_fn = w -> w / 2
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    old_traces = get_traces(state)
    old_lml_est = logsumexp(get_log_weights(state)) - log(100)
    log_priorities = p_fn.(get_log_weights(state))
    weights = exp.(log_priorities .- logsumexp(log_priorities))
    min_copies = floor.(Int, weights * 100)
    state = pf_residual_resample!(state; priority_fn=p_fn)
    new_traces = get_traces(state)
    copies = [sum([t1 == t2 for t1 in new_traces]) for t2 in old_traces]
    @test new_traces == old_traces[state.parents]
    @test all(copies .>= min_copies)
    new_lml_est = get_lml_est(state)
    @test new_lml_est ≈ old_lml_est
end

@testset "Stratified resampling" begin
    # Test that no resampling occurs if all weights are equal
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    old_traces = get_traces(state)
    state = pf_stratified_resample!(state)
    new_traces = get_traces(state)
    @test new_traces == old_traces

    # Test that the highest weight particle has the right number of copies
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    old_traces = get_traces(state)
    old_lml_est = logsumexp(get_log_weights(state)) - log(100)
    weights = get_norm_weights(state)
    max_weight, max_idx = findmax(weights)
    min_copies = floor.(Int, max_weight * 100)
    state = pf_stratified_resample!(state; sort_particles=true)
    new_traces = get_traces(state)
    copies = sum([tr == old_traces[max_idx] for tr in new_traces])
    @test new_traces == old_traces[state.parents]
    @test copies >= min_copies
    new_lml_est = get_lml_est(state)
    @test new_lml_est ≈ old_lml_est

    # Same test but with a custom priority function
    p_fn = w -> w / 2
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    old_traces = get_traces(state)
    old_lml_est = logsumexp(get_log_weights(state)) - log(100)
    log_priorities = p_fn.(get_log_weights(state))
    weights = exp.(log_priorities .- logsumexp(log_priorities))
    max_weight, max_idx = findmax(weights)
    min_copies = floor.(Int, max_weight * 100)
    state = pf_stratified_resample!(state; sort_particles=true, priority_fn=p_fn)
    new_traces = get_traces(state)
    copies = sum([tr == old_traces[max_idx] for tr in new_traces])
    @test new_traces == old_traces[state.parents]
    @test copies >= min_copies
    new_lml_est = get_lml_est(state)
    @test new_lml_est ≈ old_lml_est
end

@testset "Blockwise resampling of separate views" begin
    # Iterate over methods and priority functions
    methods = [:multinomial, :residual, :stratified]
    priority_fns = [nothing, w -> w / 2]
    
    for method in methods, p_fn in priority_fns
        state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
        old_traces_full = copy(get_traces(state))
        old_lml_est_full = logsumexp(get_log_weights(state)) - log(100)
        parents_full = Int[]

        # Resample each block / view independently
        block_size = 50
        for idx in 1:block_size:100
            block_idxs = idx:(idx+block_size-1)
            substate = state[block_idxs]
            old_traces = copy(get_traces(substate))
            old_lml_est = get_lml_est(substate)
            substate = pf_resample!(substate, method; priority_fn=p_fn)
            new_traces = get_traces(substate)
            new_lml_est = get_lml_est(substate)
            @test new_traces == old_traces[substate.parents]
            @test new_lml_est ≈ old_lml_est
            append!(parents_full, block_idxs[substate.parents])
        end

        # Test traces and log ML estimate after resampling blocks
        new_traces_full = get_traces(state)
        @test new_traces_full == old_traces_full[parents_full]
        new_lml_est_full = get_lml_est(state)
        @test new_lml_est_full ≈ old_lml_est_full
    end
end

end