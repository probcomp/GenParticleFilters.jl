@testset "Particle resampling" begin

@testset "Multinomial resampling" begin
    # Test that new traces have the correct parents
    state = pf_initialize(line_model, (10,), generate_line(10), 100)
    old_traces = get_traces(state)
    state = pf_multinomial_resample!(state)
    new_traces = get_traces(state)
    @test new_traces == old_traces[state.parents]
end

@testset "Residual resampling" begin
    # Test that no resampling occurs if all weights are equal
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    old_traces = get_traces(state)
    state = pf_residual_resample!(state)
    new_traces = get_traces(state)
    @test new_traces == old_traces

    # Test that at least the minimum number of copies are resampled
    state = pf_initialize(line_model, (10,), generate_line(10), 100)
    old_traces = get_traces(state)
    weights = get_norm_weights(state)
    min_copies = floor.(Int, weights * 100)
    state = pf_residual_resample!(state)
    new_traces = get_traces(state)
    copies = [sum([t1 == t2 for t1 in new_traces]) for t2 in old_traces]
    @test new_traces == old_traces[state.parents]
    @test all(copies .>= min_copies)

    # Same test but with a custom priority function
    p_fn = w -> w / 2
    state = pf_initialize(line_model, (10,), generate_line(10), 100)
    old_traces = get_traces(state)
    log_priorities = p_fn.(get_log_weights(state))
    weights = exp.(log_priorities .- logsumexp(log_priorities))
    min_copies = floor.(Int, weights * 100)
    state = pf_residual_resample!(state; priority_fn=p_fn)
    new_traces = get_traces(state)
    copies = [sum([t1 == t2 for t1 in new_traces]) for t2 in old_traces]
    @test new_traces == old_traces[state.parents]
    @test all(copies .>= min_copies)
end

@testset "Stratified resampling" begin
    # Test that no resampling occurs if all weights are equal
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    old_traces = get_traces(state)
    state = pf_stratified_resample!(state)
    new_traces = get_traces(state)
    @test new_traces == old_traces

    # Test that the highest weight particle has the right number of copies
    state = pf_initialize(line_model, (10,), generate_line(10), 100)
    old_traces = get_traces(state)
    weights = get_norm_weights(state)
    max_weight, max_idx = findmax(weights)
    min_copies = floor.(Int, max_weight * 100)
    state = pf_stratified_resample!(state; sort_particles=true)
    new_traces = get_traces(state)
    copies = sum([tr == old_traces[max_idx] for tr in new_traces])
    @test new_traces == old_traces[state.parents]
    @test copies >= min_copies

    # Same test but with a custom priority function
    p_fn = w -> w / 2
    state = pf_initialize(line_model, (10,), generate_line(10), 100)
    old_traces = get_traces(state)
    log_priorities = p_fn.(get_log_weights(state))
    weights = exp.(log_priorities .- logsumexp(log_priorities))
    max_weight, max_idx = findmax(weights)
    min_copies = floor.(Int, max_weight * 100)
    state = pf_stratified_resample!(state; sort_particles=true, priority_fn=p_fn)
    new_traces = get_traces(state)
    copies = sum([tr == old_traces[max_idx] for tr in new_traces])
    @test new_traces == old_traces[state.parents]
    @test copies >= min_copies
end
    
end