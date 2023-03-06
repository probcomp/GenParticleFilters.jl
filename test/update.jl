@testset "Particle update" begin

@testset "Update with default proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), line_choicemap(10))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with stratification" begin
    outlier_strata = [outlier_choicemap(1, false), outlier_choicemap(1, true)]
    observations = line_choicemap(1)
    # Test contiguous stratification
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (1,), (UnknownChange(),), observations,
                       outlier_strata; layout=:contiguous)
    for (k, val) in zip([50, 100], [false, true])
        traces = get_traces(state[(k-50+1):k])
        @test all(tr[:line => 1 => :outlier] == val for tr in traces)
    end
    @test all(w != 0 for w in get_log_weights(state))
    # Test interleaved stratification
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (1,), (UnknownChange(),), observations,
                       outlier_strata; layout=:interleaved)
    for (k, val) in zip(1:2, [false, true])
        traces = get_traces(state[k:2:100])
        @test all(tr[:line => 1 => :outlier] == val for tr in traces)
    end
    @test all(w != 0 for w in get_log_weights(state))
end

@gen outlier_propose(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]
@gen outlier_reverse(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.1) for i in idxs]

@testset "Update with custom proposal" begin
    # Test standard version
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), line_choicemap(10),
                       outlier_propose, (1:10,))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
    # Test stratified version
    outlier_strata = [outlier_choicemap(1, false), outlier_choicemap(1, true)]
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (2,), (UnknownChange(),), line_choicemap(2),
                       outlier_strata, outlier_propose, ([2],))
    for (k, val) in zip(1:2, [false, true])
        traces = get_traces(state[k:2:100]) # Interleaved indices by default
        @test all(tr[:line => 1 => :outlier] == val for tr in traces)
        @test all(tr[:line => 2 => :y] == 0 for tr in traces)
        @test all(tr[:line => 2 => :outlier] == false for tr in traces)
    end
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom forward and backward proposals" begin
    # Test standard version
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), choicemap(),
                       outlier_propose, (1:10,), outlier_reverse, (1:10,))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
    # Test stratified version
    outlier_strata = [outlier_choicemap(1, false), outlier_choicemap(1, true)]
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (2,), (UnknownChange(),), line_choicemap(2),
                       outlier_strata, outlier_propose, ([2],),
                       outlier_reverse, ([2],))
    for (k, val) in zip(1:2, [false, true])
        traces = get_traces(state[k:2:100]) # Interleaved indices by default
        @test all(tr[:line => 1 => :outlier] == val for tr in traces)
        @test all(tr[:line => 2 => :y] == 0 for tr in traces)
        @test all(tr[:line => 2 => :outlier] == false for tr in traces)
    end
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom proposal and trace transform" begin
    # Define proposal and trace transform
    @gen proposal(tr, idxs) = [{(:outlier, i)} ~ bernoulli(0.0) for i in idxs]
    @transform remap (p_old, _) to (p_new, _) begin
        idxs = get_args(p_old)[2]
        for i in idxs
            @copy(p_old[(:outlier, i)], p_new[:line => i => :outlier])
        end
    end
    # Test standard version
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), line_choicemap(10),
                       proposal, (1:10,), remap)
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
    # Test stratified version
    outlier_strata = [outlier_choicemap(1, false), outlier_choicemap(1, true)]
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (2,), (UnknownChange(),), line_choicemap(2),
                       outlier_strata, proposal, ([2],), remap)
    for (k, val) in zip(1:2, [false, true])
        traces = get_traces(state[k:2:100]) # Interleaved indices by default
        @test all(tr[:line => 1 => :outlier] == val for tr in traces)
        @test all(tr[:line => 2 => :y] == 0 for tr in traces)
        @test all(tr[:line => 2 => :outlier] == false for tr in traces)
    end
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom bidirectional proposals and trace transform" begin
    # Define forward and backward kernels
    @gen function fwd_kernel(tr, idxs)
        if (flip ~ bernoulli(0.5))
            [{(:outlier, i)} ~ bernoulli(0.0) for i in idxs]
        else
            slope ~ uniform_discrete(0, 0)
        end
    end
    @gen function bwd_kernel(tr, idxs)
        if (flip ~ bernoulli(0.5))
            [{(:outlier, i)} ~ bernoulli(0.1) for i in idxs]
        else
            slope ~ uniform_discrete(-2, 2)
        end
    end
    # Define trace transform
    @transform line_transform (p_old, q_fwd) to (p_new, q_bwd) begin
        flip = @read(q_fwd[:flip], :discrete)
        @write(q_bwd[:flip], flip, :discrete)
        if flip
            idxs = get_args(q_fwd)[2]
            for i in idxs
                @copy(q_fwd[(:outlier, i)], p_new[:line => i => :outlier])
                @copy(p_old[:line => i => :outlier], q_bwd[(:outlier, i)])
            end
        else
            @copy(q_fwd[:slope], p_new[:slope])
            @copy(p_old[:slope], q_bwd[:slope])
        end
    end
    is_involution!(line_transform)
    # Test standard version
    state = pf_initialize(line_model, (5,), line_choicemap(5), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), choicemap(),
                       fwd_kernel, (1:10,), bwd_kernel, (1:5,),
                       line_transform; check=true)
    @test all(tr[:line => 5 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 5 => :outlier] == false || tr[:slope] == 0
              for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
    # Test stratified version
    outlier_strata = [outlier_choicemap(10, false), outlier_choicemap(10, true)]
    state = pf_initialize(line_model, (5,), line_choicemap(5), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), choicemap(),
                       outlier_strata, fwd_kernel, (1:9,), bwd_kernel, (1:5,),
                       line_transform; check=true)
    for (k, val) in zip(1:2, [false, true])
        traces = get_traces(state[k:2:100]) # Interleaved indices by default
        @test all(tr[:line => 10 => :outlier] == val for tr in traces)
        @test all(tr[:line => 5 => :y] == 0 for tr in traces)
        @test all(tr[:line => 5 => :outlier] == false || tr[:slope] == 0
                  for tr in get_traces(state))
    end
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with different proposals per view" begin
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    substate = pf_update!(state[1:50], (10,), (UnknownChange(),), line_choicemap(10))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(substate))
    @test all(w != 0 for w in get_log_weights(substate))
    substate = pf_update!(state[51:end], (10,), (UnknownChange(),),
                          line_choicemap(10), outlier_propose, (10,))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(substate))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(substate))
    @test all(w != 0 for w in get_log_weights(state))
end
    
end