@testset "Particle update" begin

@testset "Update with default proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), line_choicemap(10))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
end

@gen outlier_propose(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]
@gen outlier_reverse(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.1) for i in idxs]

@testset "Update with custom proposal" begin
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), line_choicemap(10),
                       outlier_propose, (1:10,))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom forward and backward proposals" begin
    state = pf_initialize(line_model, (10,), line_choicemap(10), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), choicemap(),
                       outlier_propose, (1:10,), outlier_reverse, (1:10,))
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom proposal and trace transform" begin
    @gen proposal(tr, idxs) = [{(:outlier, i)} ~ bernoulli(0.0) for i in idxs]
    @transform remap (p_old, _) to (p_new, _) begin
        idxs = get_args(p_old)[2]
        for i in idxs
            @copy(p_old[(:outlier, i)], p_new[:line => i => :outlier])
        end
    end
    state = pf_initialize(line_model, (0,), choicemap(), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), line_choicemap(10),
                        proposal, (1:10,), remap)
    @test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
    @test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom bidirectional proposals and trace transform" begin
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
    state = pf_initialize(line_model, (5,), line_choicemap(5), 100)
    state = pf_update!(state, (10,), (UnknownChange(),), choicemap(),
                       fwd_kernel, (1:10,), bwd_kernel, (1:5,),
                       line_transform, true)
    @test all(tr[:line => 5 => :y] == 0 for tr in get_traces(state))
    @test all(tr[:line => 5 => :outlier] == false || tr[:slope] == 0
            for tr in get_traces(state))
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