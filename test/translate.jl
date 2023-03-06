@testset "Trace translators" begin

@gen function model(T::Int)
    for t in 1:T
        x = {(:x, t)} ~ normal(0, 1)
        y = {(:y, t)} ~ normal(x, 1)
    end
end

@testset "ExtendingTraceTranslator" begin
    # Define forward proposal
    @gen function proposal(trace, t::Int)
        x ~ normal(0, 1) # Proposal to next value of x
    end
    # Define trace transform
    @transform transform_f (q_fwd, _) to (p_new, _) begin
        t = get_args(q_fwd)[2]
        x = @read(q_fwd[:x], :continuous)
        @write(p_new[(:x, t)], 2*x, :continuous)
    end
    # Construct trace translator
    translator = ExtendingTraceTranslator(
        p_new_args=(1,),
        new_observations=choicemap(((:y, 1), 0)),
        q_forward=proposal,
        q_forward_args=(1,),
        transform=transform_f
    )
    # Generate initial trace
    trace, _ = generate(model, (0,))
    for _ in 1:10
        # Run trace translator
        new_trace, log_weight = translator(trace; check=true)
        # Manually compute expected weight
        x, y = new_trace[(:x, 1)], new_trace[(:y, 1)]
        obs_weight = logpdf(normal, y, x, 1)
        model_weight = logpdf(normal, x, 0, 1)
        prop_weight = logpdf(normal, x, 0, 2)
        expected = obs_weight + model_weight - prop_weight
        # Check if actual weight is equal to expected weight
        @test log_weight ≈ expected
    end
end

@testset "UpdatingTraceTranslator" begin
    # Define forward and backward kernels
    @gen function fwd_kernel(trace)
        u ~ bernoulli(0.25) # Dummy auxiliary variable
        x ~ normal(0, 1) # Proposal to next value of x
    end
    @gen function bwd_kernel(trace)
        u ~ bernoulli(0.75) # Dummy auxiliary variable
    end
    # Define forward and backward trace transforms
    @transform transform_fwd (p_old, q_fwd) to (p_new, q_bwd) begin
        t = get_args(p_old)[1] + 1
        u = @read(q_fwd[:u], :discrete)
        @write(q_bwd[:u], u, :discrete)
        x = @read(q_fwd[:x], :continuous)
        @write(p_new[(:x, t)], 2*x, :continuous)
    end
    @transform transform_bwd (p_old, q_fwd) to (p_new, q_bwd) begin
        t = get_args(p_old)[1]
        u = @read(q_fwd[:u], :discrete)
        @write(q_bwd[:u], u, :discrete)
        x = @read(p_old[(:x, t)], :continuous)
        @write(q_bwd[:x], 0.5*x, :continuous)
    end
    pair_bijections!(transform_fwd, transform_bwd)
    # Construct trace translator
    translator = UpdatingTraceTranslator(
        p_new_args=(1,),
        new_observations=choicemap(((:y, 1), 0)),
        q_forward=fwd_kernel,
        q_backward=bwd_kernel,
        transform=transform_fwd
    )
    # Generate initial trace
    trace, _ = generate(model, (0,))
    for _ in 1:10
        # Run trace translator
        new_trace, log_weight = translator(trace; check=true)
        # Manually compute possible expected weights
        x, y = new_trace[(:x, 1)], new_trace[(:y, 1)]
        obs_weight = logpdf(normal, y, x, 1)
        model_weight = logpdf(normal, x, 0, 1)
        prop_weight = logpdf(normal, x, 0, 2)
        aux_weight_1 = log(0.25) - log(0.75)
        aux_weight_2 = log(0.75) - log(0.25)
        expected_1 = obs_weight + model_weight - prop_weight + aux_weight_1
        expected_2 = obs_weight + model_weight - prop_weight + aux_weight_2
        # Check if actual weight is equal to one of the expected weights
        @test log_weight ≈ expected_1 || log_weight ≈ expected_2
    end
end

end