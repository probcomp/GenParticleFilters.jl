@testset "Particle rejuvenation" begin

@testset "Move-reweight kernel" begin
    out_addr = :line => 1 => :outlier
    observations = choicemap((:line => 1 => :y, 0))
    trace, _ = generate(line_model, (1,), observations)
    slope, outlier = trace[:slope], trace[out_addr]

    # Test selection variant
    expected_w = (out_old, out_new, slope) -> begin
        logpdf(normal, 0, slope, out_new ? 10. : 1.) -
        logpdf(normal, 0, slope, out_old ? 10. : 1.)
    end
    trs_ws = [move_reweight(trace, select(out_addr)) for i in 1:100]
    @test all(w ≈ expected_w(outlier, tr[out_addr], slope) for (tr, w) in trs_ws)

    # Test proposal variant
    expected_w = (out_old, out_new, slope) -> begin 
        logpdf(bernoulli, out_new, 0.1) - logpdf(bernoulli, out_old, 0.1) +
        logpdf(normal, 0, slope, out_new ? 10. : 1.) -
        logpdf(normal, 0, slope, out_old ? 10. : 1.) +
        (out_old == out_new ? 0.0 :
            logpdf(bernoulli, out_old, 0.9) - logpdf(bernoulli, out_old, 0.1))
    end
    @gen outlier_propose(tr, idx) = {:line => idx => :outlier} ~ bernoulli(0.9)
    trs_ws = [move_reweight(trace, outlier_propose, (1,)) for i in 1:100]
    @test all(w ≈ expected_w(outlier, tr[out_addr], slope) for (tr, w) in trs_ws)
end

@testset "Move-accept rejuvenation" begin
    # Log which particles were rejuvenated
    buffer = IOBuffer()
    logger = SimpleLogger(buffer, Logging.Debug)
    state = pf_initialize(line_model, (10,), generate_line(10, 1.), 100)
    old_traces = get_traces(state)
    with_logger(logger) do
        pf_move_accept!(state, mh, (select(:slope),), 1; check=false)
    end

    # Extract acceptances from debug log
    lines = split(String(take!(buffer)), "\n")
    lines = filter(s -> occursin("Accepted: ", s), lines)
    accepts = [match(r".*Accepted: (\w+).*", l).captures[1] for l in lines]
    accepts = parse.(Bool, accepts)

    # Check that only traces that were accepted are rejuvenated
    new_traces = get_traces(state)
    @test all(a ? t1 !== t2 : t1 === t2
            for (a, t1, t2) in zip(accepts, old_traces, new_traces))
end

@testset "Move-reweight rejuvenation" begin
    # Log which particles were rejuvenated
    buffer = IOBuffer()
    logger = SimpleLogger(buffer, Logging.Debug)
    state = pf_initialize(line_model, (10,), generate_line(10, 1.), 100)
    old_weights = copy(get_log_weights(state))
    with_logger(logger) do
        pf_move_reweight!(state, move_reweight, (select(:slope),), 1; check=false)
    end
    new_weights = copy(get_log_weights(state))

    # Extract relative weights from debug log
    lines = split(String(take!(buffer)), "\n")
    lines = filter(s -> occursin("Rel. Weight: ", s), lines)
    rel_weights = [match(r".*Rel\. Weight: (.+)\s*", l).captures[1] for l in lines]
    rel_weights = parse.(Float64, rel_weights)

    # Check that weights are adjusted accordingly
    @test all(isapprox.(new_weights, old_weights .+ rel_weights; atol=1e-3))
end

@testset "Rejuvenation on separate views" begin
    # Log which particles were rejuvenated
    buffer = IOBuffer()
    logger = SimpleLogger(buffer, Logging.Debug)
    state = pf_initialize(line_model, (10,), generate_line(10, 1.), 100)
    old_traces = get_traces(state)[1:50]
    old_weights = get_log_weights(state)[51:end]

    kern_args = (select(:slope),)
    with_logger(logger) do
        pf_rejuvenate!(state[1:50], mh, kern_args, 1; method=:move)
        pf_rejuvenate!(state[51:end], move_reweight, kern_args, 1; method=:reweight)
    end

    # Extract acceptances and relative weights from debug log
    lines = split(String(take!(buffer)), "\n")
    a_lines = filter(s -> occursin("Accepted: ", s), lines)
    accepts = [match(r".*Accepted: (\w+).*", l).captures[1] for l in a_lines]
    accepts = parse.(Bool, accepts)
    r_lines = filter(s -> occursin("Rel. Weight: ", s), lines)
    rel_weights = [match(r".*Rel\. Weight: (.+)\s*", l).captures[1] for l in r_lines]
    rel_weights = parse.(Float64, rel_weights)

    # Check that only traces that were accepted are rejuvenated
    new_traces = get_traces(state)[1:50]
    @test all(a ? t1 !== t2 : t1 === t2
            for (a, t1, t2) in zip(accepts, old_traces, new_traces))
    # Check that weights are adjusted accordingly
    new_weights = get_log_weights(state)[51:end]
    @test all(isapprox.(new_weights, old_weights .+ rel_weights; atol=1e-3))
end

end