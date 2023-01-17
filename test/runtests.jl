using GenParticleFilters, Gen, Test, Logging

@gen (static) function line_step(t::Int, x::Float64, slope::Float64)
    x = x + 1
    outlier ~ bernoulli(0.1)
    y ~ normal(x * slope, outlier ? 10.0 : 1.0)
    return x
end

line_unfold = Unfold(line_step)

@gen (static) function line_model(n::Int)
    slope ~ uniform_discrete(-2, 2)
    line ~ line_unfold(n, 0, slope)
    return line
end

@load_generated_functions()

generate_line(n::Int, slope::Float64=0.) =
    choicemap([(:line => i => :y, i*slope) for i in 1:n]...)

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

@testset "Particle update" begin

@testset "Update with default proposal" begin
state = pf_initialize(line_model, (0,), choicemap(), 100)
state = pf_update!(state, (10,), (UnknownChange(),), generate_line(10))
@test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
@test all(w != 0 for w in get_log_weights(state))
end

@gen outlier_propose(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.0) for i in idxs]
@gen outlier_reverse(tr, idxs) =
    [{:line => i => :outlier} ~ bernoulli(0.1) for i in idxs]

@testset "Update with custom proposal" begin
state = pf_initialize(line_model, (0,), choicemap(), 100)
state = pf_update!(state, (10,), (UnknownChange(),), generate_line(10),
                   outlier_propose, (1:10,))
@test all(tr[:line => 10 => :y] == 0 for tr in get_traces(state))
@test all(tr[:line => 10 => :outlier] == false for tr in get_traces(state))
@test all(w != 0 for w in get_log_weights(state))
end

@testset "Update with custom forward and backward proposals" begin
state = pf_initialize(line_model, (10,), generate_line(10), 100)
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
state = pf_update!(state, (10,), (UnknownChange(),), generate_line(10),
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
state = pf_initialize(line_model, (5,), generate_line(5), 100)
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
substate = pf_update!(state[1:50], (10,), (UnknownChange(),), generate_line(10))
@test all(tr[:line => 10 => :y] == 0 for tr in get_traces(substate))
@test all(w != 0 for w in get_log_weights(substate))
substate = pf_update!(state[51:end], (10,), (UnknownChange(),),
                      generate_line(10), outlier_propose, (10,))
@test all(tr[:line => 10 => :y] == 0 for tr in get_traces(substate))
@test all(tr[:line => 10 => :outlier] == false for tr in get_traces(substate))
@test all(w != 0 for w in get_log_weights(state))
end

end

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

@testset "Utility functions" begin
@gen model() = x ~ normal(0, 0)
state = pf_initialize(model, (), choicemap(), 100)

@test mean(state, :x) == 0
@test mean(state) == 0
@test var(state, :x) == 0
@test var(state) == 0

weights = get_norm_weights(state)
@test sum(weights) ≈ 1.0
ess = get_ess(state)
@test ess ≈ sum(weights)^2 / sum(weights .^ 2)
end
