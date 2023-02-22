@testset "Utility functions" begin
    @gen model() = x ~ normal(0, 0)
    state = pf_initialize(model, (), choicemap(), 100)
    
    log_weights = get_log_norm_weights(state)
    @test sum(exp.(log_weights)) ≈ 1.0
    weights = get_norm_weights(state)
    @test sum(weights) ≈ 1.0
    ess = get_ess(state)
    @test ess ≈ sum(weights)^2 / sum(weights .^ 2)
end