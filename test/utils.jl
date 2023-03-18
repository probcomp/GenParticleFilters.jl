@testset "Utility functions" begin
    @gen model() = x ~ normal(0, 0)
    state = pf_initialize(model, (), choicemap(), 100)
    
    log_weights = get_log_norm_weights(state)
    @test sum(exp.(log_weights)) ≈ 1.0
    weights = get_norm_weights(state)
    @test sum(weights) ≈ 1.0
    ess = get_ess(state)
    @test ess ≈ sum(weights)^2 / sum(weights .^ 2)

    strata = choiceproduct((:a, [1, 2])) |> collect
    @test choicemap((:a , 1)) in strata
    @test choicemap((:a , 2)) in strata    

    strata = choiceproduct((:a, [1, 2]), (:b, [3])) |> collect
    @test choicemap((:a , 1), (:b, 3)) in strata
    @test choicemap((:a , 2), (:b, 3)) in strata    

    strata = choiceproduct(Dict(:a => [1, 2], :b => [3])) |> collect
    @test choicemap((:a , 1), (:b, 3)) in strata
    @test choicemap((:a , 2), (:b, 3)) in strata    
end