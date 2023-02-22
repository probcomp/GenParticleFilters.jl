@testset "Statistics functions" begin
    @gen function model()
        x ~ uniform_discrete(1, 1)
        y ~ uniform_discrete(2, 2)
        return x + y
    end
    
    state = pf_initialize(model, (), choicemap(), 100)
    
    @test mean((x, y) -> x^2 + y^2, state, :x, :y) ≈ 5
    @test mean(x -> x * 2, state, :x) ≈ 2
    @test mean(state, :x) ≈ 1
    @test mean(state) ≈ 3
    
    @test var((x, y) -> x^2 + y^2, state, :x, :y) ≈ 0    atol=1e-6
    @test var(x -> x * 2, state, :x) ≈ 0                 atol=1e-6
    @test var(state, :x) ≈ 0                             atol=1e-6
    @test var(state) ≈ 0                                 atol=1e-6
    
    ps = proportionmap((x, y) -> x^2 + y^2, state, :x, :y)
    @test ps[5] ≈ 1 && length(ps) == 1
    ps = proportionmap(x -> x * 2, state, :x)
    @test ps[2] ≈ 1 && length(ps) == 1
    ps = proportionmap(state, :x)
    @test ps[1] ≈ 1 && length(ps) == 1
    ps = proportionmap(state)
    @test ps[3] ≈ 1 && length(ps) == 1 
end