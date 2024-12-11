# 1. First, activate the package environment
using Pkg
Pkg.activate(".")

# 2. Load required packages
using Test
using Distributions
using Graphs
using BangBang

# 3. Load JuliaBUGS and its submodule
using JuliaBUGS
using JuliaBUGS.ProbabilisticGraphicalModels
using JuliaBUGS.ProbabilisticGraphicalModels:
    BayesianNetwork,
    add_stochastic_vertex!,
    add_edge!,
    condition,
    marginal_distribution,
    eliminate_variables

@testset "Mixed Graph - Variable Elimination" begin
    bn = BayesianNetwork{Symbol}()
    
    # X1 ~ Uniform(0,1)
    add_stochastic_vertex!(bn, :X1, Uniform(0, 1), false)
    
    # X2 ~ Bernoulli(X1)
    function x2_distribution(x1)
        return Bernoulli(x1)
    end
    add_stochastic_vertex!(bn, :X2, x2_distribution, false)
    add_edge!(bn, :X1, :X2)
    
    # X3 ~ Normal(μ(X2), 1)
    function x3_distribution(x2)
        return Normal(x2 == 1 ? 10.0 : 2.0, 1.0)
    end
    add_stochastic_vertex!(bn, :X3, x3_distribution, false)
    add_edge!(bn, :X2, :X3)
    
    # Test graph structure
    @test has_edge(bn.graph, 1, 2)  # X1 -> X2
    @test has_edge(bn.graph, 2, 3)  # X2 -> X3
    
    # Test conditional distributions
    # Test X2's distribution given X1
    bn_cond_x1 = condition(bn, Dict(:X1 => 0.7))
    marginal_x2 = marginal_distribution(bn_cond_x1, :X2)
    @test marginal_x2 isa Bernoulli
    @test mean(marginal_x2) ≈ 0.7
    
    # Test X3's distribution given X2
    bn_cond_x2_0 = condition(bn, Dict(:X2 => 0))
    marginal_x3_0 = marginal_distribution(bn_cond_x2_0, :X3)
    @test marginal_x3_0 isa Normal
    @test mean(marginal_x3_0) ≈ 2.0
    @test std(marginal_x3_0) ≈ 1.0
    
    bn_cond_x2_1 = condition(bn, Dict(:X2 => 1))
    marginal_x3_1 = marginal_distribution(bn_cond_x2_1, :X3)
    @test marginal_x3_1 isa Normal
    @test mean(marginal_x3_1) ≈ 10.0
    @test std(marginal_x3_1) ≈ 1.0
    
    # Test full chain inference
    ordered_vertices = [1, 2]  # Eliminate X1, then X2
    query_id = 3              # Query X3
    result = eliminate_variables(bn, ordered_vertices, query_id, Dict{Symbol,Any}())
    
    # The result should be a mixture of Normal distributions
    @test result isa MixtureModel
end

@testset "Marginal Distribution P(X3|X1)" begin
    bn = BayesianNetwork{Symbol}()
    
    # X1 ~ Uniform(0,1)
    add_stochastic_vertex!(bn, :X1, Uniform(0, 1), false)
    
    # X2 ~ Bernoulli(X1)
    add_stochastic_vertex!(bn, :X2, x1 -> Bernoulli(x1), false)
    add_edge!(bn, :X1, :X2)
    
    # X3 ~ Normal(μ(X2), 1)
    add_stochastic_vertex!(bn, :X3, x2 -> Normal(x2 == 1 ? 10.0 : 2.0, 1.0), false)
    add_edge!(bn, :X2, :X3)
    
    # Test P(X3|X1=0.7)
    bn_cond = condition(bn, Dict(:X1 => 0.7))
    marginal_x3 = marginal_distribution(bn_cond, :X3)
    
    @test marginal_x3 isa MixtureModel
    @test length(marginal_x3.components) == 2
    @test marginal_x3.components[1] isa Normal
    @test marginal_x3.components[2] isa Normal
    
    # When X1 = 0.7:
    # P(X2=0) = 0.3, P(X2=1) = 0.7
    @test marginal_x3.prior.p ≈ [0.3, 0.7]
    
    # Component means should be 2 and 10
    @test mean(marginal_x3.components[1]) ≈ 2.0
    @test mean(marginal_x3.components[2]) ≈ 10.0
    
    # Overall mean should be weighted average
    @test mean(marginal_x3) ≈ 2.0 * 0.3 + 10.0 * 0.7
end