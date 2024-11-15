using Test
using Distributions
using Graphs
using JuliaBUGS.ProbabilisticGraphicalModels:
    BayesianNetwork,
    add_stochastic_vertex!,
    add_deterministic_vertex!,
    add_edge!,
    condition,
    decondition,
    ancestral_sampling,
    is_conditionally_independent,
    variable_elimination
@testset "BayesianNetwork" begin
    @testset "Adding vertices" begin
        bn = BayesianNetwork{Symbol}()

        # Test adding stochastic vertex
        id1 = add_stochastic_vertex!(bn, :A, Normal(0, 1), false)
        @test id1 == 1
        @test length(bn.names) == 1
        @test bn.names[1] == :A
        @test bn.names_to_ids[:A] == 1
        @test bn.is_stochastic[1] == true
        @test bn.is_observed[1] == false
        @test length(bn.stochastic_ids) == 1

        # Test adding deterministic vertex
        f(x) = x^2
        id2 = add_deterministic_vertex!(bn, :B, f)
        @test id2 == 2
        @test length(bn.names) == 2
        @test bn.names[2] == :B
        @test bn.names_to_ids[:B] == 2
        @test bn.is_stochastic[2] == false
        @test bn.is_observed[2] == false
        @test length(bn.deterministic_ids) == 1
    end

    @testset "Adding edges" begin
        bn = BayesianNetwork{Symbol}()
        add_stochastic_vertex!(bn, :A, Normal(0, 1), false)
        add_stochastic_vertex!(bn, :B, Normal(0, 1), false)

        add_edge!(bn, :A, :B)
        @test has_edge(bn.graph, 1, 2)
    end

    @testset "conditioning and deconditioning" begin
        bn = BayesianNetwork{Symbol}()
        # Add some vertices
        add_stochastic_vertex!(bn, :A, Normal(0, 1), false)
        add_stochastic_vertex!(bn, :B, Normal(0, 1), false)
        add_stochastic_vertex!(bn, :C, Normal(0, 1), false)

        # Test conditioning
        bn_cond = condition(bn, Dict(:A => 1.0))
        @test bn_cond.is_observed[1] == true
        @test bn_cond.values[:A] == 1.0
        @test bn_cond.is_observed[2] == false
        @test bn_cond.is_observed[3] == false

        # Ensure original bn is not mutated
        @test bn.is_observed[1] == false
        @test !haskey(bn.values, :A)

        # Test conditioning multiple variables
        bn_cond2 = condition(bn_cond, Dict(:B => 2.0))
        @test bn_cond2.is_observed[1] == true
        @test bn_cond2.is_observed[2] == true
        @test bn_cond2.values[:A] == 1.0
        @test bn_cond2.values[:B] == 2.0

        # Ensure bn_cond is not mutated
        @test bn_cond.is_observed[2] == false
        @test !haskey(bn_cond.values, :B)

        # Test deconditioning
        bn_decond = decondition(bn_cond2, [:A])
        @test bn_decond.is_observed[1] == false
        @test bn_decond.is_observed[2] == true
        @test !haskey(bn_decond.values, :A)
        @test bn_decond.values[:B] == 2.0

        # Ensure bn_cond2 is not mutated
        @test bn_cond2.is_observed[1] == true
        @test bn_cond2.values[:A] == 1.0

        # Test deconditioning all
        bn_decond_all = decondition(bn_cond2)
        @test all(.!bn_decond_all.is_observed)
        @test all(values(bn_decond_all.values) .=== nothing)

        # Ensure bn_cond2 is still not mutated
        @test bn_cond2.is_observed[1] == true
        @test bn_cond2.is_observed[2] == true
        @test bn_cond2.values[:A] == 1.0
        @test bn_cond2.values[:B] == 2.0
    end

    @testset "Simple ancestral sampling" begin
        bn = BayesianNetwork{Symbol}()
        # Add stochastic vertices
        add_stochastic_vertex!(bn, :A, Normal(0, 1), false)
        add_stochastic_vertex!(bn, :B, Normal(1, 2), false)
        # Add deterministic vertex C = A + B
        add_deterministic_vertex!(bn, :C, (a, b) -> a + b)
        add_edge!(bn, :A, :C)
        add_edge!(bn, :B, :C)
        samples = ancestral_sampling(bn)
        @test haskey(samples, :A)
        @test haskey(samples, :B)
        @test haskey(samples, :C)
        @test samples[:A] isa Number
        @test samples[:B] isa Number
        @test samples[:C] ≈ samples[:A] + samples[:B]
    end

    @testset "Complex ancestral sampling" begin
        bn = BayesianNetwork{Symbol}()
        add_stochastic_vertex!(bn, :μ, Normal(0, 2), false)
        add_stochastic_vertex!(bn, :σ, LogNormal(0, 0.5), false)
        add_stochastic_vertex!(bn, :X, Normal(0, 1), false)
        add_stochastic_vertex!(bn, :Y, Normal(0, 1), false)
        add_deterministic_vertex!(bn, :X_scaled, (μ, σ, x) -> x * σ + μ)
        add_deterministic_vertex!(bn, :Y_scaled, (μ, σ, y) -> y * σ + μ)
        add_deterministic_vertex!(bn, :Sum, (x, y) -> x + y)
        add_deterministic_vertex!(bn, :Product, (x, y) -> x * y)
        add_deterministic_vertex!(bn, :N, () -> 2.0)
        add_deterministic_vertex!(bn, :Mean, (s, n) -> s / n)
        add_edge!(bn, :μ, :X_scaled)
        add_edge!(bn, :σ, :X_scaled)
        add_edge!(bn, :X, :X_scaled)
        add_edge!(bn, :μ, :Y_scaled)
        add_edge!(bn, :σ, :Y_scaled)
        add_edge!(bn, :Y, :Y_scaled)
        add_edge!(bn, :X_scaled, :Sum)
        add_edge!(bn, :Y_scaled, :Sum)
        add_edge!(bn, :X_scaled, :Product)
        add_edge!(bn, :Y_scaled, :Product)
        add_edge!(bn, :Sum, :Mean)
        add_edge!(bn, :N, :Mean)
        samples = ancestral_sampling(bn)

        @test all(
            haskey(samples, k) for
            k in [:μ, :σ, :X, :Y, :X_scaled, :Y_scaled, :Sum, :Product, :Mean, :N]
        )

        @test all(samples[k] isa Number for k in keys(samples))
        @test samples[:X_scaled] ≈ samples[:X] * samples[:σ] + samples[:μ]
        @test samples[:Y_scaled] ≈ samples[:Y] * samples[:σ] + samples[:μ]
        @test samples[:Sum] ≈ samples[:X_scaled] + samples[:Y_scaled]
        @test samples[:Product] ≈ samples[:X_scaled] * samples[:Y_scaled]
        @test samples[:Mean] ≈ samples[:Sum] / samples[:N]
        @test samples[:N] ≈ 2.0
        @test samples[:σ] > 0
        # Multiple samples test
        n_samples = 1000
        means = zeros(n_samples)
        for i in 1:n_samples
            samples = ancestral_sampling(bn)
            means[i] = samples[:Mean]
        end

        @test mean(means) ≈ 0 atol = 0.5
        @test std(means) > 0
    end

    @testset "Bayes Ball" begin
        @testset "Chain Structure (A → B → C)" begin
            bn = BayesianNetwork{Symbol}()

            add_stochastic_vertex!(bn, :A, Normal(), false)
            add_stochastic_vertex!(bn, :B, Normal(), false)
            add_stochastic_vertex!(bn, :C, Normal(), false)

            add_edge!(bn, :A, :B)
            add_edge!(bn, :B, :C)

            @test is_conditionally_independent(bn, :A, :C, [:B])
            @test !is_conditionally_independent(bn, :A, :C, Symbol[])
        end

        @testset "Fork Structure (A ← B → C)" begin
            println("\nTesting Fork Structure")
            bn = BayesianNetwork{Symbol}()

            add_stochastic_vertex!(bn, :A, Normal(), false)
            add_stochastic_vertex!(bn, :B, Normal(), false)
            add_stochastic_vertex!(bn, :C, Normal(), false)

            add_edge!(bn, :B, :A)
            add_edge!(bn, :B, :C)

            println("Graph structure:")
            println("Edges: ", collect(edges(bn.graph)))

            result = is_conditionally_independent(bn, :A, :C, Symbol[])
            println("Result for A ⊥ C | ∅: $result")
        end

        @testset "Collider Structure (A → B ← C)" begin
            bn = BayesianNetwork{Symbol}()

            add_stochastic_vertex!(bn, :A, Normal(), false)
            add_stochastic_vertex!(bn, :B, Normal(), false)
            add_stochastic_vertex!(bn, :C, Normal(), false)

            add_edge!(bn, :A, :B)
            add_edge!(bn, :C, :B)

            @test is_conditionally_independent(bn, :A, :C, Symbol[])
            @test !is_conditionally_independent(bn, :A, :C, [:B])
        end

        @testset "Complex Structure" begin
            bn = BayesianNetwork{Symbol}()

            for v in [:A, :B, :C, :D, :E]
                add_stochastic_vertex!(bn, v, Normal(), false)
            end

            # Create structure:
            #     A → B → D
            #         ↓   ↑
            #         C → E
            add_edge!(bn, :A, :B)
            add_edge!(bn, :B, :C)
            add_edge!(bn, :B, :D)
            add_edge!(bn, :C, :E)
            add_edge!(bn, :E, :D)

            @test is_conditionally_independent(bn, :A, :E, [:B, :C])
            @test !is_conditionally_independent(bn, :A, :E, [:B])
            @test !is_conditionally_independent(bn, :A, :E, Symbol[])
        end

        @testset "Using Observed Variables" begin
            bn = BayesianNetwork{Symbol}()

            add_stochastic_vertex!(bn, :A, Normal(), false)
            add_stochastic_vertex!(bn, :B, Normal(), true)  # B is observed
            add_stochastic_vertex!(bn, :C, Normal(), false)

            add_edge!(bn, :A, :B)
            add_edge!(bn, :B, :C)

            @test is_conditionally_independent(bn, :A, :C)

            bn_decond = decondition(bn)
            @test !is_conditionally_independent(bn_decond, :A, :C)
        end

        @testset "Error Handling" begin
            bn = BayesianNetwork{Symbol}()

            add_stochastic_vertex!(bn, :A, Normal(), false)
            add_stochastic_vertex!(bn, :B, Normal(), false)

            @test_throws KeyError is_conditionally_independent(bn, :A, :NonExistent)
            @test_throws KeyError is_conditionally_independent(bn, :NonExistent, :B)
            @test_throws KeyError is_conditionally_independent(bn, :A, :B, [:NonExistent])
        end
    end

    @testset "Variable Elimination Tests" begin
        println("\nTesting Variable Elimination")

        @testset "Simple Chain Network (Z → X → Y)" begin
            # Create a simple chain network: Z → X → Y
            bn = BayesianNetwork{Symbol}()
            
            # Add vertices with specific distributions
            println("Adding vertices...")
            add_stochastic_vertex!(bn, :Z, Categorical([0.7, 0.3]), false)  # P(Z)
            add_stochastic_vertex!(bn, :X, Normal(0, 1), false)             # P(X|Z)
            add_stochastic_vertex!(bn, :Y, Normal(1, 2), false)             # P(Y|X)
            
            # Add edges
            println("Adding edges...")
            add_edge!(bn, :Z, :X)
            add_edge!(bn, :X, :Y)
            
            # Test case 1: P(X | Y=1.5)
            println("\nTest case 1: P(X | Y=1.5)")
            evidence1 = Dict(:Y => 1.5)
            query1 = :X
            result1 = variable_elimination(bn, query1, evidence1)
            @test result1 isa Number
            @test result1 >= 0
            println("P(X | Y=1.5) = ", result1)
            
            # Test case 2: P(X | Z=1)
            println("\nTest case 2: P(X | Z=1)")
            evidence2 = Dict(:Z => 1)
            query2 = :X
            result2 = variable_elimination(bn, query2, evidence2)
            @test result2 isa Number
            @test result2 >= 0
            println("P(X | Z=1) = ", result2)
            
            # Test case 3: P(Y | Z=1)
            println("\nTest case 3: P(Y | Z=1)")
            evidence3 = Dict(:Z => 1)
            query3 = :Y
            result3 = variable_elimination(bn, query3, evidence3)
            @test result3 isa Number
            @test result3 >= 0
            println("P(Y | Z=1) = ", result3)
        end
    end
end
