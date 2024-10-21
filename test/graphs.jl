using Graphs, MetaGraphsNext
using JuliaBUGS
using JuliaBUGS:
    markov_blanket, dfs_find_stochastic_boundary_and_deterministic_variables_en_route

module GraphsTest
using JuliaBUGS: JuliaBUGS
using Graphs, MetaGraphsNext

export TestNode

struct TestNode
    node_type::Int
end

# overload the functions for testing purposes
function JuliaBUGS.is_model_parameter(
    g::MetaGraph{Int,<:SimpleDiGraph,Int,TestNode}, v::Int
)
    return g[v].node_type == 1
end
function JuliaBUGS.is_observation(g::MetaGraph{Int,<:SimpleDiGraph,Int,TestNode}, v::Int)
    return g[v].node_type == 2
end
function JuliaBUGS.is_deterministic(g::MetaGraph{Int,<:SimpleDiGraph,Int,TestNode}, v::Int)
    return g[v].node_type == 3
end
function JuliaBUGS.is_stochastic(g::MetaGraph{Int,<:SimpleDiGraph,Int,TestNode}, v::Int)
    return g[v].node_type != 3
end
end

@testset "find_generated_quantities_variables" begin
    using .GraphsTest

    function generate_random_dag(num_nodes::Int, p::Float64=0.3)
        graph = SimpleGraph(num_nodes)
        for i in 1:num_nodes
            for j in 1:num_nodes
                if i != j && rand() < p
                    add_edge!(graph, i, j)
                end
            end
        end

        graph = Graphs.random_orientation_dag(graph) # ensure the random graph is a DAG
        vertices_description = [i => TestNode(rand(1:3)) for i in 1:nv(graph)]
        edges_description = [Tuple(e) => nothing for e in Graphs.edges(graph)]
        return MetaGraph(graph, vertices_description, edges_description)
    end

    # `transitiveclosure` has time complexity O(|E|⋅|V|), not fit for large graphs
    # but easy to implement and understand, here we use it for reference
    function find_generated_quantities_variables_with_transitive_closure(
        g::MetaGraph{Int,<:SimpleDiGraph,Label,VertexData}
    ) where {Label,VertexData}
        _transitive_closure = Graphs.transitiveclosure(g.graph)
        generated_quantities_variables = Set{Label}()
        for v_id in vertices(g.graph)
            if !JuliaBUGS.is_observation(g, v_id)
                if all(
                    !Base.Fix1(JuliaBUGS.is_observation, g),
                    outneighbors(_transitive_closure, v_id),
                )
                    push!(generated_quantities_variables, MetaGraphsNext.label_for(g, v_id))
                end
            end
        end

        return generated_quantities_variables
    end

    @testset "random DAG with $num_nodes nodes and $p probability of edge" for num_nodes in
                                                                               [
            10, 20, 100, 500, 1000
        ],
        p in [0.1, 0.3, 0.5]

        g = generate_random_dag(num_nodes, p)
        @test find_generated_quantities_variables(g) ==
            find_generated_quantities_variables_with_transitive_closure(g)
    end
end

@testset "markov_blanket" begin
    using .GraphsTest

    """ Mermaid code for visualizing the test graph
    ```mermaid
    graph TD
        1((1: Parameter)) --> 2((2: Deterministic))
        1 --> 3((3: Parameter))
        2 --> 4((4: Deterministic))
        3 --> 5((5: Observation))
        4 --> 6((6: Deterministic))
        5 --> 6
        5 --> 7((7: Observation))
        6 --> 8((8: Parameter))
        7 --> 8

        classDef parameter fill:#f9f,stroke:#333,stroke-width:2px;
        classDef deterministic fill:#bfb,stroke:#333,stroke-width:2px;
        classDef observation fill:#bbf,stroke:#333,stroke-width:2px;

        class 1,3,8 parameter;
        class 2,4,6 deterministic;
        class 5,7 observation;
    ```
    """

    g = MetaGraph(SimpleDiGraph(); label_type=Int, vertex_data_type=TestNode)

    g[1] = TestNode(1)
    g[2] = TestNode(3)
    g[3] = TestNode(1)
    g[4] = TestNode(3)
    g[5] = TestNode(2)
    g[6] = TestNode(3)
    g[7] = TestNode(2)
    g[8] = TestNode(1)

    add_edge!(g, 1, 2)
    add_edge!(g, 1, 3)
    add_edge!(g, 2, 4)
    add_edge!(g, 3, 5)
    add_edge!(g, 4, 6)
    add_edge!(g, 5, 6)
    add_edge!(g, 5, 7)
    add_edge!(g, 6, 8)
    add_edge!(g, 7, 8)

    # Test single node Markov blanket
    @test dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
        g, 1, MetaGraphsNext.outneighbor_labels
    ) == (Set([3, 8]), Set([4, 6, 2]))
    @test dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
        g, 1, MetaGraphsNext.inneighbor_labels
    ) == (Set(), Set())
    @test markov_blanket(g, 1) == Set(collect(1:8)) # should contains all the nodes

    @test dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
        g, 5, MetaGraphsNext.outneighbor_labels
    ) == (Set([7, 8]), Set([6]))
    @test dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
        g, 5, MetaGraphsNext.inneighbor_labels
    ) == (Set([3]), Set())
    @test markov_blanket(g, 5) == Set([1, 2, 3, 4, 5, 6, 7, 8])

    @test markov_blanket(g, 3) == Set([1, 3, 5])
    @test markov_blanket(g, 7) == Set([1, 2, 4, 5, 6, 7, 8])
    @test markov_blanket(g, 8) == Set([1, 2, 4, 5, 6, 7, 8])

    @test markov_blanket(g, [1, 3]) == Set([1, 2, 3, 4, 5, 6, 7, 8])
    @test markov_blanket(g, (3, 7)) == Set([1, 2, 3, 4, 5, 6, 7, 8])
end
