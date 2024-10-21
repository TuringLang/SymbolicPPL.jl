struct NodeInfo{F}
    is_stochastic::Bool
    is_observed::Bool
    node_function_expr::Expr
    node_function::F
    node_args::Tuple{Vararg{Symbol}}
    loop_vars::NamedTuple
end

const BUGSGraph = MetaGraph{
    Int,Graphs.SimpleDiGraph{Int},<:VarName,<:NodeInfo,Nothing,Nothing,<:Any,Float64
}

is_stochastic(g::BUGSGraph, v::VarName) = g[v].is_stochastic
is_model_parameter(g::BUGSGraph, v::VarName) = g[v].is_stochastic && !g[v].is_observed
is_observation(g::BUGSGraph, v::VarName) = g[v].is_stochastic && g[v].is_observed
is_deterministic(g::BUGSGraph, v::VarName) = !g[v].is_stochastic

"""
    find_generated_quantities_variables(g::BUGSGraph)

Find all the generated quantities variables in the graph.

Generated quantities variables are variables that do not affect the sampling process. 
They are variables that do not have any descendant variables that are observed.
"""
function find_generated_quantities_variables(
    g::MetaGraph{Int,<:SimpleDiGraph,Label,VertexData}
) where {Label,VertexData}
    generated_quantities_variables = Set{Label}()
    can_reach_observations = Dict{Label,Bool}()

    for n in labels(g)
        if !is_observation(g, n)
            if !dfs_can_reach_observations(g, n, can_reach_observations)
                push!(generated_quantities_variables, n)
            end
        end
    end
    return generated_quantities_variables
end

function dfs_can_reach_observations(g, n, can_reach_observations)
    if haskey(can_reach_observations, n)
        return can_reach_observations[n]
    end

    if is_observation(g, n)
        can_reach_observations[n] = true
        return true
    end

    can_reach = false
    for child in MetaGraphsNext.outneighbor_labels(g, n)
        if dfs_can_reach_observations(g, child, can_reach_observations)
            can_reach = true
            break
        end
    end

    can_reach_observations[n] = can_reach
    return can_reach
end

"""
    markov_blanket(g::BUGSGraph, v)

Find the Markov blanket of variable(s) `v` in graph `g`. `v` can be a single `VarName` or a vector/tuple of `VarName`.

The Markov Blanket of a variable is the set of variables that shield the variable from the rest of the
network. Effectively, the Markov blanket of a variable is the set of its parents, its children, and 
its children's other parents (reference: https://en.wikipedia.org/wiki/Markov_blanket).

In the case of a vector of variables, the Markov Blanket is the union of the Markov Blankets of each variable 
minus the variables themselves[1].

This function returns a `Set` of `VarName`s, containing the Markov blanket of the variable(s) `v` in graph `g`, deterministic
variables that are on the path from `v` to their Markov blankets, and the variables `v` itself(themselves).

[1] Liu, X.-Q., & Liu, X.-S. (2018). Markov Blanket and Markov 
Boundary of Multiple Variables. Journal of Machine Learning Research, 19(43), 1–50.
"""
function markov_blanket(g::MetaGraph{Int,<:SimpleDiGraph,L,VD}, v::L) where {L,VD}
    if !is_stochastic(g, v)
        throw(ArgumentError("Variable $v is logical, so it has no Markov blanket."))
    end

    parents, deterministic_variables_en_route_parents = dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
        g, v, MetaGraphsNext.inneighbor_labels
    )
    children, deterministic_variables_en_route_children = dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
        g, v, MetaGraphsNext.outneighbor_labels
    )

    co_parents, deterministic_variables_en_route_co_parents = Set{L}(), Set{L}()
    for child in children
        co_parents_child, deterministic_variables_en_route_co_parents_child = dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
            g, child, MetaGraphsNext.inneighbor_labels
        )
        union!(co_parents, co_parents_child)
        union!(
            deterministic_variables_en_route_co_parents,
            deterministic_variables_en_route_co_parents_child,
        )
    end

    blanket = union!(
        parents,
        children,
        co_parents,
        deterministic_variables_en_route_parents,
        deterministic_variables_en_route_children,
        deterministic_variables_en_route_co_parents,
    )
    push!(blanket, v)
    return blanket
end

function markov_blanket(
    g::MetaGraph{Int,<:SimpleDiGraph,L,VD}, vs::AbstractVector{<:L}
) where {L,VD}
    return reduce((acc, vn) -> union!(acc, markov_blanket(g, vn)), vs; init=Set{L}())
end

function dfs_find_stochastic_boundary_and_deterministic_variables_en_route(
    g::MetaGraph{Int,<:SimpleDiGraph,L,VD}, v::L, neighbor_func::F
) where {L,VD,F}
    stochastic_neighbors = Set{L}()
    deterministic_variables_en_route = Set{L}()
    stack = VarName[v]
    visited = Set{L}()

    while !isempty(stack)
        current = pop!(stack)
        current in visited && continue

        if is_deterministic(g, current)
            push!(deterministic_variables_en_route, current)
        end

        push!(visited, current)

        for neighbor in neighbor_func(g, current)
            if !(neighbor in visited)
                if is_stochastic(g, neighbor)
                    push!(stochastic_neighbors, neighbor)
                    # and stop (not pushing to stack)
                else
                    push!(stack, neighbor)
                end
            end
        end
    end

    return stochastic_neighbors, deterministic_variables_en_route
end
