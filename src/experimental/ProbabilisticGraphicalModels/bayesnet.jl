module BayesianNetworkModule

using Graphs
using Distributions

###############################################################################
# 1) BayesianNetwork definition (mutable + node_types)
###############################################################################

mutable struct BayesianNetwork{V,T,F}
    graph::SimpleDiGraph{T}
    names::Vector{V}
    names_to_ids::Dict{V,T}
    values::Dict{V,Any}
    distributions::Vector{Any}            # Distribution or function returning a Distribution
    deterministic_functions::Vector{F}    # (unused here)
    stochastic_ids::Vector{T}
    deterministic_ids::Vector{T}
    is_stochastic::BitVector
    is_observed::BitVector
    node_types::Vector{Symbol}            # e.g. :discrete or :continuous
end

"""
Construct an empty BayesianNetwork with symbol names.
"""
function BayesianNetwork{V}() where {V}
    return BayesianNetwork(
        SimpleDiGraph{Int}(),
        V[],
        Dict{V,Int}(),
        Dict{V,Any}(),
        Any[],
        Any[],
        Int[],
        Int[],
        BitVector(),
        BitVector(),
        Symbol[],
    )
end

###############################################################################
# 2) Graph Helpers
###############################################################################

"""
    condition(bn::BayesianNetwork{V}, values::Dict{V,Any}) where {V}

Condition the Bayesian Network on the values of some variables. Return a new Bayesian Network with the conditioned graph.
"""
function condition(
    bn::BayesianNetwork{V}, conditioning_variables_and_values::Dict{V,<:Any}
) where {V}
    is_observed = copy(bn.is_observed)
    values = copy(bn.values)
    bn_new = BangBang.setproperties!!(bn; is_observed=is_observed, values=values)
    return condition!(bn_new, conditioning_variables_and_values)
end

"""
    condition!(bn::BayesianNetwork{V}, values::Dict{V,Any}) where {V}

Condition the Bayesian Network on the values of some variables. Mutating version of [`condition`](@ref).
"""
function condition!(
    bn::BayesianNetwork{V}, conditioning_variables_and_values::Dict{V,<:Any}
) where {V}
    for (name, value) in conditioning_variables_and_values
        id = bn.names_to_ids[name]
        if !bn.is_stochastic[id]
            throw(ArgumentError("Variable $name is not stochastic, cannot condition on it"))
        elseif bn.is_observed[id]
            @warn "Variable $name is already observed, overwriting its value"
        else
            bn.is_observed[id] = true
        end
        bn.values[name] = value
    end
    return bn
end

function decondition(bn::BayesianNetwork{V}) where {V}
    conditioned_variables_ids = findall(bn.is_observed)
    return decondition(bn, bn.names[conditioned_variables_ids])
end

function decondition!(bn::BayesianNetwork{V}) where {V}
    conditioned_variables_ids = findall(bn.is_observed)
    return decondition!(bn, bn.names[conditioned_variables_ids])
end

function decondition(bn::BayesianNetwork{V}, variables::Vector{V}) where {V}
    is_observed = copy(bn.is_observed)
    values = copy(bn.values)
    bn_new = BangBang.setproperties!!(bn; is_observed=is_observed, values=values)
    return decondition!(bn_new, variables)
end

function decondition!(bn::BayesianNetwork{V}, deconditioning_variables::Vector{V}) where {V}
    for name in deconditioning_variables
        id = bn.names_to_ids[name]
        if !bn.is_stochastic[id]
            throw(
                ArgumentError("Variable $name is not stochastic, cannot decondition on it")
            )
        elseif !bn.is_observed[id]
            throw(ArgumentError("Variable $name is not observed, cannot decondition on it"))
        end
        bn.is_observed[id] = false
        delete!(bn.values, name)
    end
    return bn
end


"""
Add a stochastic vertex to the BayesianNetwork.
- `dist` can be a `Distribution` or a function returning a `Distribution`.
- `node_type` can be `:discrete` or `:continuous`.
- `is_observed` defaults to `false`.
"""
function add_stochastic_vertex!(
    bn::BayesianNetwork{V,T},
    name::V,
    dist::Any,
    node_type::Symbol = :continuous;  
    is_observed::Bool = false
)::T where {V,T}
    Graphs.add_vertex!(bn.graph) || return 0
    id = nv(bn.graph)
    push!(bn.distributions, dist)
    push!(bn.is_stochastic, true)
    push!(bn.is_observed, is_observed)
    push!(bn.names, name)
    bn.names_to_ids[name] = id
    push!(bn.stochastic_ids, id)
    push!(bn.node_types, node_type)
    return id
end

"""
Add a deterministic vertex to the BayesianNetwork.
- `f` is a function that defines how this node is computed from its parents.
"""
function add_deterministic_vertex!(bn::BayesianNetwork{V,T}, name::V, f::F)::T where {T,V,F}
    Graphs.add_vertex!(bn.graph) || return 0
    id = nv(bn.graph)
    push!(bn.deterministic_functions, f)
    push!(bn.is_stochastic, false)
    push!(bn.is_observed, false)
    push!(bn.names, name)
    bn.names_to_ids[name] = id
    push!(bn.deterministic_ids, id)
    push!(bn.node_types, :deterministic)
    return id
end

"""
Add a directed edge `from -> to` in the BayesianNetwork's graph.
"""
function add_edge!(bn::BayesianNetwork{V,T}, from::V, to::V)::Bool where {T,V}
    from_id = bn.names_to_ids[from]
    to_id   = bn.names_to_ids[to]
    return Graphs.add_edge!(bn.graph, from_id, to_id)
end

"""
    ancestral_sampling(bn::BayesianNetwork{V}) where {V}

Perform ancestral sampling on a Bayesian network to generate one sample from the joint distribution.

Ancestral sampling works by:
1. Finding a topological ordering of the nodes
2. Sampling from each node in order, using the already-sampled parent values for conditional distributions

### Return Value
The function returns a `Dict{V, Any}` where:
- Each key is a variable name (of type `V`) in the Bayesian Network.
- Each value is the sampled value for that variable, which can be of any type (`Any`).

This dictionary represents a single sample from the joint distribution of the Bayesian Network, capturing the dependencies and conditional relationships defined in the network structure.

"""
function ancestral_sampling(bn::BayesianNetwork{V}) where {V}
    ordered_vertices = Graphs.topological_sort_by_dfs(bn.graph)
    samples = Dict{V,Any}()

    for vertex_id in ordered_vertices
        vertex_name = bn.names[vertex_id]
        if bn.is_observed[vertex_id]
            samples[vertex_name] = bn.values[vertex_name]
            continue
        end
        if bn.is_stochastic[vertex_id]
            dist_idx = findfirst(id -> id == vertex_id, bn.stochastic_ids)
            samples[vertex_name] = rand(bn.distributions[dist_idx])
        else
            # deterministic node
            parent_ids = Graphs.inneighbors(bn.graph, vertex_id)
            parent_values = [samples[bn.names[pid]] for pid in parent_ids]
            func_idx = findfirst(id -> id == vertex_id, bn.deterministic_ids)
            samples[vertex_name] = bn.deterministic_functions[func_idx](parent_values...)
        end
    end

    return samples
end

"""
    is_conditionally_independent(bn::BayesianNetwork, X::Vector{V}, Y::Vector{V}, Z::Vector{V}) where {V}

Test whether sets of variables X and Y are conditionally independent given set Z in a Bayesian Network using the Bayes Ball algorithm.

# Arguments
- `bn::BayesianNetwork`: The Bayesian Network structure
- `X::Vector{V}`: First set of variables to test for independence
- `Y::Vector{V}`: Second set of variables to test for independence
- `Z::Vector{V}`: Set of conditioning variables (can be empty)

# Returns
- `true`: if X and Y are conditionally independent given Z (X ⊥ Y | Z)
- `false`: if X and Y are conditionally dependent given Z

# Description
The Bayes Ball algorithm determines conditional independence by checking if there exists an active path between 
variables in X and Y given Z. The algorithm follows these rules:
- In a chain (A → B → C): B blocks the path if conditioned
- In a fork (A ← B → C): B blocks the path if conditioned
- In a collider (A → B ← C): B opens the path if conditioned 
# Examples
```
"""
function is_conditionally_independent(
    bn::BayesianNetwork{V}, X::Vector{V}, Y::Vector{V}, Z::Vector{V}
) where {V}
    isempty(X) && throw(ArgumentError("X cannot be empty"))
    isempty(Y) && throw(ArgumentError("Y cannot be empty"))

    x_ids = Set([bn.names_to_ids[x] for x in X])
    y_ids = Set([bn.names_to_ids[y] for y in Y])
    z_ids = Set([bn.names_to_ids[z] for z in Z])

    # Check if any variable in X or Y is in Z
    if !isempty(intersect(x_ids, z_ids)) || !isempty(intersect(y_ids, z_ids))
        return true
    end

    # Add observed variables to conditioning set
    for (id, is_obs) in enumerate(bn.is_observed)
        if is_obs
            push!(z_ids, id)
        end
    end

    # Track visited nodes and their directions
    n_vertices = nv(bn.graph)
    visited_up = falses(n_vertices)   # Visited going up (from child to parent)
    visited_down = falses(n_vertices) # Visited going down (from parent to child)

    # Queue entries are (node_id, going_up)
    queue = Tuple{Int,Bool}[]

    # Start from all X nodes
    for x_id in x_ids
        push!(queue, (x_id, true))   # Try going up
        push!(queue, (x_id, false))  # Try going down
    end

    while !isempty(queue)
        current_id, going_up = popfirst!(queue)

        # Skip if we've visited this node in this direction
        if (going_up && visited_up[current_id]) || (!going_up && visited_down[current_id])
            continue
        end

        # Mark as visited in current direction
        if going_up
            visited_up[current_id] = true
        else
            visited_down[current_id] = true
        end

        # If we reached a Y node, path is active
        if current_id in y_ids
            return false
        end

        is_conditioned = current_id in z_ids
        parents = inneighbors(bn.graph, current_id)
        children = outneighbors(bn.graph, current_id)

        if is_conditioned
            # If conditioned:
            # - In a chain/fork: blocks the path
            # - In a collider or descendant of collider: allows going up to parents
            if length(parents) > 1 || !isempty(children)  # Is collider or has children
                for parent in parents
                    push!(queue, (parent, true))  # Can only go up to parents
                end
            end
        else
            # If not conditioned:
            if going_up
                # Going up: can visit parents
                for parent in parents
                    push!(queue, (parent, true))
                end
            else
                # Going down: can visit children
                for child in children
                    push!(queue, (child, false))
                end
            end

            # At starting nodes (X), we can go both up and down
            if current_id in x_ids
                if going_up
                    for child in children
                        push!(queue, (child, false))
                    end
                else
                    for parent in parents
                        push!(queue, (parent, true))
                    end
                end
            end
        end
    end

    return true
end

# Single variable version with Z
function is_conditionally_independent(
    bn::BayesianNetwork{V}, X::V, Y::V, Z::Vector{V}
) where {V}
    return is_conditionally_independent(bn, [X], [Y], Z)
end

###############################################################################
# 3) Parent/Distribution Helpers
###############################################################################

function parent_ids(bn::BayesianNetwork, node_id::Int)
    return inneighbors(bn.graph, node_id)
end

function parent_values(bn::BayesianNetwork, node_id::Int)
    pids = parent_ids(bn, node_id)
    sort!(pids)
    vals = Any[]
    for pid in pids
        varname = bn.names[pid]
        if !haskey(bn.values, varname)
            error("Missing value for parent $varname of node id=$node_id")
        end
        push!(vals, bn.values[varname])
    end
    return vals
end

function get_distribution(bn::BayesianNetwork, node_id::Int)::Distribution
    stored = bn.distributions[node_id]
    if stored isa Distribution
        return stored
    elseif stored isa Function
        pvals = parent_values(bn, node_id)  # gather parents' assigned values
        return stored(pvals...)
    else
        error("Node $node_id has invalid distribution entry.")
    end
end

function is_discrete_node(bn::BayesianNetwork, node_id::Int)
    return bn.node_types[node_id] == :discrete
end

###############################################################################
# 4) Logpdf Computation
###############################################################################

"""
Compute the sum of log probabilities for all **stochastic** nodes
using the current values assigned in `bn.values`.
If any distribution or parent's value is missing or invalid, returns -Inf.
"""
function compute_full_logpdf(bn::BayesianNetwork)
    logp = 0.0
    for sid in bn.stochastic_ids
        varname = bn.names[sid]
        if haskey(bn.values, varname)
            # ensure parents assigned
            for pid in parent_ids(bn, sid)
                if !haskey(bn.values, bn.names[pid])
                    return -Inf
                end
            end
            dist = get_distribution(bn, sid)
            val  = bn.values[varname]
            lpdf = logpdf(dist, val)
            if isinf(lpdf)
                return -Inf
            end
            logp += lpdf
        end
    end
    return logp
end

###############################################################################
# 5) Naive Summation of Discrete Configurations
###############################################################################

"""
Naive recursion:
Enumerate all discrete node values for unobserved discrete nodes.
Returns a *probability sum*, i.e. sum over exp(logpdf).
"""
function sum_discrete_configurations(
    bn::BayesianNetwork,
    discrete_ids::Vector{Int},
    idx::Int
)::Float64
    if idx > length(discrete_ids)
        return exp( compute_full_logpdf(bn) )
    else
        node_id = discrete_ids[idx]
        dist = get_distribution(bn, node_id)
        total_prob = 0.0
        for val in support(dist)
            bn.values[ bn.names[node_id] ] = val
            total_prob += sum_discrete_configurations(bn, discrete_ids, idx+1) * pdf(dist, val)
        end
        delete!(bn.values, bn.names[node_id])
        return total_prob
    end
end

###############################################################################
# 6) create_log_posterior (Naive Only)
###############################################################################

"""
Creates a log_posterior function that merges unobserved values + sums out
unobserved discrete nodes (naive recursion).
Returns log(prob_sum).
"""
function create_log_posterior(bn::BayesianNetwork)
    function log_posterior(unobserved_values::Dict{Symbol,Float64})
        old_values = copy(bn.values)
        try
            # Merge unobserved
            for (k, v) in unobserved_values
                bn.values[k] = v
            end

            # Identify unobserved discrete IDs
            unobs_discrete_ids = Int[]
            for sid in bn.stochastic_ids
                if !bn.is_observed[sid]
                    varname = bn.names[sid]
                    if !haskey(bn.values, varname) && is_discrete_node(bn, sid)
                        push!(unobs_discrete_ids, sid)
                    end
                end
            end

            if isempty(unobs_discrete_ids)
                # no discrete marginalization => direct logpdf
                return compute_full_logpdf(bn)
            else
                # naive recursion
                prob_sum = sum_discrete_configurations(bn, unobs_discrete_ids, 1)
                return log(prob_sum)
            end
        finally
            bn.values = old_values
        end
    end
    return log_posterior
end

end  # module
