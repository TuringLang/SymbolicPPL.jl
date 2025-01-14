using Distributions
using Printf

#######################
# 1) Utility Functions
#######################
logistic(x) = 1 / (1 + exp(-x))

#######################
# 2) BayesNet Struct
#######################
"""
BayesNet{T}:
  - nodes::Dict{Symbol,T}  # each entry is either a Distribution or a function returning a Distribution
  - edges::Dict{Symbol,Vector{Symbol}}  # a map from node => list_of_parents
"""
struct BayesNet{T}
    nodes::Dict{Symbol,T}
    edges::Dict{Symbol,Vector{Symbol}}
end

"""
create_bayes_net(nodes, edges) -> BayesNet
Given dictionaries of node definitions and edges, builds a BayesNet.
"""
function create_bayes_net(nodes::Dict{Symbol,Any}, edges::Dict{Symbol,Vector{Symbol}})
    return BayesNet(nodes, edges)
end

"""
create_sequential_net_n(nodes) -> BayesNet
Given a list of distribution/callable objects `nodes` of length n,
construct a chain X1 -> X2 -> ... -> Xn, where each Xi depends on X(i-1).
"""
function create_sequential_net_n(nodes::Vector{Any})
    n = length(nodes)
    node_dict = Dict{Symbol,Any}()
    edge_dict = Dict{Symbol,Vector{Symbol}}()

    for i in 1:n
        node_sym = Symbol("X$i")
        node_dict[node_sym] = nodes[i]

        if i == 1
            edge_dict[node_sym] = Symbol[]
        else
            edge_dict[node_sym] = [Symbol("X$(i-1)")]
        end
    end

    return BayesNet(node_dict, edge_dict)
end

###############################
# 3) create_log_posterior
###############################
function create_log_posterior(net::BayesNet{T}, observations::Dict{Symbol,Float64}) where {T}

    function compute_root_prior_prob(current_values::Dict{Symbol,Float64}, indent="")
        prob = 1.0
        for (node, node_or_callable) in net.nodes
            parents = get(net.edges, node, Symbol[])
            if isempty(parents) && haskey(current_values, node)
                val = current_values[node]
                dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable()
                prob *= pdf(dist, val)
            end
        end
        return prob
    end

    function compute_root_prior_logprob(current_values::Dict{Symbol,Float64}, indent="")
        logp = 0.0
        for (node, node_or_callable) in net.nodes
            parents = get(net.edges, node, Symbol[])
            if isempty(parents) && haskey(current_values, node)
                val = current_values[node]
                dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable()
                logp += logpdf(dist, val)
            end
        end
        return logp
    end

    function find_discrete_nodes_to_marginalize(known_values::Dict{Symbol,Float64})
        nodes_to_marginalize = Symbol[]

        function get_intermediate_nodes()
            intermediates = Symbol[]
            for (child, parents) in net.edges
                if haskey(known_values, child) || any(p -> haskey(known_values, p), parents)
                    current = child
                    while haskey(net.edges, current)
                        pnodes = net.edges[current]
                        if isempty(pnodes)
                            break
                        end
                        for p in pnodes
                            if !haskey(known_values, p) && !(p in intermediates)
                                push!(intermediates, p)
                            end
                        end
                        current = pnodes[1]
                    end
                end
            end
            return intermediates
        end

        intermediate_nodes = get_intermediate_nodes()

        for node in intermediate_nodes
            if !haskey(known_values, node)
                parents = get(net.edges, node, Symbol[])
                node_dist = isempty(parents) ? net.nodes[node] : net.nodes[node](zeros(length(parents))...)
                if node_dist isa DiscreteDistribution
                    push!(nodes_to_marginalize, node)
                end
            end
        end

        return nodes_to_marginalize
    end

    function topological_sort(node, nodes_to_marginalize, visited, sorted_nodes)
        if node in visited
            return
        end
        push!(visited, node)
        if haskey(net.edges, node)
            for parent in net.edges[node]
                if parent in nodes_to_marginalize
                    topological_sort(parent, nodes_to_marginalize, visited, sorted_nodes)
                end
            end
        end
        if node in nodes_to_marginalize
            push!(sorted_nodes, node)
        end
    end

    function marginalize_recursive(nodes_to_marginalize::Vector{Symbol}, current_values::Dict{Symbol,Float64}, depth::Int=0)
        if isempty(nodes_to_marginalize)
            prob = compute_root_prior_prob(current_values)
            for (obs_node, obs_val) in observations
                parents = get(net.edges, obs_node, Symbol[])
                if !isempty(parents) && all(p -> haskey(current_values, p), parents)
                    parent_vals = [current_values[p] for p in parents]
                    node_or_callable = net.nodes[obs_node]
                    node_dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable(parent_vals...)
                    prob *= pdf(node_dist, obs_val)
                elseif isempty(parents) && !haskey(current_values, obs_node)
                    node_or_callable = net.nodes[obs_node]
                    dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable()
                    prob *= pdf(dist, obs_val)
                else
                    return 0.0
                end
            end
            return prob
        else
            current_node = nodes_to_marginalize[1]
            remaining_nodes = nodes_to_marginalize[2:end]
            parents = get(net.edges, current_node, Symbol[])
            if !all(p -> haskey(current_values, p), parents)
                return 0.0
            end
            parent_vals = [current_values[p] for p in parents]
            node_or_callable = net.nodes[current_node]
            node_dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable(parent_vals...)
            sum_likelihood = 0.0
            for val in support(node_dist)
                new_values = copy(current_values)
                new_values[current_node] = val
                p_val = pdf(node_dist, val)
                child_prob = marginalize_recursive(remaining_nodes, new_values, depth + 1)
                sum_likelihood += p_val * child_prob
            end
            return sum_likelihood
        end
    end

    function log_posterior(values::Dict{Symbol,Float64})
        known_values = merge(values, observations)
        nodes_to_marginalize = find_discrete_nodes_to_marginalize(known_values)

        sorted_nodes = Symbol[]
        visited = Set{Symbol}()
        for node in nodes_to_marginalize
            topological_sort(node, nodes_to_marginalize, visited, sorted_nodes)
        end

        if isempty(sorted_nodes)
            log_p = compute_root_prior_logprob(known_values)
            for (obs_node, obs_val) in observations
                parents = get(net.edges, obs_node, Symbol[])
                if isempty(parents) && !haskey(known_values, obs_node)
                    node_or_callable = net.nodes[obs_node]
                    dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable()
                    log_p += logpdf(dist, obs_val)
                elseif all(p -> haskey(known_values, p), parents)
                    parent_vals = [known_values[p] for p in parents]
                    node_or_callable = net.nodes[obs_node]
                    node_dist = node_or_callable isa Distribution ? node_or_callable : node_or_callable(parent_vals...)
                    log_p += logpdf(node_dist, obs_val)
                else
                    return -Inf
                end
            end
            return log_p
        else
            likelihood = marginalize_recursive(sorted_nodes, known_values)
            return log(likelihood)
        end
    end

    return log_posterior
end

###################################################
# 4) evaluate_model Function
###################################################
function evaluate_model(net::BayesNet, observations::Dict{Symbol,Float64}, X1_values, description::AbstractString)
    println("\n=== $description ===")
    println("Observations: ", observations)

    log_posterior_fn = create_log_posterior(net, observations)
    results = [(x1, log_posterior_fn(Dict(:X1 => x1))) for x1 in X1_values]

    max_logp = maximum(last.(results))
    normalized = [(x1, exp(lp - max_logp)) for (x1, lp) in results]

    for (x1, p) in normalized
        @printf("  X1 = %.1f => normalized posterior = %.4f\n", x1, p)
    end
end

##################################
# 5) Example Usage
##################################
model_5_nodes = create_sequential_net_n([
    Normal(0, 1),
    x1 -> Bernoulli(logistic(x1)),
    x2 -> Bernoulli(x2 == 1 ? 0.7 : 0.3),
    x3 -> Bernoulli(x3 == 1 ? 0.8 : 0.2),
    x4 -> Normal(x4 == 1 ? 3.0 : -3.0, 1.0),
])

X1_values = 0.0:0.5:1.5

println("\n=== Running test cases on the 5-node model ===")

evaluate_model(model_5_nodes, Dict(:X4 => 1.0, :X5 => 2.0), X1_values, "5-Node Model (X4=1.0, X5=2.0, marginalizing X2,X3)")
evaluate_model(model_5_nodes, Dict(:X5 => 2.0), X1_values, "5-Node Model (X5=2.0, marginalizing X2,X3,X4)")
evaluate_model(model_5_nodes, Dict(:X2 => 1.0, :X3 => 1.0, :X4 => 1.0, :X5 => 2.0), X1_values, "5-Node Model (all observed)")

println("\nDone.")
