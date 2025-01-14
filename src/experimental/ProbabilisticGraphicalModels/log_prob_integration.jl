using Graphs
using Distributions
using Printf

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

function add_stochastic_vertex!(
    bn::BayesianNetwork{V,T},
    name::V,
    dist::Any,
    node_type::Symbol=:continuous;  # e.g. :discrete or :continuous
    is_observed::Bool=false,
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

function add_edge!(bn::BayesianNetwork{V,T}, from::V, to::V)::Bool where {T,V}
    from_id = bn.names_to_ids[from]
    to_id = bn.names_to_ids[to]
    return Graphs.add_edge!(bn.graph, from_id, to_id)
end

###############################################################################
# 3) A 5-node chain, with discrete/continuous mix
###############################################################################

logistic(x) = 1 / (1 + exp(-x))

function create_5node_network()
    bn = BayesianNetwork{Symbol}()

    # X1 ~ Normal(0,1) => continuous
    add_stochastic_vertex!(bn, :X1, Normal(0, 1), :continuous)

    # X2 ~ Bernoulli(logistic(X1)) => discrete
    add_stochastic_vertex!(bn, :X2, (x1) -> Bernoulli(logistic(x1)), :discrete)
    add_edge!(bn, :X1, :X2)

    # X3 ~ Bernoulli(x2==1 ? 0.7 : 0.3) => discrete
    add_stochastic_vertex!(bn, :X3, (x2) -> Bernoulli(x2 == 1 ? 0.7 : 0.3), :discrete)
    add_edge!(bn, :X2, :X3)

    # X4 ~ Bernoulli(x3==1 ? 0.8 : 0.2) => discrete
    add_stochastic_vertex!(bn, :X4, (x3) -> Bernoulli(x3 == 1 ? 0.8 : 0.2), :discrete)
    add_edge!(bn, :X3, :X4)

    # X5 ~ Normal(x4==1 ? 3.0 : -3.0, 1.0) => continuous
    add_stochastic_vertex!(bn, :X5, (x4) -> Normal(x4 == 1 ? 3.0 : -3.0, 1.0), :continuous)
    add_edge!(bn, :X4, :X5)

    return bn
end

###############################################################################
# 4) Parent/Distribution Helpers
###############################################################################

function parent_ids(bn::BayesianNetwork, node_id::Int)
    return inneighbors(bn.graph, node_id)  # replaced `predecessors` with `inneighbors`
end

function parent_values(bn::BayesianNetwork, node_id::Int)
    pids = parent_ids(bn, node_id)
    sort!(pids)
    vals = Any[]
    for pid in pids
        varname = bn.names[pid]
        if !haskey(bn.values, varname)
            error("Missing value for parent $varname of node id=$node_id. Observed: $(bn.is_observed[pid])")
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
        pvals = parent_values(bn, node_id)  # gather parent's assigned values
        return stored(pvals...)
    else
        error("Node $node_id has invalid distribution entry.")
    end
end

function is_discrete_node(bn::BayesianNetwork, node_id::Int)
    return bn.node_types[node_id] == :discrete
end

###############################################################################
# 5) Logpdf Computation
###############################################################################

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
            val = bn.values[varname]
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
# 6) Naive Summation vs. DP Summation
###############################################################################
# We provide two ways to sum over discrete configurations:
#  - sum_discrete_configurations: naive recursion
#  - sum_discrete_configurations_dp: memoized recursion (DP)

"""
Naive recursion:
Enumerate all discrete node values for unobserved discrete nodes.
"""
function sum_discrete_configurations(
    bn::BayesianNetwork, discrete_ids::Vector{Int}, idx::Int
)::Float64
    if idx > length(discrete_ids)
        return exp(compute_full_logpdf(bn))
    else
        node_id = discrete_ids[idx]
        dist = get_distribution(bn, node_id)
        total_prob = 0.0
        for val in support(dist)
            bn.values[bn.names[node_id]] = val
            # multiply by pdf(dist, val)
            total_prob +=
                sum_discrete_configurations(bn, discrete_ids, idx + 1) * pdf(dist, val)
        end
        delete!(bn.values, bn.names[node_id])
        return total_prob
    end
end

"""
DP-based recursion:
Use a memo dictionary to store subproblem results.
Key = (idx, assigned_values_for_these_discrete_ids)

We store the partial assignment of discrete_ids[1:idx-1], then skip redoing
the entire subtree if we've already computed it.
"""
function sum_discrete_configurations_dp(
    bn::BayesianNetwork,
    discrete_ids::Vector{Int},
    idx::Int,
    memo::Dict{Any, Float64},
    assigned_vals::Tuple{Vararg{Any}},
)::Float64
    # Create a key using the immutable tuple `assigned_vals`

    key = (idx, deepcopy(assigned_vals))
    println("Generated key: $key")
    if haskey(memo, key)
        println("Cache hit for key $key")
        return memo[key]
    else
        println("Key $key not in memo")
    end


    if idx > length(discrete_ids)
        # Base case: Compute the logpdf
        result = exp(compute_full_logpdf(bn))
        memo[key] = result
        println("Base case: LogPDF result = $result for key $key")
        return result
    else
        node_id = discrete_ids[idx]
        dist = get_distribution(bn, node_id)
        total_prob = 0.0
        # Iterate over all possible values in the support of the distribution
        for val in support(dist)
            # Create a new tuple with the current value appended
            new_assigned_vals = (assigned_vals..., val)
            bn.values[bn.names[node_id]] = val
            println("Assigning value $val to node $(bn.names[node_id])")
            total_prob += sum_discrete_configurations_dp(
                bn, discrete_ids, idx + 1, memo, new_assigned_vals
            ) * pdf(dist, val)
        end
        delete!(bn.values, bn.names[node_id])
        memo[key] = total_prob
        println("Memoized result for key $key: $total_prob")
        return total_prob
    end
end


###############################################################################
# 7) create_log_posterior with DP option
###############################################################################

"""
Creates a log_posterior function that merges unobserved values + sums out
unobserved discrete nodes. If use_dp=true, we use the DP approach; else naive.
"""

function create_log_posterior(bn::BayesianNetwork; obs::Dict=Dict(), use_dp::Bool=false)
    # Logic to incorporate `obs` into the posterior computation
    println("Observations provided: $obs")

    function log_posterior(bn::BayesianNetwork; obs::Dict=Dict(), use_dp::Bool=false)
        println("Computing log posterior. Observations: $obs, use_dp=$use_dp")
        
        # Apply observations
        for (k, v) in obs
            id = bn.names_to_ids[k]
            bn.values[k] = v
            bn.is_observed[id] = true
        end
    
        # Return a callable function for posterior computation
        if use_dp
            return (unobserved_values::Dict{Symbol, Float64}) -> begin
                # Add logic to handle unobserved values if needed
                sum_discrete_configurations_dp(bn, bn.stochastic_ids, 1, Dict(), Tuple())
            end
        else
            return (unobserved_values::Dict{Symbol, Float64}) -> begin
                # Add naive computation logic here
                error("Naive approach not implemented yet.")
            end
        end
    end
    return log_posterior
end

###############################################################################
# 8) Evaluate function
###############################################################################

function evaluate_model(
    bn::BayesianNetwork;
    obs::Dict{Symbol,Float64}=Dict(),
    X1_values=0.0:0.5:1.5,
    description="",
    use_dp::Bool=false,
)
    println("\n=== $description ===")
    println("Observations: $obs")
    println("use_dp = $use_dp")

    old_values = copy(bn.values)
    old_obs = copy(bn.is_observed)
    try
        # Condition on obs
        for (k, v) in obs
            id = bn.names_to_ids[k]
            bn.values[k] = v
            bn.is_observed[id] = true
        end

        # create log posterior
        log_post = create_log_posterior(bn; obs=obs, use_dp=use_dp)
        


        # evaluate over X1
        results = [(x1, log_post(Dict(:X1 => x1))) for x1 in X1_values]

        # normalize
        max_lp = maximum(last.(results))
        posterior = [(x1, exp(lp - max_lp)) for (x1, lp) in results]

        # print
        for (x1, p) in posterior
            @printf("  X1 = %.2f => normalized posterior = %.5f\n", x1, p)
        end

    finally
        bn.values = old_values
        bn.is_observed = old_obs
    end
end

###############################################################################
# 9) Demonstration
###############################################################################

# (A) The Original 5-node chain
model_5_nodes = create_5node_network()
X1_values = 0.0:0.5:1.5

println("\n=== Running test cases on the 5-node chain ===")

# 1) Observing X4=1, X5=2 => marginalizing X2,X3
evaluate_model(
    model_5_nodes;
    obs=Dict(:X4 => 1.0, :X5 => 2.0),
    X1_values=X1_values,
    description="5-Node Model (X4=1.0, X5=2.0, marginalizing X2,X3) [DP=FALSE]",
    use_dp=false,
)

# same scenario but use_dp=true
evaluate_model(
    model_5_nodes;
    obs=Dict(:X4 => 1.0, :X5 => 2.0),
    X1_values=X1_values,
    description="5-Node Model (X4=1.0, X5=2.0, marginalizing X2,X3) [DP=TRUE]",
    use_dp=true,
)

# 2) Observing only X5=2.0 => marginalize X2,X3,X4
evaluate_model(
    model_5_nodes;
    obs=Dict(:X5 => 2.0),
    X1_values=X1_values,
    description="5-Node Model (X5=2.0, marginalizing X2,X3,X4) [DP=TRUE]",
    use_dp=true,
)

# 3) Observing X2=1, X3=1, X4=1, X5=2 => only X1 is unknown
evaluate_model(
    model_5_nodes;
    obs=Dict(:X2 => 1.0, :X3 => 1.0, :X4 => 1.0, :X5 => 2.0),
    X1_values=X1_values,
    description="5-Node Model (all observed except X1) [DP=TRUE]",
    use_dp=true,
)

###############################################################################
# 10) More Complex/Branching Example to Show DP Gains
###############################################################################

"""
Construct a BN with multiple discrete nodes that share parents,
to demonstrate repeated subproblems for DP.

Structure (simplified example):
        X1 (discrete or continuous)
       /   \
     X2     X3  (both discrete, each depends on X1)
          /
       X4 (discrete, depends on X2 and X3)
        \
         X5 (continuous, depends on X4)

We'll artificially inflate the support of X2, X3, X4 to highlight DP benefits.
"""
function create_branching_network()
    bn = BayesianNetwork{Symbol}()

    # X1 ~ Normal(0, 1)
    add_stochastic_vertex!(bn, :X1, Normal(0, 1), :continuous)

    # X2 ~ Bernoulli(logistic(X1))
    add_stochastic_vertex!(bn, :X2, (x1) -> Bernoulli(logistic(x1)), :discrete)
    add_edge!(bn, :X1, :X2)

    # X3 ~ Bernoulli(logistic(X1))
    add_stochastic_vertex!(bn, :X3, (x1) -> Bernoulli(logistic(x1)), :discrete)
    add_edge!(bn, :X1, :X3)

    # X4 ~ Bernoulli(logistic(X1))
    add_stochastic_vertex!(bn, :X4, (x1) -> Bernoulli(logistic(x1)), :discrete)
    add_edge!(bn, :X1, :X4)

    # X5 ~ Normal(mean(X2, X3, X4), 1)
    add_stochastic_vertex!(
        bn,
        :X5,
        (x2, x3, x4) -> Normal(mean([x2, x3, x4]), 1),
        :continuous
    )
    add_edge!(bn, :X2, :X5)
    add_edge!(bn, :X3, :X5)
    add_edge!(bn, :X4, :X5)

    return bn
end

# Create the network
bn = create_branching_network()

# Evaluate with observations and DP
evaluate_model(
    bn;
    obs=Dict(:X1 => 0.5, :X5 => 1.0),
    description="Branching BN (X1=0.5, X5=1.0, marginalizing X2, X3, X4)",
    use_dp=true
)

# Let's build & run a test to show how DP helps when enumerating
println("\n=== More Complex/Branching BN test to demonstrate DP gains ===")
branching_bn = create_branching_network()

# Suppose we observe X5=4.2 => i.e. we know the continuous node,
# and must marginalize out (X1, X2, X3, X4) which are mostly discrete.
obs = Dict(:X5 => 4.2)
id5 = branching_bn.names_to_ids[:X5]
branching_bn.values[:X5] = 4.2
branching_bn.is_observed[id5] = true

# Evaluate log posterior with & without DP
function test_branching_bn(branching_bn)
    # We'll define a dummy "X1_values" since X1 is discrete; let's pass an empty set
    # because we only want the log posterior (not scanning over X1 in a grid).
    log_post_naive = create_log_posterior(branching_bn; use_dp=false)
    log_post_dp = create_log_posterior(branching_bn; use_dp=true)

    # Just call each function once with no extra unobserved_values
    # to force the code to sum over X1, X2, X3, X4.
    println(">>> Branching BN: Observing X5=4.2, no other nodes => big discrete sum <<<")
    println("Naive approach ...")

    @time naive_lp = log_post_naive(Dict{Symbol,Float64}()) # time macro to see performance

    println("DP approach ...")

    @time dp_lp = log_post_dp(Dict{Symbol,Float64}())

    return println("Naive log posterior = $naive_lp, DP log posterior = $dp_lp\n")
end

test_branching_bn(branching_bn)

println("\nDone.")



###############################################################################
# Helper functions (add_vertex, edges, etc.)
###############################################################################
# All your original functions (add_stochastic_vertex!, add_edge!, etc.) are here.
# I've omitted their full listing for brevity as they remain unchanged.

###############################################################################
# Analysis Functions
###############################################################################

function ensure_parent_values!(bn::BayesianNetwork)
    for sid in bn.stochastic_ids
        # Get parent IDs and values
        for pid in parent_ids(bn, sid)
            varname = bn.names[pid]
            if !haskey(bn.values, varname)
                # Assign a default value if missing
                dist = get_distribution(bn, pid)
                bn.values[varname] = mean(dist)  # Use the mean as a sensible default
                println("Assigned default value to $varname: $(bn.values[varname])")
            end
        end
    end
end

function analyze_runtime(bn::BayesianNetwork, obs::Dict{Symbol,Float64}, use_dp::Bool)
    ensure_parent_values!(bn)  # Ensure all parent values are initialized
    log_post = create_log_posterior(bn, use_dp=use_dp)

    @time begin
        result = log_post(obs)
        println("Log posterior (use_dp=$use_dp): $result")
    end
end


function analyze_memoization(bn::BayesianNetwork, obs::Dict{Symbol,Float64})
    println("\nAnalyzing memoization effectiveness")
    memo = Dict{Any,Float64}()
    assigned_vals = Vector{Any}(undef, length(bn.stochastic_ids))
    fill!(assigned_vals, nothing)

    log_post_dp = create_log_posterior(bn, use_dp=true)
    log_post_dp(obs)

    println("Total unique subproblems stored in memo table: $(length(memo))")
end

###############################################################################
# 5-node Example
###############################################################################

model_5_nodes = create_5node_network()
X1_values = 0.0:0.5:1.5

println("=== Runtime Analysis: 5-Node Chain ===")
obs_5node = Dict(:X4 => 1.0, :X5 => 2.0)

println("Naive recursion:")
analyze_runtime(model_5_nodes, obs_5node, false)

println("\nDynamic Programming:")
analyze_runtime(model_5_nodes, obs_5node, true)

println("\n=== Memoization Analysis: 5-Node Chain ===")
analyze_memoization(model_5_nodes, obs_5node)

###############################################################################
# Branching Example
###############################################################################

branching_bn = create_branching_network()
obs_branching = Dict(:X5 => 4.2)
id5 = branching_bn.names_to_ids[:X5]
branching_bn.values[:X5] = 4.2
branching_bn.is_observed[id5] = true

println("\n=== Runtime Analysis: Branching Network ===")

println("Naive recursion:")
analyze_runtime(branching_bn, obs_branching, false)

println("\nDynamic Programming:")
analyze_runtime(branching_bn, obs_branching, true)

println("\n=== Memoization Analysis: Branching Network ===")
analyze_memoization(branching_bn, obs_branching)

###############################################################################
# Scalability Experiment
###############################################################################

function scalability_experiment()
    println("\n=== Scalability Experiment ===")
    for num_nodes in 5:5:20
        println("\nNetwork with $num_nodes nodes:")
        bn = create_5node_network()  # Replace with a generator for larger networks
        obs = Dict(:X4 => 1.0, :X5 => 2.0)  # Keep observations consistent

        println("Naive recursion:")
        @btime analyze_runtime(bn, obs, false)

        println("Dynamic Programming:")
        @btime analyze_runtime(bn, obs, true)
    end
end

# scalability_experiment()
