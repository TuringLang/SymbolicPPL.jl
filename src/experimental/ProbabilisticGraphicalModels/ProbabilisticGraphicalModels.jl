module ProbabilisticGraphicalModels

using BangBang
using Graphs
using Distributions

include("bayesnet.jl")

export BayesianNetwork,
    add_stochastic_vertex!,
    add_deterministic_vertex!,
    add_edge!,
    condition,
    condition!,
    decondition,
    decondition!,
    ancestral_sampling,
    is_conditionally_independent,
    marginal_distribution,
    eliminate_variables

end
