module JuliaBUGS

using AbstractMCMC
using AbstractPPL
using Accessors
using ADTypes
using BangBang
using Bijectors: Bijectors
using Distributions
using Graphs, MetaGraphsNext
using LinearAlgebra
using LogDensityProblems, LogDensityProblemsAD
using MacroTools
using OrderedCollections: OrderedDict
using Random
using StaticArrays

import Base: ==, hash, Symbol, size
import Distributions: truncated
import AbstractPPL: AbstractContext, evaluate!!

export @bugs
export compile, initialize!

export @varname

include("BUGSPrimitives/BUGSPrimitives.jl")
using .BUGSPrimitives

include("parser/Parser.jl")
using .Parser

include("utils.jl")
using .CompilerUtils

include("graphs.jl")
include("compiler_pass.jl")
include("model.jl")
include("logdensityproblems.jl")
include("gibbs.jl")

include("BUGSExamples/BUGSExamples.jl")

function check_input(input::NamedTuple)
    for (k, v) in pairs(input)
        if v isa AbstractArray
            if !(eltype(v) <: Union{Int,Float64,Missing})
                error(
                    "For array input, only Int, Float64, or Missing types are supported. Received: $(typeof(v)).",
                )
            end
        elseif v === missing
            error("Scalars cannot be missing. Received: $k")
        elseif !(v isa Union{Int,Float64})
            error("Scalars must be of type Int or Float64. Received: $k")
        end
    end
    return input
end
function check_input(input::Dict{KT,VT}) where {KT,VT}
    if isempty(input)
        return NamedTuple()
    end
    if KT === Symbol
        return check_input(NamedTuple(input))
    else
        ks = map(identity, collect(keys(input)))
        if eltype(ks) === Symbol
            return check_input(NamedTuple(ks, vs))
        else
            error(
                "When the input isa Dict, the keys must be of type Symbol. Received: $(typeof(ks)).",
            )
        end
    end
end
function check_input(input)
    return error("Input must be of type NamedTuple or Dict. Received: $(typeof(input)).")
end

function determine_array_sizes(model_def, data)
    pass = CollectVariables(model_def, data)
    analyze_block(pass, model_def; warn_loop_bounds=Ref(true))
    non_data_scalars, non_data_array_sizes = post_process(pass)
    return non_data_scalars, non_data_array_sizes
end

function check_repeated_assignments(model_def, data, array_sizes)
    pass = CheckRepeatedAssignments(model_def, data, array_sizes)
    analyze_block(pass, model_def)
    conflicted_scalars, conflicted_arrays = post_process(pass)
    return conflicted_scalars, conflicted_arrays
end

function compute_data_transformation(
    non_data_scalars, non_data_array_sizes, model_def, data
)
    eval_env = create_eval_env(non_data_scalars, non_data_array_sizes, data)
    has_new_val = true
    pass = DataTransformation(eval_env, false, Ref(false))
    while has_new_val
        pass.new_value_added = false
        analyze_block(pass, model_def)
        has_new_val = pass.new_value_added
    end
    return concretize_eval_env(pass.env)
end

function finish_checking_repeated_assignments(
    conflicted_scalars, conflicted_arrays, eval_env
)
    for scalar in conflicted_scalars
        if eval_env[scalar] isa Missing
            error("$scalar is assigned by both logical and stochastic variables.")
        end
    end

    for (array_name, conflict_array) in pairs(conflicted_arrays)
        missing_values = ismissing.(eval_env[array_name])
        conflicts = conflict_array .& missing_values
        if any(conflicts)
            error(
                "$(array_name)[$(join(Tuple.(findall(conflicts)), ", "))] is assigned by both logical and stochastic variables.",
            )
        end
    end
end

function create_graph(model_def, eval_env)
    pass = AddVertices(model_def, eval_env)
    analyze_block(pass, model_def)
    pass = AddEdges(pass.env, pass.g, pass.vertex_id_tracker)
    analyze_block(pass, model_def)
    return pass.g
end

function semantic_analysis(model_def, data)
    non_data_scalars, non_data_array_sizes = determine_array_sizes(model_def, data)
    conflicted_scalars, conflicted_arrays = check_repeated_assignments(
        model_def, data, non_data_array_sizes
    )
    eval_env = compute_data_transformation(
        non_data_scalars, non_data_array_sizes, model_def, data
    )
    finish_checking_repeated_assignments(conflicted_scalars, conflicted_arrays, eval_env)
    return eval_env
end

"""
    compile(model_def, data[, initial_params])

Compile the model with model definition and data. Optionally, initializations can be provided. 
If initializations are not provided, values will be sampled from the prior distributions. 
"""
function compile(model_def::Expr, data::NamedTuple, initial_params::NamedTuple=NamedTuple())
    data = check_input(data)
    eval_env = semantic_analysis(model_def, data)
    model_def = concretize_colon_indexing(model_def, eval_env)
    g = create_graph(model_def, eval_env)
    nonmissing_eval_env = NamedTuple{keys(eval_env)}(
        map(
            v -> begin
                if v === missing
                    return 0.0
                elseif v isa AbstractArray
                    if eltype(v) === Missing
                        return zeros(size(v)...)
                    elseif Missing <: eltype(v)
                        return coalesce.(v, zero(nonmissingtype(eltype(v))))
                    end
                end
                return v
            end,
            values(eval_env),
        ),
    )
    return BUGSModel(g, nonmissing_eval_env, initial_params)
end

"""
    @register_primitive(expr)

Currently, only function defined in the `BUGSPrimitives` module can be used in the model definition. 
This macro allows the user to register a user-defined function or distribution to be used in the model definition.

Example:
```julia
julia> @register_primitive function f(x) # function
    return x + 1
end

julia> JuliaBUGS.f(1)
2

julia> @register_primitive d(x) = Normal(0, x^2) # distribution

julia> JuliaBUGS.d(1)
Distributions.Normal{Float64}(μ=0.0, σ=1.0)
```
"""
macro register_primitive(expr)
    def = MacroTools.splitdef(expr)
    func_name = def[:name]
    func_expr = MacroTools.combinedef(def)
    return quote
        @eval JuliaBUGS begin
            # export $func_name
            $func_expr
        end
    end
end

"""
    @register_primitive(func)

`@register_primitive` can also be used to register function without definition.

Example
```julia
julia> f(x) = x + 1

julia> @register_primitive(f)

julia> JuliaBUGS.f(1)
2
```
"""
macro register_primitive(func::Symbol)
    return quote
        @eval JuliaBUGS begin
            $func = Main.$func
        end
    end
end
macro register_primitive(funcs::Vararg{Symbol})
    exprs = Expr(:block)
    for func in funcs
        push!(exprs.args, :($func = Main.$func))
    end
    return quote
        @eval JuliaBUGS begin
            $exprs
        end
    end
end

"""
    gen_chains

Generate a `Chains` object from the samples and statistics generated by `AbstractMCMC.sample`.
Only defined with `MCMCChains` extension.
"""
function gen_chains end

end
