# AbstractBUGSModel cannot subtype `AbstractPPL.AbstractProbabilisticProgram` (which subtypes `AbstractMCMC.AbstractModel`)
# because it will then dispatched to https://github.com/TuringLang/AbstractMCMC.jl/blob/d7c549fe41a80c1f164423c7ac458425535f624b/src/sample.jl#L81
# instead of https://github.com/TuringLang/AbstractMCMC.jl/blob/d7c549fe41a80c1f164423c7ac458425535f624b/src/logdensityproblems.jl#L90
abstract type AbstractBUGSModel end

"""
    FlattenedGraphNodeData{TNF,TNA,TV}

Pre-compute the values of the nodes in the model to avoid lookups from MetaGraph.
"""
struct FlattenedGraphNodeData{TNF,TV}
    sorted_nodes::Vector{<:VarName}
    is_stochastic_vals::Vector{Bool}
    is_observed_vals::Vector{Bool}
    node_function_vals::TNF
    loop_vars_vals::TV
end

function FlattenedGraphNodeData(
    g::BUGSGraph,
    sorted_nodes::Vector{<:VarName}=VarName[
        label_for(g, node) for node in topological_sort(g)
    ],
)
    is_stochastic_vals = Array{Bool}(undef, length(sorted_nodes))
    is_observed_vals = Array{Bool}(undef, length(sorted_nodes))
    node_function_vals = Array{Any}(undef, length(sorted_nodes))
    loop_vars_vals = Array{Any}(undef, length(sorted_nodes))
    for (i, vn) in enumerate(sorted_nodes)
        (; is_stochastic, is_observed, node_function, loop_vars) = g[vn]
        is_stochastic_vals[i] = is_stochastic
        is_observed_vals[i] = is_observed
        node_function_vals[i] = node_function
        loop_vars_vals[i] = loop_vars
    end
    return FlattenedGraphNodeData(
        sorted_nodes,
        is_stochastic_vals,
        is_observed_vals,
        map(identity, node_function_vals),
        map(identity, loop_vars_vals),
    )
end

"""
    BUGSModel

The `BUGSModel` object is used for inference and represents the output of compilation. It implements the
[`LogDensityProblems.jl`](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
struct BUGSModel{
    base_model_T<:Union{<:AbstractBUGSModel,Nothing},T<:NamedTuple,TNF,TV,data_T
} <: AbstractBUGSModel
    " Indicates whether the model parameters are in the transformed space. "
    transformed::Bool

    "The length of the parameters vector in the original (constrained) space."
    untransformed_param_length::Int
    "The length of the parameters vector in the transformed (unconstrained) space."
    transformed_param_length::Int
    "A dictionary mapping the names of the variables to their lengths in the original (constrained) space."
    untransformed_var_lengths::Dict{<:VarName,Int}
    "A dictionary mapping the names of the variables to their lengths in the transformed (unconstrained) space."
    transformed_var_lengths::Dict{<:VarName,Int}

    "A `NamedTuple` containing the values of the variables in the model, all the values are in the constrained space."
    evaluation_env::T
    "A vector containing the names of the model parameters (unobserved stochastic variables)."
    parameters::Vector{<:VarName}
    "An `FlattenedGraphNodeData` object containing pre-computed values of the nodes in the model. For each topological order, this needs to be recomputed."
    flattened_graph_node_data::FlattenedGraphNodeData{TNF,TV}

    "An instance of `BUGSGraph`, representing the dependency graph of the model."
    g::BUGSGraph

    "If not `Nothing`, the model is a conditioned model; otherwise, it's the model returned by `compile`."
    base_model::base_model_T

    # for serialization, save the original model definition and data
    model_def::Expr
    data::data_T
end

function Base.show(io::IO, model::BUGSModel)
    # Print model type and dimension
    space_type =
        model.transformed ? "transformed (unconstrained)" : "original (constrained)"
    dim = if model.transformed
        model.transformed_param_length
    else
        model.untransformed_param_length
    end
    printstyled(io, "BUGSModel"; bold=true, color=:blue)
    println(io, " (parameters are in ", space_type, " space, with dimension ", dim, "):\n")

    # Group and print parameters
    printstyled(io, "  Model parameters:\n"; bold=true, color=:yellow)
    grouped_params = Dict{Symbol,Vector{VarName}}()
    for param in model.parameters
        sym = AbstractPPL.getsym(param)
        push!(get!(grouped_params, sym, VarName[]), param)
    end
    for (sym, params) in grouped_params
        param_str = length(params) == 1 ? string(params[1]) : "$(join(params, ", "))"
        print(io, "    ")
        printstyled(io, param_str; color=:cyan)
        println(io)
    end
    println(io)

    # Print variable info
    printstyled(io, "  Variable sizes and types:\n"; bold=true, color=:yellow)
    for (name, value) in pairs(model.evaluation_env)
        type_str = if isa(value, Number)
            "type = $(typeof(value))"
        else
            "size = $(size(value)), type = $(typeof(value))"
        end
        print(io, "    ")
        printstyled(io, name; color=:cyan)
        print(io, ": ")
        printstyled(io, type_str; color=:green)
        println(io)
    end
    return nothing
end

"""
    parameters(model::BUGSModel)

Return a vector of `VarName` containing the names of the model parameters (unobserved stochastic variables).
"""
parameters(model::BUGSModel) = model.parameters

"""
    variables(model::BUGSModel)

Return a vector of `VarName` containing the names of all the variables in the model.
"""
variables(model::BUGSModel) = collect(labels(model.g))

function BUGSModel(
    g::BUGSGraph,
    evaluation_env::NamedTuple,
    model_def::Expr,
    data::NamedTuple,
    initial_params::NamedTuple=NamedTuple(),
    is_transformed::Bool=true,
)
    flattened_graph_node_data = FlattenedGraphNodeData(g)
    parameters = VarName[]
    untransformed_param_length, transformed_param_length = 0, 0
    untransformed_var_lengths, transformed_var_lengths = Dict{VarName,Int}(),
    Dict{VarName,Int}()

    for (i, vn) in enumerate(flattened_graph_node_data.sorted_nodes)
        is_stochastic = flattened_graph_node_data.is_stochastic_vals[i]
        is_observed = flattened_graph_node_data.is_observed_vals[i]
        node_function = flattened_graph_node_data.node_function_vals[i]
        loop_vars = flattened_graph_node_data.loop_vars_vals[i]

        if !is_stochastic
            value = Base.invokelatest(node_function, evaluation_env, loop_vars)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        elseif !is_observed
            push!(parameters, vn)
            dist = Base.invokelatest(node_function, evaluation_env, loop_vars)

            untransformed_var_lengths[vn] = length(dist)
            # not all distributions are defined for `Bijectors.transformed`
            transformed_var_lengths[vn] = if Bijectors.bijector(dist) == identity
                untransformed_var_lengths[vn]
            else
                length(Bijectors.transformed(dist))
            end
            untransformed_param_length += untransformed_var_lengths[vn]
            transformed_param_length += transformed_var_lengths[vn]

            initialization = try
                AbstractPPL.get(initial_params, vn)
            catch _
                missing
            end
            if !ismissing(initialization)
                evaluation_env = BangBang.setindex!!(evaluation_env, initialization, vn)
            else
                init_value = try
                    rand(dist)
                catch e
                    error(
                        "Failed to sample from the prior distribution of $vn, consider providing initialization values for $vn or it's parents: $(collect(MetaGraphsNext.inneighbor_labels(g, vn))...).",
                    )
                end
                evaluation_env = BangBang.setindex!!(evaluation_env, init_value, vn)
            end
        end
    end
    return BUGSModel(
        is_transformed,
        untransformed_param_length,
        transformed_param_length,
        untransformed_var_lengths,
        transformed_var_lengths,
        evaluation_env,
        parameters,
        flattened_graph_node_data,
        g,
        nothing,
        model_def,
        data,
    )
end

function BUGSModel(
    model::BUGSModel,
    g::BUGSGraph,
    parameters::Vector{<:VarName},
    sorted_nodes::Vector{<:VarName},
    evaluation_env::NamedTuple=model.evaluation_env,
)
    return BUGSModel(
        model.transformed,
        sum(model.untransformed_var_lengths[v] for v in parameters),
        sum(model.transformed_var_lengths[v] for v in parameters),
        model.untransformed_var_lengths,
        model.transformed_var_lengths,
        evaluation_env,
        parameters,
        FlattenedGraphNodeData(g, sorted_nodes),
        g,
        isnothing(model.base_model) ? model : model.base_model,
        model.model_def,
        model.data,
    )
end

function Serialization.serialize(s::Serialization.AbstractSerializer, model::BUGSModel)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    Serialization.serialize(s, typeof(model))
    Serialization.serialize(s, model.transformed)
    Serialization.serialize(s, model.model_def)
    Serialization.serialize(s, model.data)
    Serialization.serialize(s, model.evaluation_env)
    return nothing
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{<:BUGSModel})
    transformed = Serialization.deserialize(s)
    model_def = Serialization.deserialize(s)
    data = Serialization.deserialize(s)
    evaluation_env = Serialization.deserialize(s)
    # use evaluation_env as initialization to restore the values
    model = compile(model_def, data, evaluation_env)
    return settrans(model, transformed)
end

"""
    initialize!(model::BUGSModel, initial_params::NamedTuple)

Initialize the model with a NamedTuple of initial values, the values are expected to be in the original space.
"""
function initialize!(model::BUGSModel, initial_params::NamedTuple)
    check_input(initial_params)
    for (i, vn) in enumerate(model.flattened_graph_node_data.sorted_nodes)
        is_stochastic = model.flattened_graph_node_data.is_stochastic_vals[i]
        is_observed = model.flattened_graph_node_data.is_observed_vals[i]
        node_function = model.flattened_graph_node_data.node_function_vals[i]
        loop_vars = model.flattened_graph_node_data.loop_vars_vals[i]
        if !is_stochastic
            value = Base.invokelatest(node_function, model.evaluation_env, loop_vars)
            BangBang.@set!! model.evaluation_env = setindex!!(
                model.evaluation_env, value, vn
            )
        elseif !is_observed
            initialization = try
                AbstractPPL.get(initial_params, vn)
            catch _
                missing
            end
            if !ismissing(initialization)
                BangBang.@set!! model.evaluation_env = setindex!!(
                    model.evaluation_env, initialization, vn
                )
            else
                BangBang.@set!! model.evaluation_env = setindex!!(
                    model.evaluation_env,
                    rand(Base.invokelatest(node_function, model.evaluation_env, loop_vars)),
                    vn,
                )
            end
        end
    end
    return model
end

"""
    initialize!(model::BUGSModel, initial_params::AbstractVector)

Initialize the model with a vector of initial values, the values can be in transformed space if `model.transformed` is set to true.
"""
function initialize!(model::BUGSModel, initial_params::AbstractVector)
    evaluation_env, _ = AbstractPPL.evaluate!!(model, initial_params)
    return BangBang.setproperty!!(model, :evaluation_env, evaluation_env)
end

"""
    getparams(model::BUGSModel)

Extract the parameter values from the model as a flattened vector, in an order consistent with
the what `LogDensityProblems.logdensity` expects.
"""
function getparams(model::BUGSModel)
    param_length = if model.transformed
        model.transformed_param_length
    else
        model.untransformed_param_length
    end

    param_vals = Vector{Float64}(undef, param_length)
    pos = 1
    for v in model.parameters
        if !model.transformed
            val = AbstractPPL.get(model.evaluation_env, v)
            len = model.untransformed_var_lengths[v]
            if val isa AbstractArray
                param_vals[pos:(pos + len - 1)] .= vec(val)
            else
                param_vals[pos] = val
            end
        else
            (; node_function, loop_vars) = model.g[v]
            dist = node_function(model.evaluation_env, loop_vars)
            transformed_value = Bijectors.transform(
                Bijectors.bijector(dist), AbstractPPL.get(model.evaluation_env, v)
            )
            len = model.transformed_var_lengths[v]
            if transformed_value isa AbstractArray
                param_vals[pos:(pos + len - 1)] .= vec(transformed_value)
            else
                param_vals[pos] = transformed_value
            end
        end
        pos += len
    end
    return param_vals
end

"""
    getparams(T::Type{<:AbstractDict}, model::BUGSModel)

Extract the parameter values from the model into a dictionary of type T.
If model.transformed is true, returns parameters in transformed space.
"""
function getparams(T::Type{<:AbstractDict}, model::BUGSModel)
    d = T()
    for v in model.parameters
        value = AbstractPPL.get(model.evaluation_env, v)
        if !model.transformed
            d[v] = value
        else
            (; node_function, loop_vars) = model.g[v]
            dist = node_function(model.evaluation_env, loop_vars)
            d[v] = Bijectors.transform(Bijectors.bijector(dist), value)
        end
    end
    return d
end

"""
    settrans(model::BUGSModel, bool::Bool=!(model.transformed))

The `BUGSModel` contains information for evaluation in both transformed and untransformed spaces. The `transformed` field
indicates the current "mode" of the model.

This function enables switching the "mode" of the model.
"""
function settrans(model::BUGSModel, bool::Bool=!(model.transformed))
    return BangBang.setproperty!!(model, :transformed, bool)
end

function AbstractPPL.condition(
    model::BUGSModel,
    d::Dict{<:VarName,<:Any},
    sorted_nodes=Nothing, # support cached sorted Markov blanket nodes
)
    new_evaluation_env = deepcopy(model.evaluation_env)
    for (p, value) in d
        new_evaluation_env = setindex!!(new_evaluation_env, value, p)
    end
    return AbstractPPL.condition(
        model, collect(keys(d)), new_evaluation_env; sorted_nodes=sorted_nodes
    )
end

function AbstractPPL.condition(
    model::BUGSModel,
    var_group::Vector{<:VarName},
    evaluation_env::NamedTuple=model.evaluation_env,
    sorted_nodes=Nothing,
)
    check_var_group(var_group, model)
    new_parameters = setdiff(model.parameters, var_group)

    sorted_blanket_with_vars = if sorted_nodes isa Nothing
        model.flattened_graph_node_data.sorted_nodes
    else
        filter(
            vn -> vn in union(markov_blanket(model.g, new_parameters), new_parameters),
            model.flattened_graph_node_data.sorted_nodes,
        )
    end

    g = copy(model.g)
    for vn in sorted_blanket_with_vars
        if vn in new_parameters
            continue
        end
        ni = g[vn]
        if ni.is_stochastic && !ni.is_observed
            ni = @set ni.is_observed = true
            g[vn] = ni
        end
    end

    new_model = BUGSModel(
        model, g, new_parameters, sorted_blanket_with_vars, evaluation_env
    )
    return BangBang.setproperty!!(new_model, :g, g)
end

function AbstractPPL.decondition(model::BUGSModel, var_group::Vector{<:VarName})
    check_var_group(var_group, model)
    base_model = model.base_model isa Nothing ? model : model.base_model

    new_parameters = [
        v for v in base_model.flattened_graph_node_data.sorted_nodes if
        v in union(model.parameters, var_group)
    ] # keep the order

    markov_blanket_with_vars = union(
        markov_blanket(base_model.g, new_parameters), new_parameters
    )
    sorted_blanket_with_vars = filter(
        vn -> vn in markov_blanket_with_vars,
        base_model.flattened_graph_node_data.sorted_nodes,
    )

    new_model = BUGSModel(
        model, model.g, new_parameters, sorted_blanket_with_vars, base_model.evaluation_env
    )
    evaluate_env, _ = evaluate!!(new_model)
    return BangBang.setproperty!!(new_model, :evaluation_env, evaluate_env)
end

function check_var_group(var_group::Vector{<:VarName}, model::BUGSModel)
    non_vars = filter(var -> var ∉ labels(model.g), var_group)
    logical_vars = filter(var -> !model.g[var].is_stochastic, var_group)
    isempty(non_vars) || error("Variables $(non_vars) are not in the model")
    return isempty(logical_vars) || error(
        "Variables $(logical_vars) are not stochastic variables, conditioning on them is not supported",
    )
end

function AbstractPPL.evaluate!!(rng::Random.AbstractRNG, model::BUGSModel; sample_all=true)
    logp = 0.0
    evaluation_env = deepcopy(model.evaluation_env)
    for (i, vn) in enumerate(model.flattened_graph_node_data.sorted_nodes)
        is_stochastic = model.flattened_graph_node_data.is_stochastic_vals[i]
        is_observed = model.flattened_graph_node_data.is_observed_vals[i]
        node_function = model.flattened_graph_node_data.node_function_vals[i]
        loop_vars = model.flattened_graph_node_data.loop_vars_vals[i]
        if_sample = sample_all || !is_observed # also sample if not observed, only sample conditioned variables if sample_all is true
        if !is_stochastic
            value = node_function(evaluation_env, loop_vars)
            evaluation_env = setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(evaluation_env, loop_vars)
            if if_sample
                value = rand(rng, dist) # just sample from the prior
            else
                value = AbstractPPL.get(evaluation_env, vn)
            end
            if model.transformed
                # see below for why we need to transform the value
                value_transformed = Bijectors.transform(Bijectors.bijector(dist), value)
                logp +=
                    Distributions.logpdf(dist, value) + Bijectors.logabsdetjac(
                        Bijectors.inverse(Bijectors.bijector(dist)), value_transformed
                    )
            else
                logp += Distributions.logpdf(dist, value)
            end
            evaluation_env = setindex!!(evaluation_env, value, vn)
        end
    end
    return evaluation_env, logp
end

function AbstractPPL.evaluate!!(model::BUGSModel)
    logp = 0.0
    evaluation_env = deepcopy(model.evaluation_env)
    for (i, vn) in enumerate(model.flattened_graph_node_data.sorted_nodes)
        is_stochastic = model.flattened_graph_node_data.is_stochastic_vals[i]
        node_function = model.flattened_graph_node_data.node_function_vals[i]
        loop_vars = model.flattened_graph_node_data.loop_vars_vals[i]
        if !is_stochastic
            value = node_function(evaluation_env, loop_vars)
            evaluation_env = setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(evaluation_env, loop_vars)
            value = AbstractPPL.get(evaluation_env, vn)
            if model.transformed
                # although the values stored in `evaluation_env` are in their original space, 
                # here we behave as accepting a vector of parameters in the transformed space
                # this is so that we have consistent logp values between
                # (1) set values in original space then evaluate (2) directly evaluate with the values in transformed space 
                value_transformed = Bijectors.transform(Bijectors.bijector(dist), value)
                logp +=
                    Distributions.logpdf(dist, value) + Bijectors.logabsdetjac(
                        Bijectors.inverse(Bijectors.bijector(dist)), value_transformed
                    )
            else
                logp += Distributions.logpdf(dist, value)
            end
        end
    end
    return evaluation_env, logp
end

function AbstractPPL.evaluate!!(model::BUGSModel, flattened_values::AbstractVector)
    evaluation_env, (logprior, loglikelihood, tempered_logjoint) = _tempered_evaluate!!(
        model, flattened_values; temperature=1.0
    )
    return evaluation_env, tempered_logjoint
end

"""
    _tempered_evaluate!!(model::BUGSModel, flattened_values::AbstractVector; temperature=1.0)

Evaluating the model with the given model parameter values, returns updated evaluation environment 
and a NamedTuple of logprior, loglikelihood and tempered logjoint (where tempered logjoint is the logjoint 
whose loglikelihood component scaled by the given temperature).
"""
function _tempered_evaluate!!(
    model::BUGSModel, flattened_values::AbstractVector; temperature=1.0
)
    var_lengths = if model.transformed
        model.transformed_var_lengths
    else
        model.untransformed_var_lengths
    end

    evaluation_env = deepcopy(model.evaluation_env)
    current_idx = 1
    logprior, loglikelihood = 0.0, 0.0
    for (i, vn) in enumerate(model.flattened_graph_node_data.sorted_nodes)
        is_stochastic = model.flattened_graph_node_data.is_stochastic_vals[i]
        is_observed = model.flattened_graph_node_data.is_observed_vals[i]
        node_function = model.flattened_graph_node_data.node_function_vals[i]
        loop_vars = model.flattened_graph_node_data.loop_vars_vals[i]
        if !is_stochastic
            value = node_function(evaluation_env, loop_vars)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(evaluation_env, loop_vars)
            if !is_observed
                l = var_lengths[vn]
                if model.transformed
                    b = Bijectors.bijector(dist)
                    b_inv = Bijectors.inverse(b)
                    reconstructed_value = reconstruct(
                        b_inv,
                        dist,
                        view(flattened_values, current_idx:(current_idx + l - 1)),
                    )
                    value, logjac = Bijectors.with_logabsdet_jacobian(
                        b_inv, reconstructed_value
                    )
                else
                    value = reconstruct(
                        dist, view(flattened_values, current_idx:(current_idx + l - 1))
                    )
                    logjac = 0.0
                end
                current_idx += l
                logprior += logpdf(dist, value) + logjac
                evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
            else
                loglikelihood += logpdf(dist, AbstractPPL.get(evaluation_env, vn))
            end
        end
    end
    return evaluation_env,
    (
        logprior=logprior,
        loglikelihood=loglikelihood,
        tempered_logjoint=logprior + temperature * loglikelihood,
    )
end

# "incoherent" means that the values of deterministic variables may not be consistent with
# the values, because node functions are not evaluated. This function initializes only the
# stochastic unobserved variables without updating deterministic nodes.
function _incoherent_initialize!(model::BUGSModel, flattened_values::AbstractVector)
    # Get appropriate variable lengths based on transformed state
    var_lengths =
        model.transformed ? model.transformed_var_lengths : model.untransformed_var_lengths

    # Create copy of evaluation environment to avoid modifying original
    evaluation_env = deepcopy(model.evaluation_env)
    current_idx = 1
    summed_logjac = 0.0

    # Iterate through nodes in topological order
    for (i, vn) in enumerate(model.flattened_graph_node_data.sorted_nodes)
        is_stochastic = model.flattened_graph_node_data.is_stochastic_vals[i]
        is_observed = model.flattened_graph_node_data.is_observed_vals[i]

        # Skip deterministic and observed nodes
        if !is_stochastic || is_observed
            continue
        end

        # Get distribution and variable length
        node_function = model.flattened_graph_node_data.node_function_vals[i]
        loop_vars = model.flattened_graph_node_data.loop_vars_vals[i]
        dist = node_function(evaluation_env, loop_vars)
        l = var_lengths[vn]

        if model.transformed
            b = Bijectors.bijector(dist)
            b_inv = Bijectors.inverse(b)
            # Get values from flattened vector and reconstruct
            reconstructed_value = reconstruct(
                b_inv, dist, view(flattened_values, current_idx:(current_idx + l - 1))
            )
            value, logjac = Bijectors.with_logabsdet_jacobian(b_inv, reconstructed_value)
            summed_logjac += logjac # logjac is correct, but logpdf(dist, value) is not
        else
            value = reconstruct(
                dist, view(flattened_values, current_idx:(current_idx + l - 1))
            )
        end

        # Update index and environment
        current_idx += l
        evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
    end

    return BangBang.setproperty!!(model, :evaluation_env, evaluation_env), summed_logjac
end
