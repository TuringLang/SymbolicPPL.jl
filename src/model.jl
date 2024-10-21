# AbstractBUGSModel cannot subtype `AbstractPPL.AbstractProbabilisticProgram` (which subtypes `AbstractMCMC.AbstractModel`)
# because it will then dispatched to https://github.com/TuringLang/AbstractMCMC.jl/blob/d7c549fe41a80c1f164423c7ac458425535f624b/src/sample.jl#L81
# instead of https://github.com/TuringLang/AbstractMCMC.jl/blob/d7c549fe41a80c1f164423c7ac458425535f624b/src/logdensityproblems.jl#L90
abstract type AbstractBUGSModel end

"""
    BUGSModel

The `BUGSModel` object is used for inference and represents the output of compilation. It implements the
[`LogDensityProblems.jl`](https://github.com/tpapp/LogDensityProblems.jl) interface.
"""
struct BUGSModel{base_model_T<:Union{<:AbstractBUGSModel,Nothing},T<:NamedTuple} <:
       AbstractBUGSModel
    " Indicates whether the model parameters are in the transformed space. "
    transformed::Bool

    "The length of the parameters vector in the original (constrained) space."
    untransformed_param_length::Int
    "The length of the parameters vector in the transformed (unconstrained) space."
    transformed_param_length::Int
    "A dictionary mapping the names of the variables to their lengths in the original (constrained) space."
    untransformed_var_lengths::OrderedDict{<:VarName,Int}
    "A dictionary mapping the names of the variables to their lengths in the transformed (unconstrained) space."
    transformed_var_lengths::OrderedDict{<:VarName,Int}

    "A `NamedTuple` containing the values of the variables in the model, all the values are in the constrained space."
    evaluation_env::T
    "A vector containing the names of the model parameters (unobserved stochastic variables)."
    parameters::Vector{<:VarName}
    "A vector containing the names of all the variables in the model, sorted in topological order."
    sorted_nodes::Vector{<:VarName}

    "An instance of `BUGSGraph`, representing the dependency graph of the model."
    g::BUGSGraph

    "If not `Nothing`, the model is a conditioned model; otherwise, it's the model returned by `compile`."
    base_model::base_model_T
end

function Base.show(io::IO, m::BUGSModel)
    if m.transformed
        println(
            io,
            "BUGSModel (transformed, with dimension $(m.transformed_param_length)):",
            "\n",
        )
    else
        println(
            io,
            "BUGSModel (untransformed, with dimension $(m.untransformed_param_length)):",
            "\n",
        )
    end
    println(io, "  Model parameters:")
    println(io, "    ", join(m.parameters, ", "), "\n")
    println(io, "  Variable values:")
    return println(io, "$(m.evaluation_env)")
end

"""
    parameters(m::BUGSModel)

Return a vector of `VarName` containing the names of the model parameters (unobserved stochastic variables).
"""
parameters(m::BUGSModel) = m.parameters

"""
    variables(m::BUGSModel)

Return a vector of `VarName` containing the names of all the variables in the model.
"""
variables(m::BUGSModel) = collect(labels(m.g))

function prepare_arg_values(
    args::Tuple{Vararg{Symbol}}, evaluation_env::NamedTuple, loop_vars::NamedTuple{lvars}
) where {lvars}
    return NamedTuple{args}(Tuple(
        map(args) do arg
            if arg in lvars
                loop_vars[arg]
            else
                AbstractPPL.get(evaluation_env, @varname($arg))
            end
        end,
    ))
end

function BUGSModel(
    g::BUGSGraph,
    evaluation_env::NamedTuple,
    initial_params::NamedTuple=NamedTuple();
    is_transformed::Bool=true,
)
    sorted_nodes = VarName[label_for(g, node) for node in topological_sort(g)]
    parameters = VarName[]
    untransformed_param_length, transformed_param_length = 0, 0
    untransformed_var_lengths, transformed_var_lengths = OrderedDict{VarName,Int}(),
    OrderedDict{VarName,Int}()

    for vn in sorted_nodes
        (; is_stochastic, is_observed, node_function, node_args, loop_vars) = g[vn]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic
            value = Base.invokelatest(node_function; args...)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        elseif !is_observed
            push!(parameters, vn)
            dist = Base.invokelatest(node_function; args...)

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
                        """
                        Failed to sample from the prior distribution of $vn, consider providing 
                        initialization values for $vn or it's parents: 
                        $(collect(MetaGraphsNext.inneighbor_labels(g, vn))...).
                        """,
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
        sorted_nodes,
        g,
        nothing,
    )
end

function BUGSModel(
    model::BUGSModel;
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
        sorted_nodes,
        model.g,
        isnothing(model.base_model) ? model : model.base_model,
    )
end

"""
    initialize!(model::BUGSModel, initial_params::NamedTuple)

Initialize the model with a NamedTuple of initial values, the values are expected to be in the original space.
"""
function initialize!(model::BUGSModel, initial_params::NamedTuple)
    check_input(initial_params)
    evaluation_env = model.evaluation_env
    for vn in model.sorted_nodes
        (; node_function, node_args, loop_vars) = model.g[vn]
        args = prepare_arg_values(node_args, model.evaluation_env, loop_vars)
        if is_deterministic(model.g, vn)
            value = Base.invokelatest(node_function; args...)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        elseif is_model_parameter(model.g, vn)
            initialization = try
                AbstractPPL.get(initial_params, vn)
            catch _
                missing
            end
            if !ismissing(initialization)
                evaluation_env = BangBang.setindex!!(evaluation_env, initialization, vn)
            else
                evaluation_env = BangBang.setindex!!(
                    evaluation_env, rand(Base.invokelatest(node_function; args...)), vn
                )
            end
        end
    end
    return BangBang.setproperty!!(model, :evaluation_env, evaluation_env)
end

"""
    initialize!(model::BUGSModel, initial_params::AbstractVector)

Initialize the model with a vector of initial values, the values can be in transformed 
space if `model.transformed` is set to true.
"""
function initialize!(model::BUGSModel, initial_params::AbstractVector)
    evaluation_env, _ = AbstractPPL.evaluate!!(model, LogDensityContext(), initial_params)
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
            (; node_function, node_args, loop_vars) = model.g[v]
            args = prepare_arg_values(node_args, model.evaluation_env, loop_vars)
            dist = node_function(; args...)
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
    settrans(model::BUGSModel, bool::Bool=!(model.transformed))

The `BUGSModel` contains information for evaluation in both transformed and untransformed spaces. The `transformed` field
indicates the current "mode" of the model.

This function enables switching the "mode" of the model.
"""
function settrans(model::BUGSModel, bool::Bool=!(model.transformed))
    return BangBang.setproperty!!(model, :transformed, bool)
end

function create_sub_model(
    model::BUGSModel,
    model_parameters_in_submodel::Vector{<:VarName},
    all_variables_in_submodel::Vector{<:VarName},
)
    return BUGSModel(model, model_parameters_in_submodel, all_variables_in_submodel)
end

function AbstractPPL.condition(
    model::BUGSModel, variables_to_condition_on_and_values::Dict{<:VarName,<:Any}
)
    evaluation_env = model.evaluation_env
    for (variable, value) in pairs(variables_to_condition_on_and_values)
        evaluation_env = BangBang.setindex!!(evaluation_env, value, variable)
    end
    return AbstractPPL.condition(
        model, collect(keys(variables_to_condition_on_and_values)), evaluation_env
    )
end
function AbstractPPL.condition(
    model::BUGSModel,
    variables_to_condition_on::Vector{<:VarName},
    evaluation_env::NamedTuple=model.evaluation_env,
)
    BangBang.setproperty!!(model, :evaluation_env, evaluation_env)
    for vn in variables_to_condition_on
        if !model.g[vn].is_stochastic
            throw(
                ArgumentError(
                    "$vn is not a stochastic variable, conditioning on it is not supported"
                ),
            )
        elseif model.g[vn].is_observed
            @warn "$vn is already an observed variable, conditioning on it won't have any effect"
        else
            new_g = copy(model.g)
            new_g[vn] = BangBang.setproperty!!(model.g[vn], :is_observed, true)
            model = BangBang.setproperty!!(model, :g, new_g)
        end
    end
    return model
end

function AbstractPPL.decondition(model::BUGSModel, var_group::Vector{<:VarName})
    for vn in var_group
        if !model.g[vn].is_stochastic
            throw(
                ArgumentError(
                    "$vn is not a stochastic variable, deconditioning it is not supported"
                ),
            )
        elseif !model.g[vn].is_observed
            @warn "$vn is already treated as model parameter, deconditioning it won't have any effect"
        else
            new_g = copy(model.g)
            new_g[vn] = BangBang.setproperty!!(model.g[vn], :is_observed, false)
            model = BangBang.setproperty!!(model, :g, new_g)
        end
    end
    return model
end

"""
    DefaultContext

Use values in varinfo to compute the log joint density.
"""
struct DefaultContext <: AbstractPPL.AbstractContext end

"""
    SamplingContext

Do an ancestral sampling of the model parameters. Also accumulate log joint density.
"""
@kwdef struct SamplingContext{T<:Random.AbstractRNG} <: AbstractPPL.AbstractContext
    rng::T = Random.default_rng()
end

"""
    LogDensityContext

Use the given values to compute the log joint density.
"""
struct LogDensityContext <: AbstractPPL.AbstractContext end

function AbstractPPL.evaluate!!(model::BUGSModel, rng::Random.AbstractRNG)
    return evaluate!!(model, SamplingContext(rng))
end
function AbstractPPL.evaluate!!(model::BUGSModel, ctx::SamplingContext)
    evaluation_env = deepcopy(model.evaluation_env) # TODO: a lot of the arrays are not modified
    logp = 0.0
    for vn in model.sorted_nodes
        (; node_function, node_args, loop_vars) = model.g[vn]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic(model.g, vn)
            value = node_function(; args...)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(; args...)
            if is_observation(model.g, vn)
                value = AbstractPPL.get(evaluation_env, vn)
            else
                value = rand(ctx.rng, dist)
                evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
            end
            if model.transformed
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

function AbstractPPL.evaluate!!(model::BUGSModel)
    return AbstractPPL.evaluate!!(model, DefaultContext())
end
function AbstractPPL.evaluate!!(model::BUGSModel, ::DefaultContext)
    (; sorted_nodes, g, evaluation_env) = model
    logp = 0.0
    for vn in sorted_nodes
        (; is_stochastic, node_function, node_args, loop_vars) = g[vn]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic
            value = node_function(; args...)
            evaluation_env = setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(; args...)
            value = AbstractPPL.get(evaluation_env, vn)
            if model.transformed
                # although the values stored in `evaluation_env` are in their original space, 
                # here we behave as accepting a vector of parameters in the transformed space
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

function AbstractPPL.evaluate!!(
    model::BUGSModel, ::LogDensityContext, flattened_values::AbstractVector
)
    param_lengths = if model.transformed
        model.transformed_param_length
    else
        model.untransformed_param_length
    end

    if length(flattened_values) != param_lengths
        error(
            "The length of `flattened_values` does not match the length of the parameters in the model",
        )
    end

    var_lengths = if model.transformed
        model.transformed_var_lengths
    else
        model.untransformed_var_lengths
    end

    sorted_nodes = model.sorted_nodes
    g = model.g
    evaluation_env = deepcopy(model.evaluation_env)
    current_idx = 1
    logp = 0.0
    for vn in sorted_nodes
        (; is_stochastic, node_function, node_args, loop_vars) = g[vn]
        args = prepare_arg_values(node_args, evaluation_env, loop_vars)
        if !is_stochastic
            value = node_function(; args...)
            evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
        else
            dist = node_function(; args...)
            if vn in model.parameters
                l = var_lengths[vn]
                if model.transformed
                    b = Bijectors.bijector(dist)
                    b_inv = Bijectors.inverse(b)
                    reconstructed_value = reconstruct(
                        b_inv, dist, flattened_values[current_idx:(current_idx + l - 1)]
                    )
                    value, logjac = Bijectors.with_logabsdet_jacobian(
                        b_inv, reconstructed_value
                    )
                else
                    value = reconstruct(
                        dist, flattened_values[current_idx:(current_idx + l - 1)]
                    )
                    logjac = 0.0
                end
                current_idx += l
                logp += logpdf(dist, value) + logjac
                evaluation_env = BangBang.setindex!!(evaluation_env, value, vn)
            else
                logp += logpdf(dist, AbstractPPL.get(evaluation_env, vn))
            end
        end
    end
    return evaluation_env, logp
end
