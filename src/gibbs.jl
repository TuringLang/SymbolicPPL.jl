struct Gibbs{ODT<:OrderedDict,ADT<:ADTypes.AbstractADType} <: AbstractMCMC.AbstractSampler
    sampler_map::ODT
    adtype::ADT
end

function Gibbs(sampler_map::ODT) where {ODT<:OrderedDict}
    return Gibbs(sampler_map, ADTypes.AutoReverseDiff(; compile=false))
end

function verify_sampler_map(model::BUGSModel, sampler_map::OrderedDict)
    all_variables_in_keys = Set(vcat(keys(sampler_map)...))
    model_parameters = Set(model.parameters)

    # Check for extra variables in sampler_map that are not in model parameters
    extra_variables = setdiff(all_variables_in_keys, model_parameters)
    if !isempty(extra_variables)
        throw(
            ArgumentError(
                "Sampler map contains variables not in the model: $extra_variables"
            ),
        )
    end

    # Check for model parameters not covered by sampler_map
    left_over_variables = setdiff(model_parameters, all_variables_in_keys)
    if !isempty(left_over_variables)
        throw(
            ArgumentError(
                "Some model parameters are not covered by the sampler map: $left_over_variables",
            ),
        )
    end

    return true
end

"""
    _create_submodel_for_gibbs_sampling(model::BUGSModel, variables_to_update::Vector{<:VarName})

Internal function to create a conditioned model for Gibbs sampling. This is different from conditioning, because conditioning
only marks a model parameter as observation, while the function effectively creates a sub-model with only the variables in the
Markov blanket of the variables that are being updated.
"""
function _create_submodel_for_gibbs_sampling(model::BUGSModel, variables_to_update::VarName)
    return _create_submodel_for_gibbs_sampling(model, [variables_to_update])
end
function _create_submodel_for_gibbs_sampling(
    model::BUGSModel, variables_to_update::NTuple{N,<:VarName}
) where {N}
    return _create_submodel_for_gibbs_sampling(model, collect(variables_to_update))
end
function _create_submodel_for_gibbs_sampling(
    model::BUGSModel, variables_to_update::Vector{<:VarName}
)
    _markov_blanket = markov_blanket(model.g, variables_to_update)
    mb_without_variables_to_update = setdiff(_markov_blanket, variables_to_update)
    model_parameters_in_mb = filter(
        v -> is_stochastic(model.g, v) && !is_observation(model.g, v),
        mb_without_variables_to_update,
    )
    sub_model = BUGSModel(
        model; parameters=variables_to_update, sorted_nodes=collect(_markov_blanket)
    )
    return condition(sub_model, collect(model_parameters_in_mb))
end

struct GibbsState{T,S,C}
    evaluation_env::T
    sub_model_cache::C
    sub_states::S
end

"""
    gibbs_internal(rng, sub_model, sampler, state, adtype)

Internal function to perform Gibbs sampling. This function should first update the 
sampler state with the correct log density and then do a single step of the sampler.
It should return the `evaluation_env` and the updated sampler state.
"""
function gibbs_internal end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensitymodel::AbstractMCMC.LogDensityModel{<:BUGSModel},
    sampler::Gibbs;
    model=logdensitymodel.logdensity,
    kwargs...,
)
    verify_sampler_map(model, sampler.sampler_map)

    submodel_cache = Vector{BUGSModel}(undef, length(sampler.sampler_map))
    sub_states = Any[]
    for (i, variable_group) in enumerate(keys(sampler.sampler_map))
        local_sampler = sampler.sampler_map[variable_group]
        submodel = _create_submodel_for_gibbs_sampling(model, variable_group)
        if local_sampler isa MHFromPrior
            evaluation_env, logp = evaluate!!(submodel, DefaultContext())
            state = MHState(evaluation_env, logp)
        else
            sublogdensitymodel = AbstractMCMC.LogDensityModel(
                LogDensityProblemsAD.ADgradient(sampler.adtype, submodel)
            )
            _, state = AbstractMCMC.step(rng, sublogdensitymodel, local_sampler)
        end
        submodel_cache[i] = submodel
        push!(sub_states, state)
    end

    return getparams(model),
    GibbsState(model.evaluation_env, submodel_cache, map(identity, sub_states))
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    logdensitymodel::AbstractMCMC.LogDensityModel{<:BUGSModel},
    sampler::Gibbs,
    state::GibbsState;
    model=logdensitymodel.logdensity,
    kwargs...,
)
    evaluation_env = state.evaluation_env
    for (i, vs) in enumerate(keys(sampler.sampler_map))
        sub_model = BangBang.setproperty!!(
            state.sub_model_cache[i], :evaluation_env, evaluation_env
        )
        evaluation_env, new_sub_state = gibbs_internal(
            rng, sub_model, sampler.sampler_map[vs], state.sub_states[i], sampler.adtype
        )
        state.sub_states[i] = new_sub_state
    end
    model = BangBang.setproperty!!(model, :evaluation_env, evaluation_env)
    return getparams(model),
    GibbsState(evaluation_env, state.sub_model_cache, state.sub_states)
end

struct MHFromPrior <: AbstractMCMC.AbstractSampler end

struct MHState{T}
    evaluation_env::T
    logp::Float64
end

function gibbs_internal(
    rng::Random.AbstractRNG,
    sub_model::BUGSModel,
    ::MHFromPrior,
    state::MHState,
    adtype::ADTypes.AbstractADType,
)
    evaluation_env, logp = evaluate!!(sub_model, DefaultContext())
    proposed_evaluation_env, logp_proposed = evaluate!!(sub_model, SamplingContext())

    if logp_proposed - logp > log(rand(rng))
        evaluation_env = proposed_evaluation_env
        logp = logp_proposed
    end

    return evaluation_env, MHState(evaluation_env, logp)
end
