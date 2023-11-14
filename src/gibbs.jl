struct WithinGibbs{S} <: AbstractMCMC.AbstractSampler
    sampler_map::S
end

struct MHFromPrior end

struct HMCSampler
    # m::Dict{Union{VarName,Vector{VarName}},HMC}
end

abstract type AbstractGibbsState end

struct GibbsState <: AbstractGibbsState
    varinfo
    markov_blanket_cache
    sorted_nodes_cache
end

ensure_vector(x) = x isa Union{Number,VarName} ? [x] : x

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    l_model::AbstractMCMC.LogDensityModel{BUGSModel},
    sampler::WithinGibbs{T};
    model=l_model.logdensity,
    kwargs...,
) where {T}
    vi = deepcopy(model.varinfo)
    markov_blanket_cache = Dict{Any,Any}()
    sorted_nodes_cache = Dict{Any,Any}()
    for v in model.parameters
        mb_model = JuliaBUGS.MarkovBlanketBUGSModel(model, v)
        markov_blanket_cache[v] = ensure_vector(mb_model.members)
        sorted_nodes_cache[v] = ensure_vector(mb_model.sorted_nodes)
    end
    return getparams(model, vi), GibbsState(vi, markov_blanket_cache, sorted_nodes_cache)
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    l_model::AbstractMCMC.LogDensityModel{BUGSModel},
    sampler::WithinGibbs{T},
    state::AbstractGibbsState;
    model=l_model.logdensity,
    kwargs...,
) where {T}
    vi = gibbs_steps(rng, model, sampler, state)
    return getparams(model, vi),
    GibbsState(vi, state.markov_blanket_cache, state.sorted_nodes_cache)
end

function gibbs_steps end

function gibbs_steps(
    rng::Random.AbstractRNG,
    model::BUGSModel,
    ::WithinGibbs{MHFromPrior},
    state,
    var_iterator=model.parameters,
)
    vi = state.varinfo
    for v in var_iterator
        ni = model.g[v]
        args = (; (getsym(arg) => vi[arg] for arg in ni.node_args)...)
        dist = _eval(ni.node_function_expr.args[2], args)

        transformed_original = ensure_vector(Bijectors.link(dist, vi[v]))
        transformed_proposal = ensure_vector(Bijectors.link(dist, rand(rng, dist)))

        mb_model = JuliaBUGS.MarkovBlanketBUGSModel(
            vi,
            ensure_vector(v),
            state.markov_blanket_cache[v],
            state.sorted_nodes_cache[v],
            model,
        )
        vi_proposed, logp_proposed = evaluate!!(
            mb_model, LogDensityContext(), transformed_proposal
        )
        vi, logp = evaluate!!(mb_model, LogDensityContext(), transformed_original)
        logr = logp_proposed - logp
        if logr > log(rand(rng))
            vi = vi_proposed
        end
    end
    return vi
end

function AbstractMCMC.bundle_samples(
    ts,
    logdensitymodel::AbstractMCMC.LogDensityModel{JuliaBUGS.BUGSModel},
    sampler::WithinGibbs{ST},
    state,
    ::Type{T};
    discard_initial=0,
    kwargs...,
) where {ST,T}
    return JuliaBUGS.gen_chains(
        logdensitymodel, ts, [], []; discard_initial=discard_initial, kwargs...
    )
end
