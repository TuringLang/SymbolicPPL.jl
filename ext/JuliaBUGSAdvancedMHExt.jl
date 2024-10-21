module JuliaBUGSAdvancedMHExt

using AbstractMCMC: AbstractMCMC
using AdvancedMH: AdvancedMH
using MCMCChains: MCMCChains
using JuliaBUGS: JuliaBUGS
using JuliaBUGS: Accessors, ADTypes, LogDensityProblems, LogDensityProblemsAD, Random

function AbstractMCMC.bundle_samples(
    ts::Vector{<:AdvancedMH.Transition},
    logdensitymodel,
    sampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    params = [t.params for t in ts]
    stats_names = [:lp]
    stats_values = [t.lp for t in ts]

    return JuliaBUGS.gen_chains(
        logdensitymodel,
        params,
        stats_names,
        stats_values;
        discard_initial=discard_initial,
        thinning=thinning,
        kwargs...,
    )
end

function JuliaBUGS.gibbs_internal(
    rng::Random.AbstractRNG,
    sub_model::JuliaBUGS.BUGSModel,
    sampler::AdvancedMH.MHSampler,
    state::AdvancedMH.Transition,
    adtype::ADTypes.AbstractADType,
)
    state = Accessors.@set state.lp = LogDensityProblems.logdensity(sub_model, state.params)

    logdensitymodel = AbstractMCMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(adtype, sub_model)
    )
    _, s = AbstractMCMC.step(rng, logdensitymodel, sampler, state)
    return JuliaBUGS.initialize!(sub_model, s.params).evaluation_env, s
end

end
