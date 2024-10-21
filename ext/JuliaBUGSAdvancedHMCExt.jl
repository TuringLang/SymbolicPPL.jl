module JuliaBUGSAdvancedHMCExt

using AbstractMCMC: AbstractMCMC
using AdvancedHMC: AdvancedHMC
using MCMCChains: MCMCChains
using JuliaBUGS:
    JuliaBUGS, Accessors, ADTypes, LogDensityProblems, LogDensityProblemsAD, Random

function AbstractMCMC.bundle_samples(
    ts::Vector{<:AdvancedHMC.Transition},
    logdensitymodel,
    sampler,
    state,
    chain_type::Type{MCMCChains.Chains};
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    params = [t.z.θ for t in ts]
    stats_names = collect(keys(merge((; lp=ts[1].z.ℓπ.value), AdvancedHMC.stat(ts[1]))))
    stats_values = [
        vcat([ts[i].z.ℓπ.value..., collect(values(AdvancedHMC.stat(ts[i])))...]) for
        i in eachindex(ts)
    ]

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
    sampler::AdvancedHMC.HMC,
    state::AdvancedHMC.HMCState,
    adtype::ADTypes.AbstractADType,
)
    # update the log density in the state
    hamiltonian = AdvancedHMC.Hamiltonian(state.metric, sub_model)
    state = Accessors.@set state.transition.z = AdvancedHMC.phasepoint(
        hamiltonian, state.transition.z.θ, state.transition.z.r
    )

    logdensitymodel = AbstractMCMC.LogDensityModel(
        LogDensityProblemsAD.ADgradient(adtype, sub_model)
    )
    _, s = AbstractMCMC.step(rng, logdensitymodel, sampler, state; n_adapts=0)
    return initialize!(sub_model, s.transition.z.θ).evaluation_env, s
end

end
