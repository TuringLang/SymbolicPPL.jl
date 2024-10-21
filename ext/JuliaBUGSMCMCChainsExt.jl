module JuliaBUGSMCMCChainsExt

using AbstractMCMC: AbstractMCMC
using MCMCChains: Chains
using JuliaBUGS:
    JuliaBUGS, AbstractPPL, BUGSPrimitives, LogDensityProblems, LogDensityProblemsAD

function AbstractMCMC.bundle_samples(
    ts,
    logdensitymodel::AbstractMCMC.LogDensityModel{<:JuliaBUGS.BUGSModel},
    sampler::JuliaBUGS.Gibbs,
    state,
    ::Type{Chains};
    discard_initial=0,
    kwargs...,
)
    return JuliaBUGS.gen_chains(
        logdensitymodel, ts, [], []; discard_initial=discard_initial, kwargs...
    )
end

function get_bugsmodel(model::AbstractMCMC.LogDensityModel{<:JuliaBUGS.BUGSModel})
    return model.logdensity
end

function get_bugsmodel(
    model::AbstractMCMC.LogDensityModel{<:LogDensityProblemsAD.ADGradientWrapper}
)
    ad_wrapper = model.logdensity
    return Base.parent(ad_wrapper)::JuliaBUGS.BUGSModel
end

function JuliaBUGS.gen_chains(
    model::AbstractMCMC.LogDensityModel,
    samples,
    stats_names,
    stats_values;
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    return JuliaBUGS.gen_chains(
        get_bugsmodel(model),
        samples,
        stats_names,
        stats_values;
        discard_initial=discard_initial,
        thinning=thinning,
        kwargs...,
    )
end

function JuliaBUGS.gen_chains(
    model::JuliaBUGS.BUGSModel,
    samples,
    stats_names,
    stats_values;
    discard_initial=0,
    thinning=1,
    kwargs...,
)
    param_vars = model.parameters
    g = model.g

    generated_vars = JuliaBUGS.find_generated_quantities_variables(g)
    generated_vars = [v for v in model.sorted_nodes if v in generated_vars] # keep the order

    param_vals = []
    generated_quantities = []
    for i in axes(samples)[1]
        evaluation_env = first(
            JuliaBUGS.evaluate!!(model, JuliaBUGS.LogDensityContext(), samples[i])
        )
        push!(
            param_vals,
            [AbstractPPL.get(evaluation_env, param_var) for param_var in param_vars],
        )
        push!(
            generated_quantities,
            [
                AbstractPPL.get(evaluation_env, generated_var) for
                generated_var in generated_vars
            ],
        )
    end

    param_name_leaves = collect(
        Iterators.flatten([
            collect(varname_leaves(vn, param_vals[1][i])) for
            (i, vn) in enumerate(param_vars)
        ],),
    )
    generated_varname_leaves = collect(
        Iterators.flatten([
            collect(varname_leaves(vn, generated_quantities[1][i])) for
            (i, vn) in enumerate(generated_vars)
        ],),
    )

    # some of the values may be arrays
    flattened_param_vals = [collect(Iterators.flatten(p)) for p in param_vals]
    flattened_generated_quantities = [
        collect(Iterators.flatten(gq)) for gq in generated_quantities
    ]

    vals = [
        convert(
            Vector{Real},
            vcat(
                flattened_param_vals[i],
                flattened_generated_quantities[i],
                isempty(stats_values) ? [] : stats_values[i],
            ),
        ) for i in axes(samples)[1]
    ]

    @assert length(vals[1]) ==
        length(param_name_leaves) +
            length(generated_varname_leaves) +
            length(stats_names)

    return Chains(
        vals,
        vcat(Symbol.(param_name_leaves), Symbol.(generated_varname_leaves), stats_names),
        (
            parameters=vcat(Symbol.(param_name_leaves), Symbol.(generated_varname_leaves)),
            internals=stats_names,
        );
        start=discard_initial + 1,
        thin=thinning,
    )
end

# utils: copied from DynamicPPL

varname_leaves(vn::JuliaBUGS.VarName, ::Real) = [vn]
function varname_leaves(vn::JuliaBUGS.VarName, val::AbstractArray{<:Union{Real,Missing}})
    return (
        JuliaBUGS.VarName(vn, Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)) for
        I in CartesianIndices(val)
    )
end
function varname_leaves(vn::JuliaBUGS.VarName, val::AbstractArray)
    return Iterators.flatten(
        varname_leaves(
            JuliaBUGS.VarName(vn, Accessors.IndexLens(Tuple(I)) ∘ getoptic(vn)), val[I]
        ) for I in CartesianIndices(val)
    )
end
function varname_leaves(vn::JuliaBUGS.VarName, val::NamedTuple)
    iter = Iterators.map(keys(val)) do sym
        optic = Accessors.PropertyLens{sym}()
        varname_leaves(JuliaBUGS.VarName(vn, optic ∘ getoptic(vn)), optic(val))
    end
    return Iterators.flatten(iter)
end

end
