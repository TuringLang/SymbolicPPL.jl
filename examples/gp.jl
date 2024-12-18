using JuliaBUGS

using CSV, DataDeps, DataFrames

ENV["DATADEPS_ALWAYS_ACCEPT"] = true
register(
    DataDep(
        "putting",
        "Putting data from BDA",
        "http://www.stat.columbia.edu/~gelman/book/data/golf.dat",
        "fc28d83896af7094d765789714524d5a389532279b64902866574079c1a977cc",
    ),
)

fname = joinpath(datadep"putting", "golf.dat")
df = CSV.read(fname, DataFrame; delim=' ', ignorerepeated=true)

data = (d=df.distance, n=df.n, y=df.y, jitter=1e-6, N=length(df.distance))

using AbstractGPs, LogExpFunctions

model_def = @bugs begin
    v ~ Gamma(2, 1)
    l ~ Gamma(4, 1)
    f_latent[1:N] ~ gp_predict(v, l, d[1:N], jitter)
    y[1:N] ~ y_distribution(n[1:N], f_latent[1:N])
end

JuliaBUGS.@register_primitive GP with_lengthscale SEKernel

# all variables have numerical types in BUGS, so we write a function that returns the GP predictions
JuliaBUGS.@register_primitive function gp_predict(v, l, d, jitter)
    return GP(v * with_lengthscale(SEKernel(), l))(d, jitter)
end

JuliaBUGS.@register_primitive function y_distribution(n, f_latent)
    return product_distribution(Binomial.(n, logistic.(f_latent)))
end

model = compile(model_def, data)

using LogDensityProblems, LogDensityProblemsAD
using AbstractMCMC, AdvancedHMC
using MCMCChains

using ReverseDiff

ad_model = ADgradient(:ReverseDiff, model; compile=Val(true))

samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    1500;
    chain_type=Chains,
    n_adapts=500,
    discard_initial=500,
)
