using JuliaBUGS

using Distributions
using DifferentialEquations

using LogDensityProblems, LogDensityProblemsAD
using AbstractMCMC, AdvancedHMC
using MCMCChains

using ForwardDiff

function SIR!(
    du,  # buffer for the updated differential equation
    u,   # current state
    p,   # parameters
    t,    # current time
)
    N = 763  # population
    S, I, R = u
    β, γ = p
    du[1] = dS = -β * I * S / N
    du[2] = dI = β * I * S / N - γ * I
    return du[3] = dR = γ * I
end

JuliaBUGS.@register_primitive function solve_ode(u0, p)
    tspan = (0.0, 14.0)
    prob = DifferentialEquations.ODEProblem(SIR!, u0, tspan, p)
    sol = solve(prob; saveat=1.0)
    return sol[2, 2:15]
end

JuliaBUGS.@register_primitive function NegativeBinomial2(μ, ϕ)
    p = 1 / (1 + μ / ϕ)
    r = ϕ
    return NegativeBinomial(r, p)
end

sir_bugs_model = @bugs begin
    β ~ truncated(Normal(2, 1), 0, nothing)
    γ ~ truncated(Normal(0.4, 0.5), 0, nothing)
    ϕ⁻¹ ~ Exponential(1 / 5)
    ϕ = inv(ϕ⁻¹)

    p[1] = β
    p[2] = γ

    predicted[1:14] = solve_ode(u0[:], p[:])

    for i in 1:14
        I_data[i] ~ NegativeBinomial2(predicted[i] + 1e-5, ϕ)
    end

    # generated quantities
    R0 = β / γ
    recovery_time = 1 / γ
    infected[1:14] = predicted[:]
end

data = (
    I_data=[3, 8, 26, 76, 225, 298, 258, 233, 189, 128, 68, 29, 14, 4], u0=[762.0, 1.0, 0.0]
)
inits = (β=2, γ=0.5, ϕ⁻¹=0.2)
model = compile(sir_bugs_model, data, inits)

# use `ForwardDiff` this time
ad_model = ADgradient(:ForwardDiff, model)

n_samples = 3000;
n_adapts = 1000;
initial_θ = [1.6, 7, 1.2]

samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    n_samples;
    chain_type=Chains,
    n_adapts=n_adapts,
    init_params=initial_θ,
    discard_initial=n_adapts,
)

samples_and_stats[[:β, :γ, :ϕ⁻¹]]

# MH
using AdvancedMH, LinearAlgebra

# AdvancedMH is functional, but the bundle_samples method is not yet implemented
samples_and_stats = AbstractMCMC.sample(
    model,
    AdvancedMH.RWMH(MvNormal(zeros(3), I)), # A simple random walk proposal
    n_samples;
    chain_type=Chains,
    n_adapts=n_adapts,
    init_params=initial_θ,
    discard_initial=n_adapts,
)

# Parallel sampling with AdvancedHMC
# Start Julia with multiple threads
# julia -t 4
Threads.nthreads()
n_chains = 4
samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    AbstractMCMC.MCMCThreads(),
    n_samples,
    n_chains;
    chain_type=Chains,
    n_adapts=n_adapts,
    init_params=[initial_θ for _ in 1:n_chains],
    discard_initial=n_adapts,
)

######

# Demo 1: classic examples: rats -- details covered in the slides
rats_model = JuliaBUGS.BUGSExamples.rats.model_def;
data = JuliaBUGS.BUGSExamples.rats.data;
inits = JuliaBUGS.BUGSExamples.rats.inits[1];
model = compile(rats_model, data, inits);

# Inference
ad_model = ADgradient(:ReverseDiff, model; compile=Val(true))

n_samples = 3000;
n_adapts = 1000;

initial_θ = rand(LogDensityProblems.dimension(model))

samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    n_samples;
    chain_type=Chains,
    n_adapts=n_adapts,
    init_params=initial_θ,
    discard_initial=n_adapts,
)

######

# Distributed sampling with AdvancedHMC
# Start Julia with multiple processes
# julia -p 4
using Distributed

@everywhere begin
    using JuliaBUGS
    using Distributions
    using DifferentialEquations
    using LogDensityProblems, LogDensityProblemsAD
    using AbstractMCMC, AdvancedHMC
    using MCMCChains
    using ForwardDiff
end

# Here, we define the functions for each process. 
# A more efficient practice would be to encapsulate these functions within a module and then use `@everywhere` to import the module. 
# This ensures that the functions are accessible across all processes.
@everywhere begin
    function SIR!(
        du,  # buffer for the updated differential equation
        u,   # current state
        p,   # parameters
        t,    # current time
    )
        N = 763  # population
        S, I, R = u
        β, γ = p

        du[1] = dS = -β * I * S / N
        du[2] = dI = β * I * S / N - γ * I
        return du[3] = dR = γ * I
    end

    JuliaBUGS.@register_primitive function NegativeBinomial2(μ, ϕ)
        p = 1 / (1 + μ / ϕ)
        r = ϕ
        return NegativeBinomial(r, p)
    end

    function solve_ode(u0, p)
        tspan = (0.0, 14.0)
        prob = DifferentialEquations.ODEProblem(SIR!, u0, tspan, p)
        sol = solve(prob; saveat=1.0)
        return sol[2, 2:15]
    end
    JuliaBUGS.@register_primitive solve_ode
end

n_chains = nprocs()
samples_and_stats = AbstractMCMC.sample(
    ad_model,
    AdvancedHMC.NUTS(0.65),
    AbstractMCMC.MCMCDistributed(),
    n_samples,
    n_chains;
    chain_type=Chains,
    n_adapts=n_adapts,
    init_params=[initial_θ for _ in 1:4],
    discard_initial=n_adapts,
    progress=false, # Base.TTY creating problems in distributed setting
)
