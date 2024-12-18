using JuliaBUGS

using Lux, Mooncake, Functors, FillArrays, ADTypes, DifferentiationInterface

using LinearAlgebra, Random

## data simulation

# Number of points to generate
N = 80
M = round(Int, N / 4)
rng = Random.default_rng()
Random.seed!(rng, 1234)

# Generate artificial data
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
xt1s = Array([[x1s[i] + 0.5f0; x2s[i] + 0.5f0] for i in 1:M])
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
append!(xt1s, Array([[x1s[i] - 5.0f0; x2s[i] - 5.0f0] for i in 1:M]))

x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
xt0s = Array([[x1s[i] + 0.5f0; x2s[i] - 5.0f0] for i in 1:M])
x1s = rand(rng, Float32, M) * 4.5f0;
x2s = rand(rng, Float32, M) * 4.5f0;
append!(xt0s, Array([[x1s[i] - 5.0f0; x2s[i] + 0.5f0] for i in 1:M]))

# Store all the data for later
xs = [xt1s; xt0s]
xs_hcat = Float64.(reduce(hcat, xs))
ts = [ones(2 * M); zeros(2 * M)]

alpha = 0.09
sigma = sqrt(1.0 / alpha)

## 

# Construct a neural network using Lux
nn_initial = Chain(Dense(2 => 3, tanh), Dense(3 => 2, tanh), Dense(2 => 1, σ))

# Initialize the model weights and state
ps, st = Lux.setup(rng, nn_initial)

Lux.parameterlength(nn_initial) # number of parameters in NN

function vector_to_parameters(ps_new::AbstractVector, ps::NamedTuple)
    @assert length(ps_new) == Lux.parameterlength(ps)
    i = 1
    function get_ps(x)
        z = reshape(view(ps_new, i:(i + length(x) - 1)), size(x))
        i += length(x)
        return z
    end
    return fmap(get_ps, ps)
end

const nn = StatefulLuxLayer{true}(nn_initial, nothing, st)

model_def = @bugs begin
    parameters[1:nparameters] ~ parameter_distribution(nparameters, sigma)
    predictions[1:N] = make_prediction(parameters[1:nparameters], xs[:, :])
    for i in 1:N
        ts[i] ~ Bernoulli(predictions[i])
    end
end

JuliaBUGS.@register_primitive function parameter_distribution(nparameters, sigma)
    return MvNormal(zeros(nparameters), Diagonal(abs2.(sigma .* ones(nparameters))))
end

JuliaBUGS.@register_primitive function make_prediction(parameters, xs; ps=ps, nn=nn)
    return Lux.apply(nn, f32(xs), f32(vector_to_parameters(parameters, ps)))
end

@eval JuliaBUGS begin
    ps = Main.ps
    nn = Main.nn
    Lux = Main.Lux
    f32 = Main.f32
    vector_to_parameters = Main.vector_to_parameters
end

data = (nparameters=Lux.parameterlength(nn), xs=xs_hcat, ts=ts, N=length(ts), sigma=sigma)

model = compile(model_def, data)

# ! issue with Mooncake: produce MWE
ad_model = ADgradient(AutoMooncake(; config=Mooncake.Config()), model)
ad_model = ADgradient(:ReverseDiff, model; compile=Val(true))

# sampling is slow
samples_and_stats = AbstractMCMC.sample(
    ad_model, NUTS(0.65), 3000; chain_type=Chains, n_adapts=1000, discard_initial=1000
)
