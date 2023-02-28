using Graphs, JuliaBUGS, Distributions
using JuliaBUGS:
    CollectVariables,
    DependencyGraph,
    NodeFunctions,
    ArrayElement,
    ArraySlice,
    ArrayVariable,
    program!

##
model_def = @bugsast begin
    for i in 1:N
        for j in 1:T
            Y[i, j] ~ dnorm(mu[i, j], tau_c)
            mu[i, j] = alpha[i] + beta[i] * (x[j] - xbar)
        end
        alpha[i] ~ dnorm(alpha_c, alpha_tau)
        beta[i] ~ dnorm(beta_c, beta_tau)
    end
    tau_c ~ dgamma(0.001, 0.001)
    sigma = 1 / sqrt(tau_c)
    alpha_c ~ dnorm(0.0, 1.0E-6)
    alpha_tau ~ dgamma(0.001, 0.001)
    beta_c ~ dnorm(0.0, 1.0E-6)
    beta_tau ~ dgamma(0.001, 0.001)
    alpha0 = alpha_c - xbar * beta_c
end

## original data
x = [8.0, 15.0, 22.0, 29.0, 36.0]
xbar = 22
N = 30
T = 5
Y = [
    151 199 246 283 320
    145 199 249 293 354
    147 214 263 312 328
    155 200 237 272 297
    135 188 230 280 323
    159 210 252 298 331
    141 189 231 275 305
    159 201 248 297 338
    177 236 285 350 376
    134 182 220 260 296
    160 208 261 313 352
    143 188 220 273 314
    154 200 244 289 325
    171 221 270 326 358
    163 216 242 281 312
    160 207 248 288 324
    142 187 234 280 316
    156 203 243 283 317
    157 212 259 307 336
    152 203 246 286 321
    154 205 253 298 334
    139 190 225 267 302
    146 191 229 272 302
    157 211 250 285 323
    132 185 237 286 331
    160 207 257 303 345
    169 216 261 295 333
    157 205 248 289 316
    137 180 219 258 291
    153 200 244 286 324
]

data = Dict(:x => x, :xbar => xbar, :Y => Y, :N => N, :T => T)

alpha = ones(Integer, 30) .* 250
beta = ones(Integer, 30) .* 6
alpha_c = 150
beta_c = 10
tau_c = 1
alpha_tau = 1
beta_tau = 1

initializations = Dict(
    :alpha => alpha,
    :beta => beta,
    :alpha_c => alpha_c,
    :beta_c => beta_c,
    :tau_c => tau_c,
    :alpha_tau => alpha_tau,
    :beta_tau => beta_tau,
)

## simplified data
x = [8.0, 15.0]
xbar = 22
N = 2
T = 2
Y = [
    151 199
    145 199
]

data = Dict(:x => x, :xbar => xbar, :Y => Y, :N => N, :T => T)

alpha = ones(Integer, T) .* 250
beta = ones(Integer, T) .* 6
alpha_c = 150
beta_c = 10
tau_c = 1
alpha_tau = 1
beta_tau = 1

initializations = Dict(
    :alpha => alpha,
    :beta => beta,
    :alpha_c => alpha_c,
    :beta_c => beta_c,
    :tau_c => tau_c,
    :alpha_tau => alpha_tau,
    :beta_tau => beta_tau,
)

##

vars, array_map, var_types = program!(CollectVariables(), model_def, data)
dep_graph = program!(DependencyGraph(vars, array_map), model_def, data)
node_args, node_functions, link_functions = program!(NodeFunctions(vars, array_map), model_def, data)

open("/home/sunxd/JuliaBUGS.jl/notebooks/output.jl","w+") do io
    println(io, node_functions)
end

open("/home/sunxd/JuliaBUGS.jl/notebooks/output2.jl","w+") do io
    # println(io, node_args)
    println(io, tt)
end



m = JuliaBUGS.BUGSModel(vars, array_map, var_types, dep_graph, node_args, node_functions, link_functions)

t = JuliaBUGS.Trace(m, data, initializations)
f_t = JuliaBUGS.flatten(t)
t == JuliaBUGS.unflatten(t, f_t)

bs = JuliaBUGS.get_bijectors(m, t)
ds = JuliaBUGS.get_distribution_type(m, t)

JuliaBUGS.transform_and_flatten(t, bs)

l, tt = JuliaBUGS.logdensity(m, t)

##
model_def = @bugsast begin
    a ~ dnorm(0, 1)
    b ~ dnorm(0, a)
    for i in 1:N
        c[i] ~ dnorm(a, b)
    end
    g = b * 2 + a
    d[1:3] ~ dmnorm(e[1:3], f[1:3, 1:3])
end

data = Dict(:N => 3, :f => [1 0 0; 0 1 0; 0 0 1], :e => [1, 2, 3])
initializations = Dict(:a => 1, :b => 2, :c => [1, 2, 3], :d => [4, 5, 6])


# data
x = [8.0, 15.0, 22.0, 29.0, 36.0]
xbar = 22
N = 30
T = 5
Y = [
    151 199 246 283 320
    145 199 249 293 354
    147 214 263 312 328
    155 200 237 272 297
    135 188 230 280 323
    159 210 252 298 331
    141 189 231 275 305
    159 201 248 297 338
    177 236 285 350 376
    134 182 220 260 296
    160 208 261 313 352
    143 188 220 273 314
    154 200 244 289 325
    171 221 270 326 358
    163 216 242 281 312
    160 207 248 288 324
    142 187 234 280 316
    156 203 243 283 317
    157 212 259 307 336
    152 203 246 286 321
    154 205 253 298 334
    139 190 225 267 302
    146 191 229 272 302
    157 211 250 285 323
    132 185 237 286 331
    160 207 257 303 345
    169 216 261 295 333
    157 205 248 289 316
    137 180 219 258 291
    153 200 244 286 324
]

# parameters
alpha = [250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250, 250]
beta = [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
alpha_c = 150
beta_c = 10
tau_c = 1
alpha_tau = 1
beta_tau = 1;

dnorm(μ, τ) = Normal(μ, 1 / sqrt(τ))
dgamma(a, b) = Gamma(a, 1 / b) 

struct Rats
    x
    xbar
    Y
    N::Int
    T::Int
end

function (problem::Rats)(vars)
    @unpack x, xbar, Y, N, T = problem
    @unpack alpha, beta, tau_c, alpha_c, alpha_tau, beta_c, beta_tau = vars

    mu = ones(N, T)
    for i in 1:N
        for j in 1:T
            mu[i, j] = alpha[i] + beta[i] * (x[j] - xbar)
        end
    end
    
    loglikelihood = 0.0
    for i in 1:N
        for j in 1:T
            loglikelihood += logpdf(dnorm(mu[i, j], tau_c), Y[i, j])
        end
    end
    for i in 1:N
        loglikelihood += logpdf(dnorm(alpha_c, alpha_tau), alpha[i])
        loglikelihood += logpdf(dnorm(beta_c, beta_tau), beta[i])
    end

    loglikelihood += logpdf(dgamma(0.001, 0.001), tau_c)
    loglikelihood += logpdf(dnorm(0.0, 1.0E-6), alpha_c)
    loglikelihood += logpdf(dgamma(0.001, 0.001), alpha_tau)
    loglikelihood += logpdf(dnorm(0.0, 1.0E-6), beta_c)
    loglikelihood += logpdf(dgamma(0.001, 0.001), beta_tau)
    return loglikelihood, mu
end