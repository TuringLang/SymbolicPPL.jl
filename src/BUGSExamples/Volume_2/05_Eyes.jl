name = "Eyes: Normal Mixture Model"

model_def = @bugs begin
    for i in 1:N
        y[i] ~ dnorm(mu[i], tau)
        mu[i] = lambda[T[i]]
        T[i] ~ dcat(P[])
    end
    P[1:2] ~ ddirich(alpha[])
    theta ~ dunif(0.0, 1000)
    lambda[2] = lambda[1] + theta
    lambda[1] ~ dnorm(0.0, 1.0E-6)
    tau ~ dgamma(0.001, 0.001)
    sigma = 1 / sqrt(tau)
end

original = """
model {
    for( i in 1 : N ) {
        y[i] ~ dnorm(mu[i], tau)
        mu[i] <- lambda[T[i]]
        T[i] ~ dcat(P[])
    }   
    P[1:2] ~ ddirich(alpha[])
    theta ~ dunif(0.0, 1000)
    lambda[2] <- lambda[1] + theta
    lambda[1] ~ dnorm(0.0, 1.0E-6)
    tau ~ dgamma(0.001, 0.001) 
    sigma <- 1 / sqrt(tau)
}
"""

data = (
    y = [
        529.0,
        530.0,
        532.0,
        533.1,
        533.4,
        533.6,
        533.7,
        534.1,
        534.8,
        535.3,
        535.4,
        535.9,
        536.1,
        536.3,
        536.4,
        536.6,
        537.0,
        537.4,
        537.5,
        538.3,
        538.5,
        538.6,
        539.4,
        539.6,
        540.4,
        540.8,
        542.0,
        542.8,
        543.0,
        543.5,
        543.8,
        543.9,
        545.3,
        546.2,
        548.8,
        548.7,
        548.9,
        549.0,
        549.4,
        549.9,
        550.6,
        551.2,
        551.4,
        551.5,
        551.6,
        552.8,
        552.9,
        553.2
    ],
    N = 48,
    alpha = [1, 1],
    T = [
        1,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        missing,
        2
    ]
)

inits = (lambda = [535, missing], theta = 5, tau = 0.1)
inits_alternative = (lambda = [100, missing], theta = 50, tau = 1)

reference_results = (
    var"P[1]" = (mean = 0.5982, std = 0.09059),
    var"P[2]" = (mean = 0.4018, std = 0.09059),
    var"lambda[1]" = (mean = 536.8, std = 0.9863),
    var"lambda[2]" = (mean = 548.8, std = 1.403),
    var"sigma" = (mean = 3.829, std = 0.7176)
)

eyes = Example(name, model_def, original, data, inits, inits_alternative, reference_results)
