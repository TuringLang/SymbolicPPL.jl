name = "Stagnant: a changepoint problem and an illustration of how NOT to do MCMC!)"

model_def = @bugs begin
    for i in 1:N
        Y[i] ~ dnorm(mu[i], tau)
        mu[i] = alpha + beta[J[i]] * (x[i] - var"x.change")
        J[i] = 1 + step(x[i] - var"x.change")
    end
    tau ~ dgamma(0.001, 0.001)
    alpha ~ dnorm(0.0, 1.0E-6)
    for j in 1:2
        beta[j] ~ dnorm(0.0, 1.0E-6)
    end
    sigma = 1 / sqrt(tau)
    var"x.change" ~ dunif(x[5], x[26])
end

original = """
model {
    for (i in 1:N) {
        Y[i] ~ dnorm(mu[i], tau)
        mu[i] <- alpha + beta[J[i]] * (x[i] - x.change)
        J[i] <- 1 + step(x[i] - x.change)
    }
    tau ~ dgamma(0.001, 0.001)
    alpha ~ dnorm(0.0, 1.0E-6)
    for (j in 1:2) {
        beta[j] ~ dnorm(0.0, 1.0E-6)
    }
    sigma <- 1 / sqrt(tau)
    x.change ~ dunif(x[5], x[26])
}
"""

data = (
    Y = [1.12, 1.12, 0.99, 1.03, 0.92, 0.9, 0.81, 0.83, 0.65, 0.67,
        0.6, 0.59, 0.51, 0.44, 0.43, 0.43, 0.33, 0.3, 0.25, 0.24,
        0.13, -0.01, -0.13, -0.14, -0.3, -0.33, -0.46, -0.43, -0.65],
    x = [-1.39, -1.39, -1.08, -1.08, -0.94, -0.8, -0.63, -0.63, -0.25,
        -0.25, -0.12, -0.12, 0.01, 0.11, 0.11, 0.11, 0.25, 0.25, 0.34,
        0.34, 0.44, 0.59, 0.7, 0.7, 0.85, 0.85, 0.99, 0.99, 1.19],
    N = 29)

inits = (alpha = 0.2, beta = [-0.45, -1.0], k = 16, tau = 5)
inits_alternative = (alpha = 0.6, beta = [-0.45, -1.0], k = 8, tau = 5)

reference_results = (
    alpha = (mean = 0.537, std = 0.02569),
    beta = (mean = [-0.4184, -1.014], std = [0.01511, 0.01747]),
    sigma = (mean = 0.0221, std = 0.003271),
    x_change = (mean = 0.02597, std = 0.03245)
)

stagnant = Example(
    name, model_def, original, data, inits, inits_alternative, reference_results)