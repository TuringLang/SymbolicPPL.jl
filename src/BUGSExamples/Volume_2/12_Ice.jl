name = "Ice: non-parametric smoothing in an age-cohort model"

model_def = @bugs begin
    for i in 1:I
        cases[i] ~ dpois(mu[i])
        mu[i] = expr(log(pyr[i]) + alpha[age[i]] + beta[year[i]])
    end
    betamean[1] = 2 * beta[2] - beta[3]
    Nneighs[1] = 1
    betamean[2] = (2 * beta[1] + 4 * beta[3] - beta[4]) / 5
    Nneighs[2] = 5
    for k in 3:(K - 2)
        betamean[k] = (4 * beta[k - 1] + 4 * beta[k + 1] - beta[k - 2] - beta[k + 2]) / 6
        Nneighs[k] = 6
    end
    betamean[K - 1] = (2 * beta[K] + 4 * beta[K - 2] - beta[K - 3]) / 5
    Nneighs[K - 1] = 5
    betamean[K] = 2 * beta[K - 1] - beta[K - 2]
    Nneighs[K] = 1
    for k in 1:K
        betaprec[k] = Nneighs[k] * tau
    end
    for k in 1:K
        beta[k] ~ dnorm(betamean[k], betaprec[k])
        logRR[k] = beta[k] - beta[5]
        var"tau.like"[k] = Nneighs[k] * beta[k] * (beta[k] - betamean[k])
    end
    alpha[1] = 0.0
    for j in 2:Nage
        alpha[j] ~ dnorm(0, 1.0E-6)
    end
    d = 0.0001 + sum(var"tau.like"[:]) / 2
    r = 0.0001 + K / 2
    tau ~ dgamma(r, d)
    sigma = 1 / sqrt(tau)
end

original = """
model {
    for (i in 1:I) {
        cases[i] ~ dpois(mu[i])
        log(mu[i]) <- log(pyr[i]) + alpha[age[i]] + beta[year[i]]
    }
    betamean[1] <- 2 * beta[2] - beta[3]
    Nneighs[1] <- 1
    betamean[2] <- (2 * beta[1] + 4 * beta[3] - beta[4]) / 5
    Nneighs[2] <- 5
    for (k in 3 : K - 2) {
        betamean[k] <- (4 * beta[k - 1] + 4 * beta[k + 1] - beta[k - 2] - beta[k + 2]) / 6
        Nneighs[k] <- 6
    }
    betamean[K - 1] <- (2 * beta[K] + 4 * beta[K - 2] - beta[K - 3]) / 5
    Nneighs[K - 1] <- 5
    betamean[K] <- 2 * beta[K - 1] - beta[K - 2]
    Nneighs[K] <- 1
    for (k in 1 : K) {
        betaprec[k] <- Nneighs[k] * tau
    }
    for (k in 1 : K) {
        beta[k] ~ dnorm(betamean[k], betaprec[k])
        logRR[k] <- beta[k] - beta[5]
        tau.like[k] <- Nneighs[k] * beta[k] * (beta[k] - betamean[k])
    }
    alpha[1] <- 0.0
    for (j in 2 : Nage) {
        alpha[j] ~ dnorm(0, 1.0E-6)
    }
    d <- 0.0001 + sum(tau.like[]) / 2
    r <- 0.0001 + K / 2
    tau ~ dgamma(r, d)
    sigma <- 1 / sqrt(tau)
}
"""

data = (
    age = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3,
        3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6,
        6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9,
        9, 9, 9, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11,
        11, 11, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13],
    year = [6, 7, 8, 9, 10, 11, 6, 7, 8, 9, 10, 11, 5, 6, 7, 8, 9,
        10, 5, 6, 7, 8, 9, 10, 4, 5, 6, 7, 8, 9, 4, 5, 6, 7,
        8, 9, 3, 4, 5, 6, 7, 8, 3, 4, 5, 6, 7, 8, 2,
        3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4,
        5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5],
    cases = [2, 0, 1, 1, 1, 2, 0, 2, 1, 1, 5, 5, 1, 1, 3, 7, 12, 10, 6,
        11, 9, 14, 20, 14, 7, 14, 22, 25, 29, 37, 21, 11, 29, 33,
        57, 24, 15, 8, 22, 27, 38, 52, 10, 15, 22, 26, 47, 31, 8,
        11, 17, 23, 31, 38, 8, 10, 24, 30, 53, 26, 5, 3, 10, 18,
        22, 30, 1, 7, 11, 26, 32, 17, 5, 8, 17, 32, 31],
    pyr = [41380, 43650, 49810, 58105, 57105, 76380, 39615,
        42205, 48315, 56785, 55965, 33955, 29150, 38460,
        40810, 47490, 55720, 55145, 27950, 37375, 39935,
        46895, 54980, 27810, 25055, 27040, 36400, 39355,
        46280, 54350, 24040, 26290, 35480, 38725, 45595,
        25710, 22890, 23095, 25410, 34420, 37725, 44740,
        21415, 21870, 24240, 33175, 36345, 21320, 17450,
        19765, 20255, 22760, 31695, 34705, 15350, 17720,
        18280, 20850, 29600, 15635, 9965, 12850, 15015,
        15725, 18345, 26400, 8175, 11020, 13095, 14050,
        16480, 10885, 7425, 10810, 12260, 14780, 13600],
    I = 77, Nage = 13, K = 11)

inits = (tau = 1,
    alpha = [missing, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    beta = [0.05, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

inits_alternative = (tau = 1,
    alpha = [missing, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    beta = [0.5, 0.1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

reference_results = (
    var"logRR[1]" = (mean = -1.076, std = 0.241),
    var"logRR[2]" = (mean = -0.7715, std = 0.1535),
    var"logRR[3]" = (mean = -0.4719, std = 0.08124),
    var"logRR[4]" = (mean = -0.2021, std = 0.03938),
    var"logRR[6]" = (mean = 0.1582, std = 0.04448),
    var"logRR[7]" = (mean = 0.3137, std = 0.06806),
    var"logRR[8]" = (mean = 0.4699, std = 0.08578),
    var"logRR[9]" = (mean = 0.6198, std = 0.1112),
    var"logRR[10]" = (mean = 0.7852, std = 0.1433),
    var"logRR[11]" = (mean = 0.9594, std = 0.1958),
    var"sigma" = (mean = 0.05195, std = 0.03977)
)

ice = Example(name, model_def, original, data, inits, inits_alternative, reference_results)
