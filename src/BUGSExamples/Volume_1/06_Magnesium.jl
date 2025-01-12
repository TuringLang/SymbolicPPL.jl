name = "Sensitivity to prior distributions: application to Magnesium meta-analysis"

model_def = @bugs begin
    # j indexes alternative prior distributions
    for j in 1:6
        mu[j] ~ dunif(-10, 10)
        OR[j] = exp(mu[j])

        # k indexes study number
        for k in 1:8
            theta[j, k] ~ dnorm(mu[j], var"inv.tau.sqrd"[j])
            rtx[j, k] ~ dbin(pt[j, k], nt[k])
            rtx[j, k] = rt[k]
            rcx[j, k] ~ dbin(pc[j, k], nc[k])
            rcx[j, k] = rc[k]
            pt[j, k] = logistic(theta[j, k] + phi[j, k])
            phi[j, k] = logit(pc[j, k])
            pc[j, k] ~ dunif(0, 1)
        end
    end

    # k again indexes study number
    for k in 1:8
        # log-odds ratios:
        y[k] = log(((rt[k] + 0.5) / (nt[k] - rt[k] + 0.5)) /
                   ((rc[k] + 0.5) / (nc[k] - rc[k] + 0.5)))
        # variances & precisions:
        var"sigma.sqrd"[k] = 1 / (rt[k] + 0.5) + 1 / (nt[k] - rt[k] + 0.5) +
                             1 / (rc[k] + 0.5) +
                             1 / (nc[k] - rc[k] + 0.5)
        var"prec.sqrd"[k] = 1 / var"sigma.sqrd"[k]
    end
    var"s0.sqrd" = 1 / mean(var"prec.sqrd"[1:8])

    # Prior 1: Gamma(0.001, 0.001) on inv.tau.sqrd
    var"inv.tau.sqrd"[1] ~ dgamma(0.001, 0.001)
    var"tau.sqrd"[1] = 1 / var"inv.tau.sqrd"[1]
    tau[1] = sqrt(var"tau.sqrd"[1])

    # Prior 2: Uniform(0, 50) on tau.sqrd
    var"tau.sqrd"[2] ~ dunif(0, 50)
    tau[2] = sqrt(var"tau.sqrd"[2])
    var"inv.tau.sqrd"[2] = 1 / var"tau.sqrd"[2]

    # Prior 3: Uniform(0, 50) on tau
    tau[3] ~ dunif(0, 50)
    var"tau.sqrd"[3] = tau[3] * tau[3]
    var"inv.tau.sqrd"[3] = 1 / var"tau.sqrd"[3]

    # Prior 4: Uniform shrinkage on tau.sqrd
    B0 ~ dunif(0, 1)
    var"tau.sqrd"[4] = var"s0.sqrd" * (1 - B0) / B0
    tau[4] = sqrt(var"tau.sqrd"[4])
    var"inv.tau.sqrd"[4] = 1 / var"tau.sqrd"[4]

    # Prior 5: Dumouchel on tau
    D0 ~ dunif(0, 1)
    tau[5] = sqrt(var"s0.sqrd") * (1 - D0) / D0
    var"tau.sqrd"[5] = tau[5] * tau[5]
    var"inv.tau.sqrd"[5] = 1 / var"tau.sqrd"[5]

    # Prior 6: Half-Normal on tau.sqrd
    p0 = phi(0.75) / var"s0.sqrd"
    var"tau.sqrd"[6] ~ truncated(dnorm(0, p0), 0, Inf)
    tau[6] = sqrt(var"tau.sqrd"[6])
    var"inv.tau.sqrd"[6] = 1 / var"tau.sqrd"[6]
end

original = """
model {
    #   j indexes alternative prior distributions
    for (j in 1:6) {
        mu[j] ~ dunif(-10, 10)
        odds.ratio[j] <- exp(mu[j])

        # k indexes study number
        for (k in 1:8) {
            theta[j, k] ~ dnorm(mu[j], inv.tau.sqrd[j])
            rtx[j, k] ~ dbin(pt[j, k], nt[k])
            rtx[j, k] <- rt[k]
            rcx[j, k] ~ dbin(pc[j, k], nc[k])
            rcx[j, k] <- rc[k]
            logit(pt[j, k]) <- theta[j, k] + phi[j, k]
            phi[j, k] <- logit(pc[j, k])
            pc[j, k] ~ dunif(0, 1)
        }
    }

    # k again indexes study number
    for (k in 1:8) {
        # log-odds ratios:
        y[k] <- log(((rt[k] + 0.5) / (nt[k] - rt[k] + 0.5)) / ((rc[k] + 0.5) / (nc[k] - rc[k] + 0.5)))
        # variances & precisions:
        sigma.sqrd[k] <- 1 / (rt[k] + 0.5) + 1 / (nt[k] - rt[k] + 0.5) + 1 / (rc[k] + 0.5) +
                1 / (nc[k] - rc[k] + 0.5)
        prec.sqrd[k] <- 1 / sigma.sqrd[k]
    }
    s0.sqrd <- 1 / mean(prec.sqrd[1:8])

    # Prior 1: Gamma(0.001, 0.001) on inv.tau.sqrd
    inv.tau.sqrd[1] ~ dgamma(0.001, 0.001)
    tau.sqrd[1] <- 1 / inv.tau.sqrd[1]
    tau[1] <- sqrt(tau.sqrd[1])

    # Prior 2: Uniform(0, 50) on tau.sqrd
    tau.sqrd[2] ~ dunif(0, 50)
    tau[2] <- sqrt(tau.sqrd[2])
    inv.tau.sqrd[2] <- 1 / tau.sqrd[2]

    # Prior 3: Uniform(0, 50) on tau
    tau[3] ~ dunif(0, 50)
    tau.sqrd[3] <- tau[3] * tau[3]
    inv.tau.sqrd[3] <- 1 / tau.sqrd[3]

    # Prior 4: Uniform shrinkage on tau.sqrd
    B0 ~ dunif(0, 1)
    tau.sqrd[4] <- s0.sqrd * (1 - B0) / B0
    tau[4] <- sqrt(tau.sqrd[4])
    inv.tau.sqrd[4] <- 1 / tau.sqrd[4]

    # Prior 5: Dumouchel on tau
    D0 ~ dunif(0, 1)
    tau[5] <- sqrt(s0.sqrd) * (1 - D0) / D0
    tau.sqrd[5] <- tau[5] * tau[5]
    inv.tau.sqrd[5] <- 1 / tau.sqrd[5]

    # Prior 6: Half-Normal on tau.sqrd
    p0 <- phi(0.75) / s0.sqrd
    tau.sqrd[6] ~ dnorm(0, p0)T(0,)
    tau[6] <- sqrt(tau.sqrd[6])
    inv.tau.sqrd[6] <- 1 / tau.sqrd[6]
}
"""

data = (
    rt = [1, 9, 2, 1, 10, 1, 1, 90],
    nt = [40, 135, 200, 48, 150, 59, 25, 1159],
    rc = [2, 23, 7, 1, 8, 9, 3, 118],
    nc = [36, 135, 200, 46, 148, 56, 23, 1157]
)

inits = (
    mu = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
    tau = [missing, missing, 1, missing, missing, missing],
    var"tau.sqrd" = [missing, 1, missing, missing, missing, 1],
    var"inv.tau.sqrd" = [1, missing, missing, missing, missing, missing]
)
inits_alternative = (
    mu = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    tau = [missing, missing, 0.1, missing, missing, missing],
    var"tau.sqrd" = [missing, 0.1, missing, missing, missing, 0.1],
    var"inv.tau.sqrd" = [0.1, missing, missing, missing, missing, missing]
)

reference_results = nothing

magnesium = Example(
    name, model_def, original, data, inits, inits_alternative, reference_results)
