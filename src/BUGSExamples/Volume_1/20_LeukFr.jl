name = "LeukFr: Cox regression with random effects"

model_def = @bugs begin # Set up data
    for i in 1:N
        for j in 1:T
            # risk set = 1 if obs.t >= t
            Y[i, j] = step(var"obs.t"[i] - t[j] + eps)

            # counting process jump = 1 if obs.t in [ t[j], t[j+1] )
            # i.e. if t[j] <= obs.t < t[j+1]
            dN[i, j] = Y[i, j] * step(t[j + 1] - var"obs.t"[i] - eps) * fail[i]
        end
    end

    # Model
    for j in 1:T
        for i in 1:N
            dN[i, j] ~ dpois(Idt[i, j])
            Idt[i, j] = Y[i, j] * exp(beta * Z[i] + b[pair[i]]) * dL0[j]
        end
        dL0[j] ~ dgamma(mu[j], c)
        mu[j] = var"dL0.star"[j] * c # prior mean hazard

        # Survivor function = exp(-Integral{l0(u)du})^exp(beta * z)
        var"S.treat"[j] = pow(exp(-sum(dL0[1:j])), exp(beta * -0.5))
        var"S.placebo"[j] = pow(exp(-sum(dL0[1:j])), exp(beta * 0.5))
    end
    for k in 1:Npairs
        b[k] ~ dnorm(0.0, tau)
    end
    tau ~ dgamma(0.001, 0.001)
    sigma = sqrt(1 / tau)
    c = 0.001
    r = 0.1
    for j in 1:T
        var"dL0.star"[j] = r * (t[j + 1] - t[j])
    end
    beta ~ dnorm(0.0, 0.000001)
end

original = """
model {
    # Set up data
    for(i in 1 : N) {
        for(j in 1 : T) {
            # risk set = 1 if obs.t >= t
            Y[i, j] <- step(obs.t[i] - t[j] + eps)
            
            # counting process jump = 1 if obs.t in [ t[j], t[j+1] )
            # i.e. if t[j] <= obs.t < t[j+1]
            dN[i, j] <- Y[i, j ] * step(t[j+1] - obs.t[i] - eps) * fail[i]
        }
    }
    
    # Model
    for(j in 1 : T) {
        for(i in 1 : N) {
            dN[i, j] ~ dpois(Idt[i, j])
            Idt[i, j] <- Y[i, j] * exp(beta * Z[i]+b[pair[i]]) * dL0[j]
        }
        dL0[j] ~ dgamma(mu[j], c)
        mu[j] <- dL0.star[j] * c # prior mean hazard
        
        # Survivor function = exp(-Integral{l0(u)du})^exp(beta * z)
        S.treat[j] <- pow(exp(-sum(dL0[1 : j])), exp(beta * -0.5))
        S.placebo[j] <- pow(exp(-sum(dL0[1 : j])), exp(beta * 0.5))   
    }
    for(k in 1 : Npairs) {
        b[k] ~ dnorm(0.0, tau);
    }
    tau ~ dgamma(0.001, 0.001)
    sigma <- sqrt(1 / tau)
    c <- 0.001 
    r <- 0.1
    for (j in 1 : T) {
        dL0.star[j] <- r * (t[j+1]-t[j])
    }
    beta ~ dnorm(0.0,0.000001)
}
"""

data = (
    N = 42,
    T = 17,
    eps = 0.00001,
    Npairs = 21,
    t = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 22, 23, 35],
    var"obs.t" = [
        1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23, 6,
        6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35
    ],
    pair = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        19, 18, 8, 1, 20, 6, 2, 10, 3, 14, 4, 11, 7, 9, 12, 16, 17, 5, 13, 15, 21
    ],
    fail = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0
    ],
    Z = [
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
        0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
        -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5
    ]
)

inits = (
    beta = 0.0,
    dL0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    tau = 1
)
inits_alternative = (
    beta = 1.0,
    dL0 = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
        2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    tau = 0.1
)

reference_results = nothing

leukfr = Example(
    name, model_def, original, data, inits, inits_alternative, reference_results)
