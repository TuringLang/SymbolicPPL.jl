name = "Alligators: multinomial - logistic regression"

model_def = @bugs begin
    # PRIORS
    alpha[1] = 0 # zero contrast for baseline food
    for k in 2:K
        alpha[k] ~ dnorm(0, 0.00001) # vague priors
    end
    # Loop around lakes:
    for k in 1:K
        beta[1, k] = 0
    end # corner-point contrast with first lake
    for i in 2:I
        beta[i, 1] = 0 # zero contrast for baseline food
        for k in 2:K
            beta[i, k] ~ dnorm(0, 0.00001) # vague priors
        end
    end
    # Loop around sizes:
    for k in 1:K
        gamma[1, k] = 0 # corner-point contrast with first size
    end
    for j in 2:J
        gamma[j, 1] = 0 # zero contrast for baseline food
        for k in 2:K
            gamma[j, k] ~ dnorm(0, 0.00001) # vague priors
        end
    end

    # LIKELIHOOD   
    for i in 1:I  # loop around lakes
        for j in 1:J  # loop around sizes

            # Multinomial response
            # X[i, j, 1:K] ~ dmulti(p[i, j, 1:K], n[i, j])
            # n[i, j] = sum(X[i, j])
            # for k in 1:K  # loop around foods
            #     p[i, j, k] = phi[i, j, k] / sum(phi[i, j])
            #     log(phi[i, j, k]) < -alpha[k] + beta[i, k] + gamma[j, k]
            # end

            # Fit standard Poisson regressions relative to baseline
            lambda[i, j] ~ dflat()   # vague priors
            for k in 1:K  # loop around foods
                X[i, j, k] ~ dpois(mu[i, j, k])
                mu[i, j, k] = exp(lambda[i, j] + alpha[k] + beta[i, k] + gamma[j, k])
            end
        end
    end

    # TRANSFORM OUTPUT TO ENABLE COMPARISON
    # WITH AGRESTI'S RESULTS
    for k in 1:K  # loop around foods
        for i in 1:I  # loop around lakes
            b[i, k] = beta[i, k] - mean(beta[:, k]) # sum to zero constraint
        end
        for j in 1:J  # loop around sizes
            g[j, k] = gamma[j, k] - mean(gamma[:, k]) # sum to zero constraint
        end
    end
end

original = """
model {

    # PRIORS
    alpha[1] <- 0; # zero contrast for baseline food
    for (k in 2 : K) { 
        alpha[k] ~ dnorm(0, 0.00001) # vague priors
    } 

    # Loop around lakes:
    for (k in 1 : K){  
        beta[1, k] <- 0 # corner-point contrast with first lake 
    } 
    for (i in 2 : I) {     
        beta[i, 1] <- 0; # zero contrast for baseline food
        for (k in 2 : K){  
            beta[i, k] ~ dnorm(0, 0.00001) # vague priors
        } 
    }

    # Loop around sizes:
    for (k in 1 : K){  
        gamma[1, k] <- 0 # corner-point contrast with first size 
    }  
    for (j in 2 : J) {     
        gamma[j, 1] <- 0; # zero contrast for baseline food
        for (k in 2 : K){ 
            gamma[j, k] ~ dnorm(0, 0.00001) # vague priors
        } 
    }

    # LIKELIHOOD    
    for (i in 1 : I) { # loop around lakes
        for (j in 1 : J) { # loop around sizes

            # Fit standard Poisson regressions relative to baseline
            lambda[i, j] ~ dflat() # vague priors 
            for (k in 1 : K) { # loop around foods
                X[i, j, k] ~ dpois(mu[i, j, k])
                log(mu[i, j, k]) <- lambda[i, j] + alpha[k] + beta[i, k] + gamma[j, k]
            }
        }  
    }

    # TRANSFORM OUTPUT TO ENABLE COMPARISON 
    # WITH AGRESTI'S RESULTS
    for (k in 1 : K) { # loop around foods
        for (i in 1 : I) { # loop around lakes
            b[i, k] <- beta[i, k] - mean(beta[, k]); # sum to zero constraint
        }
        for (j in 1 : J) { # loop around sizes
            g[j, k] <- gamma[j, k] - mean(gamma[, k]); # sum to zero constraint
        }
    }
}  
"""

data = (
    I = 4,
    J = 2,
    K = 5,
    X = let X = Array{Int, 3}(undef, 4, 2, 5)
        X[1, 1, :] = [23, 4, 2, 2, 8]
        X[1, 2, :] = [7, 0, 1, 3, 5]
        X[2, 1, :] = [5, 11, 1, 0, 3]
        X[2, 2, :] = [13, 8, 6, 1, 0]
        X[3, 1, :] = [5, 11, 2, 1, 5]
        X[3, 2, :] = [8, 7, 6, 3, 5]
        X[4, 1, :] = [16, 19, 1, 2, 3]
        X[4, 2, :] = [17, 1, 0, 1, 3]
        X
    end)

inits = (
    alpha = [missing, 0, 0, 0, 0],
    beta = [missing missing missing missing missing
            missing 0 0 0 0
            missing 0 0 0 0
            missing 0 0 0 0
            missing 0 0 0 0],
    gamma = [missing missing missing missing missing
             missing 0 0 0 0],
    lambda = [0 0
              0 0
              0 0
              0 0]
)
inits_alternative = (
    alpha = [missing, 1, 1, 1, 1],
    beta = [missing missing missing missing missing
            missing 2 2 2 2
            missing 2 2 2 2
            missing 2 2 2 2],
    gamma = [missing missing missing missing missing
             missing 3 3 3 3],
    lambda = [4 4
              4 4
              4 4
              4 4]
)

reference_results = (
    var"b[1,2]" = (mean = -1.809, std = 0.4705),
    var"b[1,3]" = (mean = -0.337, std = 0.614),
    var"b[1,4]" = (mean = 0.5645, std = 0.5681),
    var"b[1,5]" = (mean = 0.2861, std = 0.3644),
    var"b[2,2]" = (mean = 0.8471, std = 0.3415),
    var"b[2,3]" = (mean = 0.9382, std = 0.5172),
    var"b[2,4]" = (mean = -1.279, std = 1.01),
    var"b[2,5]" = (mean = -0.6744, std = 0.54),
    var"b[3,2]" = (mean = 1.076, std = 0.3502),
    var"b[3,3]" = (mean = 1.451, std = 0.5103),
    var"b[3,4]" = (mean = 0.9416, std = 0.5918),
    var"b[3,5]" = (mean = 0.9967, std = 0.3929),
    var"b[4,2]" = (mean = -0.1136, std = 0.2929),
    var"b[4,3]" = (mean = -2.053, std = 1.007),
    var"b[4,4]" = (mean = -0.2273, std = 0.6213),
    var"b[4,5]" = (mean = -0.6084, std = 0.4099),
    var"g[1,2]" = (mean = 0.7646, std = 0.2024),
    var"g[1,3]" = (mean = -0.1807, std = 0.3109),
    var"g[1,4]" = (mean = -0.343, std = 0.3391),
    var"g[1,5]" = (mean = 0.1898, std = 0.235),
    var"g[2,2]" = (mean = -0.7646, std = 0.2024),
    var"g[2,3]" = (mean = 0.1807, std = 0.3109),
    var"g[2,4]" = (mean = 0.343, std = 0.3391),
    var"g[2,5]" = (mean = -0.1898, std = 0.235)
)

alligators = Example(
    name, model_def, original, data, inits, inits_alternative, reference_results)
