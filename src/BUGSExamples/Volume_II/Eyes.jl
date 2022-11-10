# https://chjackson.github.io/openbugsdoc/Examples/Eyes.html

eyes = (
    name = "eyes", 
    model_def = bugsmodel"""
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
    """, 

    data = (
        y = [529.0, 530.0, 532.0, 533.1, 533.4, 533.6, 533.7, 534.1, 534.8, 535.3,
            535.4, 535.9, 536.1, 536.3, 536.4, 536.6, 537.0, 537.4, 537.5, 538.3,
            538.5, 538.6, 539.4, 539.6, 540.4, 540.8, 542.0, 542.8, 543.0, 543.5,
            543.8, 543.9, 545.3, 546.2, 548.8, 548.7, 548.9, 549.0, 549.4, 549.9,
            550.6, 551.2, 551.4, 551.5, 551.6, 552.8, 552.9,553.2], 
        N = 48, 
        alpha = [1, 1],
        T = [1, NA, NA, NA, NA, NA, NA, NA, NA, NA,
            NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,         
            NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,
            NA, NA, NA, NA, NA, NA, NA, NA, NA, NA,
            NA, NA, NA, NA, NA, NA, NA, 2]
    ),
    
    inits = [
        (lambda = [535, NA], theta = 5, tau = 0.1), 
        (lambda = [100, NA], theta = 50, tau = 1),
    ],
)
