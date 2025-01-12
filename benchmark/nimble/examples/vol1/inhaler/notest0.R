source("../../R/Rcheck.R")
d <- read.jagsdata("inhaler-data.R")
inits <- read.jagsdata("inhaler0-inits.R")
m <- jags.model("inhaler.bug", d, inits, n.chains=2)
## Check data consistency, skipping variables created in data block
check.data(m, d, skip=c("group", "response"))
update(m, 1000)
x <- coda.samples(m, c("a","beta","kappa","log.sigma","pi","sigma"),
                  n.iter=10000, thin=10)
source("bench-test1.R")
check.fun()
