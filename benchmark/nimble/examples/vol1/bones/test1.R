source("../../R/Rcheck.R")
data <- read.jagsdata("bones-data.R")
data$nGrade <- NULL #not used in this model
m <- jags.model("bones.bug", data, n.chains=2)
check.data(m, data)
update(m, 1000)
x <- coda.samples(m, c("theta"), n.iter=10000)
source("bench-test1.R")
check.fun()
