model in inhaler2.bug
data in inhaler-data.R
load glm
set factory "glm::Generic" off, type(sampler)
set factory "glm::Holmes-Held" off, type(sampler)
compile, nchains(2)
parameters in inhaler-inits.R
initialize
#samplers to foo-samplers.txt
#exit
update 1000
monitor a, thin(10)
monitor beta, thin(10)
monitor kappa, thin(10)
monitor log.sigma, thin(10)
monitor pi, thin(10)
monitor sigma, thin(10)
update 10000
coda *
exit
