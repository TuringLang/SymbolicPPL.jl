/* 
   The original formulation of the inhaler example shows poor mixing and
   requires a long run with a long thinning interval to obtain good
   estimates of the posterior distribution.
*/
model in inhaler0.bug
data in inhaler-data.R
compile, nchains(2)
parameters in inhaler0-inits.R
initialize
update 10000
monitor a, thin(100)
monitor beta, thin(100)
monitor kappa, thin(100)
monitor log.sigma, thin(100)
monitor pi, thin(100)
monitor sigma, thin(100)
update 100000
coda *
exit
