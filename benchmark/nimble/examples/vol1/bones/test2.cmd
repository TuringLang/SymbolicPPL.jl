model in bones2.bug
data in bones-data.R
load glm
compile, nchains(2)
parameters in bones-init.R
initialize
samplers to foo-samplers.txt
update 1000
monitor theta 
update 10000 
coda theta
