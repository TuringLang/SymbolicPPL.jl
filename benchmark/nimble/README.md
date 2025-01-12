# Benchmarking Nimble's Performance in Computing Log-Density and Gradient

This benchmark evaluates the performance of Nimble in computing the log-density and its gradient for various models. To compute gradients in Nimble, it is necessary to use the compiled model.

Nimble comes with [JAGS' "classic-bugs" code](https://github.com/nimble-dev/nimble/tree/devel/packages/nimble/inst/classic-bugs). But here we transcribe the models of MultiBUGS, some of them are modified so Nimble can compile them.

Additionally, there is some relevant [code](https://github.com/nimble-dev/nimble/blob/devel/packages/AD-full-tests/test-ADmodels-full.R#L1143) from Nimble's ADTest that could provide useful insights when conducting a more thorough benchmarking exercise.

## Caveats in Comparing Nimble with Stan

It's important to keep in mind that, similar to the caveats in comparing JuliaBUGS and Stan, the Stan models are generally more optimized through the use of vectorized operations. As a result, the speed should not be directly compared between Nimble and Stan. Instead, the results should be viewed as an indication of the relative performance scale between the two frameworks.
