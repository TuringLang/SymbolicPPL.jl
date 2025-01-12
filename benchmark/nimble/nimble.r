library(nimble)
library(microbenchmark)
nimbleOptions(enableDerivs = TRUE)
nimbleOptions(buildModelDerivs = TRUE)
nimbleOptions(allowDynamicIndexing = TRUE)

# functions

logLikelihood_nf <- nimbleFunction(
    setup = function(model, paramNodes) {
        # Determine nodes for calculating the log likelihood for parameters given by
        # paramNodes, ignoring any priors.
        calcNodes <- model$getDependencies(paramNodes, self = FALSE)
        # Set up the additional arguments for nimDerivs involving model$calculate
        derivsInfo <- makeModelDerivsInfo(model, paramNodes, calcNodes)
        updateNodes <- derivsInfo$updateNodes
        constantNodes <- derivsInfo$constantNodes
        # Create a parameter transformation between original and unconstrained
        # parameter spaces.
        transformer <- parameterTransform(model, paramNodes)
    },
    methods = list(
        neg_logLikelihood_p = function(p = double(1)) {
            # Put values in model and calculate negative log likelihood.
            values(model, paramNodes) <<- p
            return(-model$calculate(calcNodes))
            returnType(double())
        },
        neg_logLikelihood = function(ptrans = double(1)) {
            # Objective function for optim,
            # using transformed parameter space.
            p <- transformer$inverseTransform(ptrans)
            return(neg_logLikelihood_p(p))
            returnType(double())
        },
        gr_neg_logLikelihood = function(ptrans = double(1)) {
            # Gradient of neg log likelihood
            p <- transformer$inverseTransform(ptrans)
            d <- derivs(neg_logLikelihood_p(p),
                wrt = 1:length(p), order = 1,
                model = model, updateNodes = updateNodes,
                constantNodes = constantNodes
            )
            return(d$jacobian[1, ])
            returnType(double(1))
        },
        transform = function(p = double(1)) {
            # Give user access to the transformation ...
            return(transformer$transform(p))
            returnType(double(1))
        },
        inverse = function(ptrans = double(1)) { # ... and its inverse.
            return(transformer$inverseTransform(ptrans))
            returnType(double(1))
        }
    ),
    buildDerivs = "neg_logLikelihood_p"
)

log_joint_nf <- nimbleFunction(
    setup = function(model) {
    },
    methods = list(
        log_joint = function() {
            return(-model$calculate())
            returnType(double())
        }
    )
)

#

MODEL_VOL1_NIMBLE <- c(
    "rats",
    "pumps",
    # "dogs",
    "seeds",
    "surgical",
    "magnesium",
    "salm",
    # "equiv",
    "dyes",
    "stacks",
    "epil",
    "blocker",
    "oxford",
    "lsat",
    "bones",
    "mice",
    "kidney",
    "leuk",
    "leukfr"
)

MODEL_VOL2_NIMBLE <- c(
    "dugongs",
    "orange",
    "mvotree",
    # "biopsies",
    "eyes",
    # "hearts",
    "air",
    "cervix",
    # "jaw",
    "birats",
    "schools",
    # "ice",
    "beetles",
    "alli",
    # "endo",
    "stagnant"
    # "asia"
)

#

process_example("leukfr", current_dir, "vol1")

process_example("asia", current_dir, "vol2")

#
current_dir <- getwd()

process_example <- function(example, current_dir, example_vol) {
    cat("\nProcessing example:", example, "\n")
    model <- readBUGSmodel(paste0(example, ".bug"), dir = paste0(current_dir, "/benchmark/nimble/examples/", example_vol, "/", example))
    compiled_model <- compileNimble(model)

    # Get parameter nodes and generate random parameters
    stoch_nodes <- model$getNodeNames(stochOnly = TRUE)
    param_nodes <- stoch_nodes[!sapply(stoch_nodes, function(node) model$isData(node))]
    num_params <- length(param_nodes)
    rand_params <- runif(num_params)

    # `calculate` without compilation
    non_compiled_log_joint_result <- microbenchmark(model$calculate())

    # `calculate` with compilation
    nll_compute <- log_joint_nf(model)
    cnll_compute <- compileNimble(nll_compute, project = model)
    cnll_benchmark_result <- microbenchmark(cnll_compute$log_joint())

    # Create log likelihood model
    ll_model <- tryCatch(
        {
            logLikelihood_nf(model, paramNodes = param_nodes)
        },
        error = function(e) {
            cat("Error in logLikelihood_nf: ", e$message, "\n")
            NULL
        }
    )

    # Compile log likelihood model if created successfully
    cll_model <- if (!is.null(ll_model)) {
        tryCatch(
            {
                compileNimble(ll_model, project = model)
            },
            error = function(e) {
                cat("Error in compileNimble: ", e$message, "\n")
                NULL
            }
        )
    } else {
        NULL
    }

    # Run negative log likelihood benchmark
    nll_benchmark_result <- if (!is.null(cll_model)) {
        tryCatch(
            {
                microbenchmark(cll_model$neg_logLikelihood(rand_params))
            },
            error = function(e) {
                cat("Error in negative log likelihood benchmark: ", e$message, "\n")
                NULL
            }
        )
    } else {
        NULL
    }

    # Run gradient benchmark
    gr_benchmark_result <- if (!is.null(cll_model)) {
        tryCatch(
            {
                microbenchmark(cll_model$gr_neg_logLikelihood(rand_params))
            },
            error = function(e) {
                cat("Error in gradient benchmark: ", e$message, "\n")
                NULL
            }
        )
    } else {
        NULL
    }

    return(list(
        non_compiled_log_joint = non_compiled_log_joint_result,
        compiled_log_joint = cnll_benchmark_result,
        nll = nll_benchmark_result,
        gr = gr_benchmark_result
    ))
}

#
benchmark_results <- list()
for (example in MODEL_VOL1_NIMBLE) {
    benchmark_results[[example]] <- process_example(example, current_dir, "vol1")
}
benchmark_results_vol2 <- list()
for (example in MODEL_VOL2_NIMBLE) {
    benchmark_results_vol2[[example]] <- process_example(example, current_dir, "vol2")
}

# Print results for Volume 1
cat("\nVolume 1 Results:\n")
for (example in names(benchmark_results)) {
    cat("\nExample:", example, "\n")
    result <- benchmark_results[[example]]
    
    if (!is.null(result$non_compiled_log_joint)) {
        cat("Non-compiled log joint time (microseconds):\n")
        print(summary(result$non_compiled_log_joint)$table)
    }
    
    if (!is.null(result$compiled_log_joint)) {
        cat("Compiled log joint time (microseconds):\n") 
        print(summary(result$compiled_log_joint)$table)
    }
    
    if (!is.null(result$nll)) {
        cat("Negative log likelihood time (microseconds):\n")
        print(summary(result$nll)$table)
    }
    
    if (!is.null(result$gr)) {
        cat("Gradient time (microseconds):\n")
        print(summary(result$gr)$table)
    }
}

# Print results for Volume 2  
cat("\nVolume 2 Results:\n")
for (example in names(benchmark_results_vol2)) {
    cat("\nExample:", example, "\n")
    result <- benchmark_results_vol2[[example]]
    
    if (!is.null(result$non_compiled_log_joint)) {
        cat("Non-compiled log joint time (microseconds):\n")
        print(summary(result$non_compiled_log_joint)$table)
    }
    
    if (!is.null(result$compiled_log_joint)) {
        cat("Compiled log joint time (microseconds):\n")
        print(summary(result$compiled_log_joint)$table)
    }
    
    if (!is.null(result$nll)) {
        cat("Negative log likelihood time (microseconds):\n")
        print(summary(result$nll)$table)
    }
    
    if (!is.null(result$gr)) {
        cat("Gradient time (microseconds):\n")
        print(summary(result$gr)$table)
    }
}
