function create_log_posterior(bn::BayesianNetwork; use_dp::Bool=false)
    function log_posterior(unobserved_values::Dict{Symbol,Float64})
        # Save old BN state
        old_values = copy(bn.values)
        try
            # Merge unobserved
            for (k, v) in unobserved_values
                bn.values[k] = v
            end

            # Identify unobserved discrete IDs
            unobs_discrete_ids = Int[]
            for sid in bn.stochastic_ids
                if !bn.is_observed[sid]
                    varname = bn.names[sid]
                    if !haskey(bn.values, varname) && is_discrete_node(bn, sid)
                        push!(unobs_discrete_ids, sid)
                    end
                end
            end

            if isempty(unobs_discrete_ids)
                # no discrete marginalization => direct logpdf
                return compute_full_logpdf(bn)
            else
                # sum out discrete configurations
                if use_dp
                    # DP approach
                    memo = Dict{Any,Float64}()
                    assigned_vals = Vector{Any}(undef, length(unobs_discrete_ids))
                    for i in 1:length(unobs_discrete_ids)
                        assigned_vals[i] = nothing
                    end
                    prob_sum = sum_discrete_configurations_dp(
                        bn, unobs_discrete_ids, 1, memo, assigned_vals
                    )
                    return log(prob_sum)
                else
                    # naive recursion
                    prob_sum = sum_discrete_configurations(bn, unobs_discrete_ids, 1)
                    return log(prob_sum)
                end
            end
        finally
            bn.values = old_values
        end
    end
    return log_posterior
end
