
function todppl(g::BUGSGraph)
    expr = []
    args = Dict()
    sorted_nodes = (x->label_for(g, x)).(topological_sort_by_dfs(g))
    for n in sorted_nodes
        f = g[n].f_expr.args[2] |> MacroTools.flatten
        if isa(f, Expr)
            f.args[1] = Expr(:., :SymbolicPPL, QuoteNode(f.args[1]))
            push!(expr, Expr(:call, :(~), n, f))
        elseif isa(f, Distributions.Distribution)
            f = Expr(:call, nameof(typeof(f)), Distributions.params(f)...)
            push!(expr, Expr(:call, :(~), n, f))
        end
    end
    args = [Expr(:kw, a, g[a].data) for a in (x->label_for(g,x)).(vertices(g)) if g[a].is_data]
    ex = Expr(:function, Expr(:call, :model, Expr(:parameters, args...)), Expr(:block, expr...))
    eval(DynamicPPL.model(@__MODULE__, LineNumberNode(@__LINE__, @__FILE__), ex, false))
    println(ex)
    return model
end

function gen_variation_partition(g::BUGSGraph)
    dist_types = dry_run(g)[1]
    dt = Dict{Any, Any}()
    for k in keys(dist_types)
        if dist_types[k] <: Sampleable{<:VariateForm,Discrete}
            dt[k] = true
        else
            dt[k] = false
        end
    end

    discrete_vars = [k for k in keys(dt) if dt[k]]
    continuous_vars = [k for k in keys(dt) if !dt[k]]
    return discrete_vars, continuous_vars
end
