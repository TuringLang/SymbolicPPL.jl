struct NodeFunctions{VT} <: CompilerPass
    vars::VT    
    transformed_variables::Dict
    link_functions::Dict
    logical_node_args::Dict
    logical_node_f_exprs::Dict
    stochastic_node_args::Dict
    stochastic_node_f_exprs::Dict
end

function NodeFunctions(vars, array_sizes)
    transformed_variables = Dict()
    for (k, v) in array_sizes
        transformed_variables[k] = Array(undef, v...)
    end
    NodeFunctions(vars, collect(pairs(transformed_variables)), Dict(), Dict(), Dict(), Dict(), Dict())
end

# Generate an expression to reconstruct a given distribution object
function toexpr(dist::Distributions.Distribution)
    dist_type = typeof(dist)
    dist_params = params(dist)
    return Expr(:call, dist_type, dist_params...)
end

function process_rhs(evaluated_expr::Expr, env::Dict)
    replaced_expr = replace_vars(evaluated_expr, array_map, env)
    args = Dict()
    gen_expr = MacroTools.postwalk(replaced_expr) do sub_expr
        if sub_expr isa Var
            gen_arg = Symbol(sub_expr)
            args[sub_expr] = gen_arg
            return gen_arg
        elseif sub_expr isa Array{Var}
            gen_arg = Symbol.(sub_expr)
            for (i, v) in enumerate(sub_expr)
                args[v] = gen_arg[i]
            end
            return Expr(:call, :reshape, Expr(:vect, gen_arg...), (size(sub_expr)...))
        else
            return sub_expr
        end
    end

    gen_expr = getindex_to_ref(gen_expr)
    # TODO: BUGS do primitive type deduction in the beginning, so that when index is 1.0, it is casted to 1
    # for now, we do this at runtime
    gen_expr = MacroTools.postwalk(gen_expr) do sub_expr
        if @capture(sub_expr, arr_[idxs__])
            new_idxs = [:(cast_to_integer_if_integer($(idx))) for idx in idxs]
            return Expr(:ref, arr, new_idxs...)
        else
            return sub_expr
        end
    end

    f_expr = MacroTools.postwalk(
        MacroTools.unblock,
        MacroTools.combinedef(
            Dict(
                :args => values(args),
                :body => gen_expr,
                :kwargs => Any[],
                :whereparams => Any[],
            ),
        ),
    )

    return f_expr, keys(args)
end



function assignment!(pass::NodeFunctions, expr::Expr, env::Dict)
    assignment_type = expr.head
    lhs_expr = expr.args[1]
    rhs_expr = expr.args[2]

    link_function = Meta.isexpr(lhs_expr, :call) ? lhs.args[1] : identity
    l_var = find_variables_on_lhs(Meta.isexpr(lhs_expr, :call) ? lhs.args[2] : lhs_expr, env)
    
    l_vn = VarName(l_var)
    evaluated_rhs = eval(rhs_expr, env)
    # RHS can be evaluated, it is transformed variables, so we treat it as data later
    if evaluated_rhs isa Number || evaluated_rhs isa AbstractArray
        pass.transformed_variables = set(pass.transformed_variables, l_vn, evaluated_rhs)
        return nothing
    end

    # RHS is a Symbol
    if evaluated_rhs isa Symbol
        @assert isscalar(l_var)
        # do something
    end

    if Meta.isexpr(evaluated_expr, :ref) &&
        all(x -> x isa Union{Number, UnitRange, Colon}, evaluated_expr.args[2:end])
        # check lhs size, complication: Colon
    end

    if isa(evaled_var, Distributions.Distribution)
        evaluated_expr = dist_to_expr(evaluated_expr)
    end

    
    pass.link_functions[l_var] = link_function
    
    r_func, r_var_args = rhs(pass, expr.args[2], env)

    if expr.head == :(=)
        @assert !in(l_var, keys(pass.logical_node_args)) "Repeated assignment to $l_var"
        pass.logical_node_args[l_var] = r_var_args
        pass.logical_node_f_exprs[l_var] = r_func
    else
        @assert expr.head == :(~)
        pass.stochastic_node_args[l_var] = r_var_args
        pass.stochastic_node_f_exprs[l_var] = r_func
    end
    return nothing
end

function post_process(pass::NodeFunctions)
    data = pass.data
    vars = pass.vars
    array_map = pass.array_map
    missing_elements = pass.missing_elements
    logical_node_args = pass.logical_node_args
    logical_node_f_exprs = pass.logical_node_f_exprs
    stochastic_node_args = pass.stochastic_node_args
    stochastic_node_f_exprs = pass.stochastic_node_f_exprs
    link_functions = pass.link_functions

    array_variables = []
    for var in keys(vars)
        if !haskey(logical_node_args, var) && !haskey(stochastic_node_args, var) # variables without node functions
            @assert isa(var, ArrayElement) || isa(var, ArrayVariable)
            if var isa ArrayElement
                # then come from either ArrayVariable or ArraySlice
                source_var = filter(
                    x -> (x isa ArrayVariable || x isa ArraySlice) && x.name == var.name,
                    vcat(
                        map(
                            collect, [keys(logical_node_args), keys(stochastic_node_args)]
                        )...,
                    ),
                )
                @assert length(source_var) == 1
                array_var = first(source_var)
                logical_node_args[var] = [array_var]
                logical_node_f_exprs[var] = MacroTools.postwalk(
                    MacroTools.rmlines, :((array_var) -> array_var[$(var.indices...)])
                )
            elseif var.name in keys(array_map)
                push!(array_variables, var)
                array_elems = scalarize(var)
                logical_node_args[var] = vcat(array_elems)
                # @assert all(x -> x in keys(node_args), array_elems) # might not be true
                # arg_list = [Symbol("arg" * string(i)) for i in 1:length(array_elems)]
                f_name = Symbol("compose_" * String(Symbol(var)))
                # logical_node_f_exprs[var] = MacroTools.postwalk(
                #     MacroTools.rmlines,
                #     :(function ($f_name)($(arg_list...))
                #         args = [$(arg_list...)]
                #         return reshape(collect(args), $(size(array_map[var.name])))
                #     end),
                # )
                logical_node_f_exprs[var] = MacroTools.postwalk(
                    MacroTools.rmlines,
                    :(function ($f_name)(args::Vector)
                        return reshape(args, $(size(array_map[var.name])))
                    end),
                )
            else # data array
                # TODO: for now, handle this in logdensityproblems, this is a leak of abstraction, need to be addressed
            end
        end
    end

    for v in vcat(collect(values(missing_elements))...)
        logical_node_args[v] = []
        logical_node_f_exprs[v] = :missing
    end

    return logical_node_args,
    logical_node_f_exprs, stochastic_node_args, stochastic_node_f_exprs, link_functions,
    array_variables
end


"""
Previously, we add every array elements and array variable to the graph, because we 
want to capture fine-grain dependency. Corner case like:

x = [1, 2, missing, missing] # data
x[3] ~ dnorm
x[2:3] ~ dmnorm # TODO: check x[2:3] should be same type
y = sum(x[:])

x -> y 

in this case, variable pass will collect x[3] and y

"""