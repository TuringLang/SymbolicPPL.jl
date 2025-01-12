abstract type CompilerPass end

is_deterministic(expr::Expr) = Meta.isexpr(expr, :(=))
is_stochastic(expr::Expr) = Meta.isexpr(expr, :call) && expr.args[1] == :(~)

function analyze_block(
    pass::CompilerPass,
    expr::Expr,
    loop_vars::NamedTuple=NamedTuple();
    warn_loop_bounds::Ref{Bool}=Ref(false),
)
    if !Meta.isexpr(expr, :block)
        error("The top level expression must be a block.")
    end
    for statement in expr.args
        if is_deterministic(statement) || is_stochastic(statement)
            analyze_statement(pass, statement, loop_vars)
        elseif Meta.isexpr(statement, :for)
            loop_var, lb, ub, body = decompose_for_expr(statement)
            env = merge(pass.env, loop_vars)
            lb = Int(simple_arithmetic_eval(env, lb))
            ub = Int(simple_arithmetic_eval(env, ub))
            if lb > ub
                if warn_loop_bounds[]
                    @warn "In BUGS, if the lower bound of for loop is greater than the upper bound, the loop will be skipped."
                    warn_loop_bounds[] = false
                end
            else
                for loop_var_value in lb:ub
                    analyze_block(
                        pass,
                        body,
                        merge(loop_vars, (loop_var => loop_var_value,));
                        warn_loop_bounds=warn_loop_bounds,
                    )
                end
            end
        else
            error("Unsupported expression in top level: $statement")
        end
    end
end

@enum VariableTypes::Bool begin
    Logical
    Stochastic
end

"""
    CollectVariables

This analysis pass instantiates all possible left-hand sides for both deterministic and stochastic 
assignments. Checks include: (1) In a deterministic statement, the left-hand side cannot be 
specified by data; (2) In a stochastic statement, for a multivariate random variable, it cannot be 
partially observed. This pass also returns the sizes of the arrays in the model, determined by the 
largest indices.
"""
struct CollectVariables{data_vars} <: CompilerPass
    env::NamedTuple{data_vars}
    data_scalars::Tuple{Vararg{Symbol}}
    non_data_scalars::Tuple{Vararg{Symbol}}
    data_array_sizes::NamedTuple
    non_data_array_sizes::NamedTuple
end

function CollectVariables(model_def::Expr, data::NamedTuple{data_vars}) where {data_vars}
    for var in extract_variables_in_bounds_and_lhs_indices(model_def)
        if var ∉ data_vars
            error(
                "Variable $var is used in loop bounds or indices but not defined in the data.",
            )
        end
    end

    data_scalars, non_data_scalars = Symbol[], Symbol[]
    arrays, num_dims = Symbol[], Int[]
    data_scalars, scalars, arrays, num_dims = Symbol[], Symbol[], Symbol[], Int[]
    # `extract_variable_names_and_numdims` will check if inconsistent variables' ndims
    for (name, num_dim) in pairs(extract_variable_names_and_numdims(model_def))
        if num_dim == 0
            if name in data_vars
                push!(data_scalars, name)
            else
                push!(non_data_scalars, name)
            end
        else
            push!(arrays, name)
            push!(num_dims, num_dim)
        end
    end

    for (var, num_dim) in zip(arrays, num_dims)
        if var ∈ data_vars && num_dim != ndims(data[var])
            error(
                "Array variable $var has $num_dim dimensions in the model, but $(ndims(data[var])) dimensions in the data.",
            )
        end
    end

    data_arrays = Symbol[]
    data_array_sizes = SVector[]
    for k in data_vars
        if data[k] isa AbstractArray
            push!(data_arrays, k)
            push!(data_array_sizes, SVector(size(data[k])))
        end
    end

    non_data_arrays = Symbol[]
    non_data_array_sizes = MVector[]
    for (var, num_dim) in zip(arrays, num_dims)
        if var ∉ data_vars
            push!(non_data_arrays, var)
            push!(non_data_array_sizes, MVector{num_dim}(fill(0, num_dim)))
        end
    end

    error_on_undeclared_variables(model_def, non_data_scalars, non_data_arrays)

    return CollectVariables{data_vars}(
        data,
        Tuple(data_scalars),
        Tuple(non_data_scalars),
        NamedTuple{Tuple(data_arrays)}(Tuple(data_array_sizes)),
        NamedTuple{Tuple(non_data_arrays)}(Tuple(non_data_array_sizes)),
    )
end

function error_on_undeclared_variables(model_def::Expr, non_data_scalars, non_data_arrays)
    logical_scalars, stochastic_scalars, logical_arrays, stochastic_arrays = extract_variables_assigned_to(
        model_def
    )

    for scalar in non_data_scalars
        if !(scalar in union(logical_scalars, stochastic_scalars))
            error(
                "Scalar variable $scalar is used in the model, but it is not defined in the model or data.",
            )
        end
    end

    for array in non_data_arrays
        if !(array in union(logical_arrays, stochastic_arrays))
            error(
                "Array variable $array is used in the model, but it is not defined in the model or data.",
            )
        end
    end
end

"""
    evaluate(expr, env)

Evaluate `expr` in the environment `env`.

# Examples
```jldoctest; setup=:(using JuliaBUGS: evaluate)
julia> evaluate(:(x[1]), (x = [1, 2, 3],)) # array indexing is evaluated if possible
1

julia> evaluate(:(x[1] + 1), (x = [1, 2, 3],))
2

julia> evaluate(:(x[1:2]), NamedTuple()) |> Meta.show_sexpr # ranges are evaluated
(:ref, :x, 1:2)

julia> evaluate(:(x[1:2]), (x = [1, 2, 3],))
2-element Vector{Int64}:
 1
 2

julia> evaluate(:(x[1:3]), (x = [1, 2, missing],)) # when evaluate an array, if any element is missing, original expr is returned
:(x[1:3])

julia> evaluate(:(x[y[1] + 1] + 1), NamedTuple()) # if a ref expr can't be evaluated, it's returned as is
:(x[y[1] + 1] + 1)

julia> evaluate(:(sum(x[:])), (x = [1, 2, 3],)) # function calls are evaluated if possible
6

julia> evaluate(:(f(1)), NamedTuple()) # if a function call can't be evaluated, it's returned as is
:(f(1))
"""
evaluate(expr::Number, env) = expr
evaluate(expr::UnitRange, env) = expr
evaluate(expr::Colon, env) = expr
function evaluate(expr::Symbol, env::NamedTuple{variable_names}) where {variable_names}
    if expr == :(:)
        return Colon()
    else
        if expr in variable_names
            return env[expr] === missing ? expr : env[expr]
        else
            return expr
        end
    end
end
function evaluate(expr::Expr, env::NamedTuple{variable_names}) where {variable_names}
    if Meta.isexpr(expr, :ref)
        var, indices... = expr.args
        all_resolved = true
        for i in eachindex(indices)
            indices[i] = evaluate(indices[i], env)
            if indices[i] isa Float64
                indices[i] = Int(indices[i])
            end
            all_resolved = all_resolved && indices[i] isa Union{Int,UnitRange{Int},Colon}
        end
        if var in variable_names
            if all_resolved
                _indices = map(i -> i === Colon() ? 0 : last(i), indices) # if index is `:`, then we skip the check
                if any(_indices .> size(env[var]))
                    error(
                        "Variable $var[$(join(indices, ", "))] is used in the model, but it or its elements are not defined in the model or data.",
                    )
                end
                value = env[var][indices...]
                if is_resolved(value)
                    return value
                else
                    return Expr(:ref, var, indices...)
                end
            end
        else
            return Expr(:ref, var, indices...)
        end
    elseif Meta.isexpr(expr, :call)
        f, args... = expr.args
        all_resolved = true
        for i in eachindex(args)
            args[i] = evaluate(args[i], env)
            all_resolved = all_resolved && is_resolved(args[i])
        end
        if all_resolved
            if f === :(:)
                return UnitRange(Int(args[1]), Int(args[2]))
            elseif f ∈ BUGSPrimitives.BUGS_FUNCTIONS ∪ (:+, :-, :*, :/, :^)
                _f = getfield(BUGSPrimitives, f)
                return _f(args...)
            else
                return Expr(:call, f, args...)
            end
        else
            return Expr(:call, f, args...)
        end
    else
        error("Unsupported expression: $var")
    end
end

is_resolved(::Missing) = false
is_resolved(::Union{Int,Float64}) = true
is_resolved(::Array{<:Union{Int,Float64}}) = true
is_resolved(::Array{Missing}) = false
is_resolved(::Union{Symbol,Expr}) = false
is_resolved(::Any) = false

function is_specified_by_data(data::NamedTuple{data_keys}, var::Symbol) where {data_keys}
    if var ∉ data_keys
        return false
    else
        if data[var] isa AbstractArray
            error("In BUGS, implicit indexing on the LHS is not allowed.")
        else
            return true
        end
    end
end
function is_specified_by_data(
    data::NamedTuple{data_keys},
    var::Symbol,
    indices::Vararg{Union{Missing,Float64,Int,UnitRange{Int}}},
) where {data_keys}
    if var ∉ data_keys
        return false
    else
        values = data[var][indices...]
        if values isa AbstractArray
            if eltype(values) === Missing
                return false
            elseif eltype(values) <: Union{Int,Float64}
                return true
            else
                return any(!ismissing, values)
            end
        else
            if values isa Missing
                return false
            elseif values isa Union{Int,Float64}
                return true
            else
                error("Unexpected type: $(typeof(values))")
            end
        end
    end
end

function is_partially_specified_as_data(
    data::NamedTuple{data_keys},
    var::Symbol,
    indices::Vararg{Union{Missing,Float64,Int,UnitRange{Int}}},
) where {data_keys}
    if var ∉ data_keys
        return false
    else
        values = data[var][indices...]
        return values isa AbstractArray && any(ismissing, values) && any(!ismissing, values)
    end
end

function analyze_statement(pass::CollectVariables, expr::Expr, loop_vars::NamedTuple)
    lhs_expr = is_deterministic(expr) ? expr.args[1] : expr.args[2]
    env = merge(pass.env, loop_vars)
    v = simplify_lhs(env, lhs_expr)
    if v isa Symbol
        handle_symbol_lhs(pass, expr, v, env)
    else
        handle_ref_lhs(pass, expr, v, env)
    end
end

function handle_symbol_lhs(::CollectVariables, expr::Expr, v::Symbol, env::NamedTuple)
    if Meta.isexpr(expr, :(=)) && is_specified_by_data(env, v)
        error("Variable $v is specified by data, can't be assigned to.")
    end
end

function handle_ref_lhs(pass::CollectVariables, expr::Expr, v::Tuple, env::NamedTuple)
    var, indices... = v
    if Meta.isexpr(expr, :(=))
        if is_specified_by_data(env, var, indices...)
            error(
                "$var[$(join(indices, ", "))] is (are) specified by data, can't be assigned to.",
            )
        end
        update_array_sizes_for_assignment(pass, var, env, indices...)
    else
        if is_partially_specified_as_data(env, var, indices...)
            error(
                "$var[$(join(indices, ", "))] partially observed, not allowed, rewrite so that the variables are either all observed or all unobserved.",
            )
        end
        update_array_sizes_for_assignment(pass, var, env, indices...)
    end
end

function update_array_sizes_for_assignment(
    pass::CollectVariables,
    var::Symbol,
    ::NamedTuple{data_vars},
    indices::Vararg{Union{Int,UnitRange{Int}}},
) where {data_vars}
    # `is_specified_by_data` checks if the index is inbound
    if var ∉ data_vars
        for i in eachindex(pass.non_data_array_sizes[var])
            pass.non_data_array_sizes[var][i] = max(
                pass.non_data_array_sizes[var][i], last(indices[i])
            )
        end
    end
end

function post_process(pass::CollectVariables)
    return pass.non_data_scalars, pass.non_data_array_sizes
end

"""
    CheckRepeatedAssignments

BUGS generally forbids the same variable (scalar or array location) to appear more than once. The only exception
is when a variable appear exactly twice: one for logical assignment and one for stochastic assignment, and the variable
must be a transformed data.

In this pass, we check the following cases:
- A variable appear on the LHS of multiple logical assignments
- A variable appear on the LHS of multiple stochastic assignments
- Scalars appear on the LHS of both logical and stochastic assignments

The exceptional case will be checked after `DataTransformation` pass.
"""
struct CheckRepeatedAssignments <: CompilerPass
    env::NamedTuple
    overlap_scalars::Tuple{Vararg{Symbol}}
    logical_assignment_trackers::NamedTuple
    stochastic_assignment_trackers::NamedTuple
end

function CheckRepeatedAssignments(
    model_def::Expr, data::NamedTuple{data_vars}, non_data_array_sizes
) where {data_vars}
    # repeating assignments within deterministic and stochastic arrays are checked `extract_variables_assigned_to`
    logical_scalars, stochastic_scalars, logical_arrays, stochastic_arrays = extract_variables_assigned_to(
        model_def
    )

    overlap_scalars = Tuple(intersect(logical_scalars, stochastic_scalars))

    logical_assignment_trackers = Dict{Symbol,BitArray}()
    stochastic_assignment_trackers = Dict{Symbol,BitArray}()

    for v in logical_arrays
        # `v` can't be in data
        logical_assignment_trackers[v] = falses(non_data_array_sizes[v]...)
    end

    for v in stochastic_arrays
        array_size = if v in data_vars
            size(data[v])
        else
            non_data_array_sizes[v]
        end
        stochastic_assignment_trackers[v] = falses(array_size...)
    end

    return CheckRepeatedAssignments(
        data,
        overlap_scalars,
        NamedTuple(logical_assignment_trackers),
        NamedTuple(stochastic_assignment_trackers),
    )
end

function analyze_statement(
    pass::CheckRepeatedAssignments, expr::Expr, loop_vars::NamedTuple
)
    lhs_expr = is_deterministic(expr) ? expr.args[1] : expr.args[2]
    assignment_tracker = if is_deterministic(expr)
        pass.logical_assignment_trackers
    else
        pass.stochastic_assignment_trackers
    end

    env = merge(pass.env, loop_vars)
    lhs = simplify_lhs(env, lhs_expr)
    if !(lhs isa Symbol)
        v, indices... = lhs
        set_assignment_tracker!(assignment_tracker, v, indices...)
    end
end

function set_assignment_tracker!(
    assignment_tracker::NamedTuple, v::Symbol, indices::Vararg{Union{Int,UnitRange{Int}}}
)
    if any(assignment_tracker[v][indices...])
        indices = Tuple(findall(assignment_tracker[v][indices...]))
        error("$v already assigned at indices $indices")
    end
    if eltype(indices) == Int
        assignment_tracker[v][indices...] = true
    else
        assignment_tracker[v][indices...] .= true
    end
end

function post_process(pass::CheckRepeatedAssignments)
    suspect_arrays = Dict{Symbol,BitArray}()
    overlap_arrays = intersect(
        keys(pass.logical_assignment_trackers), keys(pass.stochastic_assignment_trackers)
    )
    for v in overlap_arrays
        if any(
            pass.logical_assignment_trackers[v] .& pass.stochastic_assignment_trackers[v]
        )
            suspect_arrays[v] =
                pass.logical_assignment_trackers[v] .&
                pass.stochastic_assignment_trackers[v]
        end
    end
    return pass.overlap_scalars, suspect_arrays
end

"""
    DataTransformation

Statements with a right-hand side that can be fully evaluated using the data are processed 
in this analysis pass, which computes these values. This achieves a similar outcome to 
Stan's `transformed parameters` block, but without requiring explicit specificity.

Conceptually, this is akin to `constant propagation` in compiler optimization, as both 
strategies aim to accelerate the optimized program by minimizing the number of operations.

It is crucial to highlight that computing data transformations plays a significant role 
in ensuring the compiler's correctness. BUGS prohibits the repetition of the same variable 
(be it a scalar or an array element) on the LHS more than once. The sole exception exists 
when the variable is computable within this pass, in which case it is regarded equivalently 
to data.
"""
mutable struct DataTransformation <: CompilerPass
    env::NamedTuple
    new_value_added::Bool
    already_warned::Ref{Bool}
end

function analyze_statement(pass::DataTransformation, expr::Expr, loop_vars)
    if is_deterministic(expr)
        lhs_expr, rhs_expr = expr.args[1], expr.args[2]
        env = merge(pass.env, loop_vars)
        lhs = simplify_lhs(env, lhs_expr)

        lhs_value = if lhs isa Symbol
            env[lhs]
        else
            var, indices... = lhs
            env[var][indices...]
        end
        # check if the value already exists
        if is_resolved(lhs_value)
            return nothing
        end

        rhs = evaluate(rhs_expr, env)
        if is_resolved(rhs)
            if !pass.already_warned[]
                displayed_lhs =
                    lhs isa Symbol ? lhs : String(lhs[1]) * "[$(join(lhs[2:end], ", "))]"
                @warn """
                Variables that can be directly computed from data are considered 'transformed data' in BUGS.
                These variables won't appear in the probabilistic graph.
                In this model, '$(displayed_lhs)' is one such variable.
                There may be others, but this warning is only shown once.
                If you're not working with existing BUGS programs, consider pre-computing these variables in Julia for efficiency.
                """
                pass.already_warned[] = true
            end
            pass.new_value_added = true
            pass_env = pass.env
            if lhs isa Symbol
                pass.env = BangBang.setproperty!!(pass_env, lhs, rhs)
            else
                var, indices... = lhs
                pass_env = pass.env
                pass.env = BangBang.setproperty!!(
                    pass_env, var, BangBang.setindex!!(pass_env[var], rhs, indices...)
                )
            end
        end
    end
end

"""
    evaluate_and_track_dependencies(var, env)

Evaluate `var` in the environment `env` while tracking its dependencies and node function arguments.

This function aims to extract two related but nuancedly different pieces of information:
    1. Fine-grained dependency information, which is used to construct the dependency graph.
    2. Variables used for node function arguments, which only care about the variable names and types (number or array), not the index.
    
The function returns three values:
    1. An evaluated `var`.
    2. A `Set` of dependency information.
    3. A `Set` of node function arguments information.

Array elements and array variables are represented by tuples in the returned value. All `Colon` indexing is assumed to be concretized.

# Examples
```jldoctest; setup = :(using JuliaBUGS: evaluate_and_track_dependencies)
julia> evaluate_and_track_dependencies(:(x[a]), (x=[missing, missing], a = missing))
(missing, (:a, (:x, 1:2)))

julia> evaluate_and_track_dependencies(:(x[a]), (x=[missing, missing], a = 1))
(missing, ((:x, 1),))

julia> evaluate_and_track_dependencies(:(x[y[1]+1]+a+1), (x=[missing, missing], y = [missing, missing], a = missing))
(missing, ((:y, 1), (:x, 1:2), :a))

julia> evaluate_and_track_dependencies(:(x[a, b]), (x = [1 2 3; 4 5 6], a = missing, b = missing))
(missing, (:a, :b, (:x, 1:2, 1:3)))

julia> evaluate_and_track_dependencies(:(getindex(x[1:2, 1:3], a, b)), (x = [1 2 3; 4 5 6], a = missing, b = missing))
(missing, (:a, :b))

julia> evaluate_and_track_dependencies(:(getindex(x[1:2, 1:3], 1, 1)), (x = [1 2 3; 4 5 6], a = missing, b = missing))
(1, ())

julia> evaluate_and_track_dependencies(:(getindex(x[1:2, 1:3], a, b)), (x = [1 2 missing; 4 5 6], a = missing, b = missing))
(missing, ((:x, 1:2, 1:3), :a, :b))
```
"""
evaluate_and_track_dependencies(var::Union{Int,Float64}, env) = var, ()
evaluate_and_track_dependencies(var::UnitRange, env) = var, ()
function evaluate_and_track_dependencies(var::Symbol, env)
    if var ∈ (:nothing, :missing, :Inf, :(:))
        return var, ()
    end
    if env[var] === missing
        return var, (var,)
    else
        return env[var], ()
    end
end
function evaluate_and_track_dependencies(var::Expr, env)
    dependencies = []
    if Meta.isexpr(var, :ref)
        v, indices... = var.args
        for i in eachindex(indices)
            ret = evaluate_and_track_dependencies(indices[i], env)
            index = ret[1]
            indices[i] = index isa Float64 ? Int(index) : index
            dependencies = union!(dependencies, ret[2])
        end

        value = nothing
        if all(i -> i isa Int || i isa UnitRange{Int}, indices)
            if any(last.(indices) .> size(env[v]))
                error(
                    "$v[$(join(indices, ", "))] is used in the model, but it or its elements are not defined in the model or data.",
                )
            end
            value = env[v][indices...]
            if is_resolved(value)
                return value, Tuple(dependencies)
            else
                push!(dependencies, (v, indices...))
            end
        else
            push!(
                dependencies,
                (
                    v,
                    [
                        is_resolved(index) ? index : 1:size(env[v])[i] for
                        (i, index) in enumerate(indices)
                    ]...,
                ),
            )
        end
        return missing, Tuple(dependencies)
    elseif Meta.isexpr(var, :call)
        f, args... = var.args
        value = nothing
        for i in eachindex(args)
            ret = evaluate_and_track_dependencies(args[i], env)
            args[i] = ret[1]
            union!(dependencies, ret[2])
        end

        value = nothing
        if all(is_resolved, args) &&
            f ∈ BUGSPrimitives.BUGS_FUNCTIONS ∪ (:+, :-, :*, :/, :^, :(:), :getindex)
            return getfield(JuliaBUGS, f)(args...), Tuple(dependencies)
        else
            return missing, Tuple(dependencies)
        end
    else
        error("Unexpected expression type: $var")
    end
end

"""
    AddVertices

This pass will add a vertex for every instance of LHS in the model. 

The node functions are the same for all the nodes whose corresponding LHS are originated from the same statement. 
The values of loop variables at the time LHS is evaluated will be saved. 

`vertex_id_tracker` tracks the vertex ID of each variable in the model. This is used to efficiently decide target 
vertices in pass `AddEdges`.
"""
mutable struct AddVertices <: CompilerPass
    const env::NamedTuple
    const g::MetaGraph
    vertex_id_tracker::NamedTuple
    const f_dict::Dict{Expr,Tuple{Tuple{Vararg{Symbol}},Expr,Any}}
end

function AddVertices(model_def::Expr, eval_env::NamedTuple)
    g = MetaGraph(DiGraph(); label_type=VarName, vertex_data_type=NodeInfo)
    vertex_id_tracker = Dict{Symbol,Any}()
    for (k, v) in pairs(eval_env)
        if v isa AbstractArray
            vertex_id_tracker[k] = zeros(Int, size(v))
        else
            vertex_id_tracker[k] = 0
        end
    end

    f_dict = build_node_functions(
        model_def, eval_env, Dict{Expr,Tuple{Tuple{Vararg{Symbol}},Expr,Any}}(), ()
    )

    return AddVertices(eval_env, g, NamedTuple(vertex_id_tracker), f_dict)
end

function build_node_functions(
    expr::Expr,
    eval_env::NamedTuple,
    f_dict::Dict{Expr,Tuple{Tuple{Vararg{Symbol}},Expr,Any}},
    loop_vars::Tuple{Vararg{Symbol}},
)
    for statement in expr.args
        if is_deterministic(statement) || is_stochastic(statement)
            lhs, rhs = if is_deterministic(statement)
                statement.args[1], statement.args[2]
            else
                statement.args[2], statement.args[3]
            end
            args, node_func_expr = make_function_expr(lhs, rhs, eval_env)
            node_func = eval(node_func_expr)
            f_dict[statement] = (args, node_func_expr, node_func)
        elseif Meta.isexpr(statement, :for)
            loop_var, _, _, body = decompose_for_expr(statement)
            build_node_functions(body, eval_env, f_dict, (loop_var, loop_vars...))
        else
            error("Unknown statement type: $statement")
        end
    end
    return f_dict
end

"""
    make_function_expr(lhs, rhs, env::NamedTuple{vars}; use_lhs_as_func_name=false)

Generate a function expression for the given right-hand side expression `rhs`. The generated function will take
a `NamedTuple` as its argument, which contains the values of the variables used in `rhs`.

# Examples
```jldoctest; setup = :(using JuliaBUGS: make_function_expr)
julia> make_function_expr(:(x[a, b]), :(x[a, b] + 1), (x = [1 2 3; 4 5 6], a = missing, b = missing))
((:a, :b, :x), :(function (evaluation_env, loop_vars)
      (; a, b, x) = evaluation_env
      (;) = loop_vars
      return x[Int(a), Int(b)] + 1
  end))

julia> make_function_expr(:(x[a, b]), :(x[a, b] + 1), (;x = [1 2 3; 4 5 6]))
((:a, :b, :x), :(function (evaluation_env, loop_vars)
      (; x) = evaluation_env
      (; a, b) = loop_vars
      return x[Int(a), Int(b)] + 1
  end))
```
"""
function make_function_expr(
    lhs, rhs, env::NamedTuple{vars}; use_lhs_as_func_name=false
) where {vars}
    args = Tuple(keys(extract_variable_names_and_numdims(rhs, ())))
    loop_vars = Tuple([v for v in args if v ∉ vars])
    variables = setdiff(args, loop_vars)
    # arg_exprs = Expr[]
    # for v in args
    #     if v ∈ vars
    #         value = env[v]
    #         if value isa Int || value isa Float64 || value isa Missing
    #             push!(arg_exprs, Expr(:(::), v, :Real))
    #         elseif value isa AbstractArray
    #             push!(arg_exprs, Expr(:(::), v, :(Array{<:Real})))
    #         else
    #             error("Unexpected argument type: $(typeof(value))")
    #         end
    #     else # loop variable
    #         push!(arg_exprs, Expr(:(::), v, :Int))
    #     end
    # end

    unpacking_expr = :((; $(variables...),) = evaluation_env)
    unpacking_loop_vars_expr = :((; $(loop_vars...),) = loop_vars)

    func_body = MacroTools.postwalk(rhs) do sub_expr
        if @capture(sub_expr, v_[indices__])
            new_indices = Any[]
            for i in eachindex(indices)
                if indices[i] isa Int # special case: already an Int
                    push!(new_indices, indices[i])
                elseif indices[i] isa Symbol || Meta.isexpr(indices[i], :ref) # cast to Int if it's a variable
                    push!(new_indices, Expr(:call, :Int, indices[i]))
                else # function and range are not casted
                    push!(new_indices, indices[i])
                end
            end
            return Expr(:ref, v, new_indices...)
        end
        return sub_expr
    end

    # if use_lhs_as_func_name
    #     func_name = if lhs isa Symbol
    #         lhs
    #     else
    #         Symbol("__", String(lhs.args[1]), "_", join(lhs.args[2:end], "_"), "__")
    #     end

    #     return args, MacroTools.@q function $func_name($(arg_exprs...))
    #         return $(func_body)
    #     end
    # else
    #     return args, MacroTools.@q function ($(arg_exprs...))
    #         return $(func_body)
    #     end
    # end

    return args, MacroTools.@q function (evaluation_env, loop_vars)
        $(unpacking_expr)
        $(unpacking_loop_vars_expr)
        return $(func_body)
    end
end

function analyze_statement(pass::AddVertices, expr::Expr, loop_vars::NamedTuple)
    lhs_expr = is_deterministic(expr) ? expr.args[1] : expr.args[2]
    env = merge(pass.env, loop_vars)
    lhs = simplify_lhs(env, lhs_expr)
    is_stochastic = false
    is_observed = false
    lhs_value = if lhs isa Symbol
        env[lhs]
    else
        var, indices... = lhs
        env[var][indices...]
    end
    if Meta.isexpr(expr, :(=))
        if is_resolved(lhs_value)
            return nothing
        end
    else
        is_stochastic = true
        if is_resolved(lhs_value)
            is_observed = true
        end
    end

    args, node_function_expr, node_function = pass.f_dict[expr]

    vn = if lhs isa Symbol
        AbstractPPL.VarName{lhs}(identity)
    else
        v, indices... = lhs
        AbstractPPL.VarName{v}(AbstractPPL.IndexLens(indices))
    end
    add_vertex!(
        pass.g,
        vn,
        NodeInfo(
            is_stochastic, is_observed, node_function_expr, node_function, args, loop_vars
        ),
    )
    if lhs isa Symbol
        pass.vertex_id_tracker = BangBang.setproperty!!(
            pass.vertex_id_tracker, lhs, code_for(pass.g, vn)
        )
    else
        v, indices... = lhs
        if any(i -> i isa UnitRange, indices)
            pass.vertex_id_tracker[v][indices...] .= code_for(pass.g, vn)
        else
            pass.vertex_id_tracker[v][indices...] = code_for(pass.g, vn)
        end
    end
end

"""
    AddEdges

This pass will add edges to the graph constructed in pass `AddVertices`. 
"""
struct AddEdges <: CompilerPass
    env::NamedTuple
    g::MetaGraph
    vertex_id_tracker::NamedTuple
end

function analyze_statement(pass::AddEdges, expr::Expr, loop_vars::NamedTuple)
    lhs_expr, rhs_expr = is_deterministic(expr) ? expr.args[1:2] : expr.args[2:3]
    env = merge(pass.env, loop_vars)
    lhs = simplify_lhs(env, lhs_expr)
    lhs_value = if lhs isa Symbol
        env[lhs]
    else
        var, indices... = lhs
        env[var][indices...]
    end
    if Meta.isexpr(expr, :(=)) && is_resolved(lhs_value)
        return nothing
    end

    _, dependencies = evaluate_and_track_dependencies(rhs_expr, env)

    lhs_vn = if lhs isa Symbol
        @varname($lhs)
    else
        v, indices... = lhs
        AbstractPPL.VarName{v}(AbstractPPL.IndexLens(indices))
    end

    for var in dependencies
        vertex_code = if var isa Symbol
            _vertex_code = pass.vertex_id_tracker[var]
            if _vertex_code isa AbstractArray
                error("$(var) is an array, but referenced as a scalar at $expr")
            end
            [_vertex_code]
        else
            v, indices... = var
            for idx in Iterators.product(indices...)
                if iszero(pass.vertex_id_tracker[v][idx...]) && ismissing(pass.env[v][idx...])
                    error(
                        "Variable $v[$(join(indices, ", "))] is referenced, but not defined, at $expr.",
                    )
                end
            end
            _vertex_code = pass.vertex_id_tracker[v][indices...]
            if _vertex_code isa AbstractArray
                filter(!iszero, _vertex_code)
            else
                iszero(_vertex_code) ? [] : [_vertex_code]
            end
        end
        vertex_labels = [label_for(pass.g, code) for code in vertex_code]
        for r in vertex_labels
            if r != lhs_vn
                add_edge!(pass.g, r, lhs_vn)
            end
        end
    end
end
