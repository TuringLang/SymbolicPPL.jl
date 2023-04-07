"""
    CompilerPass

Abstract supertype for all compiler passes. Concrete subtypes should store data needed and artifacts.
"""
abstract type CompilerPass end

"""
    program!(pass::CompilerPass, expr::Expr, env::Dict, vargs...)

All compiler pass share the same interface. `program!` is the entry point for the compiler pass. It
traverses the AST and calls `assignment!` and `tilde_assignment!` for each assignment. It also calls
`for_loop!` for each for loop. Finally, it calls `post_process` to do any post processing.
"""
function program!(pass::CompilerPass, expr::Expr, env::Dict, vargs...)
    for ex in expr.args
        if Meta.isexpr(ex, [:(=), :(~)])
            assignment!(pass, ex, env, vargs...)
        elseif Meta.isexpr(ex, :for)
            for_loop!(pass, ex, env, vargs...)
        else
            error()
        end
    end
    return post_process(pass, expr, env, vargs...)
end

function for_loop!(pass::CompilerPass, expr, env, vargs...)
    loop_var = expr.args[1].args[1]
    lb, ub = expr.args[1].args[2].args
    body = expr.args[2]
    lb, ub = eval_(lb, env), eval_(ub, env)
    @assert lb isa Int && ub isa Int "Only integer ranges are supported"
    for i in lb:ub
        for ex in body.args
            if Meta.isexpr(ex, [:(=), :(~)])
                assignment!(pass, ex, merge(env, Dict(loop_var => i)), vargs...)
            elseif Meta.isexpr(ex, :for)
                for_loop!(pass, ex, merge(env, Dict(loop_var => i)), vargs...)
            else
                error()
            end
        end
    end
end

function assignment!(::CompilerPass, expr::Expr, env::Dict, vargs...) end

function post_process(pass::CompilerPass, expr, env, vargs...) end

@enum VariableTypes begin
    Logical
    Stochastic
end

"""
    CollectVariables

This pass collects all the possible variables appear on the LHS of both logical and stochastic assignments. 
"""
struct CollectVariables <: CompilerPass
    vars::Dict{Var, VariableTypes}
    transformed_variables::Dict{Var, Union{Real, Array{<:Real}}}
end
CollectVariables() = CollectVariables(Dict{Var, VariableTypes}(), Dict{Var, Union{Real, Array{<:Real}}}())

"""
    find_variables_on_lhs(expr, env)

Find all the variables on the LHS of an assignment. The variables can be either symbols or array indexing.

# Examples
```jldoctest
julia> find_variables_on_lhs(:(x[1, 2]), Dict())
Var(:x, [1, 2])

julia> find_variables_on_lhs(:(x[1, 2:3]), Dict())
Var(:x, [1, 2:3])
```
"""
find_variables_on_lhs(e::Symbol, ::Dict) = Var(e)
function find_variables_on_lhs(expr::Expr, env::Dict)
    @assert Meta.isexpr(expr, :ref)
    idxs = map(x -> eval_(x, env), expr.args[2:end])
    return Var(expr.args[1], Tuple(idxs))
end

"""
    check_idxs(v_name::Symbol, idxs::Array, env::Dict)

Check if the indices are valid. The indices can be either numbers, unit ranges, or colons. If the variable is a data array,
check if the indices are out of bound.

# Examples
```jldoctest
julia> check_idxs(:x, (1:2,), Dict(:x => [1, missing]))
ERROR: Some elements of x[1:2] are specified by data, some are not.
[...]

julia> check_idxs(:x, (f(y), 2:3)), Dict())
ERROR: Some indices on the lhs can't be fully resolved. Argument 1: f(y). 
[...]
```
"""
function check_idxs(v_name::Symbol, idxs::NTuple, env::Dict)
    # check if some index is not resolved
    unresolved_indices = findall(x -> !isa(x, Union{Number, UnitRange, Colon}), idxs)
    if !isempty(unresolved_indices)
        msg = "Some indices on the lhs can't be fully resolved. "
        for i in unresolved_indices
            msg *= "Argument $i: $(expr.args[i+1]). "
        end
        error(msg)
    end
    # if the array is a data array, check if the index is out of bound
    if v_name in keys(env)
        @assert isequal(length(idxs), ndims(env[v_name])) "Dimension mismatch."
        for i in 1:length(idxs)
            if idxs[i] isa Number
                @assert idxs[i] <= size(env[v_name], i) "Index out of bound."
            elseif idxs[i] isa UnitRange
                @assert idxs[i].stop <= size(env[v_name], i) "Index out of bound."
            end
        end
    end
    # check colon index only allow in data array
    colon_idxs = findall(x -> x == Colon(), idxs)
    if !isempty(colon_idxs)
        if !haskey(v_name, env)
            error("Implicit indexing with colon is only supported when the array is a data array.")
        end
    end
    # if the variable is multi-dimensional and data, check they must all be missing or all provided
    if v_name in keys(env) && any(x -> x isa Union{UnitRange, Colon}, idxs)
        vs = env[v_name][idxs...]
        if !all(ismissing, vs) && !all(!ismissing, vs)
            error("Some elements of $v_name[$(idxs...)] are missing, some are not.")
        end
    end
end

function assignment!(pass::CollectVariables, expr::Expr, env::Dict)
    lhs_expr, rhs_expr = expr.args[1:2]

    v = find_variables_on_lhs(Meta.isexpr(lhs_expr, :call) ? lhs_expr.args[2] : lhs_expr, env)
    !isa(v, Scalar) && check_idxs(v.name, v.indices, env)
    !isnothing(eval_(v, env)) && Meta.isexpr(expr, :(=)) && error("$v is data, can't be assigned to.")
    
    var_type = Meta.isexpr(expr, :(=)) ? Logical : Stochastic
    haskey(pass.vars, v) && var_type == pass.vars[v] && error("Repeated assignment to $v.")
    if var_type == Logical
        rhs = eval_(rhs_expr, env)
        can_evaluate = (rhs isa Union{Number, Array{<:Number}}) ? true : false
        can_evaluate && (pass.transformed_variables[v] = rhs)
        haskey(pass.vars, v) && !can_evaluate && 
            error("$v is assigned to by both logical and stochastic assignments, 
            only allowed when the variable is a transformation of data.")
        haskey(pass.vars, v) && (var_type = Stochastic)
    end
    pass.vars[v] = var_type
end

function post_process(pass::CollectVariables, expr, env::Dict)
    array_elements = Dict([v.name => [] for v in keys(pass.vars) if v.indices != ()])
    for v in keys(pass.vars)
        !isa(v, Scalar) && push!(array_elements[v.name], v)
    end

    array_sizes = Dict{Symbol, Vector{Int}}()
    for (k, v) in array_elements
        k in keys(env) && continue # skip data arrays
        numdims = length(v[1].indices)
        @assert all(x -> length(x.indices) == numdims, v) "$k dimension mismatch."
        array_size = Vector(undef, numdims)
        for i in 1:numdims
            array_size[i] = maximum(x -> isa(x.indices[i], Number) ? x.indices[i] : x.indices[i].stop, v)
        end
        array_sizes[k] = array_size
    end

    transformed_variables = Dict()
    for tv in keys(pass.transformed_variables)
        if tv isa Scalar
            transformed_variables[tv] = pass.transformed_variables[tv]
        else
            if !haskey(transformed_variables, tv.name)
                tvs = fill(missing, array_sizes[tv.name]...)
                transformed_variables[tv.name] = convert(Array{Union{Missing, Number}}, tvs)
            end
            transformed_variables[tv.name][tv.indices...] = pass.transformed_variables[tv]
        end
    end
    for (k, v) in transformed_variables
        if !any(ismissing, v)
            transformed_variables[k] = convert(Array{Number}, v)
        end
    end

    # scalar is already checked in `assignment!`
    logical_bitmap = Dict([k => falses(v...) for (k, v) in array_sizes])
    stochastic_bitmap = deepcopy(logical_bitmap)
    for (k, v) in pass.vars
        k isa Scalar && continue
        k.name in keys(env) && continue # skip data arrays
        bitmap = k == Logical ? logical_bitmap : stochastic_bitmap
        for v_ in scalarize(k)
            if bitmap[v_.name][v_.indices...]
                error("Repeated assignment to $v_.")
            else
                bitmap[v_.name][v_.indices...] = true
            end
        end
    end

    # corner case: x[1:2] = something, x[3] = something, x[1:3] ~ dmnorm()
    overlap = Dict()
    for k in keys(logical_bitmap)
        overlap[k] = logical_bitmap[k]  .⊽ stochastic_bitmap[k]
    end

    for (k, v) in overlap
        if any(v)
            idxs = findall(v)
            for i in idxs
                !haskey(transformed_variables, k) && error("Logical and stochastic variables overlap on $k[$(i...)].")
                transformed_variables[k][i...]!= missing && continue
                error("Logical and stochastic variables overlap on $k[$(i...)].")
            end
        end
    end

    # used to check if a variable is defined on the lhs
    array_bitmap = Dict()
    for k in keys(logical_bitmap)
        array_bitmap[k] = logical_bitmap[k] .| stochastic_bitmap[k]
    end
    
    return pass.vars, array_sizes, transformed_variables, array_bitmap
end
