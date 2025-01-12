module CompilerUtils

using BangBang
using MacroTools

export create_eval_env,
    concretize_eval_env,
    decompose_for_expr,
    extract_variable_names_and_numdims,
    extract_variables_in_bounds_and_lhs_indices,
    extract_variables_assigned_to,
    concretize_colon_indexing,
    simplify_lhs,
    simple_arithmetic_eval

"""
    create_eval_env(non_data_scalars, non_data_array_sizes, data)

Constructs an `NamedTuple` containing all the variables defined or used in the program. 

Arrays given by data will only be copied if they contain `missing` values. This copy behavior ensures 
that the evaluation environment is a self-contained snapshot, avoiding unintended side effects on the input data.

Variables not given by data will be assigned `missing` values.
"""
function create_eval_env(
    non_data_scalars::Tuple{Vararg{Symbol}},
    non_data_array_sizes::NamedTuple{non_data_array_vars},
    data::NamedTuple{data_vars},
) where {data_vars,non_data_array_vars}
    eval_env = Dict{Symbol,Any}()
    for k in data_vars
        v = data[k]
        if v isa AbstractArray
            if Base.nonmissingtype(eltype(v)) === eltype(v)
                eval_env[k] = v
            elseif eltype(v) === Missing
                eval_env[k] = fill(missing, size(v)...)
            else
                eval_env[k] = copy(data[k])
            end
        else
            eval_env[k] = v
        end
    end

    for s in non_data_scalars
        eval_env[s] = missing
    end

    for a in non_data_array_vars
        eval_env[a] = fill(missing, non_data_array_sizes[a]...)
    end

    return NamedTuple(eval_env)
end

"""
    concretize_eval_env(eval_env::NamedTuple)

For arrays in `eval_env`, if its `eltype` is `Union{Missing, T}` where `T` is a concrete type, then 
it tries to convert the array to `AbstractArray{T}`. If the conversion is not possible, it leaves 
the array unchanged.

# Examples
```jldoctest; setup = :(using JuliaBUGS: concretize_eval_env)
julia> concretize_eval_env((a = Union{Missing,Int}[1, 2, 3],))
(a = [1, 2, 3],)

```
"""
function concretize_eval_env(eval_env::NamedTuple)
    for k in keys(eval_env)
        v = eval_env[k]
        if v isa AbstractArray
            try
                disallowmissing_v = convert(
                    AbstractArray{Base.nonmissingtype(eltype(v))}, v
                )
                eval_env = BangBang.setproperty!!(eval_env, k, disallowmissing_v)
            catch _
            end
        end
    end
    return eval_env
end

"""
    decompose_for_expr(expr::Expr)

Decompose a for-loop expression into its components. The function returns four items: the 
loop variable, the lower bound, the upper bound, and the body of the loop.
"""
@inline function decompose_for_expr(expr::Expr)
    loop_var::Symbol = expr.args[1].args[1]
    lb::Union{Int,Float64,Symbol,Expr} = expr.args[1].args[2].args[2]
    ub::Union{Int,Float64,Symbol,Expr} = expr.args[1].args[2].args[3]
    body::Expr = expr.args[2]
    return loop_var, lb, ub, body
end

"""
    extract_variable_names_and_numdims(expr, excluded)

Extract all the array variable names and number of dimensions from a given simple expression. 

# Examples:
```jldoctest; setup = :(using JuliaBUGS: extract_variable_names_and_numdims)
julia> extract_variable_names_and_numdims(:((a + b) * c), ())
(a = 0, b = 0, c = 0)

julia> extract_variable_names_and_numdims(:((a + b) * c), (:a,))
(b = 0, c = 0)

julia> extract_variable_names_and_numdims(:(a[i]), ())
(a = 1, i = 0)

julia> extract_variable_names_and_numdims(:(a[i]), (:i,))
(a = 1,)

julia> extract_variable_names_and_numdims(42, ())
NamedTuple()

julia> extract_variable_names_and_numdims(:x, (:x,))
NamedTuple()

julia> extract_variable_names_and_numdims(:(x[1, :]), ())
(x = 2,)
```
"""
function extract_variable_names_and_numdims(::Union{Int,Float64}, ::Tuple{Vararg{Symbol}})
    return (;)
end
function extract_variable_names_and_numdims(expr::Symbol, excluded::Tuple{Vararg{Symbol}})
    return if expr in excluded || expr in (:missing, :nothing)
        (;)
    else
        NamedTuple{(expr,)}((0,))
    end
end
function extract_variable_names_and_numdims(expr::Expr, excluded::Tuple{Vararg{Symbol}})
    variables = Dict{Symbol,Int}()
    MacroTools.prewalk(expr) do sub_expr
        if !(sub_expr isa Expr)
            return sub_expr
        end
        if @capture(sub_expr, f_(args__))
            for arg in args
                if arg isa Symbol && arg ∉ (:nothing, :missing, :Inf) && !(arg in excluded)
                    variables[arg] = 0
                end
            end
        elseif @capture(sub_expr, v_[idxs__])
            variables[v] = length(idxs)
            for idx in idxs
                if idx isa Symbol && idx !== :(:) && !(idx in excluded)
                    variables[idx] = 0
                end
            end
        end
        return sub_expr
    end
    return NamedTuple(variables)
end

"""
    extract_variable_names_and_numdims(expr::Expr)
   
Extract all the array variable names and number of dimensions. Inconsistent number of dimensions
will raise an error.

# Example:
```jldoctest
JuliaBUGS.extract_variable_names_and_numdims(
    @bugs begin
        for i in 1:N
            for j in 1:T
                Y[i, j] = _step((var"obs.t"[i] - t[j]) + eps)
                dN[i, j] = Y[i, j] * _step((t[j + 1] - var"obs.t"[i]) - eps) * fail[i]
            end
        end
        for j in 1:T
            for i in 1:N
                dN[i, j] ~ dpois(Idt[i, j])
                Idt[i, j] = Y[i, j] * exp(beta * Z[i]) * dL0[j]
            end
            dL0[j] ~ dgamma(mu[j], c)
            mu[j] = var"dL0.star"[j] * c
            var"S.treat"[j] = pow(exp(-(sum(dL0[1:j]))), exp(beta * -0.5))
            var"S.placebo"[j] = pow(exp(-(sum(dL0[1:j]))), exp(beta * 0.5))
        end
        c = 0.001
        r = 0.1
        for j in 1:T
            var"dL0.star"[j] = r * (t[j + 1] - t[j])
        end
        beta ~ dnorm(0.0, 1.0e-6)
    end
)

# output

(N = 0, T = 0, Y = 2, var"obs.t" = 1, eps = 0, t = 1, dN = 2, fail = 1, Idt = 2, Z = 1, beta = 0, dL0 = 1, mu = 1, c = 0, var"dL0.star" = 1, var"S.treat" = 1, var"S.placebo" = 1, r = 0)
```
"""
function extract_variable_names_and_numdims(expr::Expr)
    return extract_array_ndims_block(expr, (), NamedTuple())
end

function extract_array_ndims_block(
    expr::Expr, loop_vars::Tuple{Vararg{Symbol}}, array_ndims::NamedTuple{Ns,Ts}
) where {Ns,Ts}
    for stmt in expr.args
        if @capture(stmt, (lhs_ = rhs_) | (lhs_ ~ rhs_))
            array_ndims = extract_array_ndims_expr(
                rhs, loop_vars, extract_array_ndims_expr(lhs, loop_vars, array_ndims)
            )
        elseif @capture(
            stmt,
            for loop_var_ in lower_:upper_
                body_
            end
        )
            array_ndims = extract_array_ndims_block(
                body,
                (loop_vars..., loop_var),
                extract_array_ndims_expr(
                    upper,
                    loop_vars,
                    extract_array_ndims_expr(lower, loop_vars, array_ndims),
                ),
            )
        end
    end
    return array_ndims
end

function extract_array_ndims_expr(
    ::Union{Int,Float64}, ::Tuple{Vararg{Symbol}}, array_dims::NamedTuple
)
    return array_dims
end
function extract_array_ndims_expr(
    expr::Union{Symbol,Expr},
    loop_vars::Tuple{Vararg{Symbol}},
    array_ndims::NamedTuple{Ns,Ts},
) where {Ns,Ts}
    variables = extract_variable_names_and_numdims(expr, loop_vars)
    for var in intersect(keys(variables), Ns)
        if variables[var] !== array_ndims[var]
            error("Variable $var has inconsistent dimensions")
        end
    end
    return merge(array_ndims, variables)
end

"""
    extract_variables_in_bounds_and_lhs_indices(expr::Expr)

Extract all the variable names used in the bounds and indices of the arrays in the program.

# Example:
```jldoctest
JuliaBUGS.extract_variables_in_bounds_and_lhs_indices(
    @bugs begin
        for i in 1:N
            for j in 1:T
                Y[i, j] = _step((var"obs.t"[i] - t[j]) + eps)
                dN[i, j] = Y[i, j] * _step((t[j + 1] - var"obs.t"[i]) - eps) * fail[i]
            end
        end
        for j in 1:T
            for i in 1:N
                dN[i, j] ~ dpois(Idt[i, j])
                Idt[i, j] = Y[i, j] * exp(beta * Z[i]) * dL0[j]
            end
            dL0[j] ~ dgamma(mu[j], c)
            mu[j] = var"dL0.star"[j] * c
            var"S.treat"[j] = pow(exp(-(sum(dL0[1:j]))), exp(beta * -0.5))
            var"S.placebo"[j] = pow(exp(-(sum(dL0[1:j]))), exp(beta * 0.5))
        end
        c = 0.001
        r = 0.1
        for j in 1:T
            var"dL0.star"[j] = r * (t[j + 1] - t[j])
        end
        beta ~ dnorm(0.0, 1.0e-6)
    end
)

# output

(:N, :T)
```
"""
function extract_variables_in_bounds_and_lhs_indices(expr::Expr)
    return extract_variables_in_bounds_and_lhs_indices(expr, (), ())
end
function extract_variables_in_bounds_and_lhs_indices(
    expr::Expr, loop_vars::Tuple{Vararg{Symbol}}, variables::Tuple{Vararg{Symbol}}
)
    for stmt in expr.args
        if @capture(stmt, (lhs_ = rhs_) | (lhs_ ~ rhs_))
            if @capture(lhs, v_[indices__])
                for index in indices
                    variables = extract_variables_used_in_expr(index, loop_vars, variables)
                end
            end
        elseif @capture(
            stmt,
            for loop_var_ in lower_:upper_
                body_
            end
        )
            variables = extract_variables_in_bounds_and_lhs_indices(
                body,
                (loop_vars..., loop_var),
                extract_variables_used_in_expr(
                    upper,
                    loop_vars,
                    extract_variables_used_in_expr(lower, loop_vars, variables),
                ),
            )
        end
    end
    return variables
end

function extract_variables_used_in_expr(
    expr::Union{Int,Float64,Symbol,Expr},
    loop_vars::Tuple{Vararg{Symbol}},
    variables::Tuple{Vararg{Symbol}},
)::Tuple{Vararg{Symbol}}
    vars = extract_variable_names_and_numdims(expr, loop_vars)
    if vars === (;)
        return variables
    end
    return Tuple(union(variables, keys(vars)))
end

function simplify_lhs(::NamedTuple, lhs::Symbol)
    return lhs
end
function simplify_lhs(data::NamedTuple, lhs::Expr)
    if Meta.isexpr(lhs, :ref)
        var = lhs.args[1]
        indices = lhs.args[2:end]
        for i in eachindex(indices)
            indices[i] = simple_arithmetic_eval(data, indices[i])
        end
        return (var, indices...)
    else
        error(
            "LHS of a statement can only be a symbol or an indexing expression, but get $lhs.",
        )
    end
end

"""
    extract_variables_assigned_to(expr::Expr)

Returns four tuples contains the Symbol of the variable assigned to in the program.
The first tuple contains the logical scalar variables, the second tuple contains the stochastic scalar variables,
the third tuple contains the logical array variables, and the fourth tuple contains the stochastic array variables.

# Example:
```jldoctest
JuliaBUGS.extract_variables_assigned_to(
    @bugs begin
        for i in 1:N
            for j in 1:T
                Y[i, j] = _step((var"obs.t"[i] - t[j]) + eps)
                dN[i, j] = Y[i, j] * _step((t[j + 1] - var"obs.t"[i]) - eps) * fail[i]
            end
        end
        for j in 1:T
            for i in 1:N
                dN[i, j] ~ dpois(Idt[i, j])
                Idt[i, j] = Y[i, j] * exp(beta * Z[i]) * dL0[j]
            end
            dL0[j] ~ dgamma(mu[j], c)
            mu[j] = var"dL0.star"[j] * c
            var"S.treat"[j] = pow(exp(-(sum(dL0[1:j]))), exp(beta * -0.5))
            var"S.placebo"[j] = pow(exp(-(sum(dL0[1:j]))), exp(beta * 0.5))
        end
        c = 0.001
        r = 0.1
        for j in 1:T
            var"dL0.star"[j] = r * (t[j + 1] - t[j])
        end
        beta ~ dnorm(0.0, 1.0e-6)
    end
)

# output

((:c, :r), (:beta,), (Symbol("dL0.star"), :dN, :mu, Symbol("S.treat"), Symbol("S.placebo"), :Y, :Idt), (:dN, :dL0))
```
"""
function extract_variables_assigned_to(expr::Expr)
    return Tuple.(
        extract_variables_assigned_to(
            expr, Symbol[], Symbol[], Set{Symbol}(), Set{Symbol}()
        )
    )
end
function extract_variables_assigned_to(
    expr::Expr,
    logical_scalars::Vector{Symbol},
    stochastic_scalars::Vector{Symbol},
    logical_arrays::Set{Symbol},
    stochastic_arrays::Set{Symbol},
)
    for stmt in expr.args
        if @capture(stmt, (lhs_ = rhs_))
            if lhs isa Symbol
                if lhs in logical_scalars
                    error("Logical scalar variable $lhs is assigned to more than once")
                end
                push!(logical_scalars, lhs)
            else
                push!(logical_arrays, lhs.args[1])
            end
        elseif @capture(stmt, (lhs_ ~ rhs_))
            if lhs isa Symbol
                if lhs in stochastic_scalars
                    error("Stochastic scalar variable $lhs is assigned to more than once")
                end
                push!(stochastic_scalars, lhs)
            else
                push!(stochastic_arrays, lhs.args[1])
            end
        elseif @capture(
            stmt,
            for loop_var_ in lower_:upper_
                body_
            end
        )
            extract_variables_assigned_to(
                body, logical_scalars, stochastic_scalars, logical_arrays, stochastic_arrays
            )
        end
    end
    return logical_scalars, stochastic_scalars, logical_arrays, stochastic_arrays
end

"""
    concretize_colon_indexing(expr, eval_env::NamedTuple)

Replace all `Colon()`s in `expr` with the corresponding array size.

# Examples
```jldoctest
julia> JuliaBUGS.concretize_colon_indexing(:(f(x[1, :])), (x = [1 2 3 4; 5 6 7 8; 9 10 11 12],))
:(f(x[1, 1:4]))
```
"""
function concretize_colon_indexing(expr, eval_env::NamedTuple)
    return MacroTools.postwalk(expr) do sub_expr
        if Meta.isexpr(sub_expr, :ref)
            v, indices... = sub_expr.args
            for i in eachindex(indices)
                if indices[i] == :(:)
                    indices[i] = Expr(:call, :(:), 1, size(eval_env[v])[i])
                end
            end
            return Expr(:ref, v, indices...)
        end
        return sub_expr
    end
end

"""
    simple_arithmetic_eval(data, expr)

This function evaluates expressions that consist solely of arithmetic operations and indexing. It 
is specifically designed for scenarios such as calculating array indices or determining loop boundaries.

# Example:
```jldoctest; setup = :(using JuliaBUGS: simple_arithmetic_eval)
julia> simple_arithmetic_eval((a = 1, b = [1, 2]), 1)
1

julia> simple_arithmetic_eval((a = 1, b = [1, 2]), :a)
1

julia> simple_arithmetic_eval((a = 1, b = [1, 2]), :(a + b[1]))
2
```
"""
function simple_arithmetic_eval(data::NamedTuple, expr::Union{Int,UnitRange{Int}})
    return expr
end
function simple_arithmetic_eval(data::NamedTuple{names,Ts}, expr::Symbol) where {names,Ts}
    if expr ∉ names
        throw(ArgumentError("Don't know the value of $expr."))
    end
    return Int(data[expr])
end
function simple_arithmetic_eval(data::NamedTuple, expr::Expr)
    if Meta.isexpr(expr, :call)
        f, args... = expr.args
        for i in eachindex(args)
            value = simple_arithmetic_eval(data, args[i])
            if value isa UnitRange
                error("Don't know how to do arithmetic between UnitRange and Integer.")
            else
                args[i] = Int(value)
            end
        end
        if f == :+
            return sum(args)
        elseif f == :*
            return prod(args)
        else
            @assert length(args) == 2
            if f == :-
                return args[1] - args[2]
            elseif f == :/
                return Int(args[1] / args[2])
            elseif f == :(:)
                return UnitRange(Int(args[1]), Int(args[2]))
            else
                error("Don't know how to evaluate function $(string(f)).")
            end
        end
    elseif Meta.isexpr(expr, :ref)
        var, indices... = expr.args
        for i in eachindex(indices)
            indices[i] = simple_arithmetic_eval(data, indices[i])
        end
        return Int(data[var][indices...])
    else
        error("Don't know how to evaluate $expr.")
    end
end

end # module

# TODO: can't remove even with the `possible` fix in DynamicPPL, still seems to have eltype inference issue causing AD errors
# Resolves: setindex!!([1 2; 3 4], [2 3; 4 5], 1:2, 1:2) # returns 2×2 Matrix{Any}
# Alternatively, can overload BangBang.possible(
#     ::typeof(BangBang._setindex!), ::C, ::T, ::Vararg
# )
# to allow mutation, but the current solution seems create less possible problems, albeit less efficient.
function BangBang.NoBang._setindex(xs::AbstractArray, v::AbstractArray, I...)
    T = promote_type(eltype(xs), eltype(v))
    ys = similar(xs, T)
    if eltype(xs) !== Union{}
        copy!(ys, xs)
    end
    ys[I...] = v
    return ys
end

function BangBang.setindex!!(nt::NamedTuple, val, vn::VarName{sym}) where {sym}
    optic = BangBang.prefermutation(
        AbstractPPL.getoptic(vn) ∘ Accessors.PropertyLens{sym}()
    )
    return Accessors.set(nt, optic, val)
end

"""
    reconstruct([f, ]dist, val)

Reconstruct `val` so that it's compatible with `dist`.

If `f` is also provided, the reconstruct value will be
such that `f(reconstruct_val)` is compatible with `dist`.
"""
reconstruct(f, dist, val) = reconstruct(dist, val)

# No-op versions.
reconstruct(::UnivariateDistribution, val::Real) = val
reconstruct(::MultivariateDistribution, val::AbstractVector{<:Real}) = copy(val)
reconstruct(::MatrixDistribution, val::AbstractMatrix{<:Real}) = copy(val)
function reconstruct(
    ::Distribution{ArrayLikeVariate{N}}, val::AbstractArray{<:Real,N}
) where {N}
    return copy(val)
end

function reconstruct(dist::LKJCholesky, val::AbstractVector{<:Real})
    return reconstruct(dist, Matrix(reshape(val, size(dist))))
end
reconstruct(dist::LKJCholesky, val::AbstractMatrix{<:Real}) = Cholesky(val, dist.uplo, 0)
reconstruct(::LKJCholesky, val::Cholesky) = val

# NOTE: Necessary to handle product distributions of `Dirichlet` and similar.
function reconstruct(
    ::Bijectors.Inverse{<:Bijectors.SimplexBijector}, dist, val::AbstractVector
)
    (d, ns...) = size(dist)
    return reshape(val, d - 1, ns...)
end
function reconstruct(
    ::Bijectors.Inverse{Bijectors.VecCorrBijector}, ::LKJ, val::AbstractVector
)
    return copy(val)
end
function reconstruct(
    ::Bijectors.Inverse{Bijectors.VecCholeskyBijector}, ::LKJCholesky, val::AbstractVector
)
    return copy(val)
end
function reconstruct(
    ::Bijectors.Inverse{Bijectors.PDVecBijector}, ::MatrixDistribution, val::AbstractVector
)
    return copy(val)
end

reconstruct(d::Distribution, val::AbstractVector) = reconstruct(size(d), val)
reconstruct(::Tuple{}, val::AbstractVector) = val[1]
reconstruct(s::NTuple{1}, val::AbstractVector) = copy(val)
reconstruct(s::NTuple{2}, val::AbstractVector) = reshape(copy(val), s)
