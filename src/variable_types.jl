"""
    Var

A lightweight type for representing variables in a model.
"""
abstract type Var end

struct Scalar <: Var
    name::Symbol
    indices::Tuple{}
end

struct ArrayElement{N} <: Var
    name::Symbol
    indices::NTuple{N,Int}
end

struct ArrayVar{N} <: Var
    name::Symbol
    indices::NTuple{N,Union{Int,UnitRange,Colon}}
end

Var(name::Symbol) = Scalar(name, ())
function Var(name::Symbol, indices)
    indices = map(indices) do i
        if i isa AbstractFloat
            isinteger(i) && return Int(i)
            error("Indices must be integers.")
        end
        return i
    end
    all(x -> x isa Integer, indices) && return ArrayElement(name, indices)
    return ArrayVar(name, indices)
end

Base.size(::Scalar) = ()
Base.size(::ArrayElement) = ()
function Base.size(v::ArrayVar)
    if any(x -> x isa Colon, v.indices)
        error("Can't get size of an array with colon indices.")
    end
    return Tuple(map(length, v.indices))
end

Base.Symbol(v::Scalar) = v.name
function Base.Symbol(v::Var)
    return Symbol(v.name, "[", join(v.indices, ", "), "]")
end

toexpr(r::Number) = r
toexpr(r::UnitRange) = Expr(:call, :(:), r.start, r.stop)
toexpr(v::Scalar) = v.name
toexpr(v::Var) = Expr(:ref, v.name, toexpr.(v.indices)...)

function hash(v::Var, h::UInt)
    return hash(v.name, hash(v.indices, h))
end

function Base.:(==)(v1::Var, v2::Var)
    typeof(v1) != typeof(v2) && return false
    return v1.name == v2.name && v1.indices == v2.indices
end

Base.show(io::IO, v::Scalar) = print(io, v.name)
function Base.show(io::IO, v::Var)
    return print(io, v.name, "[", join(v.indices, ", "), "]")
end

function to_varname(v::Scalar)
    lens = AbstractPPL.IdentityLens()
    return AbstractPPL.VarName{v.name}(lens)
end
function to_varname(v::Var)
    lens = AbstractPPL.IndexLens(v.indices)
    return AbstractPPL.VarName{v.name}(lens)
end

"""
    scalarize(v::Var)

Return an array of `Var`s that are scalarized from `v`. If `v` is a scalar, return an array of length 1 containing `v`.
All indices of `v` must be integer or UnitRange.

# Examples
```jldoctest
julia> scalarize(Var(:x, (1, 2:3)))
2-element Vector{Var}:
 x[1, 2]
 x[1, 3]
```
"""
scalarize(v::Scalar) = [v]
scalarize(v::ArrayElement) = [v]
function scalarize(v::Var)
    collected_indices = collect(Iterators.product(v.indices...))
    scalarized_vars = Array{Var}(undef, size(collected_indices)...)
    for i in eachindex(collected_indices)
        scalarized_vars[i] = Var(v.name, collected_indices[i])
    end
    return scalarized_vars
end
