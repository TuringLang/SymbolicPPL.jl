abstract type NodeInfo end

"""
    AuxiliaryNodeInfo

Indicate the node is created by the compiler and not in the original BUGS model. These nodes
are only used to determine dependencies.
"""
struct AuxiliaryNodeInfo <: NodeInfo end

"""
    ConcreteNodeInfo

Define the information stored in each node of the BUGS graph.
"""
struct ConcreteNodeInfo <: NodeInfo
    node_type::VariableTypes
    link_function::Function
    node_function::Function
    node_args::Vector{VarName}
end

function ConcreteNodeInfo(var::Var, vars, link_functions, node_functions, node_args)
    return ConcreteNodeInfo(
        vars[var],
        eval(link_functions[var]),
        eval(node_functions[var]),
        map(v -> AbstractPPL.VarName{v.name}(AbstractPPL.IdentityLens()), node_args[var]),
    )
end

function NodeInfo(var::Var, vars, link_functions, node_functions, node_args)
    if var in keys(vars)
        return ConcreteNodeInfo(var, vars, link_functions, node_functions, node_args)
    else
        return AuxiliaryNodeInfo()
    end
end

"""
    BUGSGraph

The graph object for a BUGS model. Just an alias of `MetaGraph` with specified types.
"""
const BUGSGraph = MetaGraph{
    Int64,SimpleDiGraph{Int64},VarName,NodeInfo,Nothing,Nothing,Nothing,Float64
}

function BUGSGraph(vars, link_functions, node_args, node_functions, dependencies)
    g = MetaGraph(
        SimpleDiGraph{Int64}();
        weight_function=nothing,
        label_type=VarName,
        vertex_data_type=NodeInfo,
    )
    for l in keys(vars) # l for LHS variable
        l_vn = to_varname(l)
        check_and_add_vertex!(g, l_vn, NodeInfo(l, vars, link_functions, node_functions, node_args))
        scalarize_then_add_edge!(g, l; lhs_or_rhs=:lhs)
        for r in dependencies[l]
            r_vn = to_varname(r)
            check_and_add_vertex!(g, r_vn, NodeInfo(r, vars, link_functions, node_functions, node_args))
            add_edge!(g, r_vn, l_vn)
            scalarize_then_add_edge!(g, r; lhs_or_rhs=:rhs)
        end
    end
    return g
end

function to_varname(v::Scalar)
    lens = AbstractPPL.IdentityLens()
    return AbstractPPL.VarName{v.name}(lens)
end
function to_varname(v::Var)
    lens = AbstractPPL.IndexLens(v.indices)
    return AbstractPPL.VarName{v.name}(lens)
end

function check_and_add_vertex!(g::BUGSGraph, v::VarName, data::NodeInfo)
    if haskey(g, v)
        data isa AuxiliaryNodeInfo && return nothing
        if g[v] isa AuxiliaryNodeInfo
            set_data!(g, v, data)
        end
    else
        add_vertex!(g, v, data)
    end
end

function scalarize_then_add_edge!(g::BUGSGraph, v::Var; lhs_or_rhs=:lhs)
    scalarized_v = vcat(scalarize(v)...)
    length(scalarized_v) == 1 && return nothing
    v = to_varname(v)
    for v_elem in map(to_varname, scalarized_v)
        add_vertex!(g, v_elem, AuxiliaryNodeInfo()) # may fail, but it's ok
        if lhs_or_rhs == :lhs
            add_edge!(g, v, v_elem)
        elseif lhs_or_rhs == :rhs
            add_edge!(g, v_elem, v)
        else
            error("Unknown argument $lhs_or_rhs")
        end
    end
end

"""
    BUGSModel

The model object for a BUGS model.
"""
struct BUGSModel <: AbstractPPL.AbstractProbabilisticProgram
    param_length::Int
    varinfo::SimpleVarInfo # for uniformity, all values in varinfo are untransformed
    parameters::Vector{VarName}
    g::BUGSGraph
    sorted_nodes::Vector{VarName}
end

function BUGSModel(g, sorted_nodes, vars, array_sizes, data, inits)
    vs = initialize_var_store(data, vars, array_sizes)
    vi = SimpleVarInfo(vs)
    parameters = VarName[]
    logp = 0.0
    for vn in sorted_nodes
        g[vn] isa AuxiliaryNodeInfo && continue
        
        ni = g[vn]
        @unpack node_type, link_function, node_function, node_args = ni
        args = [vi[x] for x in node_args]
        if node_type == JuliaBUGS.Logical
            value = (node_function)(args...)
            @assert value isa Union{Number,Array{<:Number}}
            vi = setindex!!(vi, value, vn)
        else
            dist = (node_function)(args...)
            value = evaluate(vn, data)
            isnothing(value) && push!(parameters, vn)
            isnothing(value) && (value = evaluate(vn, inits))
            if !isnothing(value)
                # here the value is untransformed version
                logp += logpdf(dist, (link_function)(value))
                vi = setindex!!(vi, value, vn)
            else
                # println("initialization for $vn is not provided, sampling from prior");
                value = rand(dist)
                logp += logpdf(dist, value)
                vi = setindex!!(vi, inverse_link_function(link_function)(value), vn)
            end
        end
    end
    l = sum([_length(x) for x in parameters])
    vi = @set vi.logp = logp
    return BUGSModel(l, vi, parameters, g, sorted_nodes)
end

function initialize_var_store(data, vars, array_sizes)
    var_store = Dict{VarName,Any}()
    array_vn(k::Symbol) = AbstractPPL.VarName{Symbol(k)}(AbstractPPL.IdentityLens())
    for (k, v) in data
        vn = array_vn(k)
        var_store[vn] = v
    end
    for (k, v) in array_sizes
        vn = array_vn(k)
        haskey(var_store, vn) || (var_store[vn] = zeros(v...))
    end
    for v in keys(vars)
        if v isa Scalar
            vn = to_varname(v)
            var_store[vn] = 0.0 # TODO: assume all scalars are floating point numbers now
        end
    end
    return var_store
end

inverse_link_function(::typeof(logit)) = probit
inverse_link_function(::typeof(cloglog)) = cloglog
inverse_link_function(::typeof(log)) = exp
inverse_link_function(::typeof(probit)) = logit
inverse_link_function(identity) = identity

function evaluate(vn::VarName, env::Dict)
    sym = getsym(vn)
    ret = nothing
    try
        ret = get(env[sym], getlens(vn))
    catch _
    end
    return ismissing(ret) ? nothing : ret
end

# not reloading Base.length, the function only work for a specific subset of VarNames and should not be used elsewhere
function _length(vn::VarName)
    getlens(vn) isa Setfield.IdentityLens && return 1
    return prod([length(index_range) for index_range in getlens(vn).indices])
end

function DynamicPPL.settrans!!(m::BUGSModel)
    return @set m.vi = DynamicPPL.settrans!!(vi, transform_variables)
end

"""
    DefaultContext

Use values in varinfo to compute the log joint density.
"""
struct DefaultContext <: AbstractPPL.AbstractContext end

"""
    SamplingContext

Do an ancestral sampling of the model parameters. Also accumulate log joint density.
"""
struct SamplingContext <: AbstractPPL.AbstractContext
    rng::Random.AbstractRNG
end

"""
    loglikelihoodContext

Given values of parameters, compute the log joint density.
"""
struct loglikelihoodContext <: AbstractPPL.AbstractContext end

AbstractPPL.evaluate!!(model::BUGSModel, rng::Random.AbstractRNG) = evaluate!!(model, SamplingContext(rng))
function AbstractPPL.evaluate!!(model::BUGSModel, ctx::SamplingContext)
    @unpack param_length, varinfo, parameters, g, sorted_nodes = model
    vi = deepcopy(varinfo)
    logp = 0.0
    for vn in sorted_nodes
        g[vn] isa AuxiliaryNodeInfo && continue
        
        ni = g[vn]
        @unpack node_type, link_function, node_function, node_args = ni
        args = [vi[x] for x in node_args]
        if node_type == JuliaBUGS.Logical
            value = node_function(args...)
            setindex!!(vi, value, vn)
        else
            dist = node_function(args...)
            spl = rand(ctx.rng, dist)
            value = inverse_link_function(link_function)(spl)
            if T == DynamicPPL.DynamicTransformation
                logp += logpdf(transformed(dist), spl)
            else
                logp += logpdf(dist, spl)
            end
            vi = setindex!!(vi, value, vn)
        end
    end
    return @set vi.logp = logp
end

AbstractPPL.evaluate!!(model::BUGSModel) = AbstractPPL.evaluate!!(model, DefaultContext()) 
function AbstractPPL.evaluate!!(model::BUGSModel, ::DefaultContext)
    @unpack param_length, varinfo, parameters, g, sorted_nodes = model
    vi = deepcopy(varinfo)
    logp = 0.0
    for vn in sorted_nodes
        g[vn] isa AuxiliaryNodeInfo && continue

        ni = g[vn]
        @unpack node_type, link_function, node_function, node_args = ni
        node_type == JuliaBUGS.Logical && continue
        args = [vi[x] for x in node_args]
        dist = node_function(args...)
        if DynamicPPL.transformation(vi) == DynamicPPL.DynamicTransformation
            logp += logpdf(dist, link(dist, (link_function)(vi[vn]))) # TODO: work with this
        else
            logp += logpdf(dist, (link_function)(vi[vn]))
        end
    end
    return @set vi.logp = logp
end

function AbstractPPL.evaluate!!(model::BUGSModel, flattened_values::AbstractVector)
    @assert length(flattened_values) == model.param_length
    @unpack param_length, varinfo, parameters, g, sorted_nodes = model
    vi = deepcopy(varinfo)
    current_idx = 1; logp = 0.0
    for vn in sorted_nodes
        g[vn] isa AuxiliaryNodeInfo && continue
        
        ni = g[vn]   
        @unpack node_type, link_function, node_function, node_args = ni
        args = [vi[x] for x in node_args]
        if node_type == JuliaBUGS.Logical
            value = node_function(args...)
            setindex!!(vi, value, vn)
        else
            dist = node_function(args...)
            if vn in parameters # the value of parameter variables are stored in flattened_values
                l = _length(vn)
                value = if l == 1
                    flattened_values[current_idx]
                else
                    flattened_values[current_idx:(current_idx + l - 1)]
                end
                current_idx += l
                
                if DynamicPPL.transformation(vi) == DynamicPPL.DynamicTransformation
                    value = invlink(dist, value)
                end
                setindex!!(vi, value, vn)
                logp += logpdf(dist, (link_function)(value))
            else
                value = vi[vn]
                logp += logpdf(dist, (link_function)(value))
            end
        end
    end
    vi = @set vi.logp = logp
    return vi
end