module SymbolicPPL

include("bugsast.jl")
include("graph.jl")
include("compiler.jl")
include("primitives.jl")
include("gibbs.jl")
include("distributions.jl")
include("toturing.jl")


export @bugsast, @bugsmodel_str
export compile

include("BUGSExamples/BUGSExamples.jl")
using .BUGSExamples
export EXAMPLES, LINKS

end # module
