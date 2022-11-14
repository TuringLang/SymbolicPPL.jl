using SymbolicPPL
using SymbolicPPL:
    CompilerState,
    resolveif!,
    linkfunction,
    unroll!,
    tosymbolic,
    resolve,
    symbolic_eval,
    ref_to_symbolic!,
    ref_to_symbolic,
    addlogicalrules!,
    addstochasticrules!
using Test
using Symbolics

##

# tests for `addlogicalrules` for data
data = Dict(:N => 2, :g => [1, 2, 3])
compiler_state = CompilerState()
addlogicalrules!(data, compiler_state)
@test resolve(:N, compiler_state.logicalrules) == 2
@test resolve(ref_to_symbolic("g[2]"), compiler_state.logicalrules) == 2

# tests for unrolling facilities
expr = bugsmodel"""      
    # dummy assignment for easy understanding
    variable.0 <- 1

    # nested loop
    for (i in 1:3) {
        # constant assignment
        array.variable.0[i] <- 1
        # assignment using loop variable
        array.variable.1[i] <- i + 1
        
        # nested loops in another for loop
        for (j in 1:2) {
            # loop bound depend on loop variable
            for (k in 1:j) {
                array.variable.2[i, j, k] = 2
            }
        }
    }

    # variable loop bound that can be resolve from user input
    for (i in 1:N) {
        array.variable.3[i] <- i
    }

    for (i in 1:g[2]) {
        array.variable.4[i] <- i
    }

    # dummy assignment for easy understanding
    variable.1 <- 1
"""
unroll!(expr, compiler_state)

intended_result = bugsmodel"""
    variable.0 <- 1
    array.variable.0[1] <- 1
    array.variable.1[1] <- 1 + 1
    array.variable.2[1, 1, 1] <- 2
    array.variable.2[1, 2, 1] <- 2
    array.variable.2[1, 2, 2] <- 2
    array.variable.0[2] <- 1
    array.variable.1[2] <- 2 + 1
    array.variable.2[2, 1, 1] <- 2
    array.variable.2[2, 2, 1] <- 2
    array.variable.2[2, 2, 2] <- 2
    array.variable.0[3] <- 1
    array.variable.1[3] <- 3 + 1
    array.variable.2[3, 1, 1] <- 2
    array.variable.2[3, 2, 1] <- 2
    array.variable.2[3, 2, 2] <- 2
    array.variable.3[1] <- 1
    array.variable.3[2] <- 2
    array.variable.4[1] <- 1
    array.variable.4[2] <- 2
    variable.1 <- 1
"""
@test expr == intended_result

# tests for unrolling: corner case where need variable defined in outer loop to unroll inner loop
# rely on `N` was defined to be 2 from previous tests 
expr = bugsmodel"""      
    for (i in 1:N) {
        n[i] = i
        for(j in 1:n[i]) {
            m[i, j] = i + j
        }
    }     
"""
unroll!(expr, compiler_state)
addlogicalrules!(expr, compiler_state)
unroll!(expr, compiler_state)

intended_result = bugsmodel"""
    n[1] = 1
    m[1, 1] = 1 + 1
    n[2] = 2
    m[2, 1] = 2 + 1
    m[2, 2] = 2 + 2
"""

@test expr == intended_result

# tests for `ref_to_symbolic!`
# case 1: g[1] already exists
@test Symbolics.isequal(
    ref_to_symbolic!(Meta.parse("g[1]"), compiler_state),
    tosymbolic(Meta.parse("g[1]")),
)
# case 2: g[4] doesn't exist, so expand compiler_state.arrays
@test Symbolics.isequal(
    ref_to_symbolic!(Meta.parse("g[4]"), compiler_state),
    tosymbolic(Meta.parse("g[4]")),
)
@test Symbolics.isequal(compiler_state.arrays[:g][4], tosymbolic(Meta.parse("g[4]")))
# case 3: array h doesn't exist, create one
@test :h ∉ keys(compiler_state.arrays)
ref_to_symbolic!(Meta.parse("h[2]"), compiler_state)
@test Symbolics.isequal(compiler_state.arrays[:h][2], tosymbolic(Meta.parse("h[2]")))
# case 4: slicing 
s = ref_to_symbolic!(Meta.parse("s[2, 3:6]"), compiler_state)
@test size(s) == (4,)
s = ref_to_symbolic!(Meta.parse("s[2, :]"), compiler_state)
@test size(s) == (6,)
s = ref_to_symbolic!(Meta.parse("s[2:3, :]"), compiler_state)
@test size(s) == (2, 6)
s = ref_to_symbolic!(Meta.parse("s[N-1, :]"), compiler_state)
@test size(s) == (6, )

# tests for parse_logical_assignments!
expr = bugsmodel"""
    v[2] <- h[3] + (g[2] * d) * c
    w[6] <- f[2] / (y[4] + e)
"""

addlogicalrules!(expr, compiler_state)
@variables c d
intended_equation = tosymbolic(Meta.parse("h[3]")) + c * d * tosymbolic(Meta.parse("g[2]"))
@test Symbolics.isequal(
    compiler_state.logicalrules[tosymbolic(Meta.parse("v[2]"))],
    intended_equation,
)

# tests for `if`
# note: `if` implemented simply test if the condition evals to true, and then decides to add the exprs in or leave out 
expr = bugsmodel"""
    if (condt) {
        exprt <- 0
    }
    if (condf) {
        exprf <- 0
    }
"""
data = Dict(:condt => true, :condf => false)
compiler_state = CompilerState()
addlogicalrules!(data, compiler_state)
resolveif!(expr, compiler_state)
intended_expr = bugsmodel"""
    exprt <- 0
"""
@test expr == intended_expr

# test for link function handling
expr = bugsmodel"""
    logit(lhs) <- rhs
    cloglog(lhs) <- rhs
    log(lhs) <- rhs
    probit(lhs) <- rhs
"""

intended_expr = bugsmodel"""
    lhs <- logistic(rhs)
    lhs <- cexpexp(rhs)
    lhs <- exp(rhs)
    lhs <- phi(rhs)
"""
@test linkfunction(expr) == intended_expr

# test for symbolic_eval
@variables a b c
compiler_state = CompilerState(
    Dict{Symbol,Symbolics.Arr{Num}}(),
    Dict(a => b + c, b => 2, c => 3),
    Dict{Num,Expr}()
)
@test symbolic_eval(a, compiler_state.logicalrules) == 5

# test for function building
ex = @bugsast begin
    g ~ bar(p[1:2, 1:3])
    for i in 1:2    
        for j in 1:3
            p[i, j] = q[i, j] + i
            q[i, j] = foobar(u[1:i, 1:j])
            u[i, j] ~ dnorm(0, 1)
        end
    end
end

@register_function foo(x::Array) = sum(x)
@register_function foobar(x::Array) = sum(x)
@register_distribution bar(x::Array) = SymbolicPPL.dcat(reduce(vcat, v))

compiler_state = compile(ex, NamedTuple(), :IR)

# test for using observed stochastic variable for loop bound
let err = nothing
    try
        g = compile(
            @bugsast begin 
                a = 2
                a ~ dnorm(0, 1)
                for i = 1:a
                    b[i] ~ dnorm(0, 1)
                end
            end, 
            NamedTuple(), :IR
        )    
    catch err
    end

    @test err isa Exception
end

# test for link function handling in stochastic assignments
expr = @bugsast begin
    logit(x) ~ dnorm(0, 1)
end

expr = @bugsast begin
    logit(x) ~ dnorm(0, 1)
    x = 1
end

compile(expr, NamedTuple(), :IR)

# 
expr = @bugsast begin
    p[1] = 0.5
    p[2] = 0.5
    a[1] = 1
    for i in 1:2
        a[i] ~ dcat(p[:])
        b[a[i]] ~ dnorm(0, 1)
    end
end

compile(expr, NamedTuple(), :IR)

# stochastic indexing
expr = @bugsast begin
    p[1] = 0.5
    p[2] = 0.5
    b[1] = 1
    b[2] = 2
    for i in 1:2
        a[i] ~ dcat(p[:])
        c[i] = b[a[i]]
        d[i] ~ dnorm(c[i], 1)
    end
end

cs = compile(expr, NamedTuple(), :IR)
g = compile(expr, NamedTuple(), :Graph)

# test for array LHS
expr = @bugsast begin
    g[1:2] = sort(x[:])
    y = g[1] + 2
end

cs = compile(expr, (x = [4, 2],), :IR)
@test SymbolicPPL.resolve(SymbolicPPL.tosymbolic(:y), cs.logicalrules) == 4

# test for that order of expressions shouldn't affect colon indexing's correctness
expr1 = @bugsast begin
    u[1] = 2
    u[2] = 3
    a = mean(u[:])
end

expr2 = @bugsast begin
    a = mean(u[:])
    u[1] = 2
    u[2] = 3
end

@test Symbolics.isequal(compile(expr1, NamedTuple(), :IR).logicalrules[tosymbolic(:a)], compile(expr2, NamedTuple(), :IR).logicalrules[tosymbolic(:a)])

# corner case, need colon indexing for for loop bounds
expr = @bugsast begin
    a[1] = sum(u[1:2])
    a[2] = sum(u[2:3])

    B = sum(a[:])
    for i in 1:B
        v[i] = 1
    end
end

data = (u = [1, 2, 3],)

compile(expr, data, :IR).logicalrules

# test for multivariate distributions, simplified from `BiRate`
using SymbolicPPL: rreshape
model_def = bugsmodel"""
    for( i in 1 : 2 ) {
        beta[i , 1 : 2] ~ dmnorm(mu.beta[], R[ , ])
        for( j in 1 : 3 ) {
            Y[i, j] ~ dnorm(mu[i , j], tauC)
            mu[i, j] <- beta[i, 1] + beta[i, 2] * x[j]
        }
    }
    mu.beta[1 : 2] ~ dmnorm(mean[], prec[ , ])
    R[1 : 2 , 1 : 2] ~ dwish(Omega[ , ], 2)
    tauC ~ dgamma(0.001, 0.001)
"""

data = (
    x = [8.0, 15.0, 22.0, 29.0, 36.0], N = 30, T = 5,
    Omega = rreshape([200, 0, 0, 0.2], (2, 2)),   
    mean = [0,0],
    prec = rreshape([1.0E-6, 0, 0, 1.0E-6], (2, 2)),
    Y = rreshape([151, 199, 246, 283, 320, 145, ],(2,3)),
)
    
cs = compile(model_def, data, :IR)
@run o = SymbolicPPL.gen_output(cs)
g = compile(model_def, data, :Graph)

for k in keys(o)
    println(k)
    println(o[k])
end

# test for erroring when LHS of logical assignment is data array element
let err = nothing
    try
        cs = compile(
            @bugsast begin
                u[1] = 2
            end, 
            (u = ones(2), ), :IR
        )    
    catch err
    end

    @test err isa Exception
end
