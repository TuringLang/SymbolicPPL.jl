using Documenter
using JuliaBUGS
using JuliaBUGS.BUGSPrimitives: abs, cloglog, equals, exp, inprod, inverse, log, logdet, logfact, loggam, 
icloglog, logit, mexp, max, mean, min, phi, pow, sqrt, rank, ranked, round, sd, 
softplus, sort, _step, sum, trunc, sin, arcsin, arcsinh, cos, arccos, arccosh, tan, arctan, arctanh
using JuliaBUGS.BUGSPrimitives: dnorm, dlogis, dt, ddexp, dflat, dexp, dchisqr, dweib, dlnorm, dgamma, dpar, dgev, dgpar, df, dunif, dbeta, dmnorm,
dmt, dwish, ddirich, dbern, dbin, dcat, dpois, dgeom, dnegbin, dbetabin, dhyper, dmulti, TDistShiftedScaled, Flat, 
LeftTruncatedFlat, RightTruncatedFlat, TruncatedFlat

makedocs(;
    sitename="JuliaBUGS.jl",
    pages=[
        "Introduction" => "index.md",
        "API" => "api.md",
        "AST Translation" => "ast.md",
        "Functions" => "functions.md",
        "Distributions" => "distributions.md",
    ],
)

deploydocs(; repo="github.com/TuringLang/JuliaBUGS.jl.git")
