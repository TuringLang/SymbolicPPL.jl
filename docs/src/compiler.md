# Compiler Design

## Partial Evaluation of Variables
BUGS programs describe Directed Acyclic Graphical(DAG) models. 
Variables are mapped to vertices and assignments are mapped to edges in the DAG.
The target of the compilation is a finite-size graph and the size is fully determined at compile time.
Thus requires the ability to determine the loop bounds of for loops and all the array indices.

The unique challenge of BUGS program is that there are no chronological orders between all the assignments.
To evaluate the value of a variable, the compiler needs to consider all the assignments in the program.  

Our implementation is built upon [Symbolics.jl](https://github.com/JuliaSymbolics/Symbolics.jl). 
All the assignments are processed into rules, and evaluation of a variable is implemented using the [`substitute`](https://symbolics.juliasymbolics.org/dev/manual/expression_manipulation/#SymbolicUtils.substitute) function.  

## Array 
The challenge of support BUGS' array interface is that every element of an array can be either logical or stochastic, so we need to treat every array element as a separate variable. We won't go deep into the inner mechanism, rather let's see how array indexing work with some demos.

## Inference on Graph
We implemented a simple Metropolis-within-Gibbs sampler, given that inference algorithm is not the focus of this report, we'll just demonstrate the result. Curious reader can refer to our [implementation](https://github.com/TuringLang/SymbolicPPL.jl/blob/complier/src/gibbs.jl), and even better, we'll appreciate contributions.