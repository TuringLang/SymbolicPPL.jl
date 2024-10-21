using JuliaBUGS: CompilerUtils

@testset "Decompose for loop" begin
    ex = MacroTools.@q for i in 1:3
        x[i] = i
        for j in 1:3
            y[i, j] = i + j
        end
    end

    loop_var, lb, ub, body = JuliaBUGS.decompose_for_expr(ex)

    @test loop_var == :i
    @test lb == 1
    @test ub == 3
    @test body == MacroTools.@q begin
        x[i] = i
        for j in 1:3
            y[i, j] = i + j
        end
    end
end

@testset "BangBang.setindex!!" begin
    nt = (a=1, b=[1, 2, 3], c=[1, 2, 3])
    nt1 = BangBang.setindex!!(nt, 2, @varname(a))
    @test nt1.a == 2

    nt2 = BangBang.setindex!!(nt, 5, @varname(b[1]))
    @test nt2.b == [5, 2, 3]
    @test nt2.b === nt.b # mutation

    nt3 = BangBang.setindex!!(nt, 2, @varname(c[1]); prefer_mutation=false)
    @test nt3.c == [2, 2, 3]
    @test nt3.c !== nt.c # no mutation
end
