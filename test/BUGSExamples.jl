using SymbolicPPL
using SymbolicPPL: 
    transform_expr,
    addrules,
    addrules!,
    tograph,
    pregraph,
    ref_to_symbolic,
    SampleFromPrior
using AbstractMCMC
using Random

##

m = SymbolicPPL.BUGSExamples.EXAMPLES[:blockers];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:bones];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:dogs];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:dyes];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:epil];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:equiv];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:inhalers];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:kidney];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:leuk];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:leukfr];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:lsat];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:magnesium];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:mice];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:oxford];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:pumps];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:rats];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:salm];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:seeds];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:stacks];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:surgical_simple];
m = SymbolicPPL.BUGSExamples.EXAMPLES[:surgical_realistic];

ori_expr = transform_expr(m[:model_def])
expr, compiler_state = addrules(ori_expr, m[:data], true);
@run g = tograph(compiler_state, false);

model = compile_graphppl(model_def = m[:model_def], data = m[:data], initials = m[:inits][1]);
@run model = compile_graphppl(model_def = m[:model_def], data = m[:data], initials = m[:inits][1]);

sampler = SampleFromPrior(model);
sample, state = AbstractMCMC.step(Random.default_rng(), model, sampler);
sample, state = AbstractMCMC.step(Random.default_rng(), model, sampler, state);
samples = AbstractMCMC.sample(model, sampler, 3);
