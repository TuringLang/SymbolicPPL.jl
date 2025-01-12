module BUGSExamples

using JuliaBUGS: JuliaBUGS, @bugs
using JSON

struct Example
    name::String
    model_def::Expr
    original_syntax_program::String
    data::NamedTuple
    inits::NamedTuple
    inits_alternative::NamedTuple
    reference_results::Union{NamedTuple, Nothing}
end

#! format: off
function load_example_volume(volume_num::Int)
    if volume_num == 1
        include("./Volume_1/01_Rats.jl")
        include("./Volume_1/02_Pumps.jl")
        include("./Volume_1/03_Dogs.jl")
        include("./Volume_1/04_Seeds.jl")
        include("./Volume_1/05_Surgical.jl")
        include("./Volume_1/06_Magnesium.jl")
        include("./Volume_1/07_Salm.jl")
        include("./Volume_1/08_Equiv.jl")
        include("./Volume_1/09_Dyes.jl")
        include("./Volume_1/10_Stacks.jl")
        include("./Volume_1/11_Epil.jl")
        include("./Volume_1/12_Blocker.jl")
        include("./Volume_1/13_Oxford.jl")
        include("./Volume_1/14_LSAT.jl")
        include("./Volume_1/15_Bones.jl")
        #include("./Volume_1/16_Inhalers.jl") # chain graph contains cycle
        include("./Volume_1/17_Mice.jl")
        include("./Volume_1/18_Kidney.jl")
        include("./Volume_1/19_Leuk.jl")
        include("./Volume_1/20_LeukFr.jl")

        vol_1 = (
            rats = rats,
            pumps = pumps,
            dogs = dogs,
            seeds = seeds,
            surgical_simple = surgical_simple,
            surgical_realistic = surgical_realistic,
            magnesium = magnesium,
            salm = salm,
            equiv = equiv,
            dyes = dyes,
            stacks = stacks,
            epil = epil,
            blockers = blockers,
            oxford = oxford,
            lsat = lsat,
            bones = bones,
            #inhalers = inhalers, # chain graph is not supported
            mice = mice,
            kidney = kidney,
            leuk = leuk,
            leukfr = leukfr
        )
        return vol_1
    elseif volume_num == 2
        include("Volume_2/01_Dugongs.jl")
        include("Volume_2/02_Orange_trees.jl")
        include("Volume_2/03_Multivariate_Orange_trees.jl")
        include("Volume_2/04_Biopsies.jl")
        include("Volume_2/05_Eyes.jl")
        include("Volume_2/06_Hearts.jl")
        include("Volume_2/07_Air.jl")
        include("Volume_2/08_Cervix.jl")
        include("Volume_2/09_Jaws.jl")
        include("Volume_2/10_BiRats.jl")
        include("Volume_2/11_Schools.jl")
        # include("Volume_2/12_Ice.jl") # also contains cycles
        include("Volume_2/13_Beetles.jl")
        include("Volume_2/14_Alligators.jl")
        include("Volume_2/15_Endo.jl")
        include("Volume_2/16_Stagnant.jl")
        include("Volume_2/17_Asia.jl")
        # include("Volume_2/18_Pigs.jl")
        # include("Volume_2/19_Simulating_data.jl")
        vol_2 = (
            dugongs = dugongs,
            orange_trees = orange_trees,
            orange_trees_multivariate = orange_trees_multivariate,
            biopsies = biopsies,
            eyes = eyes,
            hearts = hearts,
            air = air,
            cervix = cervix,
            jaws = jaws,
            birats = birats,
            schools = schools,
            # ice = ice,
            beetles = beetles,
            alligators = alligators,
            endo = endo,
            stagnant = stagnant,
            asia = asia,
            # pigs = pigs,
            # simulating_data = simulating_data
        )
        return vol_2
    else
        @warn("Volume number $volume_num not supported yet.")
        return nothing
    end
end
#! format: on

const VOLUME_1 = load_example_volume(1)
const VOLUME_2 = load_example_volume(2)

end
