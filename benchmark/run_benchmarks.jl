using Pkg
Pkg.develop(; path=joinpath(@__DIR__, ".."))

using JuliaBUGS, ADTypes, ReverseDiff, MetaGraphsNext
using BridgeStan, StanLogDensityProblems

using LogDensityProblems, LogDensityProblemsAD
using Chairmarks
using OrderedCollections

## JuliaBUGS

juliabugs_result = OrderedDict()
for model_name in keys(JuliaBUGS.BUGSExamples.VOLUME_1)
    @info "benchmarking volume 1 model" model_name
    (; model_def, data, inits) = JuliaBUGS.BUGSExamples.VOLUME_1[model_name]
    model = compile(model_def, data, inits)
    ad_model = ADgradient(AutoReverseDiff(true), model)

    eval_time = Chairmarks.@be JuliaBUGS.evaluate!!(model)

    juliabugs_theta = rand(LogDensityProblems.dimension(model))
    density_time = Chairmarks.@be LogDensityProblems.logdensity($ad_model, $juliabugs_theta)
    density_and_gradient_time = Chairmarks.@be LogDensityProblems.logdensity_and_gradient(
        $ad_model, $juliabugs_theta
    )

    juliabugs_result[model_name] = (eval_time, density_time, density_and_gradient_time)
end

for model_name in keys(JuliaBUGS.BUGSExamples.VOLUME_2)
    @info "benchmarking volume 2 model" model_name
    (; model_def, data, inits) = JuliaBUGS.BUGSExamples.VOLUME_2[model_name]
    model = compile(model_def, data, inits)

    eval_time = Chairmarks.@be JuliaBUGS.evaluate!!(model)

    density_time = missing
    density_and_gradient_time = missing
    try
        ad_model = ADgradient(AutoReverseDiff(true), model)
        juliabugs_theta = rand(LogDensityProblems.dimension(model))
        density_time = Chairmarks.@be LogDensityProblems.logdensity(
            $ad_model, $juliabugs_theta
        )
        density_and_gradient_time = Chairmarks.@be LogDensityProblems.logdensity_and_gradient(
            $ad_model, $juliabugs_theta
        )
    catch e
        @warn "Model $model_name produces error: $e"
    end

    juliabugs_result[model_name] = (eval_time, density_time, density_and_gradient_time)
end

function print_juliabugs_result(juliabugs_result)
    @info "JuliaBUGS results:"
    for model_name in keys(juliabugs_result)
        @info model_name
        println("Evaluation time:")
        Base.show(stdout, "text/plain", juliabugs_result[model_name][1])
        println()
        println("Density time:")
        Base.show(stdout, "text/plain", juliabugs_result[model_name][2])
        println()
        println("Density+gradient time:")
        Base.show(stdout, "text/plain", juliabugs_result[model_name][3])
        println()
    end
end

## Stan

const STAN_BUGS_EXAMPLES_FOLDER = joinpath(dirname(@__FILE__), "stan/bugs_examples/")

const MODEL_VOL1_STAN = OrderedDict(
    :rats => (:rats, :rats),
    :pumps => (:pump, :pump),
    :dogs => (:dogs, :dogs),
    :seeds => (:seeds, :seeds),
    :surgical_realistic => (:surgical, :surgical),
    :magnesium => (:magnesium, :magnesium),
    :salm => (:salm, :salm),
    :equiv => (:equiv, :equiv),
    :dyes => (:dyes, :dyes),
    :stacks => (:stacks, :stacks_d_normal_ridge),
    :epil => (:epil, :epil),
    :blockers => (:blocker, :blocker),
    :oxford => (:oxford, :oxford),
    :lsat => (:lsat, :lsat),
    :bones => (:bones, :bones),
    :mice => (:mice, :mice),
    :kidney => (:kidney, :kidney),
    :leuk => (:leuk, :leuk),
    :leukfr => (:leukfr, :leukfr),
)

MODEL_VOL2_STAN = OrderedDict(
    :dugongs => (:dugongs, :dugongs),
    :air => (:air, :air),
    :birats => (:birats, :birats),
    :schools => (:schools, :schools),
    :beetles => (:beetles, :beetles_logit),
    :alligators => (:alli, :alli2),
)

function create_stan_logdensityproblem(volume, folder_name, file_name_prefix)
    stan_code_path = joinpath(
        STAN_BUGS_EXAMPLES_FOLDER,
        "vol$volume",
        String(folder_name),
        "$(file_name_prefix).stan",
    )
    stan_data_path = joinpath(
        STAN_BUGS_EXAMPLES_FOLDER,
        "vol$volume",
        String(folder_name),
        "$(file_name_prefix).data.json",
    )
    stan_model = BridgeStan.StanModel(stan_code_path, stan_data_path)
    return StanLogDensityProblems.StanProblem(stan_model)
end

function benchmark_stan_model(stan_logdensityproblem)
    stan_dim = LogDensityProblems.dimension(stan_logdensityproblem)
    stan_theta = rand(stan_dim)
    density_time = Chairmarks.@be LogDensityProblems.logdensity(
        $stan_logdensityproblem, $stan_theta
    )
    density_and_gradient_time = Chairmarks.@be LogDensityProblems.logdensity_and_gradient(
        $stan_logdensityproblem, $stan_theta
    )
    return (stan_dim, density_time, density_and_gradient_time)
end

function run_stan_benchmark(volume, model_name, model_path_dict)
    folder_name, file_name_prefix = model_path_dict[model_name]
    @info "model name" model_name
    stan_logdensityproblem = create_stan_logdensityproblem(
        volume, folder_name, file_name_prefix
    )
    try
        return benchmark_stan_model(stan_logdensityproblem)
    catch e
        println("Model $model_name produces error: $e")
        return missing
    end
end

function run_all_stan_benchmarks(volume, model_path_dict)
    return OrderedDict(
        model_name => run_stan_benchmark(volume, model_name, model_path_dict) for
        model_name in keys(model_path_dict)
    )
end

stan_result = run_all_stan_benchmarks(1, MODEL_VOL1_STAN)
stan_result = Dict(k => v for (k, v) in stan_result if !ismissing(v))

stan_result_vol2 = run_all_stan_benchmarks(2, MODEL_VOL2_STAN)
stan_result_vol2 = Dict(k => v for (k, v) in stan_result_vol2 if !ismissing(v))

function print_stan_result(stan_result)
    @info "Stan results:"
    for model_name in keys(stan_result)
        @info model_name
        println("Dimension:")
        Base.show(stdout, "text/plain", stan_result[model_name][1])
        println()
        println("Density time:")
        Base.show(stdout, "text/plain", stan_result[model_name][2])
        println()
        println("Density+gradient time:")
        Base.show(stdout, "text/plain", stan_result[model_name][3])
        println()
    end

    @info "Stan volume 2 results:"
    for model_name in keys(stan_result_vol2)
        @info model_name
        println("Dimension:")
        Base.show(stdout, "text/plain", stan_result_vol2[model_name][1])
        println()
        println("Density time:")
        Base.show(stdout, "text/plain", stan_result_vol2[model_name][2])
        println()
        println("Density+gradient time:")
        Base.show(stdout, "text/plain", stan_result_vol2[model_name][3])
        println()
    end
end

## create result table

using DataFrames
using PrettyTables

function extract_median_time(result)
    return OrderedDict(
        model_name => (
            Chairmarks.median(density_time).time,
            Chairmarks.median(density_and_gradient_time).time,
        ) for (model_name, (density_time, density_and_gradient_time)) in result
    )
end

stan_median_time_result = extract_median_time(merge(stan_result, stan_result_vol2))
juliabugs_median_time_result = extract_median_time(juliabugs_result)

juliabugs_median_time_result_micro = OrderedDict(
    model => (time[1] * 1e6, time[2] * 1e6) for
    (model, time) in juliabugs_median_time_result
)
stan_median_time_result_micro = OrderedDict(
    model => (time[1] * 1e6, time[2] * 1e6) for (model, time) in stan_median_time_result
)

model_parameters_count = OrderedDict()
model_data_count = OrderedDict()
for model_name in keys(JuliaBUGS.BUGSExamples.VOLUME_1)
    (; model_def, data, inits) = JuliaBUGS.BUGSExamples.VOLUME_1[model_name]
    model = compile(model_def, data, inits)
    model_parameters_count[model_name] = length(model.parameters)
    data_count = 0
    for l in labels(model.g)
        if model.g[l].is_observed
            data_count += 1
        end
    end
    model_data_count[model_name] = data_count
end
for model_name in keys(JuliaBUGS.BUGSExamples.VOLUME_2)
    (; model_def, data, inits) = JuliaBUGS.BUGSExamples.VOLUME_2[model_name]
    model = compile(model_def, data, inits)
    model_parameters_count[model_name] = length(model.parameters)
    data_count = 0
    for l in labels(model.g)
        if model.g[l].is_observed
            data_count += 1
        end
    end
    model_data_count[model_name] = data_count
end

results_df = DataFrame(;
    Model=String[],
    Parameters=Int[],
    Data=Int[],
    Stan_Density_Time=Union{Float64,String}[],
    Stan_Density_Gradient_Time=Union{Float64,String}[],
    JuliaBUGS_Density_Time=Union{Float64,String}[],
    JuliaBUGS_Density_Gradient_Time=Union{Float64,String}[],
)

for model in keys(juliabugs_median_time_result_micro)
    if model == :surgical_simple
        continue
    end
    stan_time = get(stan_median_time_result_micro, model, missing)
    juliabugs_time = juliabugs_median_time_result_micro[model]
    push!(
        results_df,
        (
            string(model),
            model_parameters_count[model],
            model_data_count[model],
            coalesce(ismissing(stan_time) ? missing : stan_time[1], "NA"),
            coalesce(ismissing(stan_time) ? missing : stan_time[2], "NA"),
            juliabugs_time[1],
            juliabugs_time[2],
        ),
    )
end

function print_custom_table(df; backend=Val(:markdown))
    headers = [
        "Model",
        "Parameter Count",
        "Data Count",
        "Stan Density Time (µs)",
        "Stan Density Gradient Time (µs)",
        "JuliaBUGS Density Time with Graph Walk (µs)",
        "JuliaBUGS Density Gradient Time with ReverseDiff.jl(compiled tape) (µs)",
    ]

    data = [df[!, col] for col in names(df)]
    table_data = hcat(data...)
    return pretty_table(table_data; header=headers, backend=backend)
end

print_custom_table(results_df; backend=Val(:markdown))
