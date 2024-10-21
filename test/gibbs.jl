using JuliaBUGS: MHFromPrior, Gibbs, OrderedDict

@testset "Simple gibbs" begin
    model_def = @bugs begin
        μ ~ Normal(0, 4)
        σ ~ Gamma(1, 1)
        for i in 1:100
            y[i] ~ Normal(μ, σ)
        end
    end

    μ_true = 2
    σ_true = 4

    y = rand(Normal(μ_true, σ_true), 100)

    model = compile(model_def, (; y=y))
    model = initialize!(model, (μ=4.0, σ=6.0))

    splr_map = OrderedDict(@varname(μ) => MHFromPrior(), @varname(σ) => MHFromPrior())
    splr = Gibbs(splr_map)

    p_s, st_init = AbstractMCMC.step(
        Random.default_rng(), AbstractMCMC.LogDensityModel(model), splr
    )

    p_s, st = AbstractMCMC.step(
        Random.default_rng(), AbstractMCMC.LogDensityModel(model), splr, st_init
    )

    chn = AbstractMCMC.sample(
        Random.default_rng(),
        model,
        splr,
        10000;
        # chain_type=MCMCChains.Chains,
    )

    σ_samples = [v[1] for v in chn[300:end]]
    μ_samples = [v[2] for v in chn[300:end]]

    @test mean(μ_samples) ≈ μ_true rtol = 0.2
    @test mean(σ_samples) ≈ σ_true rtol = 0.2
end

@testset "Linear regression" begin
    model_def = @bugs begin
        # Likelihood
        for i in 1:N
            y[i] ~ dnorm(mu[i], tau)
            mu[i] = alpha + beta * x[i]
        end

        # Priors
        alpha ~ dnorm(0, 0.01)
        beta ~ dnorm(0, 0.01)
        sigma ~ dunif(0, 10)

        # Precision
        tau = pow(sigma, -2)

        # Generated Quantities for testing purposes
        gen_quant = alpha + beta * sigma
    end

    # ground truth: alpha = 3, beta = 2, sigma = 1
    data = (
        N=10,
        x=[0.0, 1.11, 2.22, 3.33, 4.44, 5.56, 6.67, 7.78, 8.89, 10.0],
        y=[1.58, 4.80, 7.10, 8.86, 11.73, 14.52, 18.22, 18.73, 21.04, 22.93],
    )

    model = compile(model_def, data, (;))

    sampler = Gibbs(
        OrderedDict(
            @varname(alpha) => MHFromPrior(),
            @varname(beta) => MHFromPrior(),
            @varname(sigma) => MHFromPrior(),
        ),
    )

    # single step
    p_s, st_init = AbstractMCMC.step(
        Random.default_rng(), AbstractMCMC.LogDensityModel(model), sampler
    )

    # following step
    p_s, st = AbstractMCMC.step(
        Random.default_rng(), AbstractMCMC.LogDensityModel(model), sampler, st_init
    )

    # following step with sampler_map
    sampler_map = OrderedDict(
        [@varname(alpha), @varname(beta)] => HMC(0.1, 10), [@varname(sigma)] => RWMH(1)
    )
    p_s, st_init = AbstractMCMC.step(
        Random.default_rng(), AbstractMCMC.LogDensityModel(model), Gibbs(sampler_map)
    )
    p_s, st = AbstractMCMC.step(
        Random.default_rng(),
        AbstractMCMC.LogDensityModel(model),
        Gibbs(sampler_map),
        st_init,
    )

    # TODO: result checking is disabled because of speed and stability, revive this after improvement
    # sample_size = 10000
    sample_size = 100000
    chn = AbstractMCMC.sample(
        Random.default_rng(),
        model,
        Gibbs(
            OrderedDict(
                [@varname(alpha), @varname(beta)] => MHFromPrior(),
                [@varname(sigma)] => HMC(0.1, 10),
            ),
        ),
        sample_size;
        # chain_type=MCMCChains.Chains,
        discard_initial=Int(sample_size / 2),
    )

    num_to_discard = Int(sample_size / 2)
    alpha_samples = [v[1] for v in chn[num_to_discard:end]]
    beta_samples = [v[2] for v in chn[num_to_discard:end]]
    sigma_samples = [v[3] for v in chn[num_to_discard:end]]

    alpha_mean = mean(alpha_samples)
    beta_mean = mean(beta_samples)
    sigma_mean = mean(sigma_samples)

    @test chn.name_map[:parameters] == [
        :sigma
        :beta
        :alpha
        :gen_quant
    ]
    # means = mean(chn)
    # @test means[:alpha].nt.mean[1] ≈ 2.1 rtol = 0.2
    # @test means[:beta].nt.mean[1] ≈ 2.1 rtol = 0.2
    # @test means[:sigma].nt.mean[1] ≈ 0.95 rtol = 0.2
    # @test means[:gen_quant].nt.mean[1] ≈ 4.2 rtol = 0.2

    sample_size = 10000
    hmc_chn = AbstractMCMC.sample(
        Random.default_rng(),
        model,
        Gibbs(
            OrderedDict(
                @varname(alpha) => HMC(0.1, 10),
                @varname(beta) => HMC(0.1, 10),
                @varname(sigma) => HMC(0.1, 10),
            ),
        ),
        sample_size;
        discard_initial=Int(sample_size / 2),
    )

    num_to_discard = Int(sample_size / 2)
    alpha_samples = [v[1] for v in hmc_chn[num_to_discard:end]]
    beta_samples = [v[2] for v in hmc_chn[num_to_discard:end]]
    sigma_samples = [v[3] for v in hmc_chn[num_to_discard:end]]

    alpha_mean = mean(alpha_samples)
    beta_mean = mean(beta_samples)
    sigma_mean = mean(sigma_samples)

    means = mean(hmc_chn)
    @test means[:alpha].nt.mean[1] ≈ 2.2 rtol = 0.2
    @test means[:beta].nt.mean[1] ≈ 2.1 rtol = 0.2
    @test means[:sigma].nt.mean[1] ≈ 0.9 rtol = 0.2
    @test means[:gen_quant].nt.mean[1] ≈ 4.0 rtol = 0.2
end
