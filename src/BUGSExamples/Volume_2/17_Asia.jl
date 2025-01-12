name = "Asia: expert system"

model_def = @bugs begin
    smoking ~ dcat(var"p.smoking"[1:2])
    tuberculosis ~ dcat(var"p.tuberculosis"[asia, 1:2])
    lung_cancer ~ dcat(var"p.lung.cancer"[smoking, 1:2])
    bronchitis ~ dcat(var"p.bronchitis"[smoking, 1:2])
    either = max(tuberculosis, lung_cancer)
    xray ~ dcat(var"p.xray"[either, 1:2])
    dyspnoea ~ dcat(var"p.dyspnoea"[either, bronchitis, 1:2])
end

original = """
model {
    smoking ~ dcat(p.smoking[1:2])
    tuberculosis ~ dcat(p.tuberculosis[asia,1:2])
    lung.cancer ~ dcat(p.lung.cancer[smoking,1:2])
    bronchitis ~ dcat(p.bronchitis[smoking,1:2])
    either <- max(tuberculosis, lung.cancer)
    xray ~ dcat(p.xray[either,1:2])
    dyspnoea ~ dcat(p.dyspnoea[either, bronchitis,1:2])
}
"""

data = (asia = 2,
    dyspnoea = 2,
    var"p.tuberculosis" = [0.99 0.01; 0.95 0.05],  # Column-major order
    var"p.bronchitis" = [0.70 0.30; 0.40 0.60],    # Column-major order
    var"p.smoking" = [0.50, 0.50],
    var"p.lung.cancer" = [0.99 0.01; 0.90 0.10],  # Column-major order
    var"p.xray" = [0.95 0.05; 0.02 0.98],         # Column-major order
    var"p.dyspnoea" = reshape([0.9 0.1
                               0.2 0.8
                               0.3 0.7
                               0.1 0.9], (2, 2, 2))
)

inits = (smoking = 1, tuberculosis = 1, var"lung.cancer" = 1, bronchitis = 1, xray = 1)
inits_alternative = (
    smoking = 2, tuberculosis = 2, var"lung.cancer" = 2, bronchitis = 2, xray = 2)

reference_results = (
    bronchitis = (mean = 1.811, std = 0.3918),
    either = (mean = 1.185, std = 0.3885),
    lung_cancer = (mean = 1.101, std = 0.3011),
    smoking = (mean = 1.628, std = 0.4833),
    tuberculosis = (mean = 1.089, std = 0.2854),
    xray = (mean = 1.223, std = 0.4161)
)

asia = Example(name, model_def, original, data, inits, inits_alternative, reference_results)
