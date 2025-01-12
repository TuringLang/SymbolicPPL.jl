name = "Birats: a bivariate normal hierarchical model"

model_def = @bugs begin
    for i in 1:N
        beta[i, 1:2] ~ dmnorm(var"mu.beta"[:], R[:, :])
        for j in 1:T
            Y[i, j] ~ dnorm(mu[i, j], tauC)
            mu[i, j] = beta[i, 1] + beta[i, 2] * x[j]
        end
    end

    var"mu.beta"[1:2] ~ dmnorm(mean[:], prec[:, :])
    R[1:2, 1:2] ~ dwish(Omega[:, :], 2)
    tauC ~ dgamma(0.001, 0.001)
    sigma = 1 / sqrt(tauC)
end

original = """
model {
    for( i in 1 : N ) {
        beta[i , 1 : 2] ~ dmnorm(mu.beta[], R[ , ])
        for( j in 1 : T ) {
            Y[i, j] ~ dnorm(mu[i , j], tauC)
            mu[i, j] <- beta[i, 1] + beta[i, 2] * x[j]
        }
    }
    
    mu.beta[1 : 2] ~ dmnorm(mean[], prec[ , ])
    R[1 : 2 , 1 : 2] ~ dwish(Omega[ , ], 2)
    tauC ~ dgamma(0.001, 0.001)
    sigma <- 1 / sqrt(tauC)
}
"""

data = (
    x = [8.0, 15.0, 22.0, 29.0, 36.0],
    N = 30,
    T = 5,
    Omega = [200.0 0.0; 0.0 0.2],
    mean = [0, 0],
    prec = [1.0e-6 0.0; 0.0 1.0e-6],
    Y = [151 199 246 283 320
         145 199 249 293 354
         147 214 263 312 328
         155 200 237 272 297
         135 188 230 280 323
         159 210 252 298 331
         141 189 231 275 305
         159 201 248 297 338
         177 236 285 350 376
         134 182 220 260 296
         160 208 261 313 352
         143 188 220 273 314
         154 200 244 289 325
         171 221 270 326 358
         163 216 242 281 312
         160 207 248 288 324
         142 187 234 280 316
         156 203 243 283 317
         157 212 259 307 336
         152 203 246 286 321
         154 205 253 298 334
         139 190 225 267 302
         146 191 229 272 302
         157 211 250 285 323
         132 185 237 286 331
         160 207 257 303 345
         169 216 261 295 333
         157 205 248 289 316
         137 180 219 258 291
         153 200 244 286 324]
)

inits = (
    var"mu.beta" = [0, 0],
    tauC = 1,
    beta = [100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6
            100 6],
    R = [1 0; 0 1]
)
inits_alternative = (
    var"mu.beta" = [10, 10],
    tauC = 0.1,
    beta = [50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3
            50 3],
    R = [3 0; 0 3]
)

reference_results = (
    var"mu.beta[1:2][1]" = (mean = 106.6, std = 2.361),
    var"mu.beta[1:2][2]" = (mean = 6.185, std = 0.1063),
    sigma = (mean = 6.149, std = 0.4789)
)

birats = Example(
    name, model_def, original, data, inits, inits_alternative, reference_results)
