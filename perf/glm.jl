using GLM
glm(y ~ 1, DataFrame(y = float64(randbool(10))), Binomial())

n = 2_500_000; srand(1234321)
df2 = DataFrame(x1 = rand(Normal(), n),
                x2 = rand(Exponential(), n),
                ss = pool(rand(DiscreteUniform(50), n)));
beta = unshift!(rand(Normal(),52), 0.5); # "true" parameter values

## Create linear predictor and mean response
df2[:eta] = eta = ModelMatrix(ModelFrame(x1 ~ x1 + x2 + ss, df2)).m * beta;
df2[:mu] = mu = linkinv!(LogitLink(), similar(eta), eta); 

df2[:y] = float64(rand(n) .< mu);        # simulate observed responses

mm = ModelFrame(y ~ x1 + x2 + ss, df2); X = ModelMatrix(mm).m; y = model_response(mm);
@time gm6 = glm(X, y, Binomial())
@profile gm6 = glm(X, y, Binomial())

head(df2)

gm6 = glm(y ~ x1 + x2 + ss, df2, Binomial())
@time glm(y ~ x1 + x2 + ss, df2, Binomial());
