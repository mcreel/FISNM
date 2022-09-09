using CSV, DataFrames, Statistics, PrettyTables

params = Matrix(CSV.read("params100.csv", DataFrame))
mle = Matrix(CSV.read("MLE-n100.csv", DataFrame))

m = mean(mle, dims=1)
s = std(mle, dims=1)
err = mle - params
mse = mean(err.^2, dims=1)
rmse = sqrt.(mse)
bias = mean(err, dims=1)
h = ["param", "mean", "std", "bias", "mse", "rmse"]
pretty_table([["θ₁";"θ₂"] m' s'  bias' mse' rmse']; header=h)

