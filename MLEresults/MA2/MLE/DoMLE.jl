using Pkg
Pkg.activate(".")
using CSV, DataFrames, Statistics

θ = CSV.read("params_100.csv", header=0, DataFrame)
θ = Matrix(θ)'
θhat  = Matrix(CSV.read("MLE100.csv", DataFrame))
err = θhat - θ

mae = mean(abs.(err),dims=1)
bias = mean(err,dims=1)
mab = mean(abs.(bias))
armse = mean(sqrt.(mean(abs2.(err),dims=1)))
rmae = mae ./ [1.0, 0.5]
@info "avg relative MAE" avgrmae = mean(rmae)
@info "average absolute bias" mab
@info "average RMSE" armse


