using Pkg
Pkg.activate(".")
using BSON
using CSV
using DataFrames
using DifferentialEquations
using Flux
using LinearAlgebra
using Optim
using Random
using StatsBase

include("DGPs/DGPs.jl")
include("DGPs/JD.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("MSM/MSM.jl")
include("MSM/BMSM.jl")

specs = (name = "30-20", max_ret = 30, max_rv = 20, max_bv = 20)
specs = (name = "100-50", max_ret = 100, max_rv = 50, max_bv = 50)
# specs = (name = "30-20-06_0drop_7layers", max_ret = 30, max_rv = 20, max_bv = 20)
# Outside of the main() function due to world age issues
#tcn = BSON.load("models/JD/best_model_ln_bs1024_rv30_capNone.bson")[:best_model];
tcn = BSON.load("models/JD/best_model_$(specs.name).bson")[:best_model];
# tcn = BSON.load("models/JD/best_model_ln_bs1024_30-20-06_0drop_7layers.bson")[:best_model];
Flux.testmode!(tcn);


BSON.@load "statistics_$(specs.name).bson" μs σs

@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
    (x .- μs) ./ σs
end

# using infile=nothing to start
function main()

transform_seed = 1204
S = 50 # Simulations to estimate moments
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 10 # MCMC verbosity
transform_size = 100_000

# Tuning parameter
δ = 1.0e0 # this is just to define at this scope, not real value (see below)


@info "Loading data, preparing model..."
# Read SP500 data and transform it to TCN-friendly format
df = CSV.read("sp500.csv", DataFrame);
display(describe(df))
X₀ = Float32.(Matrix(df[:, [:rets, :rv, :bv]])) |>
    x -> reshape(x, size(x)..., 1) |>
    x -> permutedims(x, (2, 3, 1)) |> 
    tabular2conv |> preprocess

@info "Loading DGP, making data transform..."
# Define DGP and make transform
dgp = JD(1000)
Random.seed!(transform_seed)
pd = priordraw(dgp, transform_size)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)
# dtθ = data_transform(dgp, transform_size)

Flux.testmode!(tcn);
Random.seed!(rand(1:Int64(1e10)))
# Compute data moments
θ̂ₓ = Float64.((tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec))
display(θ̂ₓ)

# checking that model generates data similar to actual SP500
reps = 1000

d = generate(Float32.(θ̂ₓ), dgp, reps)
# Check and reject extreme samples
# idx = maximum(abs, d[1, :, :], dims=2) .≤ 50
# idx2 = mean(d[2, :, :], dims=2) .≤ 50 # Mean RV under threshold
# idx3 = mean(d[3, :, :], dims=2) .≤ 50 # Mean BV under threshold

# idx = idx .& idx2 .& idx3 |> vec
# @info "Discarding $(reps - sum(idx)) observations"
# d = d[:, idx, :]

means = mean(d, dims=[2,3]) |> vec

# means = mean(d, dims=2)

# histogram(means[1, :, :] |> vec, lab="Simulated", nbins=100)
# vline!([mean(df.rets)], lab="SP500")

# histogram(means[2, :, :] |> vec, lab="Simulated", nbins=100)
# vline!([mean(df.rv)], lab="SP500")

# histogram(means[3, :, :] |> vec, lab="Simulated", nbins=100)
# vline!([mean(df.bv)], lab="SP500")

# m = mean(d[1, :, :], dims=2) |> vec
# histogram(m, lab="")

@show means
end


main()