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
using Plots
include("DGPs/DGPs.jl")
include("DGPs/JD.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("MSM/MSM.jl")
include("MSM/BMSM.jl")


# Outside of the main() function due to world age issues
tcn = BSON.load("models/JD/best_model_ln_bs1024_rv20_new.bson")[:best_model];


# Load statistics for standardization
BSON.@load "statistics_new_20.bson" lnμs lnσs

@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
     (x .- lnμs) ./ lnσs
end


# using infile=nothing to start
function main()

transform_seed = 1204
S = 50 # Simulations to estimate moments
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 10 # MCMC verbosity

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
pd = priordraw(dgp, 200_000)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)

Flux.testmode!(tcn);
Random.seed!(rand(1:Int64(1e10)))
# Compute data moments
θ  = Float64.((tcn(X₀) |> m -> StatsBase.reconstruct(dtθ, m) |> vec))
display(θ)

# checking that model generates data similar to actual SP500
reps = 100
means = zeros(1,3)
meansθ = zeros(1,8)
for j = 1:reps
    d = generate(Float32.(θ), dgp, 1)
    data = zeros(1000,3)
    for i = 1:1000
        data[i,:] = d[:,:,i]
    end
#    display(plot(data[:,1]))
#    sleep(10)
    means .+= mean(data,dims=1)/reps
    meansθ .+= StatsBase.reconstruct(dtθ, tcn(preprocess(tabular2conv(d))))' / reps
end
@show means
@show meansθ
nothing
end

