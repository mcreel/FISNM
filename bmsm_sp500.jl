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
using MCMCChains
using StatsPlots

include("DGPs/DGPs.jl")
include("DGPs/JD.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("MSM/MSM.jl")
include("MSM/BMSM.jl")
include("samin.jl")

# Outside of the main() function due to world age issues
tcn = BSON.load("models/JD/best_model_ln_bs1024_30-20-06.bson")[:best_model];

# Load statistics for standardization
BSON.@load "statistics_30-20-06.bson" μs σs

@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
     (x .- μs) ./ σs
end

# using infile=nothing to start
function main(N,  outfile, infile)

transform_seed = 1204
S = 10 # Simulations to estimate moments
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 10 # MCMC verbosity

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
pd = priordraw(dgp, 100_000)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)

# get the TCN fitted parameters for the SP500 data
Flux.testmode!(tcn);
# Compute data moments
θtcn = Float64.((tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec))
display(θtcn)

if infile !== nothing
    BSON.@load infile chain Σp Weight δ 
    # use mean of last 90% as start for next chain
    n = size(chain,1)
    start = mean(chain[Int64(0.9*n):end,1:8], dims=1)[:]
    @info "start value: " start
    # update the weight matrix and proposal covariance
    @info "Computing covariance of the proposal and the weight ..."
    _, Σp = simmomentscov(tcn, dgp, covreps, start, dtθ=dtθ, preprocess=preprocess)
    Weight = inv(1000.0*(1+1/S)*Σp)
    # adjust proposal depending on acceptance rate
    ac = mean(chain[:,9])
    @info "acceptance rate of input chain: " ac
    mean(chain[:,9]) > 0.3  ? δ *= 1.25 : nothing
    mean(chain[:,9]) < 0.2  ? δ *= 0.75 : nothing
    @info "current δ: " δ
else  
    @info "Computing covariance of the proposal..."
    _, Σp = simmomentscov(tcn, dgp, covreps, θtcn, dtθ=dtθ, preprocess=preprocess)
    Weight = inv(1000.0*(1+1/S)*Σp)
    δ = 1e0
    @info "current δ: " δ
    lb, ub = θbounds(dgp)
    lb = Float64.(lb)
    ub = Float64.(ub)
    # use 2 step objective with initial covariance using tcn estimate. SA is used to get good start values.
    #saobj = θ -> -bmsmobjective(θtcn, θ, Weight, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess)
    #θsa, junk, junk, junk = samin(saobj, θtcn, lb, ub; rt=0.5, maxevals=1000, verbosity=3, nt=1, ns=20, coverage_ok=1)
    start = [-0.008, 0.078, -0.81, 0.77, -0.89, 0.057, 2.37, 0.045] # from the SA run
 end

# define functions for optimization and MCMC
prop = θ -> rand(MvNormal(θ, δ * Σp))
Random.seed!(rand(1:Int64(1e10)))


@info "Running MCMC..."
obj = θ -> bmsmobjective(θtcn, θ, Weight, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess)
chain = mcmc(start, Lₙ=obj, proposal=prop, N=N, burnin=burnin, verbosity=verbosity)
@info "acceptance rate: " mean(chain[:,9])
# Save chain
BSON.@save outfile chain Σp Weight δ


# # Make MCMC chain and display
ch = Chains(chain)
display(ch)
display(plot(ch))
end

