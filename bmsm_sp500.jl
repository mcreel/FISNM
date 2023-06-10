
# script to do Bayesian MSM, using two step MSM criterion
#
# two step is much faster than CUE, as the number of simulations can 
# be much less. CUE, to be reliable, need S large, which is very slow
#
# to run this, do main(n, outfile, infile) where
# n is the chain length
# outfile is the filename (BSON) to store results
# infile is the name of the outfile of a previous run, or nothing
#
# if infile is nothing, SA is used to get start values
#
# n should be long enough so that the posterior mean is reasonably reliable
#
# for experiments, you can use the same name for input and output, it will overwrite


using Pkg
Pkg.activate(".")
using BSON
using CSV
using DataFrames
using DifferentialEquations
using Flux
using LinearAlgebra
using Distributions
using Random
using StatsBase
using MCMCChains
using StatsPlots

include("DGPs/DGPs.jl")
include("DGPs/JD.jl")
include("DGPs/JDalt.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("BMSM.jl")
include("samin.jl")

# Outside of the main() function due to world age issues
tcn = BSON.load("models/JD/best_model_100-50.bson")[:best_model];

# Load statistics for standardization
BSON.@load "statistics_100-50.bson" μs σs

@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
     (x .- μs) ./ σs
end

# using infile=nothing to start
function main(N,  outfile, infile)

S = 10
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 10 # MCMC verbosity

@info "Loading data, preparing model..."
# Read SP500 data and transform it to TCN-friendly format
df = CSV.read("spy.csv", DataFrame);
display(describe(df))
X₀ = Float32.(Matrix(df[:, [:rets, :rv, :bv]])) |>
    x -> reshape(x, size(x)..., 1) |>
    x -> permutedims(x, (2, 3, 1)) |> 
    tabular2conv |> preprocess

@info "Loading DGP, making data transform..."
# Define DGP of auxilary model and make transform
dgp = JD(1000)
transform_seed = 1024
Random.seed!(transform_seed)
pd = priordraw(dgp, 100_000)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)

# define DGP of the actual model
#dgp = JDalt(1000)
# get the TCN fitted parameters for the SP500 data
Flux.testmode!(tcn);
# Compute data moments
θtcn = Float64.((tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec))
lb, ub = θbounds(dgp)
θtcn = min.(θtcn,ub)
θtcn = max.(θtcn,lb)
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
    _, Σp = simmomentscov(tcn, dgp, 500, θtcn, dtθ=dtθ, preprocess=preprocess)
    Weight = inv(1000.0*(1+1/S)*Σp)
    δ = 1e0
    @info "current δ: " δ
    lb, ub = θbounds(dgp)
    lb = Float64.(lb)
    ub = Float64.(ub)
    # use 2 step objective with initial covariance using tcn estimate. SA is used to get good start values.
#    saobj = θ -> -bmsmobjective(θtcn, θ, Weight, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess)
#    θsa, junk, junk, junk = samin(saobj, θtcn, lb, ub; rt=0.5, maxevals=100, verbosity=3, nt=1, ns=20, coverage_ok=1)
#    start = θsa
    start = Float64.(θtcn)
end

# define functions for optimization and MCMC

# MVN random walk, or occasional draw from prior
@inbounds function proposal(current, δ, Σ)
    p = rand(MvNormal(current, δ*Σ))
    p[6] = max(p[6],0)
    p[8] = max(p[8],0)
    p
end
prop = θ -> proposal(θ, δ, Σp)


@info "Running MCMC..."
Random.seed!(rand(1:Int64(1e10)))
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

