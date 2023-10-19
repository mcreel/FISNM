
# script to do Bayesian MSM, using CUE
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
using LinearAlgebra
using Distributions
using Random
using StatsBase
using MCMCChains
using StatsPlots

include("JD.jl")
include("Vanilla.jl")
include("mcmc.jl")


# use infile=nothing to start
function main()
chainlength = 2000
S = 25 # number of simulations of moments at each trial parameter
burnin = 200 # Burn-in steps
verbosity = 10 # MCMC verbosity

# load the pre-generated data sets (true params are TCN13-17 mean)
BSON.@load "ComparisonDataSets-TCN-13-17.bson" datasets
reps = size(datasets,1)
chains = Vector{Array{Float64,2}}() # holder for the chains

# true parameters, use as start values for chains
θtcn = [  # TCN results for 13-17 data
−0.01454,
0.17403,
−1.19645,
0.92747,
−0.79534,
0.00563,
3.25268,
0.03038]

# get covariance of proposal from the TCN chain
BSON.@load "30-20-sample1-sobel-chain2.bson" chain 
Σp = cov(chain[:,1:8])
start = θtcn # start at true params
δ = 0.1
# Define the proposal: MVN random walk
@inbounds function proposal(current, δ, Σ)
    p = rand(MvNormal(current, δ*Σ))
    p[6] = max(p[6],0)
    p[8] = max(p[8],0)
    p
end
prop = θ -> proposal(θ, δ, Σp)

# loop over data sets
for rep = 1:reps
    # make the data moments
    data = datasets[rep]
    mhat = MSMmoments(data)
    Random.seed!(rand(1:Int64(1e10)))
    obj = θ -> bmsmobjective(θ, mhat, S)
    chain = mcmc(start, Lₙ=obj, proposal=prop, N=chainlength, burnin=burnin, verbosity=verbosity)
    @info "rep: " rep
    @info "acceptance rate: " mean(chain[:,9])
    # Save chain
    push!(chains, chain)
    ch = Chains(chain)
    display(ch)
end
return chains
end
chains = main()
BSON.@save "VanillaChains.bson" chains 
