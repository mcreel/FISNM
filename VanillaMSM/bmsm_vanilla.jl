
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
using Flux
using DifferentialEquations
using LinearAlgebra
using Distributions
using Random
using StatsBase
using MCMCChains
using StatsPlots

include("../DGPs/DGPs.jl")
include("../DGPs/JD.jl")
include("VanillaMoments.jl") # defines the vanilla moments, and objective
include("mcmc.jl")
@views function main()
# set configuration    
S = 50 # number of simulations of moments at each trial parameter
chainlength = 200
burnin = 10
verbosity = 50 

# load the pre-generated data sets (true params are TCN13-17 mean)
BSON.@load "ComparisonDataSets-TCN-13-17.bson" datasets
reps = size(datasets,1)
results = zeros(10,reps)

# true parameters, use as start values for chains
θtrue = [  # TCN results for 13-17 data
−0.01454,
0.17403,
−1.19645,
0.92747,
−0.79534,
0.00563,
3.25268,
0.03038]
lb, ub = θbounds(JD(1000))

# define the proposal: MVN random walk (same for all runs)
# _, Σp = simmomentscov(tcn, dgp, 1000, θtrue, dtθ=dtθ, preprocess=preprocess)
#BSON.@save "Σp.bson" Σp
BSON.@load "Σp.bson" Σp
δ = 0.35
@inbounds function proposal(current, δ, Σ)
    p = rand(MvNormal(current, δ*Σ))
    p[6] = max(p[6],0)
    p[8] = max(p[8],0)
    p
end
prop = θ -> proposal(θ, δ, Σp)

# loop over data sets
Threads.@threads for rep = 1:reps
    # Compute data moments
    data = datasets[rep]   # get the data
    mhat = MSMmoments(data)

    # likelihood
    obj = θ -> bmsmobjective(θ, mhat, S)
    # get the chain, starting at true, but with burnin
    chain = mcmc(θtrue, Lₙ=obj, proposal=prop, N=chainlength, burnin=burnin, verbosity=verbosity)
    @info "rep: " rep
    @info "acceptance rate: " mean(chain[:,9])
    θhat = mean(chain[:,1:8], dims=1)
    @info "θhat: " θhat
    results[:,rep] = mean(chain, dims=1)[:]
    rmse = sqrt.(mean((results[1:8,1:rep] .- θtrue).^2, dims=2))
    @info "rmse: " rmse 
end
return results
end
results = main()
BSON.@save "VanillaResults.bson" results
