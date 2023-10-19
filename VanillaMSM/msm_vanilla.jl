
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
include("samin.jl")

# use infile=nothing to start
function main()
S = 50 # number of simulations of moments at each trial parameter

# load the pre-generated data sets (true params are TCN13-17 mean)
BSON.@load "ComparisonDataSets-TCN-13-17.bson" datasets
reps = size(datasets,1)
θhats = Vector{Vector{Float64}}() # holder for the chains

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

# initialize things for minimization
start = θtcn # start at true params
lb, ub = PriorSupport()

# loop over data sets
for rep = 1:reps
    # make the data moments
    data = datasets[rep]
    mhat = MSMmoments(data)
    Random.seed!(rand(1:Int64(1e10)))
    obj = θ -> -bmsmobjective(θ, mhat, S)
    sa_results = samin(obj, start, lb, ub, rt=0.25, nt=3, ns=3, verbosity=3, coverage_ok=1)
    θhat = sa_results[1]
    @info "rep: " rep
    @info "θhat: " θhat
    push!(θhats, θhat)
end
return θhats
end
θhats = main()
BSON.@save "Vanillaθhats.bson" θhats
