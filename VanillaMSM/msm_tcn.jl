
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
include("JD.jl")
include("../NeuralNets/utils.jl")
include("../NeuralNets/tcn_utils.jl")
include("../NeuralNets/TCN.jl")
include("TCNMoments.jl")  # defines the moments and MSM objective
include("samin.jl")

#---------- TCN stuff-------------
specs = (name = "30-20", max_ret = 30, max_rv = 20, max_bv = 20)
BSON.@load "../statistics/statistics_$(specs.name).bson" μs σs
# build the net
dgp = JD(1000)
tcn = build_tcn(dgp, dilation=2, kernel_size=32, channels=32,
    summary_size=10, dev=cpu, n_layers=7)
# load the trained parameters of net
BSON.@load "../tcn_state_$(specs.name).bson" tcn_state
Flux.loadmodel!(tcn, tcn_state)
Flux.testmode!(tcn)
@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
     (x .- μs) ./ σs
end
transform_seed = 1024
Random.seed!(transform_seed)
pd = priordraw(dgp, 100_000)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)
#---------- done with TCN stuff-------------

function main()
S = 50 # number of simulations of moments at each trial parameter

# load the pre-generated data sets (true params are TCN13-17 mean)
BSON.@load "ComparisonDataSets-TCN-13-17.bson" datasets
reps = size(datasets,1)
θhats = Vector{Vector{Float64}}() # holder for the chains

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


# initialize things for minimization
lb, ub = PriorSupport()
start = θtrue

# loop over data sets
for rep = 1:reps
    # Compute data moments
    data = datasets[rep]   # get the data
    data = Float32.(data) |>   # transform to TCN style
    x -> reshape(x, size(x)..., 1) |>
    x -> permutedims(x, (2, 3, 1)) |> 
    tabular2conv |> preprocess
    θtcn = Float64.((tcn(data) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec))  # get the tcn fit
    θtcn = min.(θtcn,ub)
    θtcn = max.(θtcn,lb)
    # do SA to get MSM estimate
    obj = θ -> -bmsmobjective(θtcn,  θ; tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess) 
    sa_results = samin(obj, start, lb, ub, rt=0.25, nt=3, ns=3, verbosity=1, coverage_ok=1)
    θhat = sa_results[1]
    @info "rep: " rep
    @info "θhat: " θhat
    push!(θhats, θhat)

end
return θhats
end
θhats = main()
BSON.@save "TCNθhats.bson" θhats
