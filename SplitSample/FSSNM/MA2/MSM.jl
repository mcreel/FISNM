using Pkg
Pkg.activate("../../../")
using Flux, CUDA, PrettyTables, Statistics, Term, MCMCChains
using LinearAlgebra, StatsPlots, Econometrics
using BSON:@load
include("../../../DGPs.jl")


# for some reason, there is a world age problem if 
# this is inside the main() function (???)
# appears related to https://github.com/JuliaIO/BSON.jl/issues/69
global const base_n=100
BSON.@load "splitsample_(n-$base_n).bson" best_model
Flux.testmode!(best_model)

@inbounds @views tabular2conv(X) = permutedims(reshape(X, size(X)..., 1), (4, 3, 1, 2))

@inbounds function maketransform(dgp)
    transform_seed = 1204
    transform_size = 100_000
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=cpu)
end

# computes a TCN fit to data generated at a given parameter
@inbounds @views function simstat(θ, simdata, shocks, dtY)
    S = size(simdata)[2]
    Threads.@threads for s = 1:S
        simdata[1,s,:] = shocks[1,s,3:end] .+ θ[1].*shocks[1,s,2:end-1] .+ θ[2].*shocks[1,s,1:end-2]
    end    
    (StatsBase.reconstruct(dtY, best_model(tabular2conv(Float32.(simdata)))))
end    

@inbounds function objective(θ, θhat, simdata, shocks, dtY)
        θbar = mean(simstat(θ, simdata, shocks, dtY), dims=2)
        sum(abs2, θhat - θbar)
end

@inbounds function main()
    n = 100 # sample size
    reps = 1
    dgp = Ma2(N=base_n)
    dtY = maketransform(dgp)
    names = ["θ₁", "θ₂"]
    S = 50 # reps used to evaluate objective
    results = [] # define to access out of loop
    for rep = 1:reps
        # true params to estimate
        θtrue::Vector{Float64} = vec(priordraw(dgp, 1))
        simdata1 = zeros(1, 1, n)
        shocks1 = randn(1, 1, n+2)
        θtcn::Vector{Float64} = vec(simstat(θtrue, simdata1, shocks1, dtY))# the sample statistic
        display(θtrue)
        display(θtcn)
        # now do MSM
        simdata = zeros(1, S, n) # make buffer for simdata
        shocks = randn(1, S, n+2)
        obj = θ -> insupport(θ) ? objective(θ, θtcn, simdata, shocks, dtY) : Inf
        lb = [-2., -1.]
        ub = [2., 1.]
        results = fmincon(obj, θtcn, [],[],lb, ub)
        #results = samin(obj, θtcn, lb, ub, rt=0.5, verbosity=2, coverage_ok=1)
    end    
    return results
end


