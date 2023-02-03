using Pkg
Pkg.activate("../../../")
using Flux, CUDA, PrettyTables, Statistics, Term, MCMCChains
using LinearAlgebra, StatsPlots, Econometrics
using BSON:@load
include("../../../DGPs.jl")



#===================================
#
#    NOTE: should make a dgp version that keep constant random draws
#    which will limit allocations, and will allow for gradient based
#    minimization (e.g., fmincon)
#
======================================#    



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
@inbounds @views function simstat(θ, simdata, dgp, S, dtY)
    generate!(simdata, θ, dgp, S)
    Float64.(StatsBase.reconstruct(dtY, best_model(tabular2conv(simdata))))
end    

@inbounds function objective(θ, simdata, θhat, S, dgp, dtY)
        θbar = mean(simstat(θ, simdata, dgp, S, dtY), dims=2)
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
        θtrue = priordraw(dgp, 1)
        simdata = zeros(Float32, 1, 1, n)
        θtcn = simstat(θtrue, simdata, dgp, 1, dtY)[:] # the sample statistic
        display(θtrue)
        display(θtcn)
        # now do MSM
        simdata = zeros(Float32, 1, S, n) # make buffer for simdata
        obj = θ -> insupport(θ) ? objective(θ, simdata, θtcn, S, dgp, dtY) : Inf
        lb = [-2., -1.]
        ub = [2., 1.]
        results = samin(obj, θtcn, lb, ub, rt=0.5, verbosity=2, coverage_ok=1)
    end    
    return results
end


