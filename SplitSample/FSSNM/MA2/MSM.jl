using Pkg
Pkg.activate("../../../")
using Flux, CUDA, PrettyTables, Statistics, Term, MCMCChains
using LinearAlgebra, Econometrics, DelimitedFiles
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
    for s = 1:S
        simdata[1,s,:] .= shocks[1,s,3:end] .+ θ[1].*shocks[1,s,2:end-1] .+ θ[2].*shocks[1,s,1:end-2]
    end    
    (StatsBase.reconstruct(dtY, best_model(tabular2conv(Float32.(simdata)))))
end    

@inbounds function objective(θ, θhat, simdata, shocks, dtY)
        θbar = mean(simstat(θ, simdata, shocks, dtY), dims=2)
        sum(abs2, θhat - θbar)
end

@inbounds function main()
    n = 100 # sample size
    reps = 500
    dgp = Ma2(N=base_n)
    dtY = maketransform(dgp)
    names = ["θ₁", "θ₂"]
    S = 20 # reps used to evaluate objective
    results = zeros(reps, 7)
    Threads.@threads for rep = 1:reps
        # true params to estimate
        θtrue::Vector{Float64} = vec(priordraw(dgp, 1))
        results[rep, 1:2] = θtrue
        simdata1 = zeros(1, 1, n)
        shocks1 = randn(1, 1, n+2)
        θtcn::Vector{Float64} = vec(simstat(θtrue, simdata1, shocks1, dtY))# the sample statistic
        results[rep, 3:4] = θtcn
        # now do MSM
        simdata = zeros(1, S, n) # make buffer for simdata
        shocks = randn(1, S, n+2)
        obj = θ -> insupport(θ) ? objective(θ, θtcn, simdata, shocks, dtY) : Inf
        lb = [-1.999, -0.999]
        ub = [1.999, 0.999]
#        results[rep, 5:6], results[rep, 7], junk = fmincon(obj, θtcn, [],[],lb, ub, tol=1e-5, iterlim=10000)
        results[rep, 5:6], results[rep, 7], junk, junk  = samin(obj, θtcn, lb, ub, rt=0.5, verbosity=1, coverage_ok=1)
        println(@green "results, rep=$rep")
        pretty_table(results[rep, :]')
    end    
    writedlm("resultsMA2.txt", results)
end


