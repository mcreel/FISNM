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
@inbounds @views function simstat(θ, simdata, dgp, S, dtY)
    generate!(simdata, θ, dgp, S)
    Float64.(StatsBase.reconstruct(dtY, best_model(tabular2conv(simdata))))
end    

@inbounds @views function mΣ(θ, simdata, dgp, S, dtY)
    Zs = simstat(θ, simdata, dgp, S, dtY)
    mean(Zs, dims=2)[:], Symmetric(cov(Zs'))
end

# MVN random walk, or occasional draw from prior
@inbounds function Proposal(current, tuning, cholV)
    current + tuning*cholV*randn(size(current))
end
   
@inbounds function objective(θ, simdata, θhat, S, dgp, dtY)
        θbar, Σp = mΣ(θ, simdata, dgp, S, dtY)
        W = inv(Σp)
        err = θhat - θbar
        dot(err, W, err)
end

@inbounds function main()
 # reps for covariance of proposa # reps for covariance of proposallfunction main()
    n = 100
    reps = 1
    dgp = Ma2(N=base_n)
    dtY = maketransform(dgp)
    # MCMC sampling
    names = ["θ₁", "θ₂"]
    S = 5 # reps used to evaluate objective
    covreps = 200 # reps for covariance of proposal
    length = 1000
    burnin = 10
    tuning = 1.0
    verbosity = false
    nthreads = 2
    # initialize args of anonymous function
    dgp = Ma2(N=n)
    for rep = 1:reps
        # true params to estimate
        θtrue = priordraw(dgp, 1)
        simdata = zeros(Float32, 1, 1, n)
        θtcn = simstat(θtrue, simdata, dgp, 1, dtY)[:] # the sample statistic
        display(θtrue)
        display(θtcn)
        # the covariance of the proposal (subsequently scaled by tuning)
        simdata = zeros(Float32, 1, covreps, n) # make buffer for simdata
        junk, Σp = mΣ(θtcn, simdata, dgp, covreps, dtY)
        Σp = Matrix(cholesky(Σp).U)'
        # now do MCMC
        simdata = zeros(Float32, 1, S, n) # make buffer for simdata
        prior(θ) = insupport(θ) ? 1. : 0.
        proposal = θ -> Proposal(θ, tuning, Σp)
        obj = θ -> -objective(θ, simdata, θtcn, S, dgp, dtY)
        @time chain = mcmc(θtcn, length, burnin, prior, obj, proposal, verbosity, nthreads)
        chain = Chains(chain[:,1:2], names)
        display(plot(chain))
        display(chain)
    end    
    #return chain
end


