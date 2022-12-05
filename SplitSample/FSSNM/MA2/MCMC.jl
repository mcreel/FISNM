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

@views tabular2conv(X) = permutedims(reshape(X, size(X)..., 1), (4, 3, 1, 2))

function maketransform(dgp)
    transform_seed = 1204
    transform_size = 100_000
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=cpu)
end

# computes a TCN fit to data generated at a given parameter
@views function simstat(θ, dgp, S, dtY)
    X = generate(θ, dgp, S)
    Float64.(StatsBase.reconstruct(dtY, best_model(tabular2conv(X))))
end    

@views function mΣ(θ, dgp, S, dtY)
    Zs = simstat(θ, dgp, S, dtY)
    mean(Zs, dims=2)[:], Symmetric(cov(Zs'))
end

# MVN random walk, or occasional draw from prior
function Proposal(current, tuning, cholV)
    current + tuning*cholV'*randn(size(current))
end
   
function objective(θ, θhat, S,  dgp, dtY, W)
    m = (θhat - mean(simstat(θ, dgp, S, dtY),dims=2))[:]
    dot(m, W, m)
end    


function main()
    n = 100
    dgp = Ma2(N=base_n)
    dtY = maketransform(dgp)
    # initialize args of anonymous function
    dgp = Ma2(N=n)
    θ = priordraw(dgp, 1)
    # true params to estimate
    θtrue = priordraw(dgp, 1)
    θtcn = simstat(θtrue, dgp, 1, dtY)[:] # the sample statistic
    display(θtrue)
    display(θtcn)


    # sample using Turing
    names = ["θ₁", "θ₂"]
    S = 20
    covreps = 1000
    length = 5000
    burnin = 1000
    tuning = 15.
    verbosity = true
    # the covariance of the proposal (scaled by tuning)
    junk, Σp = mΣ(θtcn, dgp, covreps, dtY)
    W = inv(Σp)
    #Σp = cholesky(Σp)
    # now do MCMC
    prior(θ) = insupport(θ) ? 1. : 0.
    proposal = θ -> Proposal(θ, tuning, Σp)
    obj = θ -> insupport(θ) ? -objective(θ, θtcn, S, dgp, dtY, W) : Inf
    chain = mcmc(θtcn, length, burnin, prior, obj, proposal, verbosity)
    chain = Chains(chain[:,1:2], names)
    display(plot(chain))
    display(chain)
    return chain
end


