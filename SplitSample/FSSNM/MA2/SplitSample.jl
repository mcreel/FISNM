using Pkg
Pkg.activate("../../../")
using Flux, CUDA, PrettyTables, Statistics, Term
using BSON:@load
include("../../../DGPs.jl")

# for some reason, there is a world age problem if 
# this is inside the main() function (???)
# appears related to https://github.com/JuliaIO/BSON.jl/issues/69
global const base_n=100
BSON.@load "splitsample_(n-$base_n).bson" best_model
Flux.testmode!(best_model)


function maketransform(dgp)
    transform_seed = 1204
    transform_size = 100_000
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=cpu)
end

# computes a TCN fit to data generated at a given parameter
function simstat(θ, dgp, S, dtY)
    X = generate(θ, dgp, S)
    fit = StatsBase.reconstruct(dtY, best_model(tabular2conv(X)))
    return fit
end    

function main()
    n = 100
    dgp = Ma2(N=base_n)
    dtY = maketransform(dgp)
    # initialize args of anonymous function
    dgp = Ma2(N=n)
    θ = priordraw(dgp, 1)
    S = 1
    # define the anonymous function that returns stats from parameters
    auxstat = θ -> simstat(θ, dgp, S, dtY)
    # true params to estimate
    θtrue = priordraw(dgp, 1)
    Zn = auxstat(θtrue) # the sample statistic

    # add the Turing MCMC here
    # define a mΣ function
    println("run $i")
    θ = priordraw(dgp, 1)
    S = 3
    @time fit = auxstat(θ)
    display([θ fit])
end
#=
# setting for sampling
names = [":α", ":ρ", ":σ"]
S = 100
covreps = 1000
length = 1250
nchains = 4
burnin = 0
tuning = 1.8
# the covariance of the proposal (scaled by tuning)
junk, Σp = mΣ(θhat, covreps, model, nnmodel, nninfo)

@model function MSM(m, S, model)
    θt ~ transformed_prior
    if !InSupport(invlink(@Prior, θt))
        Turing.@addlogprob! -Inf
        return
    end
    # sample from the model, at the trial parameter value, and compute statistics
    mbar, Σ = mΣ(invlink(@Prior,θt), S, model, nnmodel, nninfo)
    m ~ MvNormal(mbar, Symmetric(Σ))
end

chain = sample(MSM(m, S, model),
    MH(:θt => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(m,1)), tuning*Σp))),
    MCMCThreads(), length, nchains; init_params=Iterators.repeated(m), discard_initial=burnin)

=#



