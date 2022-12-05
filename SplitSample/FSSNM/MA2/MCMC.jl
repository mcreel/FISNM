using Pkg
Pkg.activate("../../../")
using Flux, CUDA, PrettyTables, Statistics, Term, Turing, AdvancedMH
using LinearAlgebra, StatsPlots
using BSON:@load
include("../../../DGPs.jl")

# for some reason, there is a world age problem if 
# this is inside the main() function (???)
# appears related to https://github.com/JuliaIO/BSON.jl/issues/69
global const base_n=100
BSON.@load "splitsample_(n-$base_n).bson" best_model
Flux.testmode!(best_model)

@views tabular2conv(X) = permutedims(reshape(X, size(X)..., 1), (4, 3, 1, 2))

# prior on r₁ and r₂ 
macro Prior()
    return :( arraydist([Uniform(0., 1.) for i = 1:2]) )
end

# ref https://www.cs.princeton.edu/~funk/tog02.pdf
function transform!(θ)
    r₁, r₂ = θ
    # the three vertices of the invertible region
    v₁ = [1.0, 1.0]
    v₂ = [-1.0, 1.0]
    v₃ = [0.0, -1.0]
    θ .= (1. - √r₁)*v₁ + (√r₁*(1. - r₂))*v₂ + (r₂*√r₁)*v₃
end

# draw r₁ and r₂ 
function PriorDraw()
    rand(2)
end

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



function main()
    transf = bijector(@Prior) # transforms draws from prior to draws from  ℛⁿ 
    transformed_prior = transformed(@Prior, transf) # the transformed prior
    n = 100
    dgp = Ma2(N=base_n)
    dtY = maketransform(dgp)
    # initialize args of anonymous function
    dgp = Ma2(N=n)
    θ = priordraw(dgp, 1)
    S = 1
    # true params to estimate
    θtrue = priordraw(dgp, 1)
    θtcn = simstat(θtrue, dgp, 1, dtY)[:] # the sample statistic
    display(θtrue)
    display(θtcn)

#= verify that transformations work, they do
    p = zeros(1000,2)
    for i=1:1000
        #pp = priordraw(dgp,1)
        #pp = PriorDraw()
        pp = rand(transformed_prior)
        ppp = invlink(@Prior, pp)
        transform!(ppp)
        p[i,:] = ppp
    end
=#
    # sample using Turing
    names = ["θ₁", "θ₂"]
    S = 20
    covreps = 1000
    length = 1000
    nchains = 4
    burnin = 0
    tuning = 0.1
    # the covariance of the proposal (scaled by tuning)
    #mbar, Σp = mΣ(θtcn, dgp, covreps, dtY)
    #display(mbar)
    #display(Σp)
    Σp = [1. -0.05 ;  -0.05 0.2]
    @model function MSM(θtcn, dgp, S, dtY)
        θ ~ transformed_prior   # from R2
        θ = invlink(@Prior, θ)  # now from unit square
        transform!(θ)           # now in invertible triangle
        # sample from the model, at the trial parameter value, and compute statistics
        θbar, Σ = mΣ(θ, dgp, S, dtY)
        θtcn ~ MvNormal(θbar, Σ)
    end


    chain = sample(MSM(θtcn, dgp, S, dtY),
        MH(:θ => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(θtcn,1)), tuning*Σp))),
        MCMCThreads(), length, nchains;
        init_params=Iterators.repeated([3.5,-0.04]), discard_initial=burnin)
#=
        chain = sample(MSM(θtcn, dgp, S, dtY),
        MH(:θ => AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(θtcn,1)), tuning*Σp))),
        MCMCThreads(), length, nchains; discard_initial=burnin)
=#
        display(chain)
    display(plot(chain))
    chain
end


