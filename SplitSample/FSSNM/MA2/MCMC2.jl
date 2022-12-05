using Pkg
Pkg.activate("../../../")
using Flux, CUDA, PrettyTables, Statistics, Term, MCMCChains
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

# method using threads and symmetric proposal
@views function mcmc(θ, reps::Int64, burnin::Int64, Prior::Function, lnL::Function, Proposal::Function, report::Bool, nthreads::Int64)
    perthread = reps ÷ nthreads
    chain = zeros(reps, size(θ,1)+1)
    Threads.@threads for t = 1:nthreads # collect the results from the threads
        chain[t*perthread-perthread+1:t*perthread,:] = mcmc(θ, perthread, burnin, Prior, lnL, Proposal, report) 
    end    
    return chain
end


# method symmetric proposal
# the main loop
@views function mcmc(θ, reps::Int64, burnin::Int64, Prior::Function, lnL::Function, Proposal::Function, report::Bool=true)
    reportevery = Int((reps+burnin)/10)
    lnLθ = lnL(θ)
    chain = zeros(reps, size(θ,1)+1)  #!!!!!! use a vector of vectors
    naccept = zeros(size(θ))
    for rep = 1:reps+burnin
        θᵗ = Proposal(θ) # new trial value  # MAKE THIS non-allocating!
        if report
            changed = Int.(.!(θᵗ .== θ)) # find which changed
        end    
        # MH accept/reject: only evaluate logL if proposal is in support of prior (avoid crashes)
        pt = Prior(θᵗ)
        accept = false
        if pt > 0.0
            lnLθᵗ = lnL(θᵗ)
            accept = rand() < exp(lnLθᵗ-lnLθ) * pt/Prior(θ)
            if accept
                θ = θᵗ
                lnLθ = lnLθᵗ 
            end
        end
        if report
            naccept = naccept .+ changed .* Int.(accept)
        end    
        if (mod(rep,reportevery)==0 && report)
            println("current parameters: ", round.(θ,digits=3))
            println("  acceptance rates: ", round.(naccept/reportevery,digits=3))
            naccept = naccept - naccept
        end    
        if rep > burnin
            chain[rep-burnin,:] = vcat(θ, accept)
        end    
    end
    return chain
end


    

function main()
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


    function objective(θ, θhat, S,  dgp, dtY, W)
        m = (θhat - mean(simstat(θ, dgp, S, dtY),dims=2))[:]
        dot(m, W, m)
    end    

    # sample using Turing
    names = ["θ₁", "θ₂"]
    S = 20
    covreps = 1000
    length = 1000
    burnin = 100
    tuning = 10.
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


