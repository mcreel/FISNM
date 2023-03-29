using StatsPlots
include("mcmc.jl")

function simmomentscov(tcn, dgp::DGP, S::Int, θ::Vector{Float32}; dtθ)
    # Computes moments according to the TCN for a given DGP, θ, and S
    X = tabular2conv(generate(θ, dgp, S)) # Generate data
     # Compute average simulated moments
    m = StatsBase.reconstruct(dtθ, tcn(X))
    return mean(m, dims=2)[:], Symmetric(cov(m'))
end

# MVN random walk, or occasional draw from prior
@inbounds function Proposal(current, tuning, cholV)
    Float32.(current + tuning*cholV*randn(size(current)))
end
   
# objective function for Bayesian MSM: MSM with efficient weight
@inbounds function bmsmobjective(
    θ̂ₓ::Vector{Float32}, θ⁺::Vector{Float32};
    tcn, S::Int, dtθ
)        
    # Make sure the solution is in the support
    insupport(dgp, θ⁺...) || return Inf
    # Compute simulated moments
    θ̂ₛ,Σₛ = simmomentscov(tcn, dgp, S, θ⁺, dtθ=dtθ)
    W = inv(Σₛ)
    err = θ̂ₓ - θ̂ₛ 
    dot(err, W, err)
end

# Bayesian MSM, following Chernozhukov and Hong, 2003
function bmsm(
    dgp::DGP; 
    S::Int, dtθ, model, M::Int=10, verbosity::Int=0, show_trace::Bool=false
)
    # MCMC controls
    tuning = 1.0
    covreps = 200 # simulations used to compute covariance of proposal
    length = 1000
    burnin = 100
    mcmcverbosity = true
    nthreads = 1 # threads for mcmc
    # end of MCMC controls
    k = nparams(dgp)
    θmat = zeros(Float32, 3k, M)
    @inbounds for i ∈ axes(θmat, 2)
        # Generate true parameters randomly
        X₀, θ₀ = generate(dgp, 1)
        θ₀ = θ₀ |> vec
        X₀ = X₀ |> tabular2conv
        # Data moments
        θ̂ₓ = model(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec
        # Bayesian MSM estimate by MCMC
        # the covariance of the proposal (subsequently scaled by tuning)
        junk, Σp = simmomentscov(tcn, dgp, covreps, θ̂ₓ, dtθ=dtθ)

display(θ₀)        
display(θ̂ₓ)
display(junk)
display(Σp)

        Σp = Matrix(cholesky(Σp).U)'
        # now do MCMC
        prior(θ⁺) = 1. # support check is built into objective
        proposal = θ⁺ -> Proposal(θ⁺, tuning, Σp)
        obj = θ⁺ -> -bmsmobjective(θ̂ₓ, θ⁺, tcn=model, S=S, dtθ=dtθ)
        @time chain = mcmc(θ̂ₓ, length, burnin, prior, obj, proposal, mcmcverbosity)

# this is nice for output, need MCMCChains
c2 = Chains(chain[:,1:end-1])
display(plot(c2))
display(c2)

        θ̂ₘₛₘ = mean(chain[:,1:end-1], dims=1) |> vec  

        θmat[:, i] = vcat(θ₀, θ̂ₓ, θ̂ₘₛₘ)
        if verbosity > 0 && i % verbosity == 0
            # Compute average RMSE
            armse_msm = mean(sqrt.(mean(abs2, θmat[1:k, 1:i] .- θmat[2k+1:end, 1:i], dims=2)))
            armse_tcn = mean(sqrt.(mean(abs2, θmat[1:k, 1:i] .- θmat[k+1:2k, 1:i], dims=2)))
            @info "Iteration $i" armse_msm armse_tcn
        end
    end
    return permutedims(θmat)
end
