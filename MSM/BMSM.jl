using StatsPlots
include("mcmc.jl")

# method to keep fixed weight matrix
function simmoments(tcn, dgp::DGP, S::Int, θ::Vector{Float32}; dtθ)
    # Computes moments according to the TCN for a given DGP, θ, and S
    X = tabular2conv(generate(θ, dgp, S)) # Generate data
    # Compute average simulated moments
    m = StatsBase.reconstruct(dtθ, tcn(X))
    return mean(m, dims=2)[:]
end

# method for CUE
function simmomentscov(tcn, dgp::DGP, S::Int, θ::Vector{Float32}; dtθ)
    # Computes moments according to the TCN for a given DGP, θ, and S
    X = tabular2conv(generate(θ, dgp, S)) # Generate data
    # Compute average simulated moments
    m = StatsBase.reconstruct(dtθ, tcn(X))
    return mean(m, dims=2)[:], Symmetric(cov(m'))
end

# MVN random walk, or occasional draw from prior
@inbounds function proposal(current, δ, Σ)
    Float32.(rand(MvNormal(current, δ .* Σ)))
end
   
# CUE objective
# objective function for Bayesian MSM: MSM with efficient weight
@inbounds function bmsmobjective(
    θ̂ₓ::Vector{Float32}, θ⁺::Vector{Float32};
    tcn, S::Int, dtθ, dgp
)        
    # Make sure the solution is in the support
    insupport(dgp, θ⁺) || return Inf
    # Compute simulated moments
    θ̂ₛ, Σₛ = simmomentscov(tcn, dgp, S, θ⁺, dtθ=dtθ)
    W = inv((1.0 + 1.0/S) .* Σₛ)
    err = θ̂ₓ - θ̂ₛ 
    dot(err, W, err)
end

# two step MSM objective
# objective function for Bayesian MSM: MSM with efficient weight
@inbounds function bmsmobjective(
    θ̂ₓ::Vector{Float32}, θ⁺::Vector{Float32}, Σinv;
    tcn, S::Int, dtθ, dgp
)        
    # Make sure the solution is in the support
    insupport(dgp, θ⁺) || return Inf
    # Compute simulated moments
    θ̂ₛ =  simmoments(tcn, dgp, S, θ⁺, dtθ=dtθ)
    err = Float32(sqrt(1000.)) * (θ̂ₓ - θ̂ₛ)
    dot(err, Σinv, err)
end




# Bayesian MSM, following Chernozhukov and Hong, 2003
function bmsm(
    dgp::DGP; 
    S::Int, dtθ, model, 
    B::Int=10,
    δ::Float32=1f0, # Tuning parameter (std adjustment)
    N::Int=1_000, # Chain length
    burnin::Int=100, # Burn-in steps
    Σreps::Int=200, # Simulations used to compute covariance of proposal
    verbosity::Int=10, 
    mcmc_verbosity::Int=100,
    show_summary::Bool=true
)
    k = nparams(dgp)
    θmat = zeros(Float32, 3k, B)
    @inbounds for i ∈ axes(θmat, 2)
        # Generate true parameters randomly
        X₀, θ₀ = generate(dgp, 1)
        θ₀ = θ₀ |> vec
        X₀ = X₀ |> tabular2conv
        # Data moments
        θ̂ₓ = model(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec
        
        # Bayesian MSM estimate by MCMC
        # the covariance of the proposal (subsequently scaled by tuning)
        _, Σp = simmomentscov(model, dgp, Σreps, θ̂ₓ, dtθ=dtθ)
        Σp = cholesky(Σp).L

        # now do MCMC
        prior(θ⁺) = 1. # support check is built into objective
        prop = θ⁺ -> proposal(θ⁺, δ, Σp) # Random walk proposal
        obj = θ⁺ -> -bmsmobjective(θ̂ₓ, θ⁺, tcn=model, S=S, dtθ=dtθ, dgp=dgp)
        chain = mcmc(θ̂ₓ, Lₙ=obj, proposal=prop, N=N, burnin=burnin, 
            verbosity=mcmc_verbosity)

        if show_summary
            # this is nice for output, need MCMCChains
            c2 = Chains(chain[:,1:end-1])
            display(plot(c2))
            display(c2)
        end

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
