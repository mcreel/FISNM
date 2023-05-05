using StatsPlots
include("mcmc.jl")

function simmomentscov(tcn, dgp::DGP, S::Int, θ::Vector{Float64}; 
    dtθ, preprocess::Union{Function,Nothing}=nothing)
    # Computes moments according to the TCN for a given DGP, θ, and S
    X = tabular2conv(generate(Float32.(θ), dgp, S)) # Generate data 
    !isnothing(preprocess) ? X = preprocess(X) : nothing
     # Compute average simulated moments
     m = Float64.(StatsBase.reconstruct(dtθ, tcn(X)))
    return mean(m, dims=2)[:], Symmetric(cov(m'))
end

# MVN random walk, or occasional draw from prior
@inbounds function proposal(current, δ, Σ)
    current + δ * Σ * randn(size(current))
end
   
# Continously updated MSM objective
@inbounds function bmsmobjective(
    θ̂ₓ::Vector{Float64}, θ⁺::Vector{Float64};
    tcn, S::Int, dtθ, dgp, preprocess::Union{Function,Nothing}=nothing
)        
    # Make sure the solution is in the support
    insupport(dgp, θ⁺) || return Inf
    # Compute simulated moments
    θ̂ₛ, Σₛ = simmomentscov(tcn, dgp, S, θ⁺, dtθ=dtθ, preprocess=preprocess)
    # to work with asymptotic distribution, need to scale by √n.
    W = inv(1000.0*Σₛ)
    err = sqrt(1000.0)*(θ̂ₓ - θ̂ₛ) 
    return dot(err, W, err)
end

# Two-step MSM objective: CUE is preferred, this is for fast experiments
@inbounds function bmsmobjective(
    θ̂ₓ::Vector{Float64}, θ⁺::Vector{Float64}, Σ⁻¹::AbstractMatrix{Float64};
    tcn, S::Int, dtθ, dgp, preprocess::Union{Function,Nothing}=nothing
)        
    # Make sure the solution is in the support
    insupport(dgp, θ⁺) || return Inf
    # Compute simulated moments
    θ̂ₛ = simmoments(tcn, dgp, S, θ⁺, dtθ=dtθ, preprocess=preprocess)
    err = sqrt(1000.0)*(θ̂ₓ - θ̂ₛ)
    dot(err, Σ⁻¹, err) # Σ⁻¹ should already be scaled
end

