# This computes the moments and the CUE objective function
# for TCN version.
using LinearAlgebra

function simmomentscov(tcn, dgp::DGP, S::Int, θ::Vector{Float64}; 
    dtθ, preprocess::Union{Function,Nothing}=nothing)
    # Computes moments according to the TCN for a given DGP, θ, and S
    Random.seed!(1234) # fix this so only θ varies
    X = tabular2conv(generate(Float32.(θ), dgp, S)) # Generate data
    Random.seed!(Int64(round((time()/1e6)))) # back to random
    !isnothing(preprocess) ? X = preprocess(X) : nothing
    # Compute average simulated moments
    m = Float64.(StatsBase.reconstruct(dtθ, tcn(X)))
    return mean(m, dims=2)[:], Symmetric(cov(m'))
end

  
# CUE objective, written to MAXIMIZE
@inbounds function bmsmobjective(
    θhat::Vector{Float64}, θ::Vector{Float64};
    tcn, S::Int, dtθ, dgp, preprocess::Union{Function,Nothing}=nothing
)        
    # Make sure the solution is in the support
    insupport(dgp, θ) || return -Inf
    # Compute simulated moments
    θbar, Σ = simmomentscov(tcn, dgp, S, θ, dtθ=dtθ, preprocess=preprocess)
    Σ *= dgp.N *(1+1/S) # 1 for θhat, 1/S for θbar
    isposdef(Σ) || return -Inf
    err = sqrt(dgp.N)*(θhat-θbar) 
    W = inv(Σ) # scaled for numeric accuracy
    -0.5*dot(err, W, err)
end


