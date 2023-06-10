using LinearAlgebra

function simmomentscov(tcn, dgp::DGP, S::Int, θ::Vector{Float64}; 
    dtθ, preprocess::Union{Function,Nothing}=nothing)
    # Computes moments according to the TCN for a given DGP, θ, and S
    X = tabular2conv(generate(Float32.(θ), dgp, S)) # Generate data
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
    Σ *= 1000.0*(1+1/S) # 1 for θhat, 1/S for θbar
    isposdef(Σ) || return -Inf
    err = sqrt(1000.0)*(θhat-θbar) 
    W = inv(Σ) # scaled for numeric accuracy
    -0.5*dot(err, W, err)
end

# Two-step objective, written to MAXIMIZE
@inbounds function bmsmobjective(
    θhat::Vector{Float64}, θ::Vector{Float64}, Weight;
    tcn, S::Int, dtθ, dgp, preprocess::Union{Function,Nothing}=nothing
)        
    # Make sure the solution is in the support
    insupport(dgp, θ) || return -Inf
    # Compute simulated moments
    θbar, _ = simmomentscov(tcn, dgp, S, θ, dtθ=dtθ, preprocess=preprocess)
    err = sqrt(1000.0)*(θhat-θbar) 
    -0.5*dot(err, Weight, err)
end

@views function mcmc(
    θ; # TODO: prior? not needed at present, as priors are uniform
    Lₙ::Function, proposal::Function, burnin::Int=100, N::Int=1_000,
    verbosity::Int=10
)
    Lₙθ = Lₙ(θ) # Objective at data moments value
    naccept = 0 # Number of acceptance / rejections
    accept = false
    acceptance_rate = 1f0
    chain = zeros(N, size(θ, 1) + 2)
    for i ∈ 1:burnin+N
        θᵗ = proposal(θ) # new trial value
        Lₙθᵗ = Lₙ(θᵗ) # Objective at trial value
        # Accept / reject trial value
        accept = rand() < exp(Lₙθᵗ - Lₙθ)
        if accept
            # Replace values
            θ = θᵗ
            Lₙθ = Lₙθᵗ
            # Increment number of accepted values
            naccept += 1
        end
        # Add to chain if burnin is passed
        # @info "current log-L" Lₙθ
        if i > burnin
            chain[i-burnin,:] = vcat(θ, accept, Lₙθ)
        end
        # Report
        if verbosity > 0 && mod(i, verbosity) == 0
            acceptance_rate = naccept / verbosity
            @info "Current parameters (iteration i=$i)" round.(θ, digits=3)' acceptance_rate
            naccept = 0
        end
    end
    return chain
end
