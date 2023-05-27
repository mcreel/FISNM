""" 
    chain = mcmc(θ, reps, burnin, Prior, lnL, Proposal)

    Simple MH MCMC

    You must set the needed functions, e.g.,:
    Prior = θ -> your_prior(θ, whatever other args)
    lnL = θ -> your_log_likelihood(θ, whatever other args)
    Proposal = θ -> your_proposal(θ, whatever other args)
    (optionally) mcmcProposalDensity = (θᵗ,θ) -> your_proposal_density

    then get the chain using the above syntax, or optionally,
    (non-symmetric proposal) chain = mcmc(θ, reps, burnin, Prior, lnL, Proposal, ProposalDensity, report=true)
    (example code) mcmc(): runs a simple example. edit(mcmc,()) to see the code

"""

using MCMCChains, Distributions
# method using threads and symmetric proposal
@views function mcmc(θ, reps::Int64, burnin::Int64, Prior::Function, lnL::Function, Proposal::Function, report::Bool, nthreads::Int64)
    perthread = reps ÷ nthreads
    chain = zeros(reps, size(θ,1)+1)
    Threads.@threads for t = 1:nthreads # collect the results from the threads
        chain[t*perthread-perthread+1:t*perthread,:] = <(θ, perthread, burnin, Prior, lnL, Proposal, report) 
    end    
    return chain
end

@views function adjusting_mcmc(
    θ; # TODO: prior?
    Lₙ::Function, proposal::Function, burnin::Int=100, N::Int=1_000,
    acceptance_threshold::Tuple{Float32,Float32}=(1f-1, 4f-1),
    std_adjustment_factor::Float32=1.2f0,
    std_adjustment_step::Int=100,
    curstd::Float32=1f0,
    verbosity::Int=10
)
    prop = θ⁺ -> proposal(θ⁺, curstd)
    Lₙθ = Lₙ(θ) # Objective at data moments value
    naccept = 0 # Number of acceptance / rejections
    accept = false
    acceptance_rate = 1f0
    chain = zeros(N, size(θ, 1) + 1)
    for i ∈ 1:burnin+N
        θᵗ = prop(θ) # new trial value
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
        if i > burnin
            chain[i-burnin,:] = vcat(θ, accept)
        end

        # Adjust proposal variance
        if i % std_adjustment_step == 0
            # Compute acceptance rate
            acceptance_rate = naccept / std_adjustment_step
            @info "n=$i" curstd acceptance_rate
            # Adjust proposal variance if acceptance rate is too high or too low
            if acceptance_rate < acceptance_threshold[1]
                curstd *= (1 / std_adjustment_factor)
                @info "Proposal variance decreased" curstd
            elseif acceptance_rate > acceptance_threshold[2]
                curstd *= std_adjustment_factor
                @info "Proposal variance increased" curstd
            end
            prop = θ⁺ -> proposal(θ⁺, curstd)
            # Reset acceptance rate
            naccept = 0
        end

        # Report
        if verbosity > 0 && mod(i, verbosity) == 0
            @info "Current parameters" round.(θ, digits=3)' acceptance_rate
        end
    end
    return chain
end

@views function mcmc(
    θ; # TODO: prior?
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
