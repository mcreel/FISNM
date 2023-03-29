function simmoments(tcn, dgp::DGP, S::Int, θ::Vector{Float32}; dtθ)
    # Computes moments according to the TCN for a given DGP, θ, and S
    X = tabular2conv(generate(θ, dgp, S)) # Generate data
     # Compute average simulated moments
    return mean(StatsBase.reconstruct(dtθ, tcn(X)), dims=2) |> vec
end

# Objective function
function objective(
    θ̂ₓ::Vector{Float32}, θ⁺::Vector{Float32};
    tcn, S::Int, dtθ, seed::Union{Nothing, Int}=nothing
)
    # Make sure the solution is in the support
    insupport(dgp, θ⁺...) || return Inf
    isnothing(seed) || Random.seed!(seed)
    # Compute simulated moments
    θ̂ₛ = simmoments(tcn, dgp, S, θ⁺, dtθ=dtθ)
    # Compute error
    sum(abs2, θ̂ₓ - θ̂ₛ) # MSE of data and simulated moments
end


function msm(
    dgp::DGP; 
    S::Int, dtθ, model, M::Int=10, verbosity::Int=0, show_trace::Bool=false
)
    k = nparams(dgp)
    θmat = zeros(Float32, 3k, M)
    @inbounds for i ∈ axes(θmat, 2)
        # Generate true parameters randomly
        X₀, θ₀ = generate(dgp, 1)
        θ₀ = θ₀ |> vec
        X₀ = X₀ |> tabular2conv
        # Data moments
        θ̂ₓ = model(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec
        # MSM estimate
        seed = abs(rand(Int64))
        θ̂ₘₛₘ = optimize(θ⁺ -> objective(
            θ̂ₓ, θ⁺, tcn=model, S=S, dtθ=dtθ, seed=seed), θ̂ₓ, NelderMead(),
            Optim.Options(show_trace=show_trace)).minimizer
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
