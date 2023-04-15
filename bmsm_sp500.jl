using Pkg
Pkg.activate(".")
using BSON
using CSV
using DataFrames
using DifferentialEquations
using Flux
using LinearAlgebra
using Optim
using Random
using StatsBase

include("DGPs/DGPs.jl")
include("DGPs/JD.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("MSM/MSM.jl")
include("MSM/BMSM.jl")


# Outside of the main() function due to world age issues
tcn = BSON.load("models/JD/best_model_ln_bs256_10k.bson")[:best_model];

# Load statistics for standardization
BSON.@load "statistics_new.bson" lnμs lnσs

@views function preprocess(x, y)
    # Restrict based on max. absolute returns being at most 50
    idx = (maximum(abs, x[1, :, 1, :], dims=1) |> vec) .≤ 50
    idx2 = (mean(x[1, :, 2, :], dims=1) |> vec) .≤ 3 # Mean RV under 3
    x = x[:, :, :, idx .& idx2]
    y = y[:, idx .& idx2]
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
    (x .- lnμs) ./ lnσs, y
end

@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
    (x .- lnμs) ./ lnσs
end

function main()

transform_seed = 1204
S = 200 # Simulations to estimate moments
N = 10_000 # MCMC chain length
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 50 # MCMC verbosity
filename = "chain_230415.bson"

# # Tuning parameter (TODO: parameter under logs still unclear)
# δ = use_logs ? 1f-3 : 15f-2
δ = 4f0 #15f-1


@info "Loading data, preparing model..."
# Read SP500 data and transform it to TCN-friendly format
df = CSV.read("sp500.csv", DataFrame);
X₀ = Float32.(Matrix(df[:, [:rets, :rv, :bv]])) |>
    x -> reshape(x, size(x)..., 1) |>
    x -> permutedims(x, (2, 3, 1)) |> 
    tabular2conv |> preprocess



@info "Loading DGP, making data transform..."
# Define DGP and make transform
dgp = JD(1000)
Random.seed!(transform_seed)
dtθ = data_transform(dgp, 100_000);

Flux.testmode!(tcn);
# Compute data moments
θ̂ₓ = tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec

@info "Computing covariance of the proposal..."
# Covariance of the proposal
_, Σp = simmomentscov(tcn, dgp, covreps, θ̂ₓ, dtθ=dtθ, preprocess=preprocess)
ΣpL = cholesky(Σp).L
Σ⁻¹ = inv(10_000Σp)

@info "Running MCMC..."
prop = θ⁺ -> rand(MvNormal(θ⁺, δ * Σp))
# Continuously updated objective
# obj = θ⁺ -> -bmsmobjective(θ̂ₓ, θ⁺, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess)
# Two-step objective
obj = θ⁺ -> -bmsmobjective(θ̂ₓ, θ⁺, Σ⁻¹, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess)
chain = mcmc(θ̂ₓ, Lₙ=obj, proposal=prop, N=N, burnin=burnin, verbosity=verbosity)

# Save chain
BSON.@save filename chain


# # Make MCMC chain and display
# ch = Chains(chain[:, 1:end-1])

# # Display chain
# display(plot(ch))
end

main()