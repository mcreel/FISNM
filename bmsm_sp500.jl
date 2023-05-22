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
using MCMCChains
using StatsPlots

include("DGPs/DGPs.jl")
include("DGPs/JD.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("MSM/MSM.jl")
include("MSM/BMSM.jl")


# Outside of the main() function due to world age issues
tcn = BSON.load("models/JD/best_model_ln_bs1024_30-20-06.bson")[:best_model];

# Load statistics for standardization
BSON.@load "statistics_30-20-06.bson" μs σs

@views function preprocess(x) # For X only, don't discard extreme values
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
     (x .- μs) ./ σs
end

# using infile=nothing to start
function main(N,  outfile, infile)

transform_seed = 1204
S = 100 # Simulations to estimate moments
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 10 # MCMC verbosity

@info "Loading data, preparing model..."
# Read SP500 data and transform it to TCN-friendly format
df = CSV.read("sp500.csv", DataFrame);
display(describe(df))
X₀ = Float32.(Matrix(df[:, [:rets, :rv, :bv]])) |>
    x -> reshape(x, size(x)..., 1) |>
    x -> permutedims(x, (2, 3, 1)) |> 
    tabular2conv |> preprocess

@info "Loading DGP, making data transform..."
# Define DGP and make transform
dgp = JD(1000)
Random.seed!(transform_seed)
pd = priordraw(dgp, 100_000)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)

Flux.testmode!(tcn);
Random.seed!(rand(1:Int64(1e10)))
# Compute data moments
θ̂ₓ = Float64.((tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec))
display(θ̂ₓ)

if infile !== nothing
    BSON.@load infile chain Σp δ 
    display(Σp)
    start = chain[end,:][1:end-1]
    burnin = 0
    # adjust proposal depending on acceptance rate
    ac = mean(chain[:,9])
    @info "acceptance rate of input chain: " ac
    mean(chain[:,9]) > 0.3  ? δ *= 1.5 : nothing
    mean(chain[:,9]) < 0.2  ? δ *= 0.5 : nothing
    @info "current δ: " δ 
else  
    @info "Computing covariance of the proposal..."
    # Covariance of the proposal
    _, Σp = simmomentscov(tcn, dgp, covreps, θ̂ₓ, dtθ=dtθ, preprocess=preprocess)
    start = θ̂ₓ
    δ = 1e-1
end
@info "Running MCMC..."
prop = θ⁺ -> rand(MvNormal(θ⁺, δ * Σp))
# Continuously updated objective
Random.seed!(rand(1:Int64(1e10)))
# objective is quasi-log likelihood (to be maximized)
obj = θ⁺ -> bmsmobjective(θ̂ₓ, θ⁺, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp, preprocess=preprocess)
chain = mcmc(start, Lₙ=obj, proposal=prop, N=N, burnin=burnin, verbosity=verbosity)
@info "acceptance rate: " mean(chain[:,9])
# Save chain
BSON.@save outfile chain Σp δ


# # Make MCMC chain and display
ch = Chains(chain)
display(ch)
display(plot(ch))
end

