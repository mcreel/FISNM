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

include("MSM/BMSM.jl") 


# Whether or not to use the model that was trained on log-transformed data
use_logs = false

# Outside of the main() function due to world age issues
best_model = use_logs ? BSON.load("models/JD/best_model_ln_bs256_10k.bson")[:best_model] :
    BSON.load("models/JD/best_model_bs256_10k.bson")[:best_model];

function main()

transform_seed = 1204
S = 10 # Simulations to estimate moments
N = 1_000 # MCMC chain length
burnin = 100 # Burn-in steps
covreps = 500 # Number of repetitions to estimate the proposal covariance
verbosity = 10 # MCMC verbosity
filename = "chain_mc_2step.bson"

# Tuning parameter (TODO: parameter under logs still unclear)
#δ = use_logs ? 1f-3 : 15f-2
δ = use_logs ? 1f-1 : 30f-1


@info "Loading data, preparing model..."
# Read SP500 data and transform it to TCN-friendly format
df = CSV.read("sp500.csv", DataFrame);
df.rv = min.(df.rv, 5.)
df.bv = min.(df.bv, 5.)
X₀ = Float32.(Matrix(df[:, [:rets, :rv, :bv]])) |>
    x -> reshape(x, size(x)..., 1) |>
    x -> permutedims(x, (2, 3, 1)) |> 
    tabular2conv



Flux.testmode!(best_model);

# Load statistics for standardization
BSON.@load "statistics.bson" μs σs lnμs lnσs

# When using log-transform, we have to transform the data as we receive it
@views function logstandardize(x)
    x[:, :, 2:3, :] = log.(x[:, :, 2:3, :])
    (x .- lnμs) ./ lnσs
end
# Simple standardization if we don't use the log-model
standardize(x) = (x .- μs) ./ σs

tcn(x) = use_logs ? logstandardize(x) |> best_model : standardize(x) |> best_model;

@info "Loading DGP, making data transform..."
# Define DGP and make transform
dgp = JD(1000)
Random.seed!(transform_seed)
dtθ = data_transform(dgp, 100_000);

# Compute data moments
θ̂ₓ = tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec
@info "Computing covariance of the proposal..."
# Covariance of the proposal
_, Σp = simmomentscov(tcn, dgp, covreps, θ̂ₓ, dtθ=dtθ)
Σinv = inv(Σp*Float32(10000.0))
display(θ̂ₓ)
display(Σp)
display(Σinv)
@info "Running MCMC..."
prop = θ⁺ -> rand(MvNormal(θ⁺,δ*Σp))

#obj = θ⁺ -> -bmsmobjective(θ̂ₓ, θ⁺, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp)
obj = θ⁺ -> -bmsmobjective(θ̂ₓ, θ⁺, Σinv, tcn=tcn, S=S, dtθ=dtθ, dgp=dgp)
chain = mcmc(θ̂ₓ, Lₙ=obj, proposal=prop, N=N, burnin=burnin, verbosity=verbosity)

# Save chain
BSON.@save filename chain


# Make MCMC chain and display
ch = Chains(chain[:, 1:end-1])

# Save chain
display(plot(ch))
end

main()
