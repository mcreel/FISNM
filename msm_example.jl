using BSON
using DifferentialEquations
using Flux
using LinearAlgebra
using Optim
using Random
using StatsBase

include("DGPs/DGPs.jl")
include("DGPs/MA2.jl")
include("DGPs/GARCH.jl")
include("DGPs/JD.jl")

include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

include("MSM/MSM.jl")

# ----- Example for MA2 DGP -----
N = 100
dgp = MA2(N)
dtθ = data_transform(dgp, 100_000)
# Load best TCN
tcn = BSON.load("models/MA2/23-02-11_(n-$N).bson")[:best_model];
Flux.testmode!(tcn);

# Run MSM and return predictions (matrix size M × |θ|) with each row [θ₀ θtcn θmsm]
θs = msm(dgp, S=10, dtθ=dtθ, model=tcn, M=10, verbosity=2, show_trace=true,
    preprocess=nothing)

# Final armse
armse_tcn = sqrt(mean(abs2, (θs[:, 1:2] .- θs[:, 3:4])))
armse_msm = sqrt(mean(abs2, (θs[:, 1:2] .- θs[:, 5:6])))

# ----- Example for GARCH DGP -----
N = 100
dgp = GARCH(N)
dtθ = data_transform(dgp, 100_000)
# Load best TCN
tcn = BSON.load("models/GARCH/23-02-11_(n-$N).bson")[:best_model];
Flux.testmode!(tcn);

# Run MSM and return predictions (matrix size M × |θ|) with each row [θ₀ θtcn θmsm]
θs = msm(dgp, S=10, dtθ=dtθ, model=tcn, M=10, verbosity=2)

# Final armse
armse_tcn = sqrt(mean(abs2, (θs[:, 1:3] .- θs[:, 4:6])))
armse_msm = sqrt(mean(abs2, (θs[:, 1:3] .- θs[:, 7:9])))

# ----- Example for JD DGP -----
BSON.@load "statistics_new.bson" μs σs lnμs lnσs
function restrict_data(x, y)
    # Restrict based on max. absolute returns being at most 50
    idx = (maximum(abs, x[1, :, 1, :], dims=1) |> vec) .≤ 50
    idx2 = (mean(x[1, :, 2, :], dims=1) |> vec) .≤ 3 # Mean RV under 3
    x = x[:, :, :, idx .& idx2]
    y = y[:, idx .& idx2]
    x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
    (x .- lnμs) ./ lnσs, y
end


N = 1000
dgp = JD(N)
dtθ = data_transform(dgp, 100_000)
# Load best TCN
best_model = BSON.load("best_model_ln_bs256_new2.bson")[:best_model];
Flux.testmode!(best_model);
BSON.@load "statistics.bson" μs σs qminP qmaxP qRV qBV lnμs lnσs qlnBV qlnRV
# tcn = xs -> (xs .- μs) ./ σs |> best_model; # Add standardization in front

# Run MSM and return predictions (matrix size M × |θ|) with each row [θ₀ θtcn θmsm]
θs = msm(dgp, S=10, dtθ=dtθ, model=best_model, M=20, verbosity=1, show_trace=true,
    preprocess=restrict_data)