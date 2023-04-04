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
θs = msm(dgp, S=10, dtθ=dtθ, model=tcn, M=10, verbosity=2, show_trace=true)

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
N = 1000
dgp = JD(N)
dtθ = data_transform(dgp, 100_000)
# Load best TCN
best_model = BSON.load("models/JD/best_model_bs256_10k.bson")[:best_model];
Flux.testmode!(best_model);
BSON.@load "statistics.bson" μs σs qminP qmaxP qRV qBV lnμs lnσs qlnBV qlnRV
tcn = xs -> (xs .- μs) ./ σs |> best_model; # Add standardization in front

# Run MSM and return predictions (matrix size M × |θ|) with each row [θ₀ θtcn θmsm]
θs = msm(dgp, S=10, dtθ=dtθ, model=tcn, M=20, verbosity=1, show_trace=true)