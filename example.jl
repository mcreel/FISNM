using BSON
using CUDA
using DifferentialEquations
using Flux
using LinearAlgebra
using PrettyTables
using Random
using StatsBase



include("DGPs/DGPs.jl")
include("DGPs/GARCH.jl")
include("DGPs/JD.jl")
include("DGPs/Logit.jl")
include("DGPs/MA2.jl")
include("DGPs/SV.jl")

include("NeuralNets/TCN.jl")
include("NeuralNets/utils.jl")
include("NeuralNets/rnn_utils.jl")
include("NeuralNets/tcn_utils.jl")

# Example on a single DGP
dgp = JD

runname = "test"

@info "Training DGP: $dgp"

# Create the folder if they don't exist

modelname="JD"
dirnames = ["models/$modelname", "results/$modelname"]
for d ∈ dirnames
    isdir(d) || mkdir(d)
end

train_tcn(
    DGPFunc=dgp, N=100, modelname=modelname, runname=runname,
    batchsize=10, 
    epochs=10, 
    validation_size=10, 
    validation_frequency=10,
    verbosity=10, 
    dev=cpu
)