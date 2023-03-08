using Pkg
Pkg.activate(".")
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

function main()
    dir = "data"
    files = readdir(dir)
    i = isempty(files) ? 0 : maximum(map(
        x -> parse(Int, split(x, ".")[1]), files))
    S = 20_000 # Number of simulation files
    batchsize = 1_024 # Batches per file
    verbosity = 100

    dgp = JD(N=1_000)
    
    @info "Generating samples"
    for s ∈ 1:S
        i += 1
        X, Y = generate(dgp, batchsize)
        X = tabular2conv(X)
        BSON.@save "$dir/$i.bson" X Y
        s % verbosity == 0 && @info "Finished sample $s"
    end
end

main()