using Pkg
Pkg.activate(".")
using BSON
using CUDA
using Dates
using DifferentialEquations
using Flux
using LinearAlgebra
using Plots
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
    dev = gpu
    transform_size = 100_000
    batchsize = 1_024
    passes_per_batch = 3
    validation_frequency = 50
    validation_size = 15_000
    verbosity = 10
    loss = rmse_conv # Flux.Losses.huber_loss
    datapath = "data"
    validation_loss = true
    validation_seed = 72
    # Iterate over all files in the data directory
    transform = true

    model_name = "ln_bs$(batchsize)_$(now())"

    dgp = JD(N=1000)
    
    specs = (name = "30-20", max_ret = 30, max_rv = 20, max_bv = 20)

    # Get mean and standard deviation from statistics file
    BSON.@load "statistics_$(specs.name).bson" μs σs
    function restrict_data(x, y, max_ret=specs.max_ret, max_rv=specs.max_rv, max_bv=specs.max_bv)
        # Restrict based on max. absolute returns being at most max_ret
        idx = (maximum(abs, x[1, :, 1, :], dims=1) |> vec) .≤ max_ret
        idx2 = (mean(x[1, :, 2, :], dims=1) |> vec) .≤ max_rv # Mean RV under threshold
        idx3 = (mean(x[1, :, 3, :], dims=1) |> vec) .≤ max_bv # Mean BV under threshold
        x = x[:, :, :, idx .& idx2 .& idx3]
        y = y[:, idx .& idx2 .& idx3]
        x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
        # # Restrict λ₀ and τ to be non-negative
        y[6, :] .= max.(y[6, :], 0)
        y[8, :] .= max.(y[8, :], 0)
        (x .- μs) ./ σs, y
    end

    model = build_tcn(dgp, dilation=2, kernel_size=32, channels=32,
        summary_size=10, dev=dev, dropout_rate=0.2, n_layers=6);
    opt = ADAMW()

    @info "Generating data transform"
    # Get  the data transform for the DGP (fit on a priordraw where we cap some values at zero)
    pd = priordraw(dgp, transform_size)
    pd[6, :] .= max.(pd[6, :], 0)
    pd[8, :] .= max.(pd[8, :], 0)
    dtY = fit(ZScoreTransform, dev(pd))
    # dtY = data_transform(dgp, transform_size, dev=dev)
    
    GC.gc(true)
    CUDA.reclaim()
    @info "Training model ...."
    losses, best_model = train_cnn_from_datapath!(model, opt, dgp, dtY;
        restrict_data=restrict_data, datapath=datapath, batchsize=batchsize, 
        dev=dev, passes_per_batch=passes_per_batch, 
        validation_size=validation_size, 
        validation_frequency=validation_frequency, verbosity=verbosity, 
        loss=loss, transform=transform, validation_seed=validation_seed,
        validation_loss=validation_loss)

    best_model = cpu(best_model)
    BSON.@save "best_model_$model_name.bson" best_model
    plt = plot(losses, lab="Validation loss")
    savefig(plt, "losses_$model_name.png")
end

main()
