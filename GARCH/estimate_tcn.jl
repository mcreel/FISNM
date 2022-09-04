using Pkg
Pkg.activate(".")
using BSON
using Random
cd("GARCH")
include("GarchLib.jl")
include("neuralnets.jl")
include("TCN.jl")
include("samin.jl")

function main()

    # General parameters
    thisrun = "working"
    MCreps = 1000       # Number of Monte Carlo samples for each sample size
    TrainSize = 2048    # Samples in each epoch
    N = [100, 200, 400, 800, 1600, 3200]  # Sample sizes (most useful to increase by 4X)
    testseed = 77
    trainseed = 78
    transformseed = 1204
    dev = gpu
    # TCN parameters
    dilation = 2
    kernel_size = 8
    channels = 2
    layers = [ceil(Int, necessary_layers(dilation, kernel_size, n)) for n ∈ N]
    epochs = 5_000

    Random.seed!(transformseed) # avoid sample contamination for NN training
    dtY = fit(ZScoreTransform, dev(PriorDraw(100_000))) # use a large sample for this

    @info "Starting training iterations..."
    # Iterate over different lengths of observed returns
    err_tcn = zeros(3, MCreps, length(N))
    for i = 1:size(N, 1)
        n = N[i]
        @info "Training iteration for n = $n"
        n_layers = layers[i]
        
        # Create TCN
        tcn = dev(
            Chain(
                TCN(vcat(1, [channels for _ ∈ 1:n_layers], 1), kernel_size=kernel_size, 
                    dropout_rate=0.0, dilation_factor=dilation), 
                Conv((1, 10), 1 => 1, stride = 10),
                Flux.flatten,
                Dense(n ÷ 10 => 3)
            )
        )
        # Train network 
        Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
        train_cnn!(tcn, ADAM(), dgp, n, TrainSize, dtY, epochs=epochs, dev=dev, 
            validation_loss=false, verbosity=250)
        
        # Compute network error on a new batch
        Random.seed!(testseed)
        X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
        X = tabular2conv(X) # Transform to rnn format
        # Get NNet estimate of parameters for each sample
        Flux.testmode!(tcn) # In case nnet has dropout / batchnorm
        # warmup
        Yhat = StatsBase.reconstruct(dtY, tcn(X))
        err_tcn[:, :, i] = cpu(Y - Yhat)
        # Save model as BSON
        BSON.@save "models/tcn_(n-$n)_$thisrun.bson" tcn
        println("Temporal convolutional neural network, n = $n done.")
    end
    BSON.@save "err_tcn_$thisrun.bson" err_tcn N MCreps TrainSize epochs

end
main()