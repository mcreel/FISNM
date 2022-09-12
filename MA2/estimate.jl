using Pkg
Pkg.activate(".")
using BSON
using Random


include("MA2lib.jl")
include("Testing/MakeTesting.jl")
include("../neuralnets.jl")
include("../TCN.jl")

function main()
    # 1. Save individual BSONs with errors, models, MCreps, datareps and epochs for each model / sample size
    # 2. Save a CSV of bias/RMSE where each row is a model × sample size. 

    thisrun = "experiment"
    path = "MA2"

    # General parameters
    dim_inputs = 1  
    dim_outputs = length(PriorDraw())
    MCreps = 1_000      # Number of Monte Carlo samples for each sample size (testing)
    datareps = 1_000    # Number of repetitions of drawing sample (training)
    batchsize = 32
    
    N = [100 * 2 ^ i for i ∈ 0:5] # Sample sizes
    dev = gpu
    testseed = 77
    trainseed = 78
    transformseed = 1204

    # RNN-specific parameters
    dim_hidden = 16
    rnn_epochs = 200

    # TCN-specific parameters
    dilation = 2
    kernel_size = 8
    channels = 2
    summ_size = 10 # 'Summarizing size'
    # Number of layers changes by sample size to obtain a RFS of n (sample size)
    layers = [ceil(Int, necessary_layers(dilation, kernel_size, n)) for n ∈ N]
    tcn_epochs = 5_000 # Use many more epochs than RNN due to fast training speed

    # Empty vectors to fill for dataframe
    bias = Float64[]
    rmse = Float64[]
    model = String[]
    sample_size = Int64[]
    n_epochs = Int64[]

    # Obtain datatransformation values for output
    Random.seed!(transformseed)
    dtY = fit(ZScoreTransform, dev(PriorDraw(100_000))) # Use a large sample

    # # --------------------------------------------------------------------------------------
    # # MLE Estimation
    # err_mle = zeros(dim_outputs, MCreps, length(N))
    # for i ∈ eachindex(N) # Iterate over different sample sizes
    #     n = N[i]
    #     @info "Compute MLE for n = $n ..."

    #     @info "Maximum likelihood estimation, n = $n done.\n"
    # end

    # --------------------------------------------------------------------------------------
    # TCN Estimation
    err_tcn = zeros(dim_outputs, MCreps, length(N))
    for i ∈ eachindex(N) # Iterate over different sample sizes
        n = N[i]
        @info "Computing TCN for n = $n ..."
        n_layers = layers[i]
        # Create TCN
        opt = ADAM()
        tcn = dev(
            Chain(
                TCN(
                    vcat(dim_inputs, [channels for _ ∈ 1:n_layers], 1), 
                    kernel_size=kernel_size
                ),
                Conv((1, summ_size), 1 => 1, stride=summ_size),
                Flux.flatten, 
                Dense(n ÷ summ_size => dim_outputs)
            )
        )

        # Train network
        Random.seed!(trainseed)
        train_cnn!(tcn, opt, dgp, n, datareps, dtY, batchsize=batchsize, epochs=tcn_epochs,
            dev=dev, validation_loss=false, verbosity=100)
        
        # Test network
        Random.seed!(testseed)
        X, Y = map(dev, dgp(n, MCreps))
        X = tabular2conv(X) # Transform to CNN format
        # Get NNet estimate of parameters for each sample
        Flux.testmode!(tcn) # In case net has dropout / batchnorm
        Ŷ = StatsBase.reconstruct(dtY, tcn(X))
        err_tcn[:, :, i] = cpu(Y - Ŷ)
        # Save model as BSON
        BSON.@save "$path/models/tcn_(n-$n)_$thisrun.bson" tcn
        # Add to lists for CSV results
        # push!(n_epochs, tcn_epochs)
        # push!(sample_size, n)
        # push!(model, "TCN")
        # push!(rmse, )
        # push!(bias, )
        
        @info "Temporal convolutional neural network, n = $n done.\n"
    end
    # Save BSON with results for all sample sizes / models
    BSON.@save "$path/results/err_tcn_$thisrun.bson" err_tcn N MCreps datareps tcn_epochs

    #---------------------------------------------------------------------------------------
    # RNN Estimation
    err_rnn = zeros(dim_outputs, MCreps, length(N))
    for i ∈ eachindex(N) # Iterate over different sample sizes
        n = N[i]
        @info "Computing RNN for n = $n ..."
        # Create RNN
        opt = ADAM()
        rnn = lstm_net(dim_hidden, dim_outputs,dim_inputs, dev)

        # Train network
        Random.seed!(trainseed)
        train_rnn!(rnn, opt, dgp, n, datareps, dtY, batchsize=batchsize, epochs=rnn_epochs, 
        dev=dev, validation_loss=false, verbosity=25)
        
        # Test network
        Random.seed!(testseed)
        X, Y = map(dev, dgp(n, MCreps))
        X = tabular2rnn(X) # Transform to RNN format
        # Get NNet estimate of parameters for each sample
        Flux.testmode!(rnn) # In case net has dropout / batchnorm
        Flux.reset!(rnn)
        rnn(X[1]) # Warmup
        Ŷ = mean(StatsBase.reconstruct(dtY, rnn(x)) for x ∈ X[2:end])
        err_rnn[:, :, i] = cpu(Y - Ŷ)
        # Save model as BSON
        BSON.@save "$path/models/rnn_(n-$n)_$thisrun.bson" rnn
        
        @info "Recurrent neural network, n = $n done.\n"
    end
    # Save BSON with results for all sample sizes / models
    BSON.@save "$path/results/err_rnn_$thisrun.bson" err_rnn N MCreps datareps rnn_epochs


end

main()