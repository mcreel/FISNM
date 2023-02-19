using PrettyTables
include("DGPs.jl") # Include all DGPs
include("neuralnets.jl")

function train_tcn(; 
    DGPFunc, N, modelname, runname,
    # Sizes and seeds
    validation_size = 5_000,     # The samples used to keep track of the 'best' model when training
    test_size = 5_000,           # The samples used to evaluate the final model
    transform_size = 100_000,    # The samples used in the data transformation of parameters
    train_seed = 77,             # The random seed for training
    test_seed = 78,              # The random seed for final model evaluation
    transform_seed = 1204,       # The random seed for the prior draw of the data transformation

    # Training parameters
    epochs = 200_000,             # The number of epochs used to train the model
    batchsize = 2048,             # The number of samples used in each batch
    passes_per_batch = 1,        # The number of passes of gradient descent on each batch
    validation_frequency = 5000,   # Every X epochs, we validate the model (and keep track of the best)
    validation_loss = true,      # Whether we validate or not
    verbosity = 500,             # When to print the current epoch / loss information
    loss = rmse_conv,            # The loss to use in training

    # TCN parameters
    dilation = 2,                # WARNING: This has no impact as of now!
    kernel_size = 32,             # Size of the kernel in the temporal blocks of the TCN
    channels = 32,               # Number of channels used in each temporal block
    summary_size = 10,           # Kernel size of the final pass before feedforward NN
    dev = gpu                   # The device to run the model on (cpu/gpu)
)


    # We need to create one instance of dgp before training (N doesn't matter in this case)
    dgp = DGPFunc(N=N[1])

    # Get the data transform for the DGP parameters
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=dev)

    # Create arrays for error tracking
    err = zeros(n_params(dgp), test_size, length(N))
    err_best = similar(err)

    for (i, n) ∈ enumerate(N)
        @info "Training TCN for n = $n"
        # Create the DGP
        dgp = DGPFunc(N=n)
        # Create the TCN for the DGP
        model = build_tcn(dgp, dilation=dilation, kernel_size=kernel_size, channels=channels,
            summary_size=summary_size, dev=dev)
        opt = ADAMW()

        # warm up the net with small version
        GC.gc(true)
        CUDA.reclaim() # GC and clear out cache
        Random.seed!(train_seed)
        _, best_model = train_cnn!(
            model, opt, dgp, dtY, epochs=1, batchsize=16, dev=dev, 
            passes_per_batch=1, validation_size=16,
            validation_frequency = 2, verbosity=verbosity, loss=loss
        )
        GC.gc(true)
        CUDA.reclaim() # GC and clear out cache

        # Train the network
        Random.seed!(train_seed)
        _, best_model = train_cnn!(
            model, opt, dgp, dtY, epochs=epochs, batchsize=batchsize, dev=dev, 
            passes_per_batch=passes_per_batch, validation_size=validation_size,
            validation_frequency = validation_frequency, verbosity=verbosity, loss=loss
        )

        # Test the network
        Random.seed!(test_seed)
        X, Y = generate(dgp, test_size, dev=dev)
        X = tabular2conv(X)
        # Get net estimates
        Flux.testmode!(model)
        Ŷ = StatsBase.reconstruct(dtY, model(X))
        err[:, :, i] = cpu(Ŷ - Y)
        # rmse = sqrt.(mean(abs2.(err[:, :, i]), dims=2))
        model = cpu(model)
        if validation_loss
            Flux.testmode!(best_model)
            Ŷb = StatsBase.reconstruct(dtY, best_model(X))
            # Ŷb = best_model(X) # TODO: This shouldn't be here?
            err_best[:, :, i] = cpu(Ŷb - Y)
            amae = mean(mean(abs.(err_best[:, :, i]), dims=2))
            armse = mean(sqrt.(mean(abs2.(err_best[:, :, i]), dims=2)))
            best_model = cpu(best_model)
            BSON.@save "models/$modelname/$(runname)_(n-$n).bson" model best_model
            pretty_table([n amae armse], header=["n", "amae", "armse"])
        else
            BSON.@save "models/$modelname/$(runname)_(n-$n).bson" model
        end
        @info "TCN (n = $n) done."
    end
    # Save BSON with results for all sample sizes / models
    if validation_loss
        BSON.@save "results/$modelname/err_$runname.bson" err err_best
    else    
         BSON.@save "results/$modelname/err_$runname.bson" err
    end    
end
