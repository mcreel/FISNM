using Term, PrettyTables
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

    # Training parameters: THIS IS SET SO THAT TRAINING SAMPLES ARE 10⁶, as in Akesson et al Table 1.
    epochs = 2000,             # The number of epochs used to train the model
    batchsize = 500,             # The number of samples used in each batch
    passes_per_batch = 5,        # The number of passes of gradient descent on each batch
    validation_frequency = 20,   # Every X epochs, we validate the model (and keep track of the best)
    validation_loss = true,      # Whether we validate or not
    verbosity = 1000,             # When to print the current epoch / loss information
    loss = rmse_conv,             # The loss to use in training

    # TCN parameters
    dilation = 2,                # WARNING: This has no impact as of now!
    kernel_size = 8,             # Size of the kernel in the temporal blocks of the TCN
    channels = 16,               # Number of channels used in each temporal block
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

    rmaes = zeros(length(N)) # holder for mae relative to prior mae

    for (i, n) ∈ enumerate(N)
        @info "Training TCN for n = $n"
        # Create the DGP
        dgp = DGPFunc(N=n)

        # Create the TCN for the DGP
        model = build_tcn(dgp, dilation=dilation, kernel_size=kernel_size, channels=channels,
            summary_size=summary_size, dev=dev)
        opt = ADAM()

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
        if validation_loss
            Flux.testmode!(best_model)
            Ŷb = StatsBase.reconstruct(dtY, best_model(X))
            err_best[:, :, i] = cpu(Ŷb - Y)
            rmaes[i] = mean(mean(abs.(err_best[:, :, i]),dims=2) ./ [1.0, 0.5])
        end
    end
    
    return rmaes
end
