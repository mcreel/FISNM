function build_lstm(
    d::DGP; hidden_nodes=32, hidden_layers=2, activation=tanh, dev=cpu
)
    dim_in, dim_out = nfeatures(d), nparams(d)
    dev(
        Chain(
            Dense(dim_in => hidden_nodes, activation),
            [LSTM(hidden_nodes => hidden_nodes) for _ ∈ 1:hidden_layers]...,
            Dense(hidden_nodes => dim_out)
        )
    )
end

# Trains a recurrent neural network
function train_rnn!(
    m, opt, dgp, dtY; 
    epochs=1000, batchsize=32, passes_per_batch=10, dev=cpu, loss=mse_full,
    validation_loss=true, validation_frequency=10, validation_size=2000, verbosity=1, 
    transform=true
)
    Flux.trainmode!(m) # In case we have dropout / batchnorm
    θ = Flux.params(m) # Extract parameters
    best_model = deepcopy(m)
    best_loss = Inf

    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = generate(dgp, validation_size, dev=dev)
        Xv = tabular2rnn(Xv)
        losses = zeros(epochs)
    end
    
    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = generate(dgp, batchsize, dev=dev) # Generate a new batch
        transform && StatsBase.transform!(dtY, Y)
        # Transform features to format for RNN
        X = tabular2rnn(X)
        # ----- Training ---------------------------------------------
        for _ ∈ 1:passes_per_batch
            Flux.reset!(m)
            # Compute loss and gradients
            ∇ = gradient(θ) do
                m(X[1]) # don't use first, to warm up state
                Ŷ = [m(x) for x ∈ X[2:end]]
                loss(Ŷ, Y)
            end
            Flux.update!(opt, θ, ∇) # Take gradient descent step
        end

        # Compute validation loss and print status if verbose
        if validation_loss && mod(epoch, validation_frequency)==0
            Flux.reset!(m)
            Flux.testmode!(m)
            m(Xv[1]) # Warm up state on first observation
            Ŷ = [StatsBase.reconstruct(dtY, m(x)) for x ∈ Xv[2:end]]
            current_loss = loss(Ŷ, Yv)
            if current_loss < best_loss
                best_loss = current_loss
                best_model = deepcopy(m)
            end
            losses[epoch] = current_loss
            Flux.trainmode!(m)
            epoch % verbosity == 0 && @info "$epoch / $epochs" best_loss current_loss
        else
            epoch % verbosity == 0 && @info "$epoch / $epochs"
        end
    end
    # Return losses if tracked
    if validation_loss
        losses, best_model
    else
        nothing, nothing
    end
end

function train_lstm(; 
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
    loss = rmse_full,            # The loss to use in training

    # LSTM parameters
    hidden_nodes = 32,          # Number of hidden nodes in the LSTM layers
    hidden_layers = 2,          # Number of hidden (LSTM) layers
    activation = tanh,          # Activation function from the first Dense input layer to the first hidden layer
    dev = gpu                   # The device to run the model on (cpu/gpu)
)
    on_gpu = dev == gpu
    # We need to create one instance of dgp before training (N doesn't matter in this case)
    dgp = DGPFunc(N=N[1])

    # Get the data transform for the DGP parameters
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=dev)

    # Create arrays for error tracking
    err = zeros(nparams(dgp), test_size, length(N))
    err_best = similar(err)

    for (i, n) ∈ enumerate(N)
        @info "Training LSTM for n = $n"
        # Create the DGP
        dgp = DGPFunc(N=n)
        # Create the TCN for the DGP
        model = build_lstm(dgp, hidden_nodes=hidden_nodes, hidden_layers=hidden_layers,
            activation=activation, dev=dev)
        opt = ADAMW()

        # warm up the net with small version
        GC.gc(true)
        on_gpu && CUDA.reclaim() # GC and clear out cache
        Random.seed!(train_seed)
        _, best_model = train_rnn!(
            model, opt, dgp, dtY, epochs=1, batchsize=16, dev=dev, 
            passes_per_batch=1, validation_size=16,
            validation_frequency = 2, verbosity=verbosity, loss=loss
        )
        GC.gc(true)
        on_gpu && CUDA.reclaim() # GC and clear out cache

        # Train the network
        Random.seed!(train_seed)
        _, best_model = train_rnn!(
            model, opt, dgp, dtY, epochs=epochs, batchsize=batchsize, dev=dev, 
            passes_per_batch=passes_per_batch, validation_size=validation_size,
            validation_frequency = validation_frequency, verbosity=verbosity, loss=loss
        )

        # Test the network
        Random.seed!(test_seed)
        X, Y = generate(dgp, test_size, dev=dev)
        X = tabular2rnn(X)
        # Get net estimates
        Flux.testmode!(model)
        Flux.reset!(model)
        model(X[1]) # Warm up
        Ŷ = mean(StatsBase.reconstruct(dtY, model(x)) for x ∈ X[2:end])
        err[:, :, i] = cpu(Ŷ - Y)
        # rmse = sqrt.(mean(abs2.(err[:, :, i]), dims=2))
        model = cpu(model)
        if validation_loss
            Flux.testmode!(best_model)
            Flux.reset!(best_model)
            best_model(X[1])
            Ŷb = mean(StatsBase.reconstruct(dtY, best_model(x)) for x ∈ X[2:end])
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
        @info "LSTM (n = $n) done."
    end
    # Save BSON with results for all sample sizes / models
    if validation_loss
        BSON.@save "results/$modelname/err_$runname.bson" err err_best
    else    
         BSON.@save "results/$modelname/err_$runname.bson" err
    end    

end