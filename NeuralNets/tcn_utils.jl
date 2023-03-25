# Build TCN for a given DGP
function build_tcn(d::DGP; dilation=2, kernel_size=8, channels=16, summary_size=10, dev=cpu)
    # Compute TCN dimensions and necessary layers for full RFS
    n_layers = ceil(Int, necessary_layers(dilation, kernel_size, d.N))
    dim_in, dim_out = nfeatures(d), nparams(d)
    dev(
        Chain(
            TCN(
                vcat(dim_in, [channels for _ ∈ 1:n_layers], 1),
                kernel_size=kernel_size, # TODO: BE CAREFUL! DILATION IS ACTUALLY NOT HANDLED!!!
            ),
            Conv((1, summary_size), 1 => 1, stride=summary_size),
            Flux.flatten,
            Dense(d.N ÷ summary_size => d.N ÷ summary_size, hardtanh), # this is a new layer
            Dense(d.N ÷ summary_size => dim_out)
        )
    )
end

# Train a convolutional neural network
function train_cnn!(
    m, opt, dgp, dtY;
    epochs=1000, batchsize=32, passes_per_batch=10, dev=cpu, loss=rmse_conv,
    validation_loss=true, validation_frequency=10, validation_size=2_000, verbosity=1, 
    transform=true
)
    Flux.trainmode!(m) # In case we have dropout / layer normalization
    θ = Flux.params(m) # Extract parameters
    best_model = deepcopy(m) 
    best_loss = Inf

    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = generate(dgp, validation_size, dev=dev)
        # Want to see validation RMSE on original scale => no rescaling
        Xv = tabular2conv(Xv)
        losses = zeros(epochs)
    end

    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = generate(dgp, batchsize, dev=dev) # Generate a new batch
        transform && StatsBase.transform!(dtY, Y)
        # Transform features to format for CNN
        X = tabular2conv(X)

        # ----- Training ---------------------------------------------
        for _ ∈ 1:passes_per_batch
            # Compute loss and gradients
            ∇ = gradient(θ) do
                Ŷ = m(X)
                loss(Ŷ, Y)
            end
            Flux.update!(opt, θ, ∇) # Take gradient descent step
        end
        # Compute validation loss and print status if verbose
        # Do this for the last 100 epochs, too, in case frequency is low
        if validation_loss && (mod(epoch, validation_frequency)==0 || epoch > epochs - 1000)
            Flux.testmode!(m)
            Ŷ = transform ? StatsBase.reconstruct(dtY, m(Xv)) : m(Xv)
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
        nothing
    end
end


# Train a convolutional neural network using pre-generated BSONs
function train_cnn_from_datapath!(
    m, opt, dgp, dtY;
    datapath, statsfile, batchsize=32, passes_per_batch=1, dev=cpu, loss=rmse_conv,
    validation_loss=true, validation_frequency=10, validation_size=2_000, verbosity=1, 
    transform=true, epochs=1_000, use_logs=false
)
    Flux.trainmode!(m) # In case we have dropout / layer normalization
    θ = Flux.params(m) # Extract parameters
    best_model = deepcopy(m) 
    best_loss = Inf

    # files = readdir(datapath)
    BSON.@load statsfile μs σs qminP qmaxP qRV qBV lnμs lnσs qlnBV qlnRV
    function restrict_data(X, Y, use_logs=use_logs)
        if use_logs
            X = cat(X[:, :, 1:1, :], log.(X[:, :, 2:3, :]), dims=3)
            μs = lnμs
            σs = lnσs
            qBV = qlnBV
            qRV = qlnRV
        end
        # Indexes where prices exceed
        csX = cumsum(X[:, :, 1, :], dims=2)
        idxP = (sum(csX .< qminP, dims=[1,2]) + sum(csX .> qmaxP, dims=[1,2])) |> vec .== 0
        X = X[:, :, :, idxP]
        Y = Y[:, idxP]
        # Indexes where RV exceeds
        idxRV = sum(X[:, :, 2, :] .> qRV, dims=[1,2]) |> vec .==0
        X = X[:, :, :, idxRV]
        Y = Y[:, idxRV]
        # Indexes where BV exceeds threshold
        idxBV = sum(X[:, :, 3, :] .> qBV, dims=[1,2]) |> vec .== 0
        (X[:, :, :, idxBV] .- μs) ./ σs, Y[:, idxBV]
    end


    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = generate(dgp, validation_size, dev=cpu)
        # Want to see validation RMSE on original scale => no rescaling
        Xv = tabular2conv(Xv)
        Xv, Yv = map(dev, restrict_data(Xv, Yv))
        losses = zeros(epochs)
    end

    # Compute pre-training loss
    Ŷ = transform ? StatsBase.reconstruct(dtY, m(Xv)) : m(Xv)
    pre_train_loss = loss(Ŷ, Yv)
    @info "Pre training:" pre_train_loss

    for epoch ∈ 1:epochs
        # TODO: why is this needed?
        GC.gc(true)
        CUDA.reclaim()
        # Load matching file (TODO: CHANGE THIS!!)
        BSON.@load joinpath(datapath, "$epoch.bson") X Y
        X, Y = restrict_data(X, Y)
        # Pass to device
        X, Y = map(dev, [X, Y])
        # Standardize Ys
        transform && StatsBase.transform!(dtY, Y)

        # ----- Training ---------------------------------------------
        for passes_per_batch ∈ 1:passes_per_batch
            dl = Flux.DataLoader((X, Y), batchsize=min(batchsize, size(Y, 2)),
                shuffle=true)
            # Compute loss and gradients
            for (xb, yb) ∈ dl
                ∇ = gradient(θ) do 
                    loss(m(xb), yb)
                end
                # Take gradient descent step
                Flux.update!(opt, θ, ∇)
            end
        end

        # Compute validation loss and print status if verbose
        # Do this for the last 100 epochs, too, in case frequency is low
        if validation_loss && (epoch % validation_frequency == 0)
            Flux.testmode!(m)
            Ŷ = transform ? StatsBase.reconstruct(dtY, m(Xv)) : m(Xv)
            current_loss = loss(Ŷ, Yv)
            if current_loss < best_loss
                best_loss = current_loss
                best_model = deepcopy(cpu(m))
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
        nothing
    end
end


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
    batchsize = 1024,             # The number of samples used in each batch
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
    on_gpu = dev == gpu

    # We need to create one instance of dgp before training (N doesn't matter in this case)
    dgp = DGPFunc(N=N[1])

    # Get the data transform for the DGP parameters
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=dev)

    priordraw(dgp, 1)

    # Create arrays for error tracking
    err = zeros(nparams(dgp), test_size, length(N))
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
        on_gpu && CUDA.reclaim() # GC and clear out cache
        Random.seed!(train_seed)
        _, best_model = train_cnn!(
            model, opt, dgp, dtY, epochs=1, batchsize=16, dev=dev, 
            passes_per_batch=1, validation_size=16,
            validation_frequency = 2, verbosity=verbosity, loss=loss
        )
        GC.gc(true)
        on_gpu && CUDA.reclaim() # GC and clear out cache

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


# Add bias correction at the end of a network
function bias_corrected_tcn(tcn, X, Y)
    Flux.testmode!(tcn)
    bias = mean(tcn(X) - Y, dims=2)
    Chain(deepcopy(tcn), x -> x .- bias)
end

# Computes the receptive field size for a specified dilation, kernel size, and number of layers
receptive_field_size(dilation::Int, kernel_size::Int, layers::Int) = 
    1 + (kernel_size - 1) * (dilation ^ layers - 1) / (dilation - 1)

# Minimum number of layers necessary to achieve a specified receptive field size
# (take ceil(Int, necessary_layers(...)) for final number of layers)
necessary_layers(dilation::Int, kernel_size::Int, receptive_field::Int) =
    log(dilation, (receptive_field - 1) * (dilation - 1) / (kernel_size - 1)) + 1
