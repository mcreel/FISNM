include("DGPs.jl") # Include all DGPs
include("neuralnets.jl")

function train_tcn(;
    DGPFunc, N, modelname, runname,
    )
validation_size = 2_000     # The samples used to keep track of the 'best' model when training
test_size = 5_000           # The samples used to evaluate the final model
transform_size = 100_000    # The samples used in the data transformation of parameters

train_seed = 77             # The random seed for training
test_seed = 78              # The random seed for final model evaluation
transform_seed = 1204       # The random seed for the prior draw of the data transformation

# Training parameters
epochs = 10_000
batchsize = 256
passes_per_batch = 2
validation_frequency = 10
validation_loss = true
verbosity = 500
loss = rmse_conv
dev = gpu                   # The device to run the model on (cpu/gpu)
# We need to create one instance of dgp before training (N doesn't matter in this case)
dgp = DGPFunc(N=N[1])

# Get the data transform for the DGP parameters
Random.seed!(transform_seed)
dtY = data_transform(dgp, transform_size, dev=dev)

# Create arrays for error tracking
err_tcn = zeros(n_params(dgp), test_size, length(N))
err_tcn_best = similar(err_tcn)

for (i, n) ∈ enumerate(N)
    @info "Training TCN for n = $n"
    # Create the DGP
    dgp = DGPFunc(N=n)

    # Create the TCN for the DGP
    tcn = build_tcn(dgp, dev=dev)
    opt = ADAM()

    # Train the network
    Random.seed!(train_seed)
    _, best_tcn = train_cnn!(
        tcn, opt, dgp, dtY, epochs=epochs, batchsize=batchsize, dev=dev, 
        passes_per_batch=passes_per_batch, validation_size=validation_size,
        validation_frequency = validation_frequency, verbosity=verbosity, loss=loss
    )

    # Test the network
    Random.seed!(test_seed)
    X, Y = generate(dgp, test_size, dev=dev)
    X = tabular2conv(X)
    # Get net estimates
    Flux.testmode!(tcn)
    Ŷ = StatsBase.reconstruct(dtY, tcn(X))
    err_tcn[:, :, i] = cpu(Ŷ - Y)
    if validation_loss
        Flux.testmode!(best_tcn)
        Ŷb = StatsBase.reconstruct(dtY, best_tcn(X))
        err_tcn_best[:, :, i] = cpu(Ŷb - Y)
    end
    # Save model as BSON
    BSON.@save "models/$modelname/tcn_(n-$n)_$runname.bson" tcn best_tcn

    @info "Temporal convolutional neural network, n = $(dgp.N) done.\n"
end

# Save BSON with results for all sample sizes / models
BSON.@save "results/$modelname/err_tcn_$runname.bson" err_tcn err_tcn_best
end