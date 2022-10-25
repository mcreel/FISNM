using Pkg
Pkg.activate(".")
using PrettyTables
include("DGPs.jl") # Include all DGPs
include("neuralnets.jl")
BSON.@load "models/MA2/mcx_(n-100).bson" best_model
model = deepcopy(best_model)

function more_training(model) 
    model |> gpu
    n = 100
    dgp=Ma2(N=n)
    
    # Sizes and seeds
    validation_size = 2_000     # The samples used to keep track of the 'best' model when training
    test_size = 5_000           # The samples used to evaluate the final model
    transform_size = 100_000    # The samples used in the data transformation of parameters
    train_seed = 77             # The random seed for training
    test_seed = 78              # The random seed for final model evaluation
    transform_seed = 1204       # The random seed for the prior draw of the data transformation

    # Training parameters
    epochs = 50_000             # The number of epochs used to train the model
    batchsize = 512            # The number of samples used in each batch
    passes_per_batch = 5        # The number of passes of gradient descent on each batch
    validation_frequency = 1   # Every X epochs, we validate the model (and keep track of the best)
    validation_loss = true      # Whether we validate or not
    verbosity = 500             # When to print the current epoch / loss information
    loss = rmse_conv            # The loss to use in training

    dev = gpu                   # The device to run the model on (cpu/gpu)

    # Get the data transform for the DGP parameters
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=dev)

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
    Flux.testmode!(best_model)
    Ŷb = StatsBase.reconstruct(dtY, best_model(X))
    err = cpu(Ŷb - Y)
    amae = mean(mean(abs.(err), dims=2))
    armse = mean(sqrt.(mean(abs2.(err), dims=2)))
    model = cpu(model)
    best_model = cpu(best_model)
    BSON.@save "models/ma2_n100_retrained.bson" model best_model
    pretty_table([n amae armse], header=["n", "amae", "armse"])
end
more_training(model)
