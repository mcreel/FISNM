using Pkg
Pkg.activate(".")
using Plots, Random, BSON
cd("GARCH")
include("GarchLib.jl")
include("neuralnets.jl")
include("samin.jl")



function main()
thisrun = "0815"
# General parameters

MCreps = 1000 # Number of Monte Carlo samples for each sample size
TrainSize = 2048 # samples in each epoch
epochs = 200
N = [100, 200, 400, 800, 1600, 3200]  # Sample sizes (most useful to incease by 4X)
testseed = 77
trainseed = 78
transformseed = 1204
dev = gpu

# Estimation by ML
# -----------------------------------------------
err_mle = zeros(3, MCreps, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    Random.seed!(testseed) # samples for ML and for NN use same seed
    X, Y = dgp(n, MCreps) # Generate the data according to DGP
    # Fit GARCH models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        θstart = Float64.([0.5, 0.5, 0.5])
        obj = θ -> -mean(garch11(θ, Float64.(X[:,s])))
        lb = [0.0001, 0.0, 0.0]
        ub = [1.0, 0.99, 1.0]
        θhat, junk, junk, junk= samin(obj, θstart, lb, ub, verbosity=0)
        err_mle[:, s, i] = Y[:, s] .- θhat # Compute errors
    end
    println("ML n = $n done.")
end
BSON.@save "err_mle_$thisrun.bson" err_mle Y N


# -----------------------------------------------
# NNet estimation
Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, dev(PriorDraw(100_000))) # use a large sample for this

# Iterate over different lengths of observed returns
err_nnet = zeros(3, MCreps, length(N))
for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes
    nnet = lstm_net(32, dev)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn!(nnet, AdaDelta(), dgp, n, TrainSize, dtY, epochs=epochs, dev=dev,
        validation_loss=false, verbosity=25)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    nnet(X[1]) # warmup
    Yhat = mean(StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X[2:end])
    err_nnet[:, :, i] = cpu(Y - Yhat)
    # Save model as BSON
    BSON.@save "models/nnet_(n-$n)_$thisrun.bson" nnet
    println("Neural network, n = $n done.")
end
BSON.@save "err_nnet_$thisrun.bson" err_nnet Y N MCreps TrainSize epochs

# -----------------------------------------------
# Bidirectional NNet estimation

# Iterate over different lengths of observed returns
err_bnnet = zeros(3, MCreps, length(N))
for i = 1:size(N,1)
    n = N[i]
    # Create bidirectional network with 32 hidden nodes
    bnnet = bilstm_net(32, dev)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn!(bnnet, AdaDelta(), dgp, n, TrainSize, dtY, epochs=epochs, dev=dev,
        validation_loss=false, bidirectional=true, verbosity=25)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(bnnet) # In case nnet has dropout / batchnorm
    Flux.reset!(bnnet)
     # warmup
    Yhat = StatsBase.reconstruct(dtY, bnnet(X))
    err_bnnet[:, :, i] = cpu(Y - Yhat)
    # Save model as BSON
    BSON.@save "models/bnnet_(n-$n)_$thisrun.bson" bnnet
    println("Bidirectional neural network, n = $n done.")
end
BSON.@save "err_bnnet_$thisrun.bson" err_bnnet Y N MCreps TrainSize epochs

# -----------------------------------------------
# Temporal Convolutional NNet estimation

# Iterate over different lengths of observed returns
err_tcn = zeros(3, MCreps, length(N))
for i = 1:size(N,1)
    n = N[i]
    # Create bidirectional network with 32 hidden nodes
    tcn = dev(
        Chain(
            TCN([1, 2, 2, 2, 2, 1], kernel_size=12, dropout_rate=0.0), 
            Conv((1, 10), 1 => 1, stride = 10),
            Flux.flatten,
            Dense(n ÷ 10 => 3)
        )
    )
    # Train network (use 40x the epochs for TCN as it is extremely fast to train)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_cnn!(tcn, AdaDelta(), dgp, n, TrainSize, dtY, epochs=10, dev=dev, 
        validation_loss=false, verbosity=100)
    
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
BSON.@save "err_tcn_$thisrun.bson" err_tcn Y N MCreps TrainSize epochs

end
main()

