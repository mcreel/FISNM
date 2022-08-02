using Plots, Random, BSON
include("GarchLib.jl")
include("neuralnets.jl")
include("samin.jl")

function main()
thisrun = "working"
# General parameters

MCreps = 1000 # Number of Monte Carlo samples for each sample size
TrainSize = 2048 # samples in each epoch
epochs = 200
N = [100, 200, 400, 800, 1600, 3200]  # Sample sizes (most useful to incease by 4X)
testseed = 77
trainseed = 78
transformseed = 1204


# Estimation by ML
# ------------------------------------------------------------------------------
err_mle = zeros(3, MCreps, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    Random.seed!(testseed) # samples for ML and for NN use same seed
    X, Y = dgp(n, MCreps) # Generate the data according to DGP
    # Fit GARCH models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        θstart = Float64.([0.1, 0.5, 0.5])
        obj = θ -> -mean(garch11(θ, Float64.(X[:,s])))
        lb = [0.0001, 0.0, 0.0]
        ub = [1.0, 0.99, 1.0]
        θhat, junk, junk, junk= samin(obj, θstart, lb, ub, verbosity=0)
        err_mle[:, s, i] = θhat  .- Y[:, s] # Compute errors
    end
    println("ML n = $n done.")
end
BSON.@save "err_mle_$thisrun.bson" err_mle N


# NNet estimation (nnet object must be pre-trained!)
# ------------------------------------------------------------------------------
# We standardize the outputs for the MSE to not be overly influenced by the
# parameters which are larger in absolute value than others. Because of this,
# we need to fit a data transformation on the labels. Thus, we use the dgp()
# function to generate a batch which we ONLY use to fit this transformation,
# this way we don't have the problem of fitting a transformation on our 
# test set.
Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, PriorDraw(100000)) # use a large sample for this

# Iterate over different lengths of observed returns
err_nnet = zeros(3, MCreps, length(N))
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes and 20% dropout rate
    nnet = lstm_net(32, .2)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn!(nnet, ADAM(), dgp, n, TrainSize, dtY, epochs=epochs)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    nnet(X[1]) # warmup
    Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X[2:end]])
    err_nnet[:, :, i] = Ŷ - Y
    # Save model as BSON
    BSON.@save "models/nnet_(n-$n)_$thisrun.bson" nnet
    println("Neural network, n = $n done.")
end
BSON.@save "err_nnet_$thisrun.bson" err_nnet N MCreps TrainSize epochs

end
main()

