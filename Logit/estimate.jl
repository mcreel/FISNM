using Plots, Random, BSON, Optim
include("LogitLib.jl")
include("neuralnets.jl")

function main()
thisrun = "working"
# General parameters

MCreps = 10 # Number of Monte Carlo samples for each sample size
TrainSize = 32 # samples in each epoch
epochs = 10
N = [100, 200]  # Sample sizes (most useful to incease by 4X)
testseed = 77
trainseed = 78
transformseed = 1204


# Estimation by ML
# ------------------------------------------------------------------------------
err_mle = zeros(5, MCreps, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    Random.seed!(testseed) # samples for ML and for NN use same seed
    # Fit Logit models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        Y = PriorDraw()
        data = SimulateLogit(Y,n)
        obj = θ -> -mean(LogitLikelihood(θ, data))
        θhat = Optim.optimize(obj, zeros(5), LBFGS(), 
                            Optim.Options(
                            g_tol = 1e-5,
                            x_tol = 1e-5,
                            f_tol=1e-8); autodiff=:forward).minimizer

        err_mle[:, s, i] = Y - θhat # Compute errors
    end
    println("ML n = $n done.")
end
BSON.@save "err_mle_$thisrun.bson" err_mle N

# NNet estimation (nnet object must be pre-trained!)
# ------------------------------------------------------------------------------
Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, PriorDraw(100000)) # use a large sample for this

# Iterate over different lengths of observed returns
err_nnet = zeros(5, MCreps, length(N))
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
=#
end
main()

