using Plots, Random, BSON, Optim
include("LogitLib.jl")
include("neuralnets.jl")

function main()
thisrun = "working"
# General parameters

MCreps = 100 # Number of Monte Carlo samples for each sample size
S = 5000 # number of repetitions of drawing sample
batchsize = 100
epochs = 20 # passes over each batch
N = [100, 200]  # Sample sizes (most useful to incease by 4X)
testseed = 77
trainseed = 78
transformseed = 1204

#=
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
        obj = θ -> -LogitLikelihood(θ, data)
        θhat = Optim.optimize(obj, Y, LBFGS(), # start value is true, to save time 
                            Optim.Options(
                            g_tol = 1e-5,
                            x_tol = 1e-5,
                            f_tol=1e-8); autodiff=:forward).minimizer
        err_mle[:, s, i] = Y - θhat # Compute errors
    end
    println("ML n = $n done.")
end
BSON.@save "err_mle_$thisrun.bson" err_mle N
=#

# NNet estimation (nnet object must be pre-trained!)
# -----------------------------------------------
Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, PriorDraw(100000)) # use a large sample for this

# Iterate over different lengths of observed returns
err_nnet = zeros(5, MCreps, length(N))
Random.seed!(trainseed) # avoid sample contamination for NN training
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes
    nnet = lstm_net(16)
    # Train network
    opt=AdaDelta()
    train_rnn!(nnet, opt, dgp, n, S, dtY, epochs=epochs, batchsize=batchsize)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    X = batch_timeseries(X, n, n) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    nnet(X[1]) # warmup
    Yhat = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X[2:end]])
    err_nnet[:, :, i] = Y - Yhat
    # Save model as BSON
    BSON.@save "models/nnet_(n-$n)_$thisrun.bson" nnet
    println("Neural network, n = $n done.")
end
BSON.@save "err_nnet_$thisrun.bson" err_nnet N MCreps S epochs
end
main()

