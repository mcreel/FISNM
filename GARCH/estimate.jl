using Plots, Random, BSON
include("GarchLib.jl")
include("neuralnets.jl")
include("samin.jl")

function main()
thisrun = "experiment"
# General parameters
k = 3 # number of labels
g = 1 # number of features
n_hidden = 16
MCreps = 1000 # Number of Monte Carlo samples for each sample size
datareps = 1000 # 3000 # number of repetitions of drawing sample
batchsize = 50
epochs = 10 # passes over each batch
N = [100, 200, 400, 800]  # Sample sizes (most useful to incease by 4X)
testseed = 77
trainseed = 78
transformseed = 1204

#=
# Estimation by ML
# -----------------------------------------------
err_mle = zeros(k, MCreps, length(N))
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
BSON.@save "err_mle_$thisrun.bson" err_mle N
=#
# -----------------------------------------------
# NNet estimation (nnet object must be pre-trained!)
Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, PriorDraw(100000)) # use a large sample for this

# Iterate over different lengths of observed returns
err_nnet = zeros(k, MCreps, length(N))
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    nnet = lstm_net(n_hidden, k, g)
    # Train network
    opt = ADAM()
    train_rnn!(nnet, opt, dgp, n, datareps, batchsize, epochs, dtY)
    # Compute network error on a new batch
    BSON.@load "bestmodel_$n.bson" m
    Random.seed!(testseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    X = batch_timeseries(X, n, n) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(m) # In case nnet has dropout / batchnorm
    Flux.reset!(m)
    m(X[1]) # warmup
    Yhat = StatsBase.reconstruct(dtY, mean([m(x) for x ∈ X[2:end]]))
    err_nnet[:, :, i] = Y - Yhat
    # Save model as BSON
    println("Neural network, n = $n done.")
end
BSON.@save "err_nnet_$thisrun.bson" err_nnet N MCreps datareps epochs
end
main()

