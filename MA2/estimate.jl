using Plots, Random, BSON, Optim
include("MA2lib.jl")
include("neuralnets.jl")

function main()
thisrun = "working"
# General parameters
k = 2 # number of labels
g = 1 # number of features
n_hidden = 16
MCreps = 5000 # Number of Monte Carlo samples for each sample size
BCreps = Int64(10^6)
datareps = 1000 # number of repetitions of drawing sample
batchsize = 100
epochs = 10 # passes over each batch
N = [50, 100, 200, 400, 800, 1600]  # Sample sizes (most useful to incease by 4X)
MCseed = 77
trainseed = 78
BCseed = 1024 # bias correction seed

# NNet estimation (nnet object must be pre-trained!)
# -----------------------------------------------
# Iterate over different lengths of observed returns
err_nnet = zeros(k, MCreps, length(N))
BC = zeros(k, length(N))
Random.seed!(trainseed) # avoid sample contamination for NN training
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes
    nnet = lstm_net(n_hidden, k, g)
    # Train network
    opt = ADAM()
    train_rnn!(nnet, opt, dgp, n, datareps, batchsize, epochs)
    # Compute network error on a new batch
    BSON.@load "bestmodel_$n.bson" m
    # bias correction
    Random.seed!(BCseed)
    X, Y = dgp(n, 100000) # Generate data according to DGP
    Flux.testmode!(m) # In case nnet has dropout / batchnorm
    Flux.reset!(m)
    Yhat = [m(x) for x ∈ X][end]
    BC[:,i] = mean(Yhat-Y, dims=2)
    # final fit and errors
    Random.seed!(MCseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    # Get NNet fit, with bias correction
    Flux.testmode!(m) # In case nnet has dropout / batchnorm
    Flux.reset!(m)
    m(X[1]) # warmup
    Yhat = [m(x) for x ∈ X[2:end]][end] .- BC[:,i]
    err_nnet[:, :, i] = Yhat - Y 
    # Save model as BSON
    println("Neural network, n = $n done.")
end
BSON.@save "bias_correction.bson" BC N
BSON.@save "err_nnet_$thisrun.bson" err_nnet N MCreps datareps epochs batchsize
end
main()

