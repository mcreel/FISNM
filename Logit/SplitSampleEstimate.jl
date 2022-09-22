using Plots, Random, BSON, Optim
include("LogitLib.jl")
include("neuralnets.jl")

function main()
thisrun = "working"
# General parameters
MCreps = 1000 # Number of Monte Carlo samples for each sample size
N = [50, 100, 150]  # Sample sizes (most useful to incease by 4X)
MCseed = 77

BSON.@load "bestmodel_50.bson" m

# NNet estimation (nnet object must be pre-trained!)
err_nnet = zeros(3, MCreps, length(N))
BC = zeros(3, length(N))
Random.seed!(trainseed) # avoid sample contamination for NN training
Threads.@threads for i = 1:size(N,1)
    n = N[i]

    # bias correction
    Random.seed!(BCseed)
    X, Y = dgp(n, 100000) # Generate data according to DGP
    X = batch_timeseries(X, n, n) # Transform to rnn format
    Flux.testmode!(m) # In case nnet has dropout / batchnorm
    Flux.reset!(m)
    m(X[1]) # warmup
    Yhat = [m(x) for x ∈ X[2:end]][end]
    BC[:,i] = mean(Yhat-Y, dims=2)
    # final fit and errors
    Random.seed!(MCseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    X = batch_timeseries(X, n, n) # Transform to rnn format
    # Get NNet fit, with bias correction
    Flux.testmode!(m) # In case nnet has dropout / batchnorm
    Flux.reset!(m)
    m(X[1]) # warmup
    Yhat = [m(x) for x ∈ X[2:end]][end] .- BC[:,i]
    err_nnet[:, :, i] = Yhat - Y 
    # Save model as BSON
    println("Neural network, n = $n done.")
end
BSON.@save "err_nnet_$thisrun.bson" err_nnet BC N MCreps datareps epochs batchsize
end
main()

