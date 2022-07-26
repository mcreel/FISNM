include("GarchLib.jl")
include("neuralnets.jl")
include("samin.jl")
using Plots, Random, BSON, DelimitedFiles
function main()
# General parameters
MCreps = 1000 # Number of Monte Carlo samples for each sample size
TrainSize = 2048 # samples in each epoch
N = [100, 200, 400]  # Sample sizes (most useful to incease by 4X)
testseed = 782
trainseed = 999
transformseed = 1204

#=
# Estimation by ML
# ------------------------------------------------------------------------------
err_mle = zeros(3, MCreps, length(N))
thetahat_mle = similar(err_mle)
thetatrue = similar(err_mle)
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    Random.seed!(testseed) # samples for ML and for NN use same seed
    X, Y = dgp(n, MCreps) # Generate the data according to DGP
    # Fit GARCH models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        θstart = Float64.([0.1, 0.5, 0.5])
        obj = θ -> -mean(garch11(θ, Float64.(X[:,s])))
        lb = [0.001, 0.0, 0.0]
        ub = [0.999, 0.99, 1.0]
        θhat, junk, junk, junk= samin(obj, θstart, lb, ub, verbosity=0)
        err_mle[:, s, i] = θhat  .- Y[:, s] # Compute errors
        thetahat_mle[:,s,i] = θhat
        thetatrue[:,s,i] = Y[:,s]
    end
    println("ML n = $n done.")
end
BSON.@save "err_mle.bson" err_mle
BSON.@save "thetahat_mle.bson" thetahat_mle
BSON.@save "thetatrue.bson" thetatrue
=#
BSON.@load "err_mle.bson" err_mle

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
thetahat_nnet = similar(err_nnet)
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes and 20% dropout rate
    nnet = lstm_net(32, .2)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn!(nnet, ADAM(), dgp, n, TrainSize, dtY, epochs=200)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    #[nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
    # Compute prediction and error
    #Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
    # Alternative: this is averaging prediction at each observation in sample
    Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X])
    err_nnet[:, :, i] = Ŷ - Y
    thetahat_nnet[:,:,i] = Ŷ 

    BSON.@save "err_nnet.bson" err_nnet
    BSON.@save "thetahat_nnet.bson" thetahat_nnet

    # Save model as BSON
    BSON.@save "models/nnet_(n-$n).bson" nnet
    println("Neural network, n = $n done.")
end

# Plotting the results
# ------------------------------------------------------------------------------
k = size(err_mle, 1) # Number of parameters
# Compute squared errors
err_mle² = abs2.(err_mle);
err_nnet² = abs2.(err_nnet);
# Compute RMSE for each individual parameter
rmse_mle = permutedims(reshape(sqrt.(mean(err_mle², dims=2)), k, length(N)));
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_mle_agg = mean(rmse_mle, dims=2);
rmse_nnet_agg = mean(rmse_nnet, dims=2);

plot(N, rmse_mle, xlab="Number of observations", ylab="RMSE", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (MLE)", ["ω" "β" "α"]),
    col=first(palette(:tab10), k))
plot!(N, rmse_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["ω" "β" "α"]),
    col=first(palette(:tab10), k))
plot!(N, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("rmse_benchmark.png")

# Compute bias for each individual parameter
bias_mle = permutedims(reshape(mean(err_mle, dims=2), k, length(N)));
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
# Compute bias aggregate
bias_mle_agg = mean(bias_mle, dims=2);
bias_nnet_agg = mean(bias_nnet, dims=2);

plot(N, bias_mle, xlab="Number of observations", ylab="Bias", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (MLE)", ["ω" "β" "α"]),
    col=first(palette(:tab10), k))
plot!(N, bias_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["ω" "β" "α"]),
    col=first(palette(:tab10), k))
plot!(N, bias_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("bias_benchmark.png")

end
main()

