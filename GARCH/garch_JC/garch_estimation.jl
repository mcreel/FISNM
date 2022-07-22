include("GarchLib.jl")
include("neuralnets.jl")

using Plots, PrettyTables
using Random


# General parameters
S = 1_000 # Number of samples for each estimation
N = 10:10:300 # Number of observations per sample
seed = 72 # Random seed

# Estimation using ARCHModels.jl
# ------------------------------------------------------------------------------
Random.seed!(seed)
err_pkg = zeros(5, S, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    X, Y = dgp(n, S) # Generate the data according to DGP
    # Fit GARCH models using ARCHModels.jl on each sample and compute error
    for s ∈ 1:S
        try
            est = fit(GARCH{1, 1}, X[:, s], meanspec=AR{1})
            @show est.fitted
            pretty_table([coef(est)' Y[:,s]])
            err_pkg[:, s, i] = coef(est) .- Y[:, s] # Compute errors
        catch
            println("GARCH ML crash")
        end   
    end
    println("ARCHModels.jl n = $n done.")
end

# NNet estimation (nnet object must be pre-trained!)
# ------------------------------------------------------------------------------
Random.seed!(seed)
err_nnet = zeros(5, S, length(N))
# We standardize the outputs for the MSE to not be overly influenced by the
# parameters which are larger in absolute value than others. Because of this,
# we need to fit a data transformation on the labels. Thus, we use the dgp()
# function to generate a batch which we ONLY use to fit this transformation,
# this way we don't have the problem of fitting a transformation on our 
# test set.
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N)
    # Fit data transformation which we will later use to rescale our nnet output
    X, Y = dgp(n, S)
    dtY = fit(ZScoreTransform, Y)
    # Create network with 32 hidden nodes and 20% dropout rate
    nnet = lstm_net(32, .2)
    # Train network (it seems we can still improve by going over 200 epochs!)
    train_rnn!(nnet, ADAM(), dgp, n, S, epochs=200)
    # Compute network error on a new batch
    X, Y = dgp(n, S) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    [nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
    # Compute prediction and error
    Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
    err_nnet[:, :, i] = Ŷ - Y
    # Save model as BSON
    BSON.@save "models/nnet_(n-$n).bson" nnet
    println("Neural network, n = $n done.")
end

# Plotting the results
# ------------------------------------------------------------------------------
k = size(err_pkg, 1) # Number of parameters
# Compute squared errors
err_pkg² = abs2.(err_pkg);
err_nnet² = abs2.(err_nnet);
# Compute RMSE for each individual parameter
rmse_pkg = permutedims(reshape(sqrt.(mean(err_pkg², dims=2)), k, length(N)));
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_pkg_agg = mean(rmse_pkg, dims=2);
rmse_nnet_agg = mean(rmse_nnet, dims=2);

plot(N, rmse_pkg, xlab="Number of observations", ylab="RMSE", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (ARCHModels.jl)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, rmse_pkg_agg, lab="Aggregate (ARCHModels.jl)", c=:black, lw=3)

plot!(N, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("rmse_benchmark.png")

# Compute bias for each individual parameter
bias_pkg = permutedims(reshape(mean(err_pkg, dims=2), k, length(N)));
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
# Compute bias aggregate
bias_pkg_agg = mean(bias_pkg, dims=2);
bias_nnet_agg = mean(bias_nnet, dims=2);

plot(N, bias_pkg, xlab="Number of observations", ylab="Bias", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (ARCHModels.jl)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, bias_pkg_agg, lab="Aggregate (ARCHModels.jl)", c=:black, lw=3)

plot!(N, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, bias_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("bias_benchmark.png")
