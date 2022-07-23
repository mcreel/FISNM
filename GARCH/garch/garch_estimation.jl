include("GarchLib.jl")
include("neuralnets.jl")
include("fmincon.jl")
using Plots, Random

function main()
# General parameters
S = 100 # Number of Monte Carlo samples for each sample size
N = [100, 400, 1600]  # Sample sizes (most useful to incease by 4X)
seed = 72 # Random seed

# Estimation by ML
# ------------------------------------------------------------------------------
Random.seed!(seed)
err_mle = zeros(5, S, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    X, Y = dgp(n, S) # Generate the data according to DGP
    # Fit GARCH models by ML on each sample and compute error
    for s ∈ 1:S
        try
            θstart = Float64.([mean(X[:,s]); 0.0; var(X[:,s]); 0.1; 0.1])
            obj = θ -> -mean(garch11(θ, Float64.(X[:,s])))
            lb = [-Inf, -1.0, 1e-5, 0.0, 0.0]
            ub = [Inf, 1.0, Inf, 1.0, 1.0] 
            θhat, logL, convergence  = fmincon(obj, θstart, [], [], lb, ub)
            println("convergence: $convergence")
            err_mle[:, s, i] = θhat  .- Y[:, s] # Compute errors
        catch
            println("s: $s")
            println("GARCH ML crash")
        end   
    end
    println("ML n = $n done.")
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
    X, Y = dgp(n, 1000)  # S should be large here, no? No reason to be constrained to be same as number of  MonteCarlo reps
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
    lw=2, lab=map(x -> x * " (MLE)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, rmse_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["ω" "β" "α" "μ" "ρ"]),
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
    lw=2, lab=map(x -> x * " (MLE)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, bias_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["ω" "β" "α" "μ" "ρ"]),
    col=first(palette(:tab10), k))
plot!(N, bias_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("bias_benchmark.png")

end
main()
