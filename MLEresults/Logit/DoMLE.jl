using Random, Statistics, BSON, Optim, Term, PrettyTables
include("LogitLib.jl")

function main()
# General parameters
k = 3 # number of labels
MCreps = 5000 # Number of Monte Carlo samples for each sample size
N = [100*2^i for i ∈ 0:5]  # Sample sizes (most useful to incease by 4X)
test_seed = 77

# ------------------------------------------------------------------------------
err_mle = zeros(k, MCreps, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    Random.seed!(test_seed) # samples for ML and for NN use same seed
    # Fit Logit models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        Y = PriorDraw()
        data = Logit(Y, n)
        obj = θ -> -LogitLikelihood(θ, data)
        θhat = Optim.optimize(obj, zeros(3), LBFGS(), # start value is true, to save time 
                            Optim.Options(
                            g_tol = 1e-5,
                            x_tol = 1e-6,
                            f_tol=1e-12); autodiff=:forward).minimizer
        err_mle[:, s, i] = Y - θhat # Compute errors
    end
    println("ML n = $n done.")
end
rmse = permutedims(reshape(sqrt.(mean(err_mle.^2, dims=2)), k, length(N)));
bias = permutedims(reshape(mean(err_mle, dims=2), k, length(N)));
rmse_agg = mean(rmse, dims=2);
bias_agg = mean(abs.(bias), dims=2);
println(@green "Average Absolute Bias and Average RMSE")
pretty_table([N bias_agg rmse_agg], header=["n", "Avg. Abs. Bias","Avg. RMSE"])


BSON.@save "LogitResults.bson" N err_mle rmse bias rmse_agg bias_agg
end

main()
