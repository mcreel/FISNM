using Random, Statistics, BSON, Term, PrettyTables
include("GarchML.jl")
include("samin.jl")

function main()
MCreps = 5000 # Number of Monte Carlo samples for each sample size
N = [100*2^i for i ∈ 0:5]  # Sample sizes (most useful to incease by 4X)
testseed = 77
k = 3
err_mle = zeros(k, MCreps, length(N))
for (i, n) ∈ enumerate(N) 
    Random.seed!(testseed) # samples for ML and for NN use same seed
    X, Y = dgp(n, MCreps) # Generate the data according to DGP
    # Fit GARCH models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        θstart = Y[:,s]
        obj = θ -> -mean(garch11(θ, Float64.(X[:,s])))
        lb = [0.0001, 0.0, 0.0]
        ub = [1.0, 0.99, 1.0]
        θhat, junk, junk, junk= samin(obj, θstart, lb, ub, verbosity=0, rt=0.25, maxevals=1e5, coverage_ok=1)
        err_mle[:, s, i] = θhat .- Y[:, s] # Compute errors
    end
    println("ML n = $n done.")
end

rmse = permutedims(reshape(sqrt.(mean(err_mle.^2, dims=2)), k, length(N)));
bias = permutedims(reshape(mean(err_mle, dims=2), k, length(N)));
rmse_agg = mean(rmse, dims=2);
bias_agg = mean(abs.(bias), dims=2);
println(@green "Average Absolute Bias and Average RMSE")
pretty_table([N bias_agg rmse_agg], header=["n", "Avg. Abs. Bias","Avg. RMSE"])

BSON.@save "GarchResults.bson" N err_mle rmse bias rmse_agg bias_agg
end

main()
