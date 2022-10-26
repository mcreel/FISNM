using Pkg
Pkg.activate(".")
using Flux, CUDA, PrettyTables, Statistics, Term
using BSON:@load
include("DGPs.jl")

# for some reason, there is a world age problem if 
# this is inside the main() function (???)
# appears related to https://github.com/JuliaIO/BSON.jl/issues/69
BSON.@load "models/MA2/mcx2_(n-100).bson" best_model
Flux.testmode!(best_model)
gpu(best_model)
function main()
# General parameters
test_size = 5_000 # Number of Monte Carlo samples for each sample size
test_seed = 78
transform_seed = 1204
transform_size = 100_000
dev=gpu
# get the transformation, and set seed to testing
Random.seed!(transform_seed)
dgp=Ma2(N=100)
dtY = data_transform(dgp, transform_size, dev=cpu)
Random.seed!(test_seed)
k = 2  # number of parameters
base_n = 100
Ns = [100, 200]  # larger samples to use
results = zeros(6,3)
# loop over sample sizes
for i = 1:size(Ns,1)
    n = Ns[i]
    n_splits = n ÷ base_n
    dgp=Ma2(N=n)
    X, Y = generate(dgp, test_size, dev=cpu)
    Yhat = Y-Y
    for split = 1:n_splits
        Xs = X[:,:,(split-1)*base_n+1:split*base_n]
        Xs = tabular2conv(Xs)
        Xs |> gpu
        fit = cpu(best_model(Xs))
        Yhat += StatsBase.reconstruct(dtY, fit)
    end
    Yhat = cpu(Yhat)
    Yhat ./= n_splits
    err = Yhat - Y
    amae= mean(mean(abs.(err),dims=2))
    armse = mean(sqrt.(mean(abs2.(err),dims=2)))
    pretty_table([n, amae, armse])
    results[i,:] = [n, amae, armse]
end
println(@green "Split sample average MAE and RMSe, base model n=100")
pretty_table(results, header=["n", "amae", "armse"])

end
main()
