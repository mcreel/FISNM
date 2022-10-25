using Pkg
Pkg.activate(".")
using Flux, CUDA, PrettyTables, Statistics, Term
using BSON:@load
include("DGPs.jl")

# for some reason, there is a world age problem if 
# this is inside the main() function (???)
BSON.@load "models/MA2/mcx_(n-100).bson" best_model
Flux.testmode!(best_model)

function main()
# General parameters
test_size = 5_000 # Number of Monte Carlo samples for each sample size
test_seed = 78
transform_seed = 1204
transform_size = 100_000
dev=cpu
# get the transformation, and set seed to testing
Random.seed!(transform_seed)
dgp=Ma2(N=100)
dtY = data_transform(dgp, transform_size, dev=dev)
Random.seed!(test_seed)
k = 2  # number of parameters
base_n = 100
Ns = [100, 200, 400, 800, 1600, 3200]  # larger samples to use
results = zeros(6,3)
# loop over sample sizes
for i = 1:size(Ns,1)
    n = Ns[i]
    n_splits = n ÷ base_n
    dgp=Ma2(N=n)
    X, Y = generate(dgp, test_size, dev=dev)
    Yhat = Y-Y
    for split = 1:n_splits
        Xs = X[:,:,(split-1)*base_n+1:split*base_n]
        Xs = tabular2conv(Xs)
        Yhat += StatsBase.reconstruct(dtY, best_model(Xs))
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
