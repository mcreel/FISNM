using Pkg
Pkg.activate(".")
using BSON, Flux, CUDA
include("DGPs.jl")
include("neuralnets.jl")
include("train_tcn.jl")
#function main()
# General parameters
test_size = 5 # Number of Monte Carlo samples for each sample size
test_seed = 78
transform_seed = 1204
transform_size = 100_000
dev=cpu

# the base model
BSON.@load "models/MA2/tcn-ss-n-100.bson" best_model

#best_model |> gpu
k = 2  # number of parameters
N = [100]  # larger samples to use
err_ss = zeros(k, test_size, length(N))
# loop over sample sizes
for i = 1:size(N,1)
    n = N[i]
    dgp=Ma2(N=n)
    Random.seed!(transform_seed)
    dtY = data_transform(dgp, transform_size, dev=dev)
    Random.seed!(test_seed)
    X, Y = generate(dgp, test_size, dev=dev)
    X = tabular2conv(X)
    Flux.testmode!(best_model)
    Yhat = StatsBase.reconstruct(dtY, best_model(X))
    err_ss[:, :, i] = cpu(Yhat - Y)
end

#end
#main()
