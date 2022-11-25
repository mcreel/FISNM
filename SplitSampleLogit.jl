using Pkg
Pkg.activate(".")
using Flux, CUDA, PrettyTables, Statistics, Term
using BSON:@load
include("DGPs.jl")

# for some reason, there is a world age problem if 
# this is inside the main() function (???)
# appears related to https://github.com/JuliaIO/BSON.jl/issues/69
global const base_n=100
BSON.@load "models/Logit/splitsample_(n-$base_n).bson" best_model
Flux.testmode!(best_model)
#gpu(best_model)
function main()
# General parameters
test_size = 5_000 # Number of Monte Carlo samples for each sample size
test_seed = 78
transform_seed = 1204
transform_size = 100_000
# get the transformation, and set seed to testing
Random.seed!(transform_seed)
dgp=Logit(N=base_n)
dtY = data_transform(dgp, transform_size, dev=cpu)
Random.seed!(test_seed)
k = 3  # number of parameters
Ns = [Int64(base_n*2^i) for i=0:log(2,3200/base_n)]  # larger samples to use
results = zeros(size(Ns,1),3)
# loop over sample sizes
for i = 1:size(Ns,1)
    n = Ns[i]
    @info "doing n=$n"
    n_splits = n ÷ base_n
    dgp=Logit(N=n)
    X, Y = generate(dgp, test_size, dev=cpu)
    Yhat = Y-Y
    for split = 1:n_splits
        @info "doing split=$split"
        Xs = X[:,:,(split-1)*base_n+1:split*base_n]
        Xs = tabular2conv(Xs)
 #       Xs |> gpu
#        fit = cpu(best_model(Xs))
        fit = best_model(Xs)
        Yhat += StatsBase.reconstruct(dtY, fit)
    end
    #Yhat = cpu(Yhat)
    Yhat ./= n_splits
    err = Yhat - Y
    amae= mean(mean(abs.(err),dims=2))
    armse = mean(sqrt.(mean(abs2.(err),dims=2)))
    results[i,:] = [n, amae, armse]
end
println(@green "Logit model: Split sample average MAE and RMSe, base model n=base_n")
return results
end
results = main()
pretty_table(results, header=["n", "amae", "armse"])

