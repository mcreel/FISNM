using Plots, Random, BSON, Flux
include("MA2lib.jl")
include("../NN/neuralnets.jl")

#@views function main()
whichrun = "final"
# General parameters
MCreps = 5000 # Number of Monte Carlo samples for each sample size
MCseed = 79

# use a model trained with samples of size n
# to fit longer samples
base_n = 400
BSON.@load "bestmodel_$whichrun$base_n.bson" m
BSON.@load "bias_correction$whichrun.bson" BC
BC = BC[:,3] # n=400 is 3rd size tried
k = 2 # number of parameters
N = [800, 1600]  # larger samples to use
err_nnet = zeros(k, MCreps, length(N))
# loop over sample sizes
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    Random.seed!(MCseed)
    Yhat = zeros(k, MCreps)
    Y = similar(Yhat)
    # loop over MC reps, for each sample size
    for rep = 1:MCreps
        x, y = dgp(n, 1)
        Y[:,rep] = y
        yhat = zeros(k)
        nsplits = 0
        # fit for each chunk of size base_n
        for stop in range(start=base_n, stop=n, step=50)
            nsplits +=1
            start = stop - base_n + 1
            xs = x[start:stop]
            Flux.reset!(m)
            yhat += [m(xs) for x ∈ xs][end]
        end
        yhat ./= nsplits .- BC # fit is average of fit from each chunk
        display(yhat)
        display(y)
        Yhat[:,rep] = yhat
    end    
    err_nnet[:, :, i] = Yhat - Y
end

BSON.@save "err_splitsample_$whichrun.bson" err_nnet

# Compute squared errors
err_nnet² = abs2.(err_nnet);
# Compute RMSE for each individual parameter
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_nnet_agg = mean(rmse_nnet, dims=2);

colors = palette(:default)[1:k]'
plot(N, rmse_nnet, lw=2, size=(1200, 800), ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)
savefig("rmse_splitsample_$whichrun.png")

# Compute bias for each individual parameter
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
# Compute bias aggregate
bias_nnet_agg = mean(abs.(bias_nnet), dims=2);
plot(N, bias_nnet, lw=2, size=(1200, 800), ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N, bias_nnet_agg, lab="Aggregate (abs) (NNet)", c=:black, lw=3, ls=:dash)
savefig("bias_splitsample_$whichrun.png")

#end

#main()
