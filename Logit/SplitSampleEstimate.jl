using Plots, Random, BSON, Optim
include("LogitLib.jl")
include("../NN/neuralnets.jl")

@views function main()
whichrun = "final"
# General parameters
MCreps = 5000 # Number of Monte Carlo samples for each sample size
MCseed = 77

# use a model trained with samples of size n
# to fit longer samples
base_n = 400
BSON.@load "bestmodel_$whichrun$base_n.bson" m
BSON.@load "bias_correction$whichrun.bson" BC
BC = BC[:,4] # n=400 is 4th size tried
k = 3 # number of parameters
N = [800, 1600, 3200]  # larger samples to use
err_nnet = zeros(k, MCreps, length(N))
# loop over sample sizes
for i = 1:size(N,1)
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
            yhat += [m(x) for x ∈ xs][end]
        end
        yhat ./= nsplits  # fit is average of fit from each chunk
        Yhat[:,rep] = yhat - BC
    end    
    err_nnet[:, :, i] = Yhat - Y
end

BSON.@save "err_splitsample_$whichrun.bson" err_nnet N
end

main()
