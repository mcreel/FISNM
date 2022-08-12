cd("GARCH")

using Plots, Random, BSON

include("GarchLib.jl")
include("neuralnets.jl")
include("samin.jl")

# function main()
thisrun = "test"
# General parameters

MCreps = 1000 # Number of Monte Carlo samples for each sample size
TrainSize = 2048 # samples in each epoch
N = [100, 200, 400, 800, 1600, 3200]  # Sample sizes (most useful to incease by 4X)
testseed = 782
trainseed = 999
transformseed = 1204
N = 1600
epochs = 200
rseed = 12
batchsize= 512
dev = gpu

Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, dev(PriorDraw(100_000))) # Use a large sample for standardization

all_losses = []

# Iterate over different lengths of observed returns
err_nnet = zeros(3, MCreps, length(N))
thetahat_nnet = similar(err_nnet)
Random.seed!(rseed)
# Threads.@threads for i = 1:size(N,1)
#     n = N[i]
#     # Create network with 32 hidden nodes and 20% dropout rate
#     nnet =  build_net(add_dropout=false)
#     # Train network (it seems we can still improve by going over 200 epochs!)
#     Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
#     push!(all_losses, train_rnn1!(nnet, ADAM(), dgp, n, TrainSize, dtY, 
#         epochs=epochs, batchsize=batchsize))
#     # Compute network error on a new batch
#     Random.seed!(testseed)
#     X, Y = dgp(n, MCreps) # Generate data according to DGP
#     X = max.(X,Float32(-20.0))
#     X = min.(X,Float32(20.0))
#     X = tabular2rnn(X) # Transform to rnn format
#     # Get NNet estimate of parameters for each sample
#     Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
#     Flux.reset!(nnet)
#     [nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
#     # Compute prediction and error
#     Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
#     # Alternative: this is averaging prediction at each observation in sample
#     # Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X])
#     err_nnet[:, :, i] = Ŷ - Y
#     thetahat_nnet[:,:,i] = Ŷ 
#     BSON.@save "err_nnet1_$thisrun.bson" err_nnet N MCreps TrainSize
#     # Save model as BSON
#     # BSON.@save "models/nnet_(n-$n).bson" nnet
#     println("Neural network 1, n = $n done.")
# end


# Threads.@threads for i = 1:size(N,1)
#     n = N[i]
#     # Create network with 32 hidden nodes and 20% dropout rate
#     nnet = seq2one_net(32)
#     # Train network (it seems we can still improve by going over 200 epochs!)
#     Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
#     push!(all_losses, train_seq2one!(nnet, ADAM(), dgp, n, TrainSize, dtY, 
#         epochs=epochs, batchsize=batchsize))
#     # Compute network error on a new batch
#     Random.seed!(testseed)
#     X, Y = dgp(n, MCreps) # Generate data according to DGP
#     X = max.(X,Float32(-20.0))
#     X = min.(X,Float32(20.0))
#     X = tabular2rnn(X) # Transform to rnn format
#     # Get NNet estimate of parameters for each sample
#     Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
#     Flux.reset!(nnet)
#     # Compute prediction and error
#     Ŷ = StatsBase.reconstruct(dtY, nnet(X))
#     # Alternative: this is averaging prediction at each observation in sample
#     # Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X])
#     err_nnet[:, :, i] = Ŷ - Y
#     thetahat_nnet[:,:,i] = Ŷ 
#     BSON.@save "err_nnet1_$thisrun.bson" err_nnet N MCreps TrainSize
#     # Save model as BSON
#     # BSON.@save "models/nnet_(n-$n).bson" nnet
#     println("Neural network 1, n = $n done.")
# end


m = build_bidirectional_net(dev=gpu)
# ------------------------------------------------
err_nnet = zeros(3, MCreps, length(N))
thetahat_nnet = similar(err_nnet)
Random.seed!(rseed)
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes and 20% dropout rate
    nnet = build_net(add_dropout=false, dev=gpu)# lstm_net(32, .2)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    push!(all_losses, train_rnn!(nnet, ADAM(), dgp, n, TrainSize, dtY, 
        epochs=epochs, batchsize=batchsize, dev=dev))
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
    X = max.(X,Float32(-20.0))
    X = min.(X,Float32(20.0))
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    #[nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
    # Compute prediction and error
    #Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
    # Alternative: this is averaging prediction at each observation in sample
    # Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X])
    # err_nnet[:, :, i] = cpu(Ŷ - Y)
    # thetahat_nnet[:,:,i] = cpu(Ŷ) 
    # BSON.@save "err_nnet2_$thisrun.bson" err_nnet N MCreps TrainSize
    # Save model as BSON
    # BSON.@save "models/nnet_(n-$n).bson" nnet
    println("Neural network 2, n = $n done.")
end
# ------------------------------------------------
# err_nnet = zeros(3, MCreps, length(N))
# thetahat_nnet = similar(err_nnet)
# Random.seed!(rseed)
# Threads.@threads for i = 1:size(N,1)
#     n = N[i]
#     # Create network with 32 hidden nodes and 20% dropout rate
#     nnet = build_net(add_dropout=false)# lstm_net(32, .2)
#     # Train network (it seems we can still improve by going over 200 epochs!)
#     Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
#     push!(all_losses, train_rnn3!(nnet, ADAM(), dgp, n, TrainSize, dtY, 
#         epochs=epochs, batchsize=batchsize))
#     # Compute network error on a new batch
#     Random.seed!(testseed)
#     X, Y = dgp(n, MCreps) # Generate data according to DGP
#     X = max.(X,Float32(-20.0))
#     X = min.(X,Float32(20.0))
#     X = tabular2rnn(X) # Transform to rnn format
#     # Get NNet estimate of parameters for each sample
#     Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
#     Flux.reset!(nnet)
#     #[nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
#     # Compute prediction and error
#     #Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
#     # Alternative: this is averaging prediction at each observation in sample
#     nnet(X[1])
#     Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X[2:end]])
#     err_nnet[:, :, i] = Ŷ - Y
#     thetahat_nnet[:,:,i] = Ŷ 
#     BSON.@save "err_nnet3_$thisrun.bson" err_nnet N MCreps TrainSize
#     # Save model as BSON
#     # BSON.@save "models/nnet_(n-$n).bson" nnet
#     println("Neural network 3, n = $n done.")
# end
# ------------------------------------------------------------------
err_nnet = zeros(3, MCreps, length(N))
thetahat_nnet = similar(err_nnet)
Random.seed!(rseed)
Threads.@threads for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes
    nnet = build_bidirectional_net(add_dropout=false, dev=gpu)

    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    push!(all_losses, train_rnn!(nnet, ADAM(), dgp, n, TrainSize, dtY, 
        epochs=epochs, batchsize=batchsize, dev=dev, bidirectional=true))
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = dgp(n, MCreps) # Generate data according to DGP
    X = max.(X,Float32(-20.0))
    X = min.(X,Float32(20.0))
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    #[nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
    # Compute prediction and error
    #Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
    # Alternative: this is averaging prediction at each observation in sample
    # nnet(X[1])
    # Ŷ = StatsBase.reconstruct(dtY, nnet(X))
    # err_nnet[:, :, i] = Ŷ - Y
    # thetahat_nnet[:,:,i] = Ŷ 
    # BSON.@save "err_nnet3_$thisrun.bson" err_nnet N MCreps TrainSize
    # Save model as BSON
    # BSON.@save "models/nnet_(n-$n).bson" nnet
    println("BILSTM 3, n = $n done.")
end

plot(all_losses[1], lab="Model 1")
plot!(all_losses[2], lab="BiLSTM")



BSON.@load "err_nnet1_$thisrun.bson" err_nnet
err_nnet1 = deepcopy(err_nnet)
BSON.@load "err_nnet2_$thisrun.bson" err_nnet
err_nnet2 = deepcopy(err_nnet)
BSON.@load "err_nnet3_$thisrun.bson" err_nnet
err_nnet3 = deepcopy(err_nnet)

hcat(
    sqrt.(mean(err_nnet1 .^ 2, dims = 2)),
    sqrt.(mean(err_nnet2 .^ 2, dims = 2)),
    sqrt.(mean(err_nnet3 .^ 2, dims = 2)),
    # sqrt.(mean(err_tcn .^ 2, dims=2))
)


# histogram(err_nnet1[1, :], lab="Single pred", alpha=0.6)
# histogram!(err_nnet2[1, :], lab="All", alpha=0.6)
# histogram!(err_nnet3[1, :], lab="All but first", alpha=.6)

# histogram(err_nnet1[1, :], lab="Single pred", alpha=0.6)
# histogram!(err_nnet2[1, :], lab="All", alpha=0.6)
# histogram!(err_nnet3[1, :], lab="All but first", alpha=.6)


plot(all_losses[1], lab="Model 1")
plot!(all_losses[2], lab="Model 2")
plot!(all_losses[3], lab="Model 3")

# Learning rate analysis
n_lr = 30
val_loss = Vector{Float32}(undef, n_lr)
lrs = 10 .^ range(-9, 1, length=n_lr)
for (i, η) ∈ enumerate(lrs)
    # Create network with 32 hidden nodes and 20% dropout rate
    nnet =  lstm_net(32, .2) # build_net(add_dropout=false) #
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn_fast!(nnet, ADAM(η), dgp, N, TrainSize, dtY, 
        epochs=epochs, batchsize=batchsize)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = dgp(N, MCreps) # Generate data according to DGP
    X = max.(X,Float32(-20.0))
    X = min.(X,Float32(20.0))
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    [nnet(x) for x ∈ X[1:end-1]] # Run network up to penultimate X
    # Compute prediction and error
    Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
    # Alternative: this is averaging prediction at each observation in sample
    # Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X])
    val_loss[i] = sqrt(mean((Ŷ - Y) .^ 2))
    @info "i = $i, η = $(round(η, digits=10))" val_loss[i]
end

scatter(lrs, val_loss, xaxis=:log, xticks=[10^(i) for i ∈ range(-9, 0, length=10)])


# Batchsize research
n_batchsizes = 10
val_loss = Vector{Float32}(undef, n_batchsizes)
batchsizes = 2 .^ (1:n_batchsizes)
for (i, bs) ∈ enumerate(batchsizes)
    # Create network with 32 hidden nodes and 20% dropout rate
    nnet =  build_net(add_dropout=false)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn_fast!(nnet, ADAM(), dgp, N, TrainSize, dtY, 
        epochs=epochs, batchsize=bs)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = dgp(N, MCreps) # Generate data according to DGP
    X = max.(X,Float32(-20.0))
    X = min.(X,Float32(20.0))
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate
plot(val_loss, xticks=batchsizes)

sum(sum([abs2.(Yb - vcat(x, x, x)) for x ∈ Xb]))/length(Xb)
mean(sum(abs2.(Yb - vcat(x, x, x)) for x ∈ Xb))
 error
    Ŷ = StatsBase.reconstruct(dtY, nnet(X[end]))
    # Alternative: this is averaging prediction at each observation in sample
    # Ŷ = mean([StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X])
    val_loss[i] = sqrt(mean((Ŷ - Y) .^ 2))
    @info "i = $i, batchsize = $bs" val_loss[i]
end





plot(val_loss, xticks=batchsizes)

sum(sum([abs2.(Yb - vcat(x, x, x)) for x ∈ Xb]))/length(Xb)
mean(sum(abs2.(Yb - vcat(x, x, x)) for x ∈ Xb), dims=2)


sum(sum([abs2.(Yb - vcat(x, x, x)) for x ∈ Xb]))
sum(mean(abs2.(Yb - vcat(x, x, x)) for x ∈ Xb))