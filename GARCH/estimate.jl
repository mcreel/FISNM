using Pkg
Pkg.activate(".")
using Plots, Random, BSON
cd("GARCH")
include("GarchLib.jl")
include("TCN.jl")
include("neuralnets.jl")
include("samin.jl")

function main()

thisrun = "experiment"
# General parameters
k = 3 # number of labels
g = 1 # number of features
n_hidden = 16
MCreps = 1000 # Number of Monte Carlo samples for each sample size
datareps = 1000 # 3000 # number of repetitions of drawing sample
batchsize = 50
epochs = 10 # passes over each batch
N = [100, 200, 400, 800]  # Sample sizes (most useful to incease by 4X)
testseed = 77
trainseed = 78
transformseed = 1204
dev = gpu

# TCN parameters
dilation = 2
kernel_size = 8
channels = 2
layers = [ceil(Int, necessary_layers(dilation, kernel_size, n)) for n ∈ N]
tcn_epochs = 5_000

Random.seed!(transformseed) # avoid sample contamination for NN training
dtY = fit(ZScoreTransform, dev(PriorDraw(100_000))) # use a large sample for this

# -----------------------------------------------
# Temporal Convolutional NNet estimation

# Iterate over different lengths of observed returns
err_tcn = zeros(k, MCreps, length(N))
for i = 1:size(N,1)
    n = N[i]
    n_layers = layers[i]
    # Create TCN
    tcn = dev(
        Chain(
            TCN(vcat(g, [channels for _ ∈ 1:n_layers], 1), kernel_size=kernel_size, 
                dropout_rate=0.0, dilation_factor=dilation), 
            Conv((1, 10), 1 => 1, stride = 10),
            Flux.flatten,
            Dense(n ÷ 10 => k)
        )
    )
    # Train network (use 40x the epochs for TCN as it is extremely fast to train)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_cnn!(tcn, ADAM(), dgp, n, datareps, dtY, epochs=tcn_epochs, dev=dev, 
        validation_loss=false, verbosity=100)
    
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
    X = tabular2conv(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(tcn) # In case nnet has dropout / batchnorm
     # warmup
    Yhat = StatsBase.reconstruct(dtY, tcn(X))
    err_tcn[:, :, i] = cpu(Y - Yhat)
    # Save model as BSON
    BSON.@save "models/tcn_(n-$n)_$thisrun.bson" tcn
    println("Temporal convolutional neural network, n = $n done.")
end
BSON.@save "err_tcn_$thisrun.bson" err_tcn N MCreps datareps epochs

# -----------------------------------------------
# NNet estimation

# Iterate over different lengths of observed returns
err_nnet = zeros(k, MCreps, length(N))
for i = 1:size(N,1)
    n = N[i]
    # Create network with 32 hidden nodes
    nnet = lstm_net(n_hidden, k, g, dev)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn!(nnet, AdaDelta(), dgp, n, datareps, dtY, epochs=epochs, dev=dev,
        validation_loss=false, verbosity=25)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(nnet) # In case nnet has dropout / batchnorm
    Flux.reset!(nnet)
    nnet(X[1]) # warmup
    Yhat = mean(StatsBase.reconstruct(dtY, nnet(x)) for x ∈ X[2:end])
    err_nnet[:, :, i] = cpu(Y - Yhat)
    # Save model as BSON
    BSON.@save "models/nnet_(n-$n)_$thisrun.bson" nnet
    println("Neural network, n = $n done.")
end
BSON.@save "err_nnet_$thisrun.bson" err_nnet N MCreps datareps epochs

# -----------------------------------------------
# Bidirectional NNet estimation

# Iterate over different lengths of observed returns
err_bnnet = zeros(k, MCreps, length(N))
for i = 1:size(N,1)
    n = N[i]
    # Create bidirectional network with 32 hidden nodes
    bnnet = bilstm_net(n_hidden, k, g, dev)
    # Train network (it seems we can still improve by going over 200 epochs!)
    Random.seed!(trainseed) # use other seed this, to avoid sample contamination for NN training
    train_rnn!(bnnet, AdaDelta(), dgp, n, datareps, dtY, epochs=epochs, dev=dev,
        validation_loss=false, bidirectional=true, verbosity=25)
    # Compute network error on a new batch
    Random.seed!(testseed)
    X, Y = map(dev, dgp(n, MCreps)) # Generate data according to DGP
    X = tabular2rnn(X) # Transform to rnn format
    # Get NNet estimate of parameters for each sample
    Flux.testmode!(bnnet) # In case nnet has dropout / batchnorm
    Flux.reset!(bnnet)
     # warmup
    Yhat = StatsBase.reconstruct(dtY, bnnet(X))
    err_bnnet[:, :, i] = cpu(Y - Yhat)
    # Save model as BSON
    BSON.@save "models/bnnet_(n-$n)_$thisrun.bson" bnnet
    println("Bidirectional neural network, n = $n done.")
end
BSON.@save "err_bnnet_$thisrun.bson" err_bnnet N MCreps datareps epochs

# Estimation by ML
# -----------------------------------------------
err_mle = zeros(k, MCreps, length(N))
# Iterate over different lengths of observed returns
for (i, n) ∈ enumerate(N) 
    Random.seed!(testseed) # samples for ML and for NN use same seed
    X, Y = dgp(n, MCreps) # Generate the data according to DGP
    # Fit GARCH models by ML on each sample and compute error
    Threads.@threads for s ∈ 1:MCreps
        θstart = Float64.([0.5, 0.5, 0.5])
        obj = θ -> -mean(garch11(θ, Float64.(X[:,s])))
        lb = [0.0001, 0.0, 0.0]
        ub = [1.0, 0.99, 1.0]
        θhat, junk, junk, junk= samin(obj, θstart, lb, ub, verbosity=0)
        err_mle[:, s, i] = Y[:, s] .- θhat # Compute errors
    end
    println("ML n = $n done.")
end
BSON.@save "err_mle_$thisrun.bson" err_mle N

end
main()