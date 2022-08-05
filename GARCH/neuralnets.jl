using Flux
using StatsBase

# Some helpers
tabular2rnn(X) = [X[i:i, :] for i ∈ 1:size(X, 1)]
rmse_loss(X, Y) = sqrt(mean(abs2.(X - Y)))
lstm_net(n_hidden) = Chain(
    Dense(1, n_hidden, tanh),
    LSTM(n_hidden, n_hidden),
    LSTM(n_hidden, n_hidden),
    Dense(n_hidden, 3)
)
    
# Trains a recurrent neural network
function train_rnn!(
    m, opt, dgp, n, S, dtY; 
    epochs=100, batchsize=32, dev=cpu, loss=rmse_loss)
    Flux.trainmode!(m) # In case we have dropout / batchnorm
    m = dev(m) # Pass model to device (cpu/gpu)
    θ = Flux.params(m) # Extract parameters
    # Iterate over training epochs
    for epoch ∈ 1:epochs
        println("epoch $epoch of $epochs")
        X, Y = dgp(n, S) # Generate a new batch
        # Standardize targets for MSE scaling
        # no need to do this for every sample, use a high accuracy
        # transform from large draw from prior
#        dtY = fit(ZScoreTransform, Y) 
        StatsBase.transform!(dtY, Y)

        # ----- Minibatch training ---------------------------------------------
        for idx ∈ Iterators.partition(1:S, batchsize)
            Flux.reset!(m)
            # Extract batch, transform features to format for RNN
            Xb, Yb = tabular2rnn(X[:, idx]), Y[:, idx]
            # Compute loss and gradients
            ∇ = gradient(θ) do
                # don't use first, to warm up state
                m(Xb[1])
                err = [abs2.(Yb - m(x))  for x ∈ Xb[2:end]]
                sum(sum(err))/n
            end
            Flux.update!(opt, θ, ∇) # Take gradient descent step
        end
    end
    nothing
end
