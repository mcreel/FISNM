using Flux
using StatsBase

# Some helpers
tabular2rnn(X) = [X[i:i, :] for i ∈ 1:size(X, 1)]
rmse_loss(X, Y) = sqrt(mean(abs2.(X - Y)))
lstm_net(n_hidden, dropout_rate) = Chain(
    Dense(1, n_hidden, tanh),
    Dropout(dropout_rate),
    LSTM(n_hidden, n_hidden),
    Dropout(dropout_rate),
    LSTM(n_hidden, n_hidden),
    Dense(n_hidden, 5)
)
    
# Trains a recurrent neural network
function train_rnn!(
    m, opt, dgp, n, S; 
    epochs=100, batchsize=32, dev=cpu, loss=rmse_loss
)
    Flux.trainmode!(m) # In case we have dropout / batchnorm
    m = dev(m) # Pass model to device (cpu/gpu)
    θ = Flux.params(m) # Extract parameters
    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = dgp(n, S) # Generate a new batch
        # Standardize targets for MSE scaling
        dtY = fit(ZScoreTransform, Y) 
        StatsBase.transform!(dtY, Y)

        # ----- Minibatch training ---------------------------------------------
        for idx ∈ Iterators.partition(1:S, batchsize)
            Flux.reset!(m)
            # Extract batch, transform features to format for RNN
            Xb, Yb = tabular2rnn(X[:, idx]), Y[:, idx]
            # Compute loss and gradients
            ∇ = gradient(θ) do
                # Run model up to penultimate X
                [m(x) for x ∈ Xb[1:end-1]]
                # Compute loss
                loss(m(Xb[end]), Yb)
            end
            Flux.update!(opt, θ, ∇) # Take gradient descent step
        end
    end
    nothing
end
