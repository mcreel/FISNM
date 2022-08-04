using Flux
using StatsBase

# Some helpers
tabular2rnn(X) = [X[i:i, :] for i ∈ 1:size(X, 1)]
rmse_loss(X, Y) = sqrt(mean(abs2.(X - Y)))
lstm_net(n_hidden, dropout_rate) = Chain(
    Dense(1, n_hidden, tanh),
    LSTM(n_hidden, n_hidden),
    LSTM(n_hidden, n_hidden),
    Dense(n_hidden, 3)
)

# Create batches of a time series
# The data going in is expected as kXT.
# In the simple case of s=t, the output is T/s batches of size s, ready
# to supply to the rnn. Thanks to Jonathan Chassot for this, see
# https://www.jldc.ch/posts/batching-time-series-in-flux/
function batch_timeseries(X, s::Int, r::Int)
    if isa(X, AbstractVector) # If X is passed in format T×1, reshape it
        X = permutedims(X)
    end
    T = size(X, 2)
    @assert s ≤ T "s cannot be longer than the total series"
    if s == r   # Non-overlapping case, each batch has unique elements
        X = X[:, (T % s)+1:end]         # Ensure uniform sequence lengths
        T = size(X, 2)                  # Re-store series length
       [X[:, t:s:T] for t ∈ 1:s] # Output
    else        # Overlapping case
        X = X[:, ((T - s) % r)+1:end]   # Ensure uniform sequence lengths
        T = size(X, 2)                  # Re-store series length
        [X[:, t:r:T-s+t] for t ∈ 1:s] # Output
    end
end


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
