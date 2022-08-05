using Flux
using StatsBase

# helpers
lstm_net(n_hidden) = Chain(
    Dense(6, n_hidden, leakyrelu),
    LSTM(n_hidden, n_hidden),
    LSTM(n_hidden, n_hidden),
    Dense(n_hidden, n_hidden, hardtanh),
    Dense(n_hidden, 5)
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
    epochs=100, batchsize=32, dev=cpu)
    Flux.trainmode!(m) # In case we have dropout / batchnorm
    m = dev(m) # Pass model to device (cpu/gpu)
    θ = Flux.params(m) # Extract parameters
    for r = 1:S
        X, Y  = dgp(n, batchsize)
        X = batch_timeseries(X, n, n)
        println("data loop $r  of $S") 
        for epoch ∈ 1:epochs
            Flux.reset!(m)
            ∇ = gradient(θ) do 
                # don't use first, to warm up state
                m(X[1])
                err = [abs2.(Y - m(x))  for x ∈ X[2:end]]
                sqrt(sum(sum(err))/n)
            end
            Flux.update!(opt, θ, ∇) # Take gradient descent step
        end
    end
    nothing
end
