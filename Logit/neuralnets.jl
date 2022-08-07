using Flux, StatsBase
using BSON:@save
using BSON:@load

# helpers
lstm_net(n_hidden) = 
Chain(
      LSTM(4, n_hidden),
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

function train_rnn!(m, opt, dgp, n, datareps, batchsize, epochs, dtY)
    # create test data
    testsize = 1000
    Xout, Yout  = dgp(n, testsize)
    Xout = batch_timeseries(Xout, n, n)
    StatsBase.transform!(dtY, Yout)
    # initialize tracking 
    bestsofar = 1.0e10
    bestmodel = m
    # train
    θ = Flux.params(m)
    for r = 1:datareps
        X, Y  = dgp(n, batchsize)
        X = batch_timeseries(X, n, n)
        StatsBase.transform!(dtY, Y)
        for epoch ∈ 1:epochs
            Flux.reset!(m)
            ∇ = gradient(θ) do 
                # don't use first, to warm up state
                m(X[1])
                pred = [m(x) for x ∈ X[2:end]][end]
                mean(sqrt.(mean([abs2.(Y - pred)])))
            end
            Flux.update!(opt, θ, ∇)
        end
        Flux.reset!(m)
        m(Xout[1])
        pred = [m(x) for x ∈ Xout[2:end]][end]
        current = mean(sqrt.(mean([abs2.(Yout - pred)])))
        if current < bestsofar
            bestsofar = current
            BSON.@save "bestmodel_$n.bson" m
            println("datarep: $r of $datareps")
            println("current best: $current")
        end
    end
    BSON.@load "bestmodel_$n.bson" m
end


