using Flux, StatsBase
using BSON:@save
using BSON:@load

# Some helpers
tabular2rnn(X) = [view(X, i:i, :) for i ∈ 1:size(X, 1)]
tabular2conv(X) = permutedims(reshape(X, size(X)..., 1, 1), (4, 1, 3, 2))
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

lstm_net(n_hidden, k, g) = 
Chain(
      LSTM(g, n_hidden),
      Dense(n_hidden, k)
)

function train_rnn!(m, opt, dgp, n, datareps, batchsize, epochs)
    # size to test data sets
    testsize = 5000
    # create test data
    Xout, Yout  = dgp(n, testsize)
    Xout = batch_timeseries(Xout, n, n)
    #initialize tracking 
    bestsofar = 1.0e10
    bestmodel = m
    # train
    θ = Flux.params(m)
    for r = 1:datareps
        X, Y  = dgp(n, batchsize)
        X = batch_timeseries(X, n, n)
        for epoch ∈ 1:epochs
            Flux.reset!(m)
            ∇ = gradient(θ) do
                m(X[1]) # don't use first, to warm up state
                pred = [m(x) for x ∈ X[2:end]][end]
                mean(sqrt.(mean(abs2.(Y - pred),dims=2)))
                #mean((mean(abs.(Y - pred),dims=2)))
            end
            Flux.update!(opt, θ, ∇)
        end
        # periodically check fit to test set (unscaled labels)
        # and save model if good enough improvement
        if mod(r,50) == 0
            Flux.reset!(m)
            m(Xout[1])
            pred = [m(x) for x ∈ Xout[2:end]][end]
            current = mean(sqrt.(mean(abs2.(Yout - pred),dims=2)))
            if current < bestsofar
                bestsofar = current
                BSON.@save "bestmodel_$n.bson" m
                println("datarep: $r of $datareps")
                println("current best: $current")
            end
        end    
    end
    BSON.@load "bestmodel_$n.bson" m
end
