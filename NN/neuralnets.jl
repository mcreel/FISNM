using Flux, StatsBase
using BSON:@save
using BSON:@load

# Some helpers
tabular2rnn(X) = [view(X, i:i, :) for i ∈ 1:size(X, 1)]
tabular2conv(X) = permutedims(reshape(X, size(X)..., 1, 1), (4, 1, 3, 2))

lstm_net(n_hidden, k, g) = 
Chain(
      LSTM(g, n_hidden),
      Dense(n_hidden, k)
)

function train_rnn!(m, opt, dgp, n, datareps, batchsize, epochs, thisrun)
    # size to test data sets
    testsize = 5000
    # create test data
    Xout, Yout  = dgp(n, testsize)
    #initialize tracking 
    bestsofar = 1.0e10
    bestmodel = m
    # train
    θ = Flux.params(m)
    for r = 1:datareps
        X, Y  = dgp(n, batchsize)
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
                BSON.@save "bestmodel_$thisrun$n.bson" m
                println("datarep: $r of $datareps")
                println("current best: $current")
            end
        end    
    end
end
