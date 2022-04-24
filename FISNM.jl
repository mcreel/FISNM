using Flux

function trainmodel(dgp, T, S, epochs, datareps)
    # the rnn. This is fairly good. Have tried GRU (similar) RNN (worse) and more layers (no help)
    x, y = dgp(T,1)
    k = size(x,1) # number of exogs (features)
    g = size(y,1) # number of endogs(outputs)
    m = Chain(LSTM(k=>16), Dense(16=>g)) |> gpu 
    for r = 1:datareps
        x, y  = dgp(T, S)
        X_rnn = batch_timeseries(x, T, T) |> gpu 
        θ = Flux.params(m)
        opt = ADAM(0.01)
        for epoch ∈ 1:epochs
            Flux.reset!(m)
            ∇ = gradient(θ) do 
                Flux.Losses.mse.([m(x) for x ∈ X_rnn][end], y) |> mean
            end
            Flux.update!(opt, θ, ∇)
        end
    end
    m |> cpu
    return m
end


# Create batches of a time series
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


