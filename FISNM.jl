using Flux, CUDA

function trainmodel(dgp, T, S, datareps, nodesperlayer, layers, epochs)
    # the rnn. This is fairly good. Have tried GRU (similar) RNN (worse) and more layers (no help)
    x, y = dgp(T,S)
    k = size(x,1) # number of exogs (features)
    g = size(y,1) # number of endogs(outputs)
    if layers == 2
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>g)
                 ) 
    elseif layers == 3
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  LSTM(nodesperlayer=>nodesperlayer),
                  Dense(nodesperlayer=>g)
                 )
    else
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  LSTM(nodesperlayer=>nodesperlayer),
                  LSTM(nodesperlayer=>nodesperlayer),
                  Dense(nodesperlayer=>g)
                 )
    end
    m |> gpu
    #println("CUDA memory use: ", CUDA.used_memory()) 
    θ = Flux.params(m)
    opt = ADAM(0.01)
    for r = 1:datareps
        x, y  = dgp(T, S)
        X_rnn = batch_timeseries(x, T, T)
        X_rnn |> gpu 
        y |> gpu
        println("data loop $r  of $datareps") 
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


