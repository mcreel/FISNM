using Flux, CUDA, BSON, Statistics, PrettyTables

# scale data, and save scaling info.
# the output goes into the struct
function scaledata(x, info=1)
    if info == 1
        s = std(x,dims=2)
        m = mean(x, dims=2)
        info = (m, s)
    end
    m = info[1]
    s = info[2]
    x = (x .- info[1]) ./ info[2]
    return x, info
end

function unscaledata(x, info)
    m = info[1]
    s = info[2]
    x .* info[2] .+ info[1]
end  

function loss(X,y)
    Flux.reset!(m)
    Flux.Losses.mse(y, [m(x) for x ∈ X][end])
end    

# takes training data and trains model for some epochs
# writes out model if new best is found
# model is a BSON file containing 
# m, the model configuration
# bestsofar, the best loss value so far
function trainmodel!(m, epochs, learningrate, xtrain, ytrain)
    θ = Flux.params(m)
    opt = ADAM(learningrate)
    for epoch ∈ 1:epochs
        Flux.reset!(m)
        ∇ = gradient(θ) do 
            loss(xtrain, ytrain)
        end
        Flux.update!(opt, θ, ∇)
    end   
    nothing
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

