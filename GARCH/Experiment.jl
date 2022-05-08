using Flux, CUDA
include("EgarchLib.jl")
include("../RNNSNM.jl")

function trainmodel!(m, xtrain, ytrain, epochs)
    θ = Flux.params(m)
    opt = ADAM()
    function loss(X,y)
        Flux.reset!(m)
        Flux.Losses.mse(y, [m(x) for x ∈ X][end])
    end    
    for epoch ∈ 1:epochs
      #  Flux.reset!(m)
        ∇ = gradient(θ) do 
            loss(xtrain, ytrain)
        end
        Flux.update!(opt, θ, ∇)
    end   
end

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

# make the testing data
T = 1000
S = 100
x, y = dgp(T, S)
x, xinfo = scaledata(x)
y, yinfo = scaledata(y)
x = batch_timeseries(x, T, T)

m = Chain(LSTM(1,20), Dense(20,10, tanh), Dense(10,5))

x |> gpu
@show CUDA.used_memory()
y |> gpu
@show CUDA.used_memory()
m |> gpu
@show CUDA.used_memory()

# train model
epochs = 100
@time trainmodel!(m, x, y, epochs)

Flux.reset!(m)
pred = [m(xx) for xx in x][end]
