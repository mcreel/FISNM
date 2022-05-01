using Flux, CUDA, BSON

# a structure for holding scales testing data, 
# and information to scale new data, and to unscale
# predictions
struct Data
    x::Array{Float32}
    y::Array{Float32}
    xinfo::Array{Float32}
    yinfo::Array{Float32}
end

function makedata(dgp, T, S)
    x, y = dgp(T, S)
end

# scale data, and save scaling info.
# the output goes into the struct
function prepdata(x, info=1)
    if info == 1
        s = std(x,dims=2)
        m = mean(x, dims=2)
        info = (m, s)
    end
    m = info[1]
    s = info[2]
    x = (x .- info[1]) ./ info[2]
    x, info
end

function unscale(data::Data)
end  

function makemodel(layers, nodesperlayer)
    # make the model
    if layers == 2
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>g)
                 ) 
    elseif layers == 3
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>nodesperlayer, tanh),
                  Dense(nodesperlayer=>g)
                 )
    else
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>nodesperlayer, tanh),
                  Dense(nodesperlayer=>nodesperlayer, tanh),
                  Dense(nodesperlayer=>g)
                 )
    end
    m
end


function saveall(name, m, data::Data)
    BSON.@save name, m, xtesting, xinfo, ytesting, yinfo
end    

function loadall(name, m, data::Data)
    BSON.@load name, m, data
end

function predict(data::Data, m)
end

# take a model, trained or untrained, and data, and train it
function trainmodel(model, bestsofar, data, datareps, epochs)

end

#=
function trainmodel(dgp, T, S, datareps, nodesperlayer, layers, epochs)
    # testing data
    function prep!(x, xm, xs)
        x = (x .- xm) ./ xs

    end    
    X_rnn_testing = batch_timeseries(xtesting, T, T)
    X_rnn_testing |> gpu
    ytesting |> gpu
    k = size(xtesting,1) # number of exogs (features)
    g = size(ytesting,1) # number of endogs(outputs)
    # make the model
    if layers == 2
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>g)
                 ) 
    elseif layers == 3
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>nodesperlayer, tanh),
                  Dense(nodesperlayer=>g)
                 )
    else
        m = Chain(
                  LSTM(k=>nodesperlayer),
                  Dense(nodesperlayer=>nodesperlayer, tanh),
                  Dense(nodesperlayer=>nodesperlayer, tanh),
                  Dense(nodesperlayer=>g)
                 )
    end
    m |> gpu
    θ = Flux.params(m)
    opt = ADAM()
    function loss(X,y)
        Flux.reset!(m)
        sqrt.(mean(abs2.(y-[m(x) for x ∈ X][end])))
    end    
    bestsofar = 1e6
    bestmodel = deepcopy(m)
    timesgreater = 0
    noimprovement = 100
    # train until there's too long a period of no improvement
    current = loss(X_rnn_testing, ytesting)
    print("Initial RMSE: ")
    printstyled("$current\n", color=:blue)
    for r = 1:datareps
        x, y  = dgp(T, S)
        prep!(x, xm, xs)
        prep!(y, ym, ys)
        #prep!(y, ymin, ymax)
        X_rnn = batch_timeseries(x, T, T)
        X_rnn |> gpu 
        y |> gpu
        for epoch ∈ 1:epochs
            Flux.reset!(m)
            ∇ = gradient(θ) do 
                loss(X_rnn, y)
            end
            Flux.update!(opt, θ, ∇)
        end
        current = loss(X_rnn_testing, ytesting)
        if current < bestsofar
            bestsofar = current
            timesgreater = 0
            bestmodel = deepcopy(m)
        else
            timesgreater +=1
        end
        if timesgreater > noimprovement
            break
        end
        if current == bestsofar
            print("data loop $r  of $datareps, current RMSE: ")
            printstyled("$current\n", color=:green)
        else    
            println("data loop $r  of $datareps, current RMSE: $current")
        end    
    end
    m = deepcopy(bestmodel)
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

=#
