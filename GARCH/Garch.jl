# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, CUDA, Statistics, ARCHModels, PrettyTables, Optim
using BSON: @save, @load
include("GarchLib.jl")
include("../RNNSNM.jl")

# run this to create models, if needed
# keep the definitions of the good models
function makemodels()
    model = "m1.bson"
    if !isfile(model)
        m = Chain(
            LSTM(1 =>20),
            Dense(20=>20, tanh),
            Dense(20=>5)
        ) 
        bestsofar = 1e6
        BSON.@save model  m bestsofar
    end
end

@views function main()
    # train model
    model = "m1.bson"
    @load model m bestsofar

    !isfile(model) ? makemodels() : nothing
    epochs = 10
    @load "data.bson" xtesting ytesting xtraining ytraining yinfo

    m |> gpu
    xtesting |> gpu
    ytesting |> gpu
    xtraining |> gpu
    ytraining |> gpu

    reps = 100
    for r = 1:reps
        # create batch
        T = 100 # length of sequences for training
        S = 64  # batch size
        start = rand(1:1000-T)
        stop = start+T
        x = xtesting[start:stop]
        ind = rand(1:10000,S) # indices of samples in batch
        x = [x[i][ind]' for i in 1: T]
        y = ytesting[:,ind] 
        
        # do the training with the batch
        epochs = 100
        learningrate = 0.0001
        @time trainmodel!(m, epochs, learningrate, x, y)

        @time current = loss(xtesting, ytesting)
        printstyled("Best loss so far: $bestsofar\n", color=:blue)
        println("current loss: $current")
        if current < bestsofar
            println("updating model with new best so far")
            bestsofar = copy(current)
            BSON.@save model m bestsofar
        end
    end    
end
