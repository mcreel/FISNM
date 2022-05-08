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
            Dense(1=>20, tanh),
            LSTM(20=>20),
            Dense(20, 50, tanh),
            Dense(50=>5)
        ) 
        bestsofar = 1e6
        BSON.@save model  m bestsofar
    end
end

function main(model)
    # train model
    epochs = 1
    T = 100
    S = 100
    @load "TestingData.bson" xtesting xinfo ytesting yinfo
    for i = 1:10000
    m = trainmodel(model, xtesting, xinfo, ytesting, yinfo, epochs, T, S)
    end
end
