# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, CUDA, Statistics, ARCHModels, PrettyTables, Optim
using BSON: @save, @load
include("EgarchLib.jl")
include("../RNNSNM.jl")


function main()
model = "3_16.bson"

# make model, if needed
if !isfile(model)
    layers = 3
    nodesperlayer = 16
    m = makemodel(1, 6, layers, nodesperlayer)
    bestsofar = 1e6
    BSON.@save model  m bestsofar
end

# train model
epochs = 10
T = 100
S = 50
learningrate = 0.001
@load "TestingData.bson" xtesting xinfo ytesting yinfo
for i = 1:10000
m = trainmodel(model, xtesting, xinfo, ytesting, yinfo, epochs, learningrate, T, S)
end
end
