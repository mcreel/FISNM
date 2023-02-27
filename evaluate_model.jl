using BSON: @load
using Flux
using MLUtils: flatten
using PrettyTables
using Random
using StatsBase

include("DGPs/DGPs.jl")
include("DGPs/GARCH.jl")
include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

# Only works with TCNs! (also only necessary for them)
function replace_flatten(bson_file)
    d = BSON.parse(bson_file)
    for k ∈ [:model, :best_model]
        d[k][:type][:params][1][:params][3][:name][1] = "MLUtils"
        d[k][:data][1][:data][3][:type][:name][1] = "MLUtils"
    end
    new_file = replace(bson_file, ".bson" => "_fix.bson")
    bson(new_file, d)
    new_file
end

model_path = "../FISNM/models/GARCH/23-02-11_(n-100).bson"
# model_path = "../FISNM/models/MA2/lstm_23-02-19_(n-100).bson"

model_path = replace_flatten(model_path)

@load model_path best_model

transform_seed = 1204
transform_size = 100_000
test_seed = 78
test_size = 5_000
dev = cpu


Random.seed!(transform_seed)
dgp = GARCH(N=100)
dtY = data_transform(dgp, transform_size, dev=dev)

X, Y = generate(dgp, transform_size, dev=dev)
StatsBase.transform!(dtY, Y)
X = tabular2conv(X)

debiased_model = bias_corrected_tcn(best_model, X, Y)

Random.seed!(test_seed)
X, Y = generate(dgp, test_size, dev=dev)
X = tabular2conv(X)
Flux.testmode!(debiased_model)
Ŷ = StatsBase.reconstruct(dtY, debiased_model(X))

err = Y - Ŷ
amae = mean(mean(abs.(err), dims=2))
armse = mean(sqrt.(mean(abs2.(err), dims=2)))
aabias = mean(abs, mean(err))