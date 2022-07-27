using BSON: @load
using Flux, PrettyTables, Statistics
@load "TestingData.bson" xtesting xinfo ytesting yinfo
@load "ML_results.bson" mlrmse
@load "m1.bson" m
# NN preds
pred = [m(x) for x in xtesting][end]
e = yinfo[2] .* (ytesting .- pred)
rmse = sqrt.(mean(abs2.(e), dims=2))
pretty_table([rmse mlrmse'], header = ["NN rmse", "ML rmse"])

