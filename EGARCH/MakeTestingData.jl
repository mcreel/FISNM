# this generates testing data and ML fit, rmse and bias
# the testing data is scaled and ready to batch after this
using Statistics, ARCHModels, Optim
using BSON: @save
include("../RNNSNM.jl")
include("EgarchLib.jl")
# make the testing data
T = 1000
S = 10000
S_actual = S + 500 # do more to allow for ML crashes
xtesting, ytesting = dgp(T, S_actual)
# ML for testing data
k = size(ytesting, 1)
θs_ml = zeros(S_actual,k)
for s = 1:S_actual
    y = Float64.(xtesting[1,s*T-T+1:s*T])
    try
        f = fit(EGARCH{1,1,1}, y; algorithm=LBFGS(;m=10), autodiff=:forward, meanspec=AR{1})
        θs_ml[s,:] = vcat(f.meanspec.coefs, f.spec.coefs)

    catch
        println("EGARCH ML crash")
    end    
end
# find the crashes
test = θs_ml[:,1] .!= 0.0
crashes = 100.0 - 100.0*sum(test)/S_actual
println("ML percentage crashes: $crashes")
# drop them from ML results, and remove those samples from NN data
θs_ml = θs_ml[test,:]
ytesting = ytesting[:,test]
test = Bool.(kron(test, ones(T))')
xtesting = xtesting[test]
# keep S samples
θs_ml = θs_ml[1:S,:]
ytesting = ytesting[:,1:S]
xtesting = xtesting[1:T*S]
xtesting = reshape(xtesting, 1, T*S) # 2d array
# ML results
e = ytesting' - θs_ml
mlbias = mean(e, dims=1)
mlrmse = sqrt.(mean(abs2.(e), dims=1))
@save "ML_results.bson" mlbias mlrmse crashes
# scale the testing data for NN
xtesting, xinfo = scaledata(xtesting)
ytesting, yinfo = scaledata(ytesting)
# batch it
xtesting = batch_timeseries(xtesting, T, T)
# save it
BSON.@save "TestingData.bson" xtesting xinfo ytesting yinfo
