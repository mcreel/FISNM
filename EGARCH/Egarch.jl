# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, CUDA, Statistics, ARCHModels, PrettyTables, Optim
using BSON: @save, @load
include("EgarchLib.jl")
include("../FISNM.jl")

# make the testing data
T = 1000
S = 10000
S_actual = S + 100 # do more to allow for ML crashes
testingData, testingθs  = makedata(dgp, T, S_actual)
# ML for testing data
k = size(testingθs, 1)
θs_ml = zeros(S_actual,k)
for s = 1:S_actual
    y = Float64.(testingData[1,s*T-T+1:s*T])
    try
        f = fit(EGARCH{1,1,1}, y; algorithm=LBFGS(;m=10), autodiff=:forward, meanspec=AR{1})
        θs_ml[s,:] = vcat(f.meanspec.coefs, f.spec.coefs)

    catch
        println("EGARCH ML crash")
    end    
end
test = θs_ml[:,1] .!= 0.0

crashes = 100.0 - 100.0*sum(test)/S_actual
println("ML percentage crashes: $crashes")
# drop them from ML results, and remove those samples from NN data
θs_ml = θs_ml[test,:]
testingθs = testingθs[:,test]
test = Bool.(kron(test, ones(T))')
testingData = testingData[test]
# keep S samples
θs_ml = θs_ml[1:S,:]
testingθs = testingθs[:,1:S]
testingData = testingData[1:T*S]
testingData = reshape(testingData, 1:T*S) # 2d array

e = testingθs' - θs_ml
mlbias = mean(e, dims=1)
mlrmse = sqrt.(mean(abs2.(e), dims=1))

# scale the testing data for NN
xtesting, xinfo = prepdata(x)
ytesting, yinfo = prepdata(y)
#=
function main()
# train the model
T = 1000 # observations in samples
S = 200 # number of samples in inner training loop
epochs = 10 # cycles through samples in inner training loop
datareps = 10000 # upper limit on number of runs through outer loop, where new samples are drawn
nodesperlayer = 16
layers = 2

m = trainmodel(dgp, T, S, datareps, nodesperlayer, layers, epochs)

# NN preds
Flux.reset!(m) # need to reset state in case batch size changes
T = 1000 # same as S&P data
S = 10000
x, y  = dgp(T, S)
X_rnn = batch_timeseries(x, T, T)
pred = [m(x) for x in X_rnn][end]
θs = Float64.(y')
θhats = Float64.(pred')

header = ["NN bias", "ML bias", "NN rmse", "ML rmse"]
pretty_table([nbias' mlbias' nrmse' mlrmse'], header)

@save "$layers"*"_"*"$nodesperlayer"*"_"*"$datareps"*"_"*"$epochs.bson" m nrmse mlrmse

end
main()
=#
