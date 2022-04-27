# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, CUDA, Statistics, Optim, PrettyTables
using BSON: @save, @load
include("LogitLib.jl")
include("../FISNM.jl")

# train the model
T = 100 # observations in samples
S = 100 # number of samples in inner training loop
epochs = 10 # cycles through samples in inner training loop
datareps = 10000 # number of runs through outer loop, where new samples are drawn
nodesperlayer = 64
layers = 2

m = trainmodel(dgp, T, S, datareps, nodesperlayer, layers, epochs)

# NN preds
Flux.reset!(m) # need to reset state in case batch size changes
T = 100
S = 10000
x, y  = dgp(T, S)
X_rnn = batch_timeseries(x, T, T)
pred = [m(x) for x in X_rnn][end]
θs = Float64.(y')
θhats = Float64.(pred')

# ML pred
# logit average log likelihood function
function logitlikelihood(theta, y, x)
    p = 1.0./(1.0 .+ exp.(-x*theta))
    mean(y.*log.(p) .+ (log.(1.0 .- p)).*(1.0 .- y))
end
k = size(θs,2)
θs_ml = zeros(S,k)
for s = 1:S
    xx = Float64.(x[1:k,s*T-T+1:s*T]')
    yy = Float64.(x[k+1,s*T-T+1:s*T])
    obj = θ -> -logitlikelihood(θ, yy, xx) 
    θs_ml[s,:]= Optim.optimize(obj, y[:,s] , LBFGS(), Optim.Options(g_tol = 1e-5,x_tol = 1e-6,f_tol=1e-8); autodiff=:forward).minimizer
end

d = [θs θhats θs_ml]
display(cor(d))
e = θs - θhats
nbias = mean(e, dims=1)
nrmse = sqrt.(mean(abs2.(e), dims=1))
e = θs - θs_ml
mlbias = mean(e, dims=1)
mlrmse = sqrt.(mean(abs2.(e), dims=1))
header = ["NN bias", "ML bias", "NN rmse", "ML rmse"]
pretty_table([nbias' mlbias' nrmse' mlrmse'], header)

@save "$layers"*"_"*"$nodesperlayer"*"_"*"$datareps"*"_"*"$epochs.bson" m nrmse mlrmse

