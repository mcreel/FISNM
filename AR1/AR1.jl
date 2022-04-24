# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, Statistics
include("AR1lib.jl")
include("../FISNM.jl")

# train the model
T = 100 # observations in samples
S = 100 # number of samples in inner training loop
epochs = 10 # cycles through samples in inner training loop
datareps = 100 # number of runs through outer loop, where new samples are drawn
m = trainmodel(dgp, T, S, epochs, datareps)
Flux.reset!(m)

# save the model (need to add)


# check the fit
T = 100
S = 10000
x, y  = dgp(T, S)
X_rnn = batch_timeseries(x, T, T) |> gpu 
pred = [m(x) for x in X_rnn][end]
pred = vec(vcat(pred...))
# ols pred
pred_ols = zeros(S)
for s = 1:S
    xx = vec(x[:,s*T-T+1:s*T])
    yy = xx[2:end]
    xx = xx[1:end-1]
    xx\yy
    pred_ols[s]= xx\yy
end    
p = [pred pred_ols]
e = y' .- p
sqrt.(mean(abs2.(e), dims=1))
