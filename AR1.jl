# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, Statistics
include("AR1lib.jl")
include("utilities.jl")
@views function main()
# Create a recurrent model
# Have experimented with RNN, GRU, and
# including a nonlinear dense layer in middle.
# This seems more or less optimal
m = Chain(LSTM(1=>16), Dense(16=>1)) |> gpu 
T, S = 100, 100 # T observations, S samples (each in a batch)
epochs, DataRenewalReps = 20, 100
for r = 1:DataRenewalReps
    x, y  = dgp(T, S)
    X_rnn = batch_timeseries(x, T, T) |> gpu 
    θ = Flux.params(m)
    opt = ADAM(0.01)
    for epoch ∈ 1:epochs
        Flux.reset!(m)
        ∇ = gradient(θ) do 
            Flux.Losses.mse.([m(x) for x ∈ X_rnn][end], y) |> mean
        end
        Flux.update!(opt, θ, ∇)
    end
end
m |> cpu
return m
end


m = main()
Flux.reset!(m)
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
