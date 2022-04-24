# this generates data and trains for some epochs,
# and then loops. That way, the size of the data
# on the gpu is reduced.
using Flux, Statistics, Optim
include("Logitlib.jl")
include("../FISNM.jl")

# train the model
T = 100 # observations in samples
S = 100 # number of samples in inner training loop
epochs = 10 # cycles through samples in inner training loop
datareps = 1000 # number of runs through outer loop, where new samples are drawn
m = trainmodel(dgp, T, S, epochs, datareps)
Flux.reset!(m)

# save the model (need to add)

# check the fit
T = 100
S = 1000
x, y  = dgp(T, S)
X_rnn = batch_timeseries(x, T, T) |> gpu 
pred = [m(x) for x in X_rnn][end]

θs = y'
θhats = pred'

# logit average log likelihood function
function logitlikelihood(theta, y, x)
    p = 1.0./(1.0 .+ exp.(-x*theta))
    mean(y.*log.(p) .+ (log.(1.0 .- p)).*(1.0 .- y))
end

# ML pred
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
e = [θhats .- θs θs_ml .- θs]
println("RMSEs: NN first, then ML")
sqrt.(mean(abs2.(e), dims=1))

