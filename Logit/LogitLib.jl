# a draw from prior
function PriorDraw()
    randn(5)
end

# get a set of draws from prior
function PriorDraw(n)
    draws = zeros(5,n)
    for i = 1:n
        draws[:,i] .= PriorDraw()
    end
    draws
end 
# draw a sample from Logit model
# generates a matrix of size k+1 X n from the logit model with parameter θ
# each column is a vector of regressors, plus the 0/1 outcome in last row
@views function SimulateLogit(θ, n)
    k = size(θ,1)
    x = randn(n,k)
    y = rand(n) .< 1.0 ./(1. .+ exp.(-x*θ))
    data = [x y]'
end    

function LogitLikelihood(θ, data)
    x = data[1:end-1,:]'
    y = data[end,:]
    p = 1.0./(1.0 .+ exp.(-x*θ))
    mean(y.*log.(p) .+ (log.(1.0 .- p)).*(1.0 .- y))
end


# generates S samples of length n
# returns are:
# x: 1XSn vector of data from logit model
# y: 1XSn vector of parameters used to generate each sample
@views function dgp(n, S)
    k = 5 # number of regressors in logit model
    x = zeros(k+1, n*S) # the samples, n obs in each
    y = randn(k, S)     # the parameters used to generate samples, drawn from Gausssian prior
    for s = 1:S
        x[:,s*n-n+1:s*n] = SimulateLogit(y[:,s],n)  
    end
    Float32.(x), Float32.(y)
end    

