# a draw from prior
function PriorDraw()
    randn(3)
end

# get a set of draws from prior
function PriorDraw(n)
    draws = zeros(3,n)
    for i = 1:n
        draws[:,i] .= PriorDraw()
    end
    draws
end 

function LogitLikelihood(θ, data)
    x = data[1:end-1,:]'
    y = data[end,:]
    p = 1.0./(1.0 .+ exp.(-x*θ))
    mean(y.*log.(p) .+ (log.(1.0 .- p)).*(1.0 .- y))
end

# generates a matrix of size k+1 X n from the logit model with parameter θ
# each column is a vector of regressors, plus the 0/1 outcome in last row
@views function Logit(θ, n)
    k = size(θ,1)
    data = randn(k+1,n)
    data[k+1,:] = rand(1,n) .< 1.0 ./(1. .+ exp.(-θ'*data[1:k,:]))
    data
end    

# generates S samples of length n
# number of parameters is k
# returns are:
# x: (k+1)XSn vector of data from logit model
# y: kXSn vector of parameters used to generate each sample
@views function dgp(n, S)
    k = 3 # number of regressors in logit model
    x = zeros(k+1, n*S) # the samples, n obs in each
    y = randn(k, S)     # the parameters, prior is Gaussian N(0,1) for each
    for s = 1:S
        x[:,s*n-n+1:s*n] = Logit(y[:,s],n)  
    end
    Float32.(x), Float32.(y)
end    
