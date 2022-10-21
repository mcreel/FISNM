function LogitLikelihood(θ, data)
    x = data[1:end-1,:]'
    y = data[end,:]
    p = 1.0./(1.0 .+ exp.(-x*θ))
    mean(y.*log.(p) .+ (log.(1.0 .- p)).*(1.0 .- y))
end

# generates a matrix of size k+1 X n from the logit model with parameter θ
# each column is a vector of regressors, plus the 0/1 outcome in last row
@views function logit(θ, n)
    k = size(θ,1)
    data = [rand(1,n); randn(2,n); zeros(1,n)]
    data[k+1,:] = rand(1,n) .< 1.0 ./(1. .+ exp.(-θ'*data[1:k,:]))
    data
end    

