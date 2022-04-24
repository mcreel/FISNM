# generates a matrix of size k+1 X n from the logit model with parameter θ
# each column is a vector of regressors, plus the 0/1 outcome in last row
@views function logit(θ, n)
    k = size(θ,1)
    data = randn(k+1,n)
    data[k+1,:] = rand(1,n) .< 1.0 ./(1. .+ exp.(-θ'*data[1:k,:]))
    data
end    

# generates S samples of length n
# returns are:
# x: 1XSn vector of data from logit model
# y: 1XSn vector of parameters used to generate each sample
@views function dgp(n, S)
    k = 3 # number of regressors in logit model
    x = zeros(k+1, n*S) # the samples, n obs in each
    y = randn(k, S)     # the parameters used to generate samples, drawn from Gausssian prior
    for s = 1:S
        x[:,s*n-n+1:s*n] = logit(y[1:k,s],n)  
    end
    Float32.(x), Float32.(y)
end    


