# K is the number of regressions in logit model
@Base.kwdef struct Logit <: DGP 
    K::Int = 3
    N::Int
end

# ----- Model-specific utilities -----------------------------------------------
# Generates a matrix of size k+1 X n from the logit model with parameter θ
# each column is a vector of regressors, plus the 0/1 outcome in last row
@views function simulate_logit(θ, n)
    k = size(θ,1)
    data = Float32[rand(1,n); randn(2,n); zeros(1,n)]
    data[k+1,:] = rand(1,n) .< 1.0 ./(1. .+ exp.(-θ'*data[1:k,:]))
    data
end    


# ----- DGP necessary functions ------------------------------------------------

# Prior is Gaussian N(0,1) for each parameter
priordraw(d::Logit, S::Int) = randn(Float32, d.K, S)

# Generate S samples of length N with K features and P parameters
# Returns are: (K × S × N), (P × S)
@views function generate(d::Logit, S::Int)
    y = priordraw(d, S)
    x = zeros(Float32, d.K+1, S, d.N)

    Threads.@threads for s ∈ axes(x, 2)
        x[:, s, :] = simulate_logit(y[:, s], d.N)  
    end
    
    x, y
end

nfeatures(d::Logit) = d.K + 1
nparams(d::Logit) = d.K

priorerror(d::Logit) = fill(√(2 / π), d.K)

@views function likelihood(::Logit, X, θ)
    x = X[1:end-1, :]
    y = X[end, :]
    p = 1 ./ (1 .+ exp.(-x'θ))
    mean(y .* log.(p) .+ (log.(1 .- p)) .* (1 .- y))
end