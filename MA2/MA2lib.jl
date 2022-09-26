using Distributions
using Random

# use rejection sampling to stay inside 
# identified region
function PriorDraw()
    ok = false
    θ1 = 0.
    θ2 = 0.
    while !ok
        θ1 = 4. * rand() - 2.
        θ2 = 2. * rand() - 1.
        ok = InSupport([θ1,θ2])
    end
    [θ1, θ2]
end

function InSupport(θ)
    θ1, θ2 = θ
    all([
         (θ2+θ1 > -1.0), 
         (θ2-θ1 > -1.0), 
         (θ1>-2.0),
         (θ1 < 2.0),
         (θ2 > -1.0),
         (θ2 < 1.0)
        ])
end

# get a set of draws from prior
function PriorDraw(n)
    draws = zeros(2,n)
    for i = 1:n
        draws[:,i] .= PriorDraw()
    end
    draws
end

function ma2(θ, n)
    e = randn(n+2)
    e[3:end] .+ θ[1].*e[2:end-1] .+ θ[2].*e[1:end-2]
end

    
# generates S samples of length n
# returns are:
# x: 1 X S*n vector of data from EGARCH model
# y: 2 X S*n vector of parameters used to generate each sample
@views function dgp(n, S)
    y = PriorDraw(S)     # the parameters for each sample
    x = zeros(n, S)    # the Garch data for each sample
    for s = 1:S
        x[:,s] = ma2(y[:,s], n)
    end
    Float32.(x), Float32.(y)
end    


function Σ(n, θ)
    θ1, θ2, θ3 = θ 
    Σ = zeros(n,n)
    for i = 1:n
        Σ[i,i] = 1. + θ1^2. + θ2^2.
    end    
    for i = 1:n-1    
        Σ[i,i+1] = θ1+θ1*θ2
        Σ[i+1,i] = θ1+θ1*θ2
    end
    for i = 1:n-2    
        Σ[i, i+2] = θ2
        Σ[i+2,i] = θ2
    end
    Σ .* θ3
end

function lnL(θ, data)
    n = size(data,1)
    InSupport(θ) ? log(pdf(MvNormal(zeros(n), Σ(n,θ)), data))[1] : -Inf
end