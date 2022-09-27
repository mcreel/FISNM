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
# formatted for RNN
@views function dgp(n, S)
    y = PriorDraw(S)     # the parameters for each sample
    x = zeros(1, n*S)    # the Garch data for each sample
    for s = 1:S
        x[:,n*s-n+1:s*n] = ma2(y[:,s], n)
    end
    x = [Float32.(x[:, t:n:n*S]) for t ∈ 1:n]
    x, Float32.(y)
end
