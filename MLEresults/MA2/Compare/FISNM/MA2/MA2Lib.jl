using Distributions
using Random

function ma2(θ, n)
    e = randn(n+2)
    e[3:end] .+ θ[1].*e[2:end-1] .+ θ[2].*e[1:end-2]
end

function insupport(θ)
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