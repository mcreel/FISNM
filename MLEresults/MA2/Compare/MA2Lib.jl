using Random

# Use rejection sampling to stay inside identified region
function priordraw(S::Int)
    θ = zeros(2, S)
    for i ∈ axes(θ, 2)
        ok = false
        θ1 = 0.
        θ2 = 0.
        while !ok
            θ1 = 4. * rand() - 2.
            θ2 = 2. * rand() - 1.
            ok = insupport([θ1, θ2])
        end
        θ[:, i] = [θ1, θ2]
    end
    θ
end

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
