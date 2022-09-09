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
        ok = (θ1+θ2 > -1.) & (θ1-θ2 > -1.)
    end
    [θ1, θ2]
end

function ma2(θ, n)
    e = randn(n+2)
    e[3:end] .+ θ[1].*e[2:end-1] .+ θ[2].*e[1:end-2]
end


