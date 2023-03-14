using Plots
include("JDlib.jl")

function main()
    for i = 1:100
    θ = PriorDraw()
    burnin=100
    data = JDmodel(θ, burnin, rand(1:Int64(1e10)))
    display(plot(data))
    sleep(0.5)
    end
end    
