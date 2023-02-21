include("JDlib.jl")

θ = PriorDraw()
burnin=100
data = JDmodel(θ, burnin, rand(1:Int64(1e10)))
