using Plots
include("GarchLib.jl")
prmse = std(PriorDraw(100_000), dims=2)
prmse = vcat(prmse, mean(prmse))
prmse = repeat(prmse, 1, 6)
plot(1:6, prmse')


