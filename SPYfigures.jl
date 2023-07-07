using Pkg
Pkg.activate(".")
using StatsPlots, CSV, DataFrames, Dates, DataFramesMeta

## make a single data frame
spy1 = CSV.read("spy.csv", DataFrame)
spy2 = CSV.read("spy16-19.csv", DataFrame)
@show describe(spy1)
@show describe(spy2)
s2 = @subset(spy2, :date .> Date("2017-12-04"))
data = vcat(spy1, s2)

## returns
p1 = plot(data.date, data.rets, tickfontsize=5, legend=true, label=nothing, legendfontsize=6)
vline!([Date("2016-01-11")], color=:green, width=5, alpha=0.2, label = "start date of sample 2", yformatter=:plain)
vline!([Date("2017-12-04")], color=:blue, width=5, alpha=0.2, label = "end date of sample 1", yformatter=:plain)
savefig("rets.svg")
## RV and BV
p2 = plot(data.date, data.rv, tickfontsize=5, legend=true, label="rv", legendfontsize=6)
plot!(data.date, data.bv, tickfontsize=5, legend=true, label="bv", legendfontsize=6)
vline!([Date("2016-01-11")], color=:green, width=5, alpha=0.2, label = "start date of sample 2", yformatter=:plain)
vline!([Date("2017-12-04")], color=:blue, width=5, alpha=0.2, label = "end date of sample 1", yformatter=:plain)
savefig("vol.svg")

