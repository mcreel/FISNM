using BSON
using CSV
using DataFrames
using DifferentialEquations
using Flux
using LinearAlgebra
using Random
using StatsBase
using StatsPlots

include("DGPs/DGPs.jl")
include("DGPs/JD.jl")
include("NeuralNets/utils.jl")
include("NeuralNets/tcn_utils.jl")

modelspecs = [
    (name="100-50", max_ret=100, max_rv=50, max_bv=50),
    (name="100-50-drop", max_ret=100, max_rv=50, max_bv=50),
    (name="30-20", max_ret=30, max_rv=20, max_bv=20),    
    (name="30-20-drop", max_ret=30, max_rv=20, max_bv=20),
]

dataspecs = [
    (name="SPX", filename="sp500.csv"),
    (name="SPY", filename="spy.csv"),
]

function get_stuff(spec)
    # TCN
    tcn = BSON.load("models/JD/best_model_$(spec.name).bson")[:best_model];
    Flux.testmode!(tcn);
    
    # Preprocess function
    BSON.@load "statistics_$(spec.name).bson" μs σs

    @views function preprocess(x)
        x[:, :, 2:3, :] = log1p.(x[:, :, 2:3, :]) # Log RV and BV
        (x .- μs) ./ σs
    end

    tcn, preprocess
end

mcreps = 1000
# Define arrays to hold results
θs = zeros(8, length(modelspecs), length(dataspecs))
rets = zeros(mcreps, length(modelspecs), length(dataspecs))
rvs = zeros(mcreps, length(modelspecs), length(dataspecs))
bvs = zeros(mcreps, length(modelspecs), length(dataspecs))


transform_seed = 1204
burnin = 100 # Burn-in steps
transform_size = 100_000
# Define DGP and make transform
dgp = JD(1000)
Random.seed!(transform_seed)
pd = priordraw(dgp, transform_size)
pd[6, :] .= max.(pd[6, :], 0)
pd[8, :] .= max.(pd[8, :], 0)
dtθ = fit(ZScoreTransform, pd)

dfs = [CSV.read(spec.filename, DataFrame) for spec in dataspecs]

for (i, modelspec) ∈ enumerate(modelspecs)
    tcn, pp = get_stuff(modelspec)
    for (j, dataspec) ∈ enumerate(dataspecs)
        @info "Model $(modelspec.name) on $(dataspec.name)"
        X₀ = Float32.(Matrix(dfs[j][!, [:rets, :rv, :bv]])) |>
            x -> reshape(x, size(x)..., 1) |>
            x -> permutedims(x, (2, 3, 1)) |> 
            tabular2conv |> pp
        θ̂ = Float64.((tcn(X₀) |> m -> mean(StatsBase.reconstruct(dtθ, m), dims=2) |> vec))
        θs[:, i, j] = θ̂
        # Generate data
        d = mean(generate(Float32.(θ̂), dgp, mcreps), dims=3)
        rets[:, i, j] = d[1, :, 1]
        rvs[:, i, j] = d[2, :, 1]
        bvs[:, i, j] = d[3, :, 1]
    end
end


plots = []
params = ["μ", "κ", "α", "σ", "ρ", "λ₀", "λ₁", "τ"]

xlab = repeat(["$(spec.name)" for spec ∈ modelspecs], outer=length(dataspecs))
clab = repeat(["$(dataspec.name)" for dataspec ∈ dataspecs], inner=length(modelspecs))
for i ∈ eachindex(params)
    p = groupedbar(xlab, θs[i, :, :], group = clab, bar_width=0.5, title=params[i], 
        lab="")
    hline!(p, [0], color=:black, linestyle=:dash, label="")
    push!(plots, p)
end


plot(plots..., layout=(4, 2), size=(600, 800))
savefig("figures/parameters.png")



# Plot histograms of rets / rvs / bvs
plots = []

symbs = [:rets, :rv, :bv]
for (k, dat) in enumerate([rets, rvs, bvs])
plots = []
for (j, dataspec) ∈ enumerate(dataspecs)
    subplots = []
    color = get_color_palette(:auto, 1)[j]
    for (i, modelspec) ∈ enumerate(modelspecs)
        p = histogram(dat[:, i, j], title=modelspec.name,
            alpha=0.5, normed=true, color=color, lab="", bins=30)
        vline!(p, [mean(dfs[j][!, symbs[k]])], color=:black, linestyle=:dash, 
            label="", lw=3)
        push!(subplots, p)
    end
    push!(plots, plot(subplots..., layout=(length(modelspecs), 1)))
end
plot(plots..., layout=(1, length(dataspecs)), size=(1200, 800))
savefig("figures/$(symbs[k]).png")
end