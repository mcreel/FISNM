using BSON
using OnlineStats
using Random
using StatsBase
datapath = "data"
files = readdir(datapath)

Random.seed!(72)
N = 500 # Number of files to draw from

sampled_files = sample(files, N, replace=false)

maxP = Vector{Float32}[]
minP = Vector{Float32}[]
RVs = Vector{Float32}[]
BVs = Vector{Float32}[]

# ------------------------------------------------------------------------------------------
# Compute quantiles
@info "Computing quantiles"
for i ∈ 1:N
    BSON.@load joinpath(datapath, sampled_files[i]) X
    P = cumsum(X[:, :, 1, :], dims=2)
    push!(minP, minimum(P, dims=2) |> vec)
    push!(maxP, maximum(P, dims=2) |> vec)
    push!(RVs, maximum(X[:, :, 2, :], dims=[1,2]) |> vec)
    push!(BVs, maximum(X[:, :, 3, :], dims=[1,2]) |> vec)
end

maxP = reduce(vcat, maxP)
minP = reduce(vcat, minP)
RVs = reduce(vcat, RVs)
BVs = reduce(vcat, BVs)


qBV = Float32(quantile(BVs, 0.99))
qRV = Float32(quantile(RVs, 0.99))

qminP = Float32(quantile(minP, .01))
qmaxP = Float32(quantile(maxP, .99))

function restrict_data(X)
    # Indexes where prices exceed
    csX = cumsum(X[:, :, 1, :], dims=2)
    idxP = (sum(csX .< qminP, dims=[1,2]) + sum(csX .> qmaxP, dims=[1,2])) |> vec .== 0
    X = X[:, :, :, idxP]
    # Indexes where RV exceeds
    idxRV = sum(X[:, :, 2, :] .> qRV, dims=[1,2]) |> vec .==0
    X = X[:, :, :, idxRV]
    # Indexes where BV exceeds threshold
    idxBV = sum(X[:, :, 3, :] .> qBV, dims=[1,2]) |> vec .== 0
    X[:, :, :, idxBV]
end

# ------------------------------------------------------------------------------------------
# Compute means and standard deviations

@info "Computing means and standard deviations"
μP = Mean()
μRV = Mean()
μBV = Mean()

vP = Variance()
vRV = Variance()
vBV = Variance()

@views for i ∈ 1:N
    BSON.@load joinpath(datapath, sampled_files[i]) X
    X = restrict_data(X) # Compute on restricted data only!
    fit!(μP, X[:, :, 1, :])
    fit!(vP, X[:, :, 1, :])
    fit!(μRV, X[:, :, 2, :])
    fit!(vRV, X[:, :, 2, :])
    fit!(μBV, X[:, :, 3, :])
    fit!(vBV, X[:, :, 3, :])
end

μP = Float32.(value(μP))
μRV = Float32.(value(μRV))
μBV = Float32.(value(μBV))

σP = Float32.(sqrt(value(vP)))
σRV = Float32.(sqrt(value(vRV)))
σBV = Float32.(sqrt(value(vBV)))

μs = reshape([μP, μRV, μBV], 1, 1, 3, 1)
σs = reshape([σP, σRV, σBV], 1, 1, 3, 1)

# Save everything to BSON
BSON.@save "statistics.bson" μs σs qminP qmaxP qRV qBV


@BSON.load "data/1.bson" X Y

statsfile = "statistics.bson"

X, Y = restrict_data(X, Y)