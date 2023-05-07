using Pkg
Pkg.activate(".")

using BSON
using OnlineStats
using Random
using StatsBase

function main()
    datapath = "data"
    statsfile = "statistics_new_20.bson"
    verbosity = 100

    Random.seed!(25)
    N = 5_000 # Number of files to draw from


    files = readdir(datapath)
    sampled_files = sample(files, N, replace=false)


    # Define a function to restrict data in a specified way
    function restrict_data(x, max_ret=30, max_rv=20, max_bv=max_rv)
        # Restrict based on max. absolute returns being at most max_ret
        idx = (maximum(abs, x[1, :, 1, :], dims=1) |> vec) .≤ max_ret
        idx2 = (mean(x[1, :, 2, :], dims=1) |> vec) .≤ max_rv # Mean RV at most max_rv
        idx3 = (mean(x[1, :, 3, :], dims=1) |> vec) .≤ max_bv # Mean BV at most max_bv
        x[:, :, :, idx .& idx2 .& idx3]
    end

    # Compute means and standard deviations

    @info "Computing means and standard deviations"
    μP, μRV, μlnRV, μBV, μlnBV = [Mean() for _ ∈ 1:5]
    vP, vRV, vlnRV, vBV, vlnBV = [Variance() for _ ∈ 1:5]

    @views for i ∈ 1:N
        BSON.@load joinpath(datapath, sampled_files[i]) X
        X = restrict_data(X) # Compute on restricted data only!
        # Compute logs as well
        X = cat(X, log1p.(X[:, :, 2:3, :]), dims=3)
        fit!(μP, X[:, :, 1, :])
        fit!(vP, X[:, :, 1, :])
        fit!(μRV, X[:, :, 2, :])
        fit!(vRV, X[:, :, 2, :])
        fit!(μBV, X[:, :, 3, :])
        fit!(vBV, X[:, :, 3, :])
        fit!(μlnRV, X[:, :, 4, :])
        fit!(vlnRV, X[:, :, 4, :])
        fit!(μlnBV, X[:, :, 5, :])
        fit!(vlnBV, X[:, :, 5, :])
        if i % verbosity == 0
            @info "Finished file $i"
        end
    end

    μP = Float32.(value(μP))
    μRV = Float32.(value(μRV))
    μlnRV = Float32.(value(μlnRV))
    μBV = Float32.(value(μBV))
    μlnBV = Float32.(value(μlnBV))

    σP = Float32.(sqrt(value(vP)))
    σRV = Float32.(sqrt(value(vRV)))
    σlnRV = Float32.(sqrt(value(vlnRV)))
    σBV = Float32.(sqrt(value(vBV)))
    σlnBV = Float32.(sqrt(value(vlnBV)))

    μs = reshape([μP, μRV, μBV], 1, 1, 3, 1)
    σs = reshape([σP, σRV, σBV], 1, 1, 3, 1)

    lnμs = reshape([μP, μlnRV, μlnBV], 1, 1, 3, 1)
    lnσs = reshape([σP, σlnRV, σlnBV], 1, 1, 3, 1)

    # Save everything to BSON
    BSON.@save statsfile μs σs lnμs lnσs
end

main()