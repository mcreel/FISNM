using Random, Statistics
include("../MA2lib.jl")

function MakeTestingParams(seed, reps)
    Random.seed!(seed)
    params = zeros(2,reps)
    for i = 1:reps
        params[:,i] = PriorDraw()
    end
    params
end

function main()
    mcreps = 1000000
    params = MakeTestingParams(77, mcreps)
    priormean = mean(params,dims=2)
    den = mean(abs.(params .- priormean), dims=2)
    priormean, den
end

priormean, den = main()
