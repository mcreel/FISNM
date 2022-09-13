using Random, DelimitedFiles
include("../MA2lib.jl")

function MakeTestingParams(seed, n, reps)
    Random.seed!(seed)
    params = zeros(2,reps)
    for i = 1:reps
        params[:,i] = PriorDraw()
    end
    params
end

function MakeTestingData(seed, params, n)
    Random.seed!(seed)
    reps = size(params,2)
    data = zeros(n, reps)
    for r = 1:reps
        data[:,r] = ma2(params[:,r], n)
    end    
    data
end

function main()
    ns = (100, 200, 400, 800, 1600, 3200)
    mcreps = Int64(5e3)
    for n ∈ ns
        testing_params = MakeTestingParams(77, n, mcreps)
        testing_data = MakeTestingData(78, testing_params, n)
        writedlm("params_$n.csv", testing_params)
        writedlm("data_$n.csv", testing_data)
    end
end

main()
