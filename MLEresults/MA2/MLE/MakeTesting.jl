using Random, DelimitedFiles
include("MA2Lib.jl")

function MakeTestingData(params, n)
    reps = size(params,2)
    data = zeros(n, reps)
    for r = 1:reps
        data[:,r] = ma2(params[:,r], n)
    end    
    data
end

function main()
    n = 100
    mcreps = 5000
    testing_seed = 78
    Random.seed!(testing_seed)
    testing_params = priordraw(mcreps)
    testing_data = MakeTestingData(testing_params, n)
    writedlm("params_$n.csv", testing_params)
    writedlm("data_$n.csv", testing_data)
end

main()
