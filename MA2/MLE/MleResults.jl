using BSON, DelimitedFiles, PrettyTables, Statistics

function main()
    ns = (100, 200, 400)
    for n ∈ ns
        BSON.@load "../Testing/testing_$n.bson" testing_params
        estimates = readdlm("mlefit_$n")

        err = estimates - testing_params'
        bias = mean(err, dims=1)
        mse = mean(err.^2, dims=1)
        rmse = sqrt.(mse)
        println("Results for sample size $n:") 
        pretty_table([bias' mse' rmse']; 
            header=["bias", "mse", "rmse"])
    end
end
main()
