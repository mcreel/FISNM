using BSON: @load
using StatsBase
using PrettyTables

# runname = "err_23-02-11"
# runname = "err_lstm_23-02-21"
for dgp ∈ ["GARCH", "MA2", "Logit"]
    @load "../FISNM/results/$dgp/$runname.bson" err_best


    n = map(i -> 100 * 2^(i - 1), axes(err_best, 3))
    amae = map(i -> mean(mean(abs.(err_best[:, :, i]), dims=2)), axes(err_best, 3))
    armse = map(i -> mean(sqrt.(mean(abs2.(err_best[:, :, i]), dims=2))), axes(err_best, 3))
    aabias = map(i -> mean(abs, mean(err_best[:, :, i], dims=2)), axes(err_best, 3))
    @info dgp
    pretty_table([n amae aabias armse], header=["n", "amae", "aabias", "armse"])
end