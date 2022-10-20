using DelimitedFiles, PrettyTables, Statistics, DataFrames

function main()
ns = (100, 200, 400, 800, 1600, 3200)
bias = zeros(6,2)
mse = similar(bias)
i = 0
for n ∈ ns
    i +=1
    params = readdlm("../Testing/params_$n.csv")
    estimates, junk = readdlm("MLE$n.csv",',', header=true)
    err = estimates - params'
    bias[i,:] = mean(err, dims=1)
    mse[i,:] = mean(err.^2, dims=1)
end
rmse = sqrt.(mse)
bias, mse, rmse
end
bias, mse, rmse = main()
ns = [100, 200, 400, 800, 1600, 3200]
println("bias")
pretty_table([ns bias])
println("mse")
pretty_table([ns mse])
println("rmse")
pretty_table([ns rmse])
writedlm("bias.txt", bias)
writedlm("mse.txt", mse)
writedlm("rmse.txt", rmse)
