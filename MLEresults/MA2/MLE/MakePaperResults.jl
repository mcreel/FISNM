using Statistics, DelimitedFiles, PrettyTables

b = readdlm("bias_mle.txt")
aab = mean(abs.(b[:,2:3]), dims=2)

r = readdlm("rmse_mle.txt")
armse = mean(r[:,2:3], dims=2)

results = [b[:,1] aab armse]

pretty_table(results; header=["n", "aab", "armse"])


