using BSON, Plots, Statistics

function MakePlots(whichrun)
# Plotting the results
BSON.@load "err_nnet_$whichrun.bson" err_nnet
BSON.@load "bias_correction$whichrun.bson" N
k = size(err_nnet, 1) # Number of parameters
# Compute squared errors
err_nnet² = abs2.(err_nnet);
# Compute RMSE for each individual parameter
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_nnet_agg = mean(rmse_nnet, dims=2);

colors = palette(:default)[1:k]'
plot(N, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)
savefig("rmse_benchmark_$whichrun.png")

# Compute bias for each individual parameter
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
# Compute bias aggregate
bias_nnet_agg = mean(abs.(bias_nnet), dims=2);
plot(N, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N, bias_nnet_agg, lab="Aggregate (abs) (NNet)", c=:black, lw=3, ls=:dash)
savefig("bias_benchmark_$whichrun.png")
end
