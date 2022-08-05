using BSON, Plots, Statistics

function MakePlots(whichrun)
# Plotting the results
BSON.@load "err_mle_$whichrun.bson" err_mle N
BSON.@load "err_nnet_$whichrun.bson" err_nnet

k = size(err_mle, 1) # Number of parameters
# Compute squared errors
err_mle² = abs2.(err_mle);
err_nnet² = abs2.(err_nnet);
# Compute RMSE for each individual parameter
rmse_mle = permutedims(reshape(sqrt.(mean(err_mle², dims=2)), k, length(N)));
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_mle_agg = mean(rmse_mle, dims=2);
rmse_nnet_agg = mean(rmse_nnet, dims=2);

colors = palette(:default)[1:k]'
plot(N, rmse_mle, xlab="Number of observations", ylab="RMSE", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (MLE)", ["p1" "p1" "p3" "p4" "p5"]),
    color=colors)
plot!(N, rmse_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("rmse_benchmark_$whichrun.png")

# Compute bias for each individual parameter
bias_mle = permutedims(reshape(mean(err_mle, dims=2), k, length(N)));
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
# Compute bias aggregate
bias_mle_agg = mean(bias_mle, dims=2);
bias_nnet_agg = mean(bias_nnet, dims=2);

plot(N, bias_mle, xlab="Number of observations", ylab="Bias", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (MLE)", ["lrv" "β+α" "π"]),
    color=colors)
plot!(N, bias_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["lrv" "β+α" "π"]),
    color=colors)
plot!(N, bias_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

savefig("bias_benchmark_$whichrun.png")
end
