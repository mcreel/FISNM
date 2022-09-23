using BSON, Plots, Statistics

function MakePlots(whichrun)
# Plotting the results
BSON.@load "err_mle_$whichrun.bson" err_mle N
BSON.@load "err_splitsample_$whichrun.bson" err_nnet
err_nnet_ss = copy(err_nnet) # otherwise, next line overwrites
BSON.@load "err_nnet_$whichrun.bson" err_nnet


k = size(err_mle, 1) # Number of parameters
# Compute squared errors
err_mle² = abs2.(err_mle);
err_nnet² = abs2.(err_nnet);
err_nnet_ss² = abs2.(err_nnet_ss);
# Compute RMSE for each individual parameter
rmse_mle = permutedims(reshape(sqrt.(mean(err_mle², dims=2)), k, length(N)));
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
rmse_nnet_ss = permutedims(reshape(sqrt.(mean(err_nnet_ss², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_mle_agg = mean(rmse_mle, dims=2);
rmse_nnet_agg = mean(rmse_nnet, dims=2);
rmse_nnet_ss_agg = mean(rmse_nnet_ss, dims=2);

colors = palette(:default)[1:k]'
plot(N, rmse_mle, xlab="Number of observations", ylab="RMSE", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (MLE)", ["p1" "p1" "p3" "p4" "p5"]),
    color=colors)
plot!(N, rmse_mle_agg, lab="Aggregate (MLE)", c=:black, lw=3)

plot!(N, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)

plot!(N, rmse_nnet_ss, lw=2, ls=:dot, 
    lab=map(x -> x * " (SS-NNet)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N, rmse_nnet_ss_agg, lab="Aggregate (SS-NNet)", c=:black, lw=3, ls=:dot)


savefig("rmse_benchmark_$whichrun.png")

# Compute bias for each individual parameter
bias_mle = permutedims(reshape(mean(err_mle, dims=2), k, length(N)));
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
bias_nnet_ss = permutedims(reshape(mean(err_nnet_ss, dims=2), k, length(N)));
# Compute bias aggregate
bias_mle_agg = mean(abs.(bias_mle), dims=2);
bias_nnet_agg = mean(abs.(bias_nnet), dims=2);
bias_nnet_ss_agg = mean(abs.(bias_nnet_ss), dims=2);

plot(N, bias_mle, xlab="Number of observations", ylab="Bias", size=(1200, 800), 
    lw=2, lab=map(x -> x * " (MLE)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N, bias_mle_agg, lab="Aggregate (abs) (MLE)", c=:black, lw=3)

plot!(N, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N, bias_nnet_agg, lab="Aggregate (abs) (NNet)", c=:black, lw=3, ls=:dash)

plot!(N, bias_nnet_ss, lw=2, ls=:dot, 
    lab=map(x -> x * " (SS-NNet)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N, bias_nnet_ss_agg, lab="Aggregate (abs) (SS-NNet)", c=:black, lw=3, ls=:dot)


savefig("bias_benchmark_$whichrun.png")
end
