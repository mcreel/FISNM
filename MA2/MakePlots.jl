using BSON, Plots, Statistics

function MakePlots(whichrun)

# mle
rmse_mle = [0.0891,0.0606,0.0420, 0.0286,0.0204,0.0145]
N_mle = [100,200,400,800,1600, 3200]
# normal fit
BSON.@load "err_nnet_$whichrun.bson" err_nnet N
k = size(err_nnet, 1) # Number of parameters
# Compute squared errors
err_nnet² = abs2.(err_nnet)
# Compute RMSE for each individual parameter
rmse_nnet = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)))
# Compute RMSE aggregate
rmse_nnet_agg = mean(rmse_nnet, dims=2)
# Compute bias for each individual parameter
bias_nnet = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)))
# Compute bias aggregate
bias_nnet_agg = mean(abs.(bias_nnet), dims=2)
N_nnet = N

# split sample fit
BSON.@load "err_splitsample_$whichrun.bson" err_nnet N
k = size(err_nnet, 1) # Number of parameters
# Compute squared errors
err_nnet² = abs2.(err_nnet);
# Compute RMSE for each individual parameter
rmse_ss = permutedims(reshape(sqrt.(mean(err_nnet², dims=2)), k, length(N)));
# Compute RMSE aggregate
rmse_ss_agg = mean(rmse_ss, dims=2);
# Compute bias for each individual parameter
bias_ss = permutedims(reshape(mean(err_nnet, dims=2), k, length(N)));
# Compute bias aggregate
bias_ss_agg = mean(abs.(bias_ss), dims=2);
N_ss = N


colors = palette(:default)[1:k]'
plot(N_nnet, rmse_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N_nnet, rmse_nnet_agg, lab="Aggregate (NNet)", c=:black, lw=3, ls=:dash)
plot!(N_ss, rmse_ss, lw=2, ls=:dot, 
    lab=map(x -> x * " (ss)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
plot!(N_ss, rmse_ss_agg, lab="Aggregate (ss)", c=:black, lw=3, ls=:dot)
plot!(N_mle, rmse_mle, lw=2, ls=:solid, 
    lab=map(x -> x * " (MLE)", ["p1" "p2" "p2" "p4" "p5"]),
    color=colors)
savefig("rmse_benchmark_$whichrun.png")


plot(N_nnet, bias_nnet, lw=2, ls=:dash, 
    lab=map(x -> x * " (NNet)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N_nnet, bias_nnet_agg, lab="Aggregate (abs) (NNet)", c=:black, lw=3, ls=:dash)
plot!(N_ss, bias_ss, lw=2, ls=:dot, 
    lab=map(x -> x * " (ss)", ["p1" "p2" "p3" "p4" "p5"]),
    color=colors)
plot!(N_ss, bias_ss_agg, lab="Aggregate (abs) (ss)", c=:black, lw=3, ls=:dot)
savefig("bias_benchmark_$whichrun.png")
end
