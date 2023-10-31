using BSON, Statistics, PrettyTables

θtrue = [  # TCN results for 13-17 data
−0.01454,
0.17403,
−1.19645,
0.92747,
−0.79534,
0.00563,
3.25268,
0.03038]

BSON.@load "TCNresults.bson" results
@info "results for TCN moments"
ac = results[9,:]
logℒ  = results[10,:]
@info "acceptance rate" mean(ac) minimum(ac) maximum(ac)
@info "logℒ " mean(logℒ ) minimum(logℒ ) maximum(logℒ )
bias_tcn = mean(results[1:8,:] .-θtrue, dims=2)
rmse_tcn = sqrt.(mean((results[1:8,:] .- θtrue).^2, dims=2))
@info "mean rmse TCN" mean(rmse_tcn)

BSON.@load "VanillaResults.bson" results
@info "results for Vanilla moments"
ac = results[9,:]
logℒ  = results[10,:]
@info "acceptance rate" mean(ac) minimum(ac) maximum(ac)
@info "logℒ " mean(logℒ ) minimum(logℒ ) maximum(logℒ )
bias_vanilla = mean(results[1:8,:] .-θtrue, dims=2)
rmse_vanilla = sqrt.(mean((results[1:8,:] .- θtrue).^2, dims=2))
@info "mean rmse vanilla" mean(rmse_vanilla)
params = ["μ", "κ", "α", "σ", "ρ", "λ₀", "λ₁", "τ" ]  

formatter = (v, i, j) -> (j>1) ? ft_printf("%5.3f")  : v
pretty_table([params bias_vanilla bias_tcn  rmse_vanilla rmse_tcn]; header = ["param", "bias  van.", "bias TCN", "rmse van.", "rmse TCN"], formatters=ft_printf("%5.3f"))
pretty_table([params abs.(bias_tcn) ./ abs.(bias_vanilla)  rmse_tcn ./ rmse_vanilla]; header = ["param", "rel. abs. bias", "rel. rmse"], formatters=ft_printf("%5.3f"))

