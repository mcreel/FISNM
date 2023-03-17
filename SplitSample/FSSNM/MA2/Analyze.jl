using DelimitedFiles, StatsPlots, PrettyTables, Term
using Econometrics

d = readdlm("results_50.txt")
θtrue  = d[:,1:2]
θtcn = d[:,3:4]
θmsm = d[:,5:6]
etcn = θtcn - θtrue
emsm = θmsm - θtrue
e = [etcn emsm]
println(@green "RMSE")
h = ["tcn θ₁", "tcn θ₂","msm θ₁","msm θ₂"]    
pretty_table(sqrt.(mean(abs2, e, dims=1)); header=h)
println(@green "MAE")
h = ["tcn θ₁", "tcn θ₂","msm θ₁","msm θ₂"]    
pretty_table((mean(abs, e, dims=1)); header=h)

d = [θtrue θtcn θmsm etcn emsm]
d = sortbyc(d,1)
p1 = plot(d[:,1], abs.([etcn[:,1] emsm[:,1]]))
d = sortbyc(d,2)
p2 = plot(d[:,2], abs.([etcn[:,2] emsm[:,2]]))

mtcn1 = marginalkde(d[:,1], abs.(etcn[:,1]))
mmsm1 = marginalkde(d[:,1], abs.(emsm[:,1]))
plot(mtcn1, mmsm1)

x = [θtrue θtrue.^2 θtrue[:,1].*θtrue[:,2]]
x = [ones(size(x,1)) x]
ols(etcn[:,1], x);
#plot(p1,p2)
