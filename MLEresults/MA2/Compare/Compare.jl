# computes the percentage abs error measure of Akeson et. al., to compare to other models
#
#=
The Akeson paper appears to use prior_mae =
(b-a)/4, coming from the uniform(a,b) distribution.
However, the MA2 has the invertibility restrictions,
so the prior is uniform over a triangle, not a square.
Thus, the reported E%, which is the average over the two parameters of relative MAB, is not quite correct. Here, we compute over a square, to compare to paper, and over the triangle, which is correct 
=#
using Statistics

tcn_mae = 0.0871    # average of 10 runs, done by run_tcn.jl
                    # in the FISNM folder, each using 10^6 samples to train

include("MA2Lib.jl")
reps = 100_000
p = priordraw(reps)
prior_mean = mean(p, dims=2)
prior_mae = mean(abs,prior_mean .-p)
println("mean relative avg. abs. error, correct:")
percent_mae = tcn_mae / prior_mae
@show percent_mae

println("mean relative avg. abs. error, paper formla:")
prior_mae = (4/4 + 2/4)/2
percent_mae = tcn_mae / prior_mae
@show percent_mae

