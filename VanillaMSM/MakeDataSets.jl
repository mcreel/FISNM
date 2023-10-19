using Pkg
Pkg.activate(".")
using BSON
using DifferentialEquations
using LinearAlgebra
using Distributions
using Random
using StatsBase

include("JD.jl")
reps = 2 # number of simulated data files to estimate

# simulate data using given parameters
θtcn = [  # TCN results for 13-17 data
−0.01454,
0.17403,
−1.19645,
0.92747,
−0.79534,
0.00563,
3.25268,
0.03038]
#=
θtcn = [  # TCN results for 16-19 data
 −0.00509,
  0.17166,
  −1.10610,
  0.95327,
  −0.73734,
  0.00798,
  2.89514,
  0.03243]
=#

datasets = Vector{Array{Float64,2}}()
for rep = 1:reps
    push!(datasets, simulate_jd(θtcn))
end    

BSON.@save "ComparisonDataSets-TCN-13-17.bson" datasets

