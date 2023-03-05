using Pkg
Pkg.activate(".")
using BSON
using LinearAlgebra
using Optim
using Random
using StatsBase

include("DGPs/DGPs.jl")
include("DGPs/GARCH.jl")
include("DGPs/Logit.jl")
include("DGPs/MA2.jl")


include("samin.jl")

function run_mle(;
    DGPFunc, N, modelname, runname,
    test_size = 5_000,
    test_seed = 78    
)
    dgp = DGPFunc(N=N[1])
    err = zeros(nparams(dgp), test_size, length(N))
    for (i, n) ∈ enumerate(N)
        @info "Running MLE for n = $n"
        dgp = DGPFunc(N=n)
        Random.seed!(test_seed)
        X, Y = generate(dgp, test_size)
        θstart = priorpred(dgp) # Use prior mean as a starting value
        if isa(dgp, Logit)
            @inbounds Threads.@threads for s ∈ axes(Y, 2)
                @views Ŷ = Optim.optimize(
                    θ -> -likelihood(dgp, X[:, s, :], θ), θstart, LBFGS(), 
                    Optim.Options(g_tol = 1e-5, x_tol = 1e-6, f_tol = 1e-12); 
                    autodiff=:forward
                    ).minimizer
                err[:, s, i] = Y[:, s] - Ŷ
            end
        elseif isa(dgp, GARCH)
            lb, ub = θbounds(dgp)
            @inbounds Threads.@threads for s ∈ axes(Y, 2)
                @views Ŷ, _, _, _ = samin(
                    θ -> -likelihood(dgp, X[1, s, :], θ), θstart, lb, ub, 
                    verbosity=0, rt=0.25, maxevals=1e5, coverage_ok=1
                )
                err[:, s, i] = Y[:, s] - Ŷ
            end
        elseif isa(dgp, MA2)
            Σ, Σ⁻¹ = zeros(n, n), zeros(n, n)
            @inbounds for s ∈ axes(Y, 2)
                @views Ŷ = Optim.optimize(
                    θ -> -likelihood(dgp, X[1, s, :], θ, Σ, Σ⁻¹), θstart, 
                    NelderMead()
                ).minimizer
                err[:, s, i] = Y[:, s] - Ŷ
            end
        else
            error("Not implemented yet.")
        end
    end
    BSON.@save "results/$modelname/$runname.bson" err
end




# run_mle(
#     DGPFunc=Logit, N=[100 * 2^i for i ∈ 0:3], 
#     modelname="Logit", runname="err_mle_23-02-28"
# )


run_mle(
    DGPFunc=MA2, N=[100 * 2^i for i ∈ 0:3], 
    modelname="MA2", runname="err_mle_23-03-04"
)