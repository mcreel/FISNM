using BSON
using Optim
using Random

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
        if isa(dgp, Logit)
            @inbounds Threads.@threads for s ∈ axes(Y, 2)
                @views Ŷ = Optim.optimize(
                    θ -> -likelihood(dgp, X[:, s, :], θ), priorpred(dgp), LBFGS(), 
                    Optim.Options(g_tol = 1e-5, x_tol = 1e-6, f_tol = 1e-12); 
                    autodiff=:forward
                    ).minimizer
                err[:, s, i] = Y[:, s] - Ŷ
            end
        elseif isa(dgp, GARCH)
            lb, ub = θbounds(dgp)
            θstart = priorpred(dgp)
            @inbounds Threads.@threads for s ∈ axes(Y, 2)
                @views Ŷ, _, _, _ = samin(
                    θ -> -likelihood(dgp, X[:, s, :], θ),
                    θstart, lb, ub, verbosity=0
                )
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
    DGPFunc=GARCH, N=[100 * 2^i for i ∈ 0:3], 
    modelname="GARCH", runname="err_mle_23-02-28"
)y<