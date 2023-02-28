using Optim
using Random

function run_mle(;
    DGPFunc, N, modelname, runname,
    test_size = 5_000,
    test_seed = 78    
)
    err = zeros(nparams(dgp), test_size, length(N))
    for (i, n) ∈ enumerate(N)
        @info "Running MLE for n = $n"
        dgp = DGPFunc(N=n)
        Random.seed!(test_seed)
        X, Y = generate(dgp, test_size)
        @inbounds Threads.@threads for s ∈ axes(Y, 2)
            @views Ŷ = Optim.optimize(
                θ -> -likelihood(dgp, X[:, s, :], θ), zeros(3), LBFGS(), 
                Optim.Options(g_tol = 1e-5, x_tol = 1e-6, f_tol = 1e-12); 
                autodiff=:forward
                ).minimizer
            err[:, s, i] = Y[:, s] - Ŷ
        end
    end
    BSON.@save "results/$modelname/$runname.bson" err
end

run_mle(
    DGPFunc=Logit, N=[100 * 2^i for i ∈ 0:3], 
    modelname="Logit", runname="err_mle_23-02-28"
)