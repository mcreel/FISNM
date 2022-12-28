using Pkg
Pkg.activate(".")

include("train_transformer.jl")


function main()
    DGPs = Dict(
    #    "GARCH" => Garch,
    #    "MA2" => Ma2,
       "Logit" => Logit
    )
    N = [100 * 2^i for i ∈ 0:0]
    runname = "test_transformer"

    for (k, v) ∈ DGPs
        @info "Training $k DGP."
        train_transformer(DGPFunc=v, N=N, modelname=k, runname=runname, 
            epochs=2_000, batchsize=64, n_layers=4)
    end
end


main()
