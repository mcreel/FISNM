using Pkg
Pkg.activate(".")

include("train_tcn.jl")


function main()
    DGPs = Dict(
       "GARCH" => Garch,
       "MA2" => Ma2,
       "Logit" => Logit
    )
    N = [100 * 2^i for i ∈ 0:3]
    runname = "23-02-11"

    for (k, v) ∈ DGPs
        @info "Training $k DGP."
        train_tcn(DGPFunc=v, N=N, modelname=k, runname=runname, 
            batchsize=1024, epochs=400_000, validation_size=10_000,
            validation_frequency=5_000, verbosity=5_000
        )
    end
end


main()
