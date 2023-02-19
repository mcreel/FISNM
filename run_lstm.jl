using Pkg
Pkg.activate(".")

include("train_lstm.jl")


function main()
    DGPs = Dict(
       "GARCH" => Garch,
       "MA2" => Ma2,
       "Logit" => Logit
    )
    N = [100 * 2^i for i ∈ 0:3]
    runname = "lstm_23-02-19"

    for (k, v) ∈ DGPs
        @info "Training $k DGP."
        train_lstm(DGPFunc=v, N=N, modelname=k, runname=runname, 
            batchsize=1024, epochs=20_000, validation_size=10_000,
            validation_frequency=250, verbosity=1_000
        )
    end
end


main()
