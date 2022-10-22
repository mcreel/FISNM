using Pkg
Pkg.activate(".")

include("train_ensemble.jl")


function main()
    DGPs = Dict(
#        "GARCH" => Garch, 
        "MA2" => Ma2 
#        "Logit" => Logit
    )
    N = [100 * 2^i for i ∈ 0:2]
    runname = "mc_x_MA"

    for (k, v) ∈ DGPs
        @info "Training $k DGP."
        train_tcn_ensemble(DGPFunc=v, N=N, modelname=k, runname=runname)
    end
end


main()
