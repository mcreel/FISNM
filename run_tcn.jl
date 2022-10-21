using Pkg
Pkg.activate(".")

include("train_tcn.jl")


function main()
    DGPs = Dict(
        "GARCH" => Garch, 
        "MA2" => Ma2, 
        "Logit" => Logit
    )
    N = [100 * 2^i for i ∈ 0:5]
    runname = "tcn-1510"

    for (k, v) ∈ DGPs
        @info "Training $k DGP."
        train_tcn(DGPFunc=v, N=N, modelname=k, runname=runname)
    end
end


main()
