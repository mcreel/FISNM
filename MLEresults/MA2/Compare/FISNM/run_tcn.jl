using Pkg, Statistics
Pkg.activate(".")

include("train_tcn.jl")


function main()
    DGPs = Dict(
        "MA2" => Ma2
    )
    N = [100 * 2^i for i ∈ 0:0]
    runname = "compareAkessonTable1"
    rmaes=zeros(10) # holder for relative MAE (E% of Akesson et al)
    for i = 1:10
        for (k, v) ∈ DGPs
            @info "Training $k DGP."
            rmaes[i] = train_tcn(DGPFunc=v, N=N, modelname=k, runname=runname)[1]
            println("mean so far: ")
            @show mean(rmaes[1:i])
        end
    end
end
main()
