using BSON, DelimitedFiles, Distributions, PrettyTables, Statistics
include("../MA2lib.jl")
include("samin.jl")

function Σ(n, θ)
    θ1, θ2 = θ 
    Σ = zeros(n,n)
    for i = 1:n
        Σ[i,i] = 1. + θ1^2. + θ2^2.
    end    
    for i = 1:n-1    
        Σ[i,i+1] = θ1+θ1*θ2
        Σ[i+1,i] = θ1+θ1*θ2
    end
    for i = 1:n-2    
        Σ[i, i+2] = θ2
        Σ[i+2,i] = θ2
    end
    Σ
end    

function lnL(θ, data)
    n = size(data,1)
    InSupport(θ) ? log(pdf(MvNormal(zeros(n), Σ(n,θ)), data))[1] : -Inf
end

    
function main()
    ns = (100, 200, 400, 800)
    lb = [-2., -1.]
    ub = [2.0, 1.0]
    for n ∈ ns
        data = zeros(n)
        BSON.@load "../Testing/testing_$n.bson" testing_data
        reps = size(testing_data,2)
        fit = zeros(reps, 2)
        for i = 1:reps
            obj = θ -> -(1.0/n)*lnL(θ, testing_data[:,i])
            θhat, junk, junk, junk= samin(obj, zeros(2), lb, ub, rt=0.25, functol=1e-5, paramtol=1e-4, verbosity=0)
            fit[i,:] = θhat
            mod(i,100) ==  0 ? println("$i of $reps") : nothing
        end    
        writedlm("mlefit_$n", fit)
    end    
    # print out the results
    for n ∈ ns
        BSON.@load "../Testing/testing_$n.bson" testing_params
        estimates = readdlm("mlefit_$n")
        err = estimates - testing_params'
        bias = mean(err, dims=1)
        mse = mean(err.^2, dims=1)
        rmse = sqrt.(mse)
        println("Results for sample size $n:") 
        pretty_table([bias' mse' rmse']; 
            header=["bias", "mse", "rmse"])
    end
end
main()

