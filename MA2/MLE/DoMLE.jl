using BSON, DelimitedFiles, Distributions, NLopt 

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
    log(pdf(MvNormal(zeros(n), Σ(n,θ)), data))[1]
end    

function GetMLE(obj, startval; tol = 1e-4, iterlim=0)

    # use NLopt's input form
    function objective_function(x::Vector{Float64}, grad::Vector{Float64})
        obj_func_value = obj(x)
        return(obj_func_value)
    end
    opt = Opt(:LN_COBYLA, size(startval,1))
    min_objective!(opt, objective_function)
    # impose lower and/or upper bounds
    lower_bounds!(opt, [-2, -1])
    upper_bounds!(opt, [2., 1.])
    # impose linear inequality restrictions
    inequality_constraint!(opt, (θ,g)  -> -1.0 - θ[1] - θ[2], tol)
    inequality_constraint!(opt, (θ,g)  -> -1.0 - θ[1] + θ[2], tol)
    xtol_rel!(opt, tol)
    ftol_rel!(opt, tol)
    maxeval!(opt, iterlim)
    (objvalue, xopt, flag) = NLopt.optimize(opt, startval)
    return xopt, objvalue, flag
end

function main()
    #ns = (100, 200, 400)
    ns = (200, 400)
    for n ∈ ns
        data = zeros(n)
        BSON.@load "../Testing/testing_$n.bson" testing_params testing_data
        reps = size(testing_data,2)
        fit = zeros(reps, 2)
        for i = 1:reps
            obj = θ -> -(1.0/n)*lnL(θ, testing_data[:,i])
            # use true as start value
            θhat, junk, junk = GetMLE(obj, testing_params[:,i])
            fit[i,:] = θhat
            mod(i,1000) ==  0 ? println("$i of $reps") : nothing
        end    
        writedlm("mlefit_$n", fit)
    end    
end
main()

