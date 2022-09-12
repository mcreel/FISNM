using Distributions, Plots, NLopt 
include("../MA2lib.jl")


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
    f = 0.
    θ1, θ2 = θ
    (θ2+θ1 >= -1.) & (θ2-θ1 >= -1.) ? f = log(pdf(MvNormal(zeros(n), Σ(n,θ)), data))[1] : nothing
    f
end    

function GetMLE(obj, startval; tol = 1e-10, iterlim=0)

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
    inequality_constraint!(opt, (θ,g)  -> -1.0 + θ[1] - θ[2], tol)
    xtol_rel!(opt, tol)
    ftol_rel!(opt, tol)
    maxeval!(opt, iterlim)
    (objvalue, xopt, flag) = NLopt.optimize(opt, startval)
    return xopt, objvalue, flag
end

gr()
n = 100
θtrue = [0., 0.95]
data = ma2(θtrue, n)
obj = θ -> -(1.0/n)*lnL(θ, data)
θhat, objval1, junk = GetMLE(obj, zeros(2))
θhat2, objval2, junk = GetMLE(obj, θtrue)
obj = (a,b) -> -(1.0/n)*lnL([a,b], data)
θ1 = range(-2, step=0.01, stop=2)
θ2 = range(-1, step=0.01, stop=1)
contour(θ1, θ2, (a,b)->obj(a,b),c=:viridis)
xlabel!("θ1")
ylabel!("θ2")
scatter!([θhat[1]], [θhat[2]],markersize=15)
scatter!([θhat2[1]], [θhat2[2]],markersize=15)
savefig("MA2contours.png")

