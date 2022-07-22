# estimate GARCH(1,1) with AR(1) in mean
using ARCHModels, NLopt, PrettyTables
include("GarchLib.jl")

function fmincon(obj, startval, R=[], r=[], lb=[], ub=[]; tol = 1e-10, iterlim=0)
    # the objective is an anonymous function
    function objective_function(x::Vector{Float64}, grad::Vector{Float64})
        obj_func_value = obj(x)[1,1]
        return(obj_func_value)
    end
    # impose the linear restrictions
    function constraint_function(x::Vector, grad::Vector, R, r)
        result = R*x .- r
        return result[1,1]
    end
    opt = Opt(:LN_COBYLA, size(startval,1))
    min_objective!(opt, objective_function)
    # impose lower and/or upper bounds
    if lb != [] lower_bounds!(opt, lb) end
    if ub != [] upper_bounds!(opt, ub) end
    # impose linear restrictions, by looping over the rows
    if R != []
        for i = 1:size(R,1)
            equality_constraint!(opt, (theta, g) -> constraint_function(theta, g, R[i:i,:], r[i]), tol)
        end
    end    
    xtol_rel!(opt, tol)
    ftol_rel!(opt, tol)
    maxeval!(opt, iterlim)
    (objvalue, xopt, flag) = NLopt.optimize(opt, startval)
    return xopt, objvalue, flag
end

# the likelihood function
function garch11(θ, y)
    # dissect the parameter vector
    μ, ρ, ω, β, α  = θ
    ylag = y[1:end-1]
    y = y[2:end]
    ϵ = y .- μ .- ρ*ylag
    n = size(ϵ,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    #h[1] = var(y[1:10])
    h[1] = var(y)
    for t = 2:n
        h[t] = ω + α*(ϵ[t-1])^2. + β*h[t-1]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*(ϵ.^2.)./h
end

#function main()
    X, Y = dgp(100, 1) # Generate the data according to DGP
    X = X[:]
    Y = Y[:]
    θstart = [mean(X); 0.0; var(X); 0.1; 0.1]
    obj = θ -> -mean(garch11(θ, X))
    lb = [-Inf, -1.0, 1e-5, 0.0, 0.0]
    ub = [Inf, 1.0, Inf, 1.0, 1.0] 
# ADD samin here for local max?
    θhat, logL, flag  = fmincon(obj, θstart, [], [], lb, ub)
    pretty_table([Y θhat])
    #nothing
#end
#main()
