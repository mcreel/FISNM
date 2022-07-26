using NLopt
function fmincon(obj, startval, lb, ub; tol = 1e-5, iterlim=0)
    # the objective is an anonymous function
    function objective_function(x::Vector{Float64}, grad::Vector{Float64})
        obj_func_value = obj(x)[1,1]
    end
    # impose α + β ≤ 1
    function constraint_function(x::Vector{Float64}, grad::Vector{Float64}, a, b)
        a*x[4] + b*x[5] - 1.0
    end

    opt = Opt(:LN_COBYLA, size(startval,1))
    # impose α + β ≤ 1
    # inequality_constraint!(opt, (x,grad) -> constraint_function(x,grad, 1.0, 1.0), 1e-5)
    min_objective!(opt, objective_function)
    # impose lower and/or upper bounds
    lower_bounds!(opt, lb)
    upper_bounds!(opt, ub)

    xtol_rel!(opt, tol)
    ftol_rel!(opt, tol)
    maxeval!(opt, iterlim)
    (objvalue, xopt, ret) = NLopt.optimize(opt, startval)
    (ret == :SUCCESS || ret == :XTOL_REACHED || ret == :FTOL_REACHED) ? convergence = true : convergence = false
    return xopt, objvalue, convergence
end

