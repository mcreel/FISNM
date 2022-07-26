using NLopt
function fmincon(obj, startval, lb, ub; tol = 1e-5, iterlim=0)
    # the objective is an anonymous function
    function objective_function(x::Vector{Float64}, grad::Vector{Float64})
        obj_func_value = obj(x)[1,1]
    end
    opt = Opt(:LN_COBYLA, size(startval,1))
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

