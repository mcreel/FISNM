abstract type DGP end

error_msg(t, f) = error("$f not implemented for ::$t")

# Generic functions for DGPs
priordraw(d::DGP, args...) = error_msg(typeof(d), "priordraw")
generate(d::DGP, args...) = error_msg(typeof(d), "generate")
nfeatures(d::DGP, args...) = error_msg(typeof(d), "nfeatures")
nparams(d::DGP, args...) = error_msg(typeof(d), "nparams")

# Data transform for a particular DGP
data_transform(d::DGP, S::Int; dev=cpu) = fit(ZScoreTransform, dev(priordraw(d, S)))

# Generate to specific device directly
generate(d::DGP, S::Int; dev=cpu) = map(dev, generate(d, S))

# Generate prior parameters according to uniform with lower and upper bounds
function uniformpriordraw(d::DGP, S::Int)
    lb, ub = θbounds(d)
    (ub .- lb) .* rand(Float32, size(lb, 1), S) .+ lb 
end

# Expected absolute error when using the prior mean as prediction 
# θbounds has to be defined => uniform priors only
priorerror(d::DGP) = .25abs.(reduce(-, θbounds(d)))
priorpred(d::DGP) = .5reduce(+, θbounds(d))