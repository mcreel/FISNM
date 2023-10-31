# This computes the moments and the CUE objective function
# for the plain vanilla version of MSM.
using LinearAlgebra


# [μ; κ; α; σ; ρ; λ₀; λ₁; τ]
function PriorSupport()
    #         μ,    κ,    α,   σ,    ρ,    λ₀, λ₁,    τ   
    [-.05, .01,  -6, 0.1, -.99,  -.02,  2, -.02], 
    [ .05, .30,   0, 4.0, -.50,   .10,  6,  .20]
end

function InSupport(θ)
    lb, ub = PriorSupport()
    all(θ .>= lb) && all(θ .<= ub)
end

function PriorDraw()
    lb, ub = PriorSupport()
    lb + (ub-lb) .* rand(size(lb))
end    

function Prior(θ)
    InSupport(θ) ? 1.0 : 0.0
end    




# take the elements of cov matrix below the main diag (covariances)
function offdiag2(A::AbstractMatrix)
    [A[ι] for ι in CartesianIndices(A) if ι[1] > ι[2]]
end 

using StatsBase
@views function MSMmoments(data)
    data[:,2:3] .= log.(data[:,2:3])
    # covariances between variables (there are 3)
    c = offdiag2(cor(data))
    # moments: 3 means, 3 std. dev., 3 autocors, 3 covariances = 12 moments
    m = vcat(
        mean(data,dims=1)[:],
        std(data, dims=1)[:],
	autocor(data, [1])[:],
        c[:]
       )
    return m
end    


@views function simmomentscov(θ, S)
    # Computes moments according to the TCN for a given DGP, θ, and S
    ms = zeros(S, 12)
    for s = 1:S
        data = simulate_jd(θ, 1000, burnin=100)
        ms[s,:] = MSMmoments(data)
    end
    return vec(mean(ms, dims=1)), Symmetric(cov(ms))
end

  
# CUE objective, written to MAXIMIZE
@inbounds function bmsmobjective(θ, mhat, S)      
    # Make sure parameter is the support of prior
    InSupport(θ) || return -Inf
    # Compute s:imulated moments
    mbar, Σ = simmomentscov(θ, S)
    n = 1000.0 # sample size
    Σ *= n*(1+1/S) # 1 for θhat, 1/S for θbar
    isposdef(Σ) || return -Inf
    err = sqrt(n)*(mhat - mbar) 
    W = inv(Σ) # scaled for numeric accuracy
    -0.5*dot(err, W, err)
end


