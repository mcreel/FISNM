# This computes the moments and the CUE objective function
# for the plain vanilla version of MSM.
using LinearAlgebra

# take the elements of cov matrix below the main diag (covariances)
function offdiag2(A::AbstractMatrix)
    [A[ι] for ι in CartesianIndices(A) if ι[1] > ι[2]]
end 


@views function MSMmoments(data)
    data[:,2:3] .= log.(data[:,2:3])
    # covariances between variables (there are 3)
    c = offdiag2(cov(data))
    # moments: 3 means, 3 std. dev., 6 autocovs, 3 covs = 15 moments
    m = vcat(
        mean(data,dims=1)[:],
        std(data, dims=1)[:],
        mean(data[2:end,:].*data[1:end-1,:], dims=1)[:], # 1st order autocovs
        mean(data[3:end,:].*data[1:end-2,:], dims=1)[:], # 2nd order autocovs
        c[:]
       )
    return m
end    

@views function simmomentscov(θ, S)
    # Computes moments according to the TCN for a given DGP, θ, and S
    ms = zeros(S, 15)
    Threads.@threads for s = 1:S
        data = simulate_jd(θ)
        ms[s,:] = MSMmoments(data)
    end
    return vec(mean(ms, dims=1)), Symmetric(cov(ms))
end

  
# CUE objective, written to MAXIMIZE
@inbounds function bmsmobjective(θ, mhat, S)      
    # Make sure parameter is the support of prior
    InSupport(θ) || return -Inf
    # Compute simulated moments
    mbar, Σ = simmomentscov(θ, S)
    n = 1000.0 # sample size
    Σ *= n*(1+1/S) # 1 for θhat, 1/S for θbar
    isposdef(Σ) || return -Inf
    err = sqrt(n)*(mhat - mbar) 
    W = inv(Σ) # scaled for numeric accuracy
    -0.5*dot(err, W, err)
end


