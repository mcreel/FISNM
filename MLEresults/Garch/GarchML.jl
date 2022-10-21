using Distributions, Statistics

# draw the params from their priors. Model
# is parameterized in terms of long run variance,
# beta+alpha, and beta's share of beta+alpha
function PriorDraw()
    # long run variance
    lrv = 0.0001 + 0.9999*rand()
    βplusα = 0.99*rand()
    share = rand()
    [lrv, βplusα, share]
end

# get a set of draws from prior
function PriorDraw(n)
    draws = zeros(3,n)
    for i = 1:n
        draws[:,i] .= PriorDraw()
    end
    draws
end    

@views function dgp(n, S)
    y = PriorDraw(S)     # the parameters for each sample
    x = zeros(n,S)    # the Garch data for each sample
    for s = 1:S
        x[:,s] = SimulateGarch11(y[:,s], n)
    end
    x, y
end    


# the likelihood function, alternative version with reparameterization
@views function SimulateGarch11(θ, n)
    burnin = 1000
    # dissect the parameter vector
    lrv, βplusα , share  = θ
    ω = (1.0 - βplusα)*lrv
    β = share*βplusα
    α = (1.0 - share)*βplusα
    ys = zeros(n)
    h = lrv
    y = 0.
    for t = 1:burnin + n
        h = ω + α*y^2. + β*h
        y = sqrt(h)*randn()
        t > burnin ? ys[t-burnin] = y : nothing
    end
    ys
end

# the likelihood function, alternative version with reparameterization
@views function garch11(θ, y)
    # dissect the parameter vector
    lrv, βplusα , share  = θ
    ω = (1.0 - βplusα)*lrv
    β = share*βplusα
    α = (1.0 - share)*βplusα
    n = size(y,1)
    h = zeros(n)
    h[1] = lrv
    for t = 2:n
        h[t] = ω + α*(y[t-1])^2. + β*h[t-1]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*(y.^2.)./h
end
