using Distributions, DelimitedFiles, Statistics, Flux, CUDA, BSON

# estimates GARCH with SP500 data, to get reasonable parameters
# and standard errors.
function calibrate()
    data = readdlm("sp500.csv",',')
    data = data[2:end,2]
    data = Float64.(data)
    f = fit(GARCH{1,1}, data;  meanspec=AR{1})
end

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
# generates S samples of length n
# returns are:
# x: 1 X S*n vector of data from EGARCH model
# y: 3 X S*n vector of parameters used to generate each sample
@views function dgp(n, S)
    y = PriorDraw(S)     # the parameters for each sample
    x = zeros(1, n*S)    # the Garch data for each sample
    for s = 1:S
        x[:,n*s-n+1:s*n] = SimulateGarch11(y[:,s], n)
    end
    Float32.(x), Float32.(y)
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
    ys'
end

# the likelihood function, alternative version with reparameterization
@views function garch11(θ, y)
    # dissect the parameter vector
    lrv, βplusα , share  = θ
    ω = (1.0 - βplusα)*lrv
    β = share*βplusα
    α = (1.0 - share)*βplusα
    ylag = y[1:end-1]
    y = y[2:end]
    n = size(y,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    #h[1] = var(y[1:10])
    #h[1] = var(y)
    h[1] = lrv
    for t = 2:n
        h[t] = ω + α*(y[t-1])^2. + β*h[t-1]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*(y.^2.)./h
end
