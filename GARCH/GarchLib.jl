using Distributions, DelimitedFiles, Flux, CUDA, BSON
using StatsBase

# estimates GARCH with SP500 data, to get reasonable parameters
# and standard errors.
function calibrate()
    data = readdlm("sp500.csv",',')
    data = data[2:end,2]
    data = Float64.(data)
    f = fit(GARCH{1,1}, data;  meanspec=AR{1})
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

is_possible_garch(lrv, βplusα, share) = 
    (0 < lrv ≤ 1) && (0 ≤ βplusα < 1) && (0 ≤ share ≤ 1)