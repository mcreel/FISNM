using ARCHModels, Distributions, DelimitedFiles, Statistics, Flux, CUDA, BSON

# estimates GARCH with SP500 data, to get reasonable parameters
# and standard errors.
function calibrate()
    data = readdlm("sp500.csv",',')
    data = data[2:end,2]
    data = Float64.(data)
    f = fit(GARCH{1,1}, data;  meanspec=AR{1})
end

# draw the params from their priors, informed by fit to SP500. Draws are from uniform with mean equal to estimated parameters, and over +/- 5 sds.
function PriorDraw()
    ω = 0.001 + 0.998*rand()
    βplusα = 0.99*rand()
    share = rand()
    [ω, βplusα, share]
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
    y = zeros(3, S)     # the parameters for each sample
    x = zeros(n, S)    # the Garch data for each sample
    for s = 1:S
        # the parameter vector
        θ = PriorDraw()
        ω, β_plus_α , share  = θ
        β = share*β_plus_α 
        α = (1.0 - share)*β_plus_α
        # get y and x for the sample s
        y[:,s] = θ   
        x[:,s] = simulate(GARCH{1,1}([ω,β,α]), n;warmup=1000).data
    end
    x .-= mean(x)
    Float32.(x), Float32.(y)
end    

# the likelihood function, alternative version with reparameterization
@views function garch11(θ, y)
    # dissect the parameter vector
    ω, β_plus_α , share  = θ
    β = share*β_plus_α 
    α = (1.0 - share)*β_plus_α
    ylag = y[1:end-1]
    y = y[2:end]
    n = size(y,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    #h[1] = var(y[1:10])
    h[1] = var(y)
    for t = 2:n
        h[t] = ω + α*(y[t-1])^2. + β*h[t-1]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*(y.^2.)./h
end


