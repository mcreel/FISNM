using ARCHModels, Distributions, DelimitedFiles, Statistics, Flux, CUDA, BSON

# estimates GARCH with SP500 data, to get reasonable parameters
# and standard errors.
function calibrate()
    data = readdlm("sp500.csv",',')
    data = data[2:end,2]
    data = Float64.(data)
    f = fit(GARCH{1,1}, data;  meanspec=AR{1})
end

# generates S samples of length n
# returns are:
# x: 1 X S*n vector of data from EGARCH model
# y: 6 X S*n vector of parameters used to generate each sample
function dgp(n, S)
    y = zeros(5, S)     # the parameters for each sample
    x = zeros(n, S)    # the Garch data for each sample
    ms = [0.066, -0.08, 0.047, 0.72, 0.19]
    ss = [0.019, 0.036, 0.017, 0.059, 0.049]
    for s = 1:S
        # draw the params from their priors, informed by fit to SP500. Draws are from uniform with mean equal to estimated parameters, and over +/- 5 sds.
        ok = false
        μ, ρ, ω, β, α = zeros(5)
        while !ok
            μ = ms[1] + 5.0*ss[1]*randn()
            ρ = ms[2] + 5.0*ss[2]*randn()
            ω = ms[3] + 5.0*ss[3]*randn()
            β = ms[4] + 5.0*ss[4]*randn()
            α = ms[5] + 5.0*ss[5]*randn()
            # stationary, and unconditional daily std. dev less than 2
            ok = (β + α < 1.0) && (ω > 0.0) && (α > 0.0) && (β > 0.) && (ω/(1-α-β)) < 9.0 && (ω/(1-α-β)) >= 0.1 
        end    
        # the parameter vector
        θ = [μ, ρ, ω, β, α]
        # get y and x for the sample s
        y[:,s] = θ   
        x[:,s] = simulate(GARCH{1,1}([ω,β,α]), n;warmup=1000, meanspec=AR{1}([μ, ρ])).data
    end
    Float32.(x), Float32.(y)
end    

function batch(x,y)
    n, b = size(x) # series length (n) and number of samples (b)
    x = [x[t:t,:] for t = 1:n]
    x, y
end

using Statistics
function scaledata(x)
    ql = quantile(vec(x),0.005)
    qu = quantile(vec(x),0.995)
    x = max.(x, ql)
    x = min.(x, qu)
    mn = mean(x)
    sd = std(x)
    x = Float32.((x .- mn) ./sd)
    info = [ql, qu, mn, sd]
    return x, info
end

function scaledata(x, info)
    ql, qu, mn, sd = info
    x = max.(x, ql)
    x = min.(x, qu)
    x = Float32.((x .- mn) ./sd)
end

