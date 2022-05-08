using ARCHModels, Distributions, DelimitedFiles, Statistics

# estimates EGARCH with SP500 data, to get reasonable parameters
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
    x = zeros(1,n*S)    # the Garch data for each sample
    ms = [0.066, -0.08, 0.047, 0.72, 0.19]
    ss = [0.019, 0.036, 0.017, 0.059, 0.049]
    for s = 1:S
        # draw the params from their priors, informed by fit to SP500. Draws are from uniform with mean equal to estimated parameters, and over +/- 5 sds.
        ok = false
        μ, ρ, ω, β, α = zeros(5)
        while !ok
            μ = ms[1] + 10.0*ss[1]*rand() - 5.0*ss[1]
            ρ = ms[2] + 10.0*ss[2]*rand() - 5.0*ss[2]
            ω = ms[3] + 10.0*ss[3]*rand() - 5.0*ss[3]
            β = ms[4] + 10.0*ss[4]*rand() - 5.0*ss[4]
            α = ms[5] + 10.0*ss[5]*rand() - 5.0*ss[5]
            ok = (β + α < 0.99) && (ω > 0.0) && (α > 0.0) && (β > 0.)
        end    
        # the parameter vector
        θ = [μ, ρ, ω, β, α]
        # get y and x for the sample s
        y[:,s] = θ   
        data = simulate(GARCH{1,1}([ω,β,α]), n;warmup=50, meanspec=AR{1}([μ, ρ])).data
        # drop outliers
        q01 = quantile(data,0.01)
        q99 = quantile(data,0.99)
        data .= max.(data, q01)
        data .= min.(data, q99)
        x[:,s*n-n+1:s*n] = data 
    end
    Float32.(x), Float32.(y)
end    

