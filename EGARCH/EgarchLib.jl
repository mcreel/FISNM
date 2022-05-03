using ARCHModels, Distributions, DelimitedFiles

# estimates EGARCH with SP500 data, to get reasonable parameters
# and standard errors.
function calibrate()
    data = readdlm("sp500.csv",',')
    data = data[2:end,2]
    data = Float64.(data)
    f = fit(EGARCH{1,1,1}, data;  meanspec=AR{1})
end

# generates S samples of length n
# returns are:
# x: 1 X S*n vector of data from EGARCH model
# y: 6 X S*n vector of parameters used to generate each sample
function dgp(n, S)
    y = zeros(6, S)     # the parameters for each sample
    x = zeros(1,n*S)    # the Garch data for each sample
    ms = [0.036, -0.098, -0.054, -0.25, 0.93, 0.12]
    ss = [0.018, 0.041, 0.019, 0.036, 0.018, 0.042]
    for s = 1:S
        # draw the params from their priors, informed by fit to SP500. Draws are from uniform with mean equal to estimated parameters, and over +/- 5 sds.
        μ = ms[1] + 10.0*ss[1]*rand() - 5.0*ss[1]
        ρ = ms[2] + 10.0*ss[2]*rand() - 5.0*ss[2]
        ω = ms[3] + 10.0*ss[3]*rand() - 5.0*ss[3]
        γ = ms[4] + 10.0*ss[4]*rand() - 5.0*ss[4]
        α = ms[6] + 10.0*ss[6]*rand() - 5.0*ss[6]
        β = 1.1 # make sure β is in stationary region
        while β > 0.99
            β = ms[5] + 10.0*ss[5]*rand() - 5.0*ss[5]
        end    
        # the parameter vector
        θ = [μ, ρ, ω, γ, β, α]
        # get y and x for the sample s
        y[:,s] = θ   
        x[:,s*n-n+1:s*n]  = simulate(EGARCH{1,1,1}([ω,γ,β,α]), n;warmup=50, meanspec=AR{1}([μ, ρ])).data
    end
    Float32.(x), Float32.(y)
end    


