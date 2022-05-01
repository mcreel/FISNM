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
    for s = 1:S
        # draw the params from their priors, informed by fit to SP500. Draws are from Gaussian with the estimated means and std. errors.
        μ = rand(Normal(0.037, 0.018))
        ρ = rand(Normal(-0.097, 0.041))
        ω = rand(Normal(-0.054, 0.019)) 
        γ = rand(Normal(-0.25, 0.036))
        β = 1.1 # make sure β is in stationary region
        while β > 1.0
            β = rand(Normal(0.93, 0.018))
        end    
        α = rand(Normal(0.12, 0.04))
        # the parameter vector
        θ = [μ, ρ, ω, γ, β, α]
        # get y and x for the sample s
        y[:,s] = θ   
        x[:,s*n-n+1:s*n]  = simulate(EGARCH{1,1,1}([ω,γ,β,α]), n;warmup=50, meanspec=AR{1}([μ, ρ])).data
    end
    Float32.(x), Float32.(y)
end    


