using CSV, DataFrames

# use rejection sampling to stay inside 
# identified region
function sample_from_prior()
    ok = false
    θ1 = 0.
    θ2 = 0.
    while !ok
        θ1 = 4. * rand() - 2.
        θ2 = 2. * rand() - 1.
        ok = (θ2 > -1. + θ1) & (θ2 > -1. - θ1)
    end
    [θ1, θ2]
end

function ma2(θ, n)
    e = randn(n+2)
    e[3:end] .+ θ[1].*e[2:end-1] .+ θ[2].*e[1:end-2]
end

function makedata(n, reps)
    params = zeros(reps, 2)
    data = zeros(n, reps)
    for i = 1:reps
        θ = sample_from_prior()
        params[i,:] = θ
        data[:,i] = ma2(θ, n)
    end
    params, data
end

n = 100
mcreps = 1000
params, data = makedata(n, mcreps)
df = DataFrame(params, :auto)
CSV.write("params$n.csv", df)
df = DataFrame(data, :auto)
CSV.write("data$n.csv", df)


