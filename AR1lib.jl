# generates a vector of length n from the AR1 model with parameter ρ
@views function AR1(ρ, n)
    y = zeros(n)
    y[1] = randn()/(1.0 - ρ^2.0)
    for t = 2:n
        y[t] = ρ*y[t-1] + randn()
    end
    Float32.(y)
end    

# generates S samples of length n
# returns are:
# x: 1XSn vector of data from AR1
# y: 1XSn vector of parameters used to generate each sample
@views function dgp(n, S)
    x = zeros(1, n*S)
    y = zeros(1, S)
    for s = 1:S
        ρ = 2.0*rand() - 1.0
        y[:,s] .= ρ
        x[:,s*n-n+1:s*n] = AR1(ρ,n)  
    end
    Float32.(x), Float32.(y)
end    


