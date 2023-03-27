@Base.kwdef struct GARCH <: DGP 
    N::Int
end

# ----- Model-specific utilities -----------------------------------------------
# the likelihood function, alternative version with reparameterization
function simulate_garch(
    lrv::Float32, βplusα::Float32, share::Float32, n::Int)
    ω = (1f0 - βplusα) * lrv
    β = share * βplusα
    α = (1f0 - share) * βplusα

    h, y = lrv, 0f0
    z = randn(Float32, n)
    ys = zeros(Float32, n)

    @inbounds @simd for t ∈ eachindex(ys)
        h = ω + α * y ^ 2f0 + β * h
        y = √h * z[t]
        ys[t] = y
    end

    ys
end

θbounds(::GARCH) = (Float32[.0001, 0, 0], Float32[1, .99, 1])

insupport(::GARCH, lrv::Float32, βplusα::Float32, share::Float32) = 
    (0 < lrv < 1) && (0 < βplusα < 1) && (0 < share < 1)
# ----- DGP necessary functions ------------------------------------------------

# GARCH is parameterized in terms of long run variance β+α, and β's share of β+α
priordraw(d::GARCH, S::Int) = uniformpriordraw(d, S) # [lrv; β+α; share]

# Generate S samples of length N with K features and P parameters
# Returns are: (K × S × N), (P × S)
@views function generate(d::GARCH, S::Int)
    y = priordraw(d, S)
    x = zeros(Float32, d.N, S) # the Garch data for each sample

    @inbounds Threads.@threads for s ∈ axes(x, 2)
        x[:, s] .= simulate_garch(y[:, s]..., d.N)
    end

    permutedims(reshape(x, 1, d.N, S), (1, 3, 2)), y
end

@views function generate(θ::Vector{Float32}, d::GARCH, S::Int)    
    insupport(d, θ...) || throw(ArgumentError("θ is not in support"))
    x = zeros(Float32, d.N, S) # the Garch data for each sample
    Threads.@threads for s ∈ axes(x, 2)
        x[:, s] .= simulate_garch(θ..., d.N)
    end
    # Add a dimension to x and return
    permutedims(reshape(x, 1, d.N, S), (1, 3, 2))
end


nfeatures(::GARCH) = 1
nparams(::GARCH) = 3

@views function likelihood(::GARCH, X, θ)
    lrv, βplusα , share  = θ
    ω = (1 - βplusα) * lrv
    β = share * βplusα
    α = (1 - share) * βplusα
    n = size(X, 1)
    h = zeros(n)
    X = X .^ 2
    h[1] = lrv
    @inbounds for t ∈ 2:n
        h[t] = ω + α * X[t-1] + β * h[t-1]
    end
    # Drop the constant part from the loglikelihood (-log(sqrt(2π)))
    mean(-0.5log.(h) .- X ./ (2h))
end