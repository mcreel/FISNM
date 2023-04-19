@Base.kwdef struct MA2 <: DGP 
    N::Int
end

# ----- Model-specific utilities -----------------------------------------------
insupport(::MA2, θ₁, θ₂) = (-2 < θ₁ < 2) && (-1 < θ₂ < 1) && (θ₂ - abs(θ₁) > -1)
insupport(d::MA2, θ::AbstractVector) = insupport(d, θ...)

function simulate_ma2(
    θ₁::Float32, θ₂::Float32, n::Int
)
    ϵ = randn(Float32, n+2)
    ϵ[3:end] .+ θ₁ .* ϵ[2:end-1] .+ θ₂ .* ϵ[1:end-2]
end

# MLE functions
# This fills in the covariance matrix for MA2
@inbounds @views function Σ!(θ, Σ)
    n = size(Σ,1)
    ϕ₁, ϕ₂ = θ 
    for t = 1:n
        Σ[t,t] = 1.0 + ϕ₁^2+ ϕ₂^2
    end
    for t = 1:n-1
        Σ[t,t+1] = ϕ₁*(1.0 + ϕ₂)
        Σ[t+1,t] = ϕ₁*(1.0 + ϕ₂)
    end
    for t = 1:n-2
        Σ[t,t+2] = ϕ₂ 
        Σ[t+2,t] = ϕ₂
    end
end

@views function likelihood(d::MA2, X, θ, Σ, Σ⁻¹)
    insupport(d, θ...) || return -Inf
    n = size(X, 1)
    Σ!(θ, Σ)
    Σ⁻¹ .= inv(Σ)
    # Return loglikelihood without constant part
    -log(det(Σ))/(2n) - dot(X, Σ⁻¹, X)
end

# ----- DGP necessary functions ------------------------------------------------
# Use rejection sampling to stay inside identified region
@views function priordraw(d::MA2, S::Int)::Matrix{Float32}
    θ = zeros(Float32, 2, S)
    Threads.@threads for i ∈ axes(θ, 2)
        ok = false
        θ₁ = 0f0
        θ₂ = 0f0
        while !ok
            θ₁::Float32 = 4rand() - 2
            θ₂::Float32 = 2rand() - 1
            ok = insupport(d, θ₁, θ₂)
        end
        θ[:, i] = [θ₁, θ₂]
    end

    θ
end

# Generate S samples of length N with K features and P parameters
# Returns are: (K × S × N), (P × S)
@views function generate(d::MA2, S::Int)
    y = priordraw(d, S)
    x = zeros(Float32, d.N, S)
    @inbounds Threads.@threads for s ∈ axes(x, 2)
        x[:, s] = simulate_ma2(y[:, s]..., d.N)
    end

    permutedims(reshape(x, 1, d.N, S), (1, 3, 2)), y
end

@views function generate(θ::Vector{Float32}, d::MA2, S::Int)
    insupport(d, θ...) || throw(ArgumentError("θ is not in support"))
    x = zeros(Float32, d.N, S)
    @inbounds Threads.@threads for s ∈ axes(x, 2)
        x[:, s] = simulate_ma2(θ..., d.N)
    end

    permutedims(reshape(x, 1, d.N, S), (1, 3, 2))
end

nfeatures(::MA2) = 1
nparams(::MA2) = 2

priorpred(d::MA2) = [0., 1/3]
priorerror(d::MA2) = [1., .5] # TODO !