@Base.kwdef struct MA2 <: DGP 
    N::Int
end

# ----- Model-specific utilities -----------------------------------------------
insupport(::MA2, θ₁, θ₂) = (-2 < θ₁ < 2) && (-1 < θ₂ < 1) && (θ₂ - abs(θ₁) > -1)

function simulate_ma2(θ₁::Float32, θ₂::Float32, n::Int)
    ϵ = randn(Float32, n+2)
    ϵ[3:end] .+ θ₁ .* ϵ[2:end-1] .+ θ₂ .* ϵ[1:end-2]
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
    x = zeros(Float32, 1, S, d.N)

    Threads.@threads for s ∈ axes(x, 2)
        x[1, s, :] = simulate_ma2(y[:, s]..., d.N)
    end

    x, y
end

nfeatures(::MA2) = 1
nparams(::MA2) = 2

priorpred(d::MA2) = [0., 1/3]
priorerror(d::MA2) = [1., .5] # TODO !