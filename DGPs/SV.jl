@Base.kwdef struct SV <: DGP
    N::Int
end


# ----- Model-specific utilities -----------------------------------------------
function simulate_sv(
    ϕ::Float32, ρ::Float32, σ::Float32, n::Int; 
    hmax::Float32=20f0
)
    hlag = 0f0
    ϵ = randn(Float32, n)
    η = randn(Float32, n)
    ys = zeros(Float32, n)

    @inbounds @simd for t ∈ eachindex(ys)
        # Bound the log variance at a high value
        h = min(ρ * hlag + σ * η[t], hmax)
        ys[t] = ϕ * exp(.5h) * ϵ[t]
        hlag = h
    end
    ys
end

θbounds(::SV) = (Float32[.05, 0, .05], Float32[2, .99, 1])

# ----- DGP necessary functions ------------------------------------------------
priordraw(d::SV, S::Int) = uniformpriordraw(d, S) # [ϕ; ρ; σ]

@views function generate(d::SV, S::Int)
    y = priordraw(d, S)
    x = zeros(Float32, 1, S, d.N)

    Threads.@threads for s ∈ axes(x, 2)
        x[1, s, :] = simulate_sv(y[:, s]..., d.N)
    end

    x, y
end

nfeatures(::SV) = 1
nparams(::SV) = 3