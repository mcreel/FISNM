@Base.kwdef struct JD <: DGP
    N::Int
end

# ----- Model-specific utilities -----------------------------------------------
isweekday(d::Int)::Bool = (d % 7) % 6 != 0

# [μ; κ; α; σ; ρ; λ₀; λ₁; τ]
θbounds(::JD) = (
    #         μ,    κ,    α,   σ,    ρ,    λ₀, λ₁,    τ   
    Float32[-.05, .01, -10, 0.1, -.99,  -.02,  3, -.02], 
    Float32[ .05, .30,   0, 4.0, -.50,   .05,  6,  .05]
)

function insupport(dgp::JD, θ)
    lb, ub = θbounds(dgp)
    all(θ .>= lb) && all(θ .<=ub)
end

function diffusion(μ,κ,α,σ,ρ,u0,tspan)
    f = function (du,u,p,t)
        du[1] = μ # drift in log prices
        du[2] = κ.*(α.-u[2]) # mean reversion in shocks
    end
    g = function (du,u,p,t)
        du[1] = exp(u[2]/2.0)
        du[2] = σ
    end
    noise = CorrelatedWienerProcess!([1.0 ρ;ρ 1.0], tspan[1], zeros(2), zeros(2))
    sde_f = SDEFunction{true}(f,g)
    SDEProblem(sde_f,g,u0,tspan,noise=noise)
end


@views function simulate_jd(θ, n::Int; burnin::Int=100)
    trading_days = n # TODO: slightly ugly.
    days = round(Int, 1.4 * (trading_days + burnin)) # Add weekends (x + x/5*2 = 1.4x)
    min_per_day = 1_440 # Minutes per day
    min_per_tic = 10 # Minutes between tics, lower for better accuracy
    tics = round(Int, min_per_day / min_per_tic) # Number of tics per day
    dt = 1/tics # Divisions per day
    closing = round(Int, 390 / min_per_tic) # Tic at closing (390 = 6.5 * 60)

    # Solve the diffusion
    μ, κ, α, σ, ρ, λ₀, λ₁, τ = θ
    τ = max(0, τ) # The prior allows for negative measurement error, to allow an accumulation at zero
    u₀ = [μ; α]
    prob = diffusion(μ, κ, α, σ, ρ, u₀, (0., days))
    λ₀⁺ = max(0, λ₀) # The prior allows for negative rate, to allow an accumulation at zero

    # # Jump in log price
    rate(u, p, t) = λ₀⁺

    # Jump is random sign time λ₁ times current std. dev.
    function affect!(integrator)
        integrator.u[1] = integrator.u[1] + rand([-1., 1.]) * λ₁ * exp(integrator.u[2] / 2)
        nothing
    end

    jump = ConstantRateJump(rate, affect!)
    jump_prob = JumpProblem(prob, Direct(), jump)

    # Do the simulation
    sol = solve(jump_prob, SRIW1(), dt=dt, adaptive=false)

    # Get log price, with measurement error 
    # Trick: we only need very few log prices, 39 per trading day, use smart filtering
    lnPs = (
        [sol(t)[1] + τ * randn() for t ∈ Iterators.take(p, closing)]
        for (_, p) ∈ Iterators.drop(
            Iterators.filter(
                x -> isweekday(x[1]), 
                enumerate(Iterators.partition(dt:dt:days, tics))), 
            burnin - 1)
    )

    # Get log price at end of trading days We will compute lag, so lose first
    lnP_trading = zeros(Float64, trading_days + 1)
    rv = zeros(Float64, trading_days + 1)
    bv = zeros(Float64, trading_days + 1) 

    p₋₁ = 0.
    @inbounds for (t, p) ∈ enumerate(lnPs)
        r = abs.(diff([p₋₁; p]))
        bv[t] = dot(r[2:end], r[1:end-1])
        rv[t] = dot(r[2:end], r[2:end])
        p₋₁ = p[end]
        lnP_trading[t] = p[end]
    end
    
    [diff(lnP_trading) rv[2:end] π/2 .* bv[2:end]]
end


# ----- DGP necessary functions ------------------------------------------------
priordraw(d::JD, S::Int) = uniformpriordraw(d, S) # [μ; κ; α; σ; ρ; λ₀; λ₁; τ]

@views function generate(d::JD, S::Int)
    y = priordraw(d, S)
    x = zeros(Float32, d.N, 3, S)
    Threads.@threads for s ∈ axes(x, 3)
        x[:, :, s] = simulate_jd(y[:, s], d.N)
    end
    permutedims(x, (2, 3, 1)), y
end

@views function generate(θ::Vector{Float32}, d::JD, S::Int)
    # insupport(d, θ...) || throw(ArgumentError("θ is not in support"))
    x = zeros(Float32, d.N, 3, S)
    Threads.@threads for s ∈ axes(x, 3)
        x[:, :, s] = simulate_jd(θ, d.N)
    end
    permutedims(x, (2, 3, 1))
end


nfeatures(::JD) = 3
nparams(::JD) = 8

