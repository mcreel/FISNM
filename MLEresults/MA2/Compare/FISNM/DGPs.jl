using Random, Statistics
include("MA2/MA2Lib.jl")
include("TCN.jl")
include("NNEnsemble.jl")


abstract type DGP end

# ------------------------------- MA2 -------------------------------
@Base.kwdef mutable struct Ma2 <: DGP 
    N::Int
end

# Use rejection sampling to stay inside identified region
function priordraw(::Ma2, S::Int)
    θ = zeros(2, S)
    for i ∈ axes(θ, 2)
        ok = false
        θ1 = 0.
        θ2 = 0.
        while !ok
            θ1 = 4. * rand() - 2.
            θ2 = 2. * rand() - 1.
            ok = insupport([θ1, θ2])
        end
        θ[:, i] = [θ1, θ2]
    end
    θ
end

# ==========================================================================================
# Data Generating Processes
#   Generate S samples of length N with K features and P parameters
#   Returns are: (K × S × N), (P × S)

# ------------------------------- MA2 -------------------------------
@views function generate(d::Ma2, S::Int)
    y = priordraw(d, S)     # the parameters for each sample
    x = zeros(1, S, d.N)    # the Garch data for each sample
    for s ∈ axes(x, 2)
        x[1, s, :] = ma2(y[:, s], d.N)
    end
    Float32.(x), Float32.(y)
end

# Generate to specific device directly
generate(d::DGP, S::Int; dev=cpu) = map(dev, generate(d, S))

# ==========================================================================================
# Useful functions
n_features(d::DGP) = 1
n_params(::Ma2) = 2

# Data transform for a particular dgp
data_transform(d::DGP, S::Int; dev=cpu) = fit(ZScoreTransform, dev(priordraw(d, S)))

# Build TCN for a given DGP
function build_tcn(d::DGP; dilation=2, kernel_size=8, channels=16, summary_size=10, dev=cpu)
    # Compute TCN dimensions and necessary layers for full RFS
    n_layers = ceil(Int, necessary_layers(dilation, kernel_size, d.N))
    dim_in, dim_out = n_features(d), n_params(d)
    dev(
        Chain(
            TCN(
                vcat(dim_in, [channels for _ ∈ 1:n_layers], 1),
                kernel_size=kernel_size, # TODO: BE CAREFUL! DILATION IS ACTUALLY NOT HANDLED!!!
            ),
            Conv((1, summary_size), 1 => 1, stride=summary_size),
            Flux.flatten,
            Dense(d.N ÷ summary_size => dim_out)
        )
    )
end

# Build TCNEnsemble for a particular DGP
function build_tcn_ensemble(
    d::DGP, opt_func, n_models::Int=10; 
    dilation=2, kernel_size=8, channels=16, summary_size=10, dev=cpu
)
    TCNEnsemble(
        [build_tcn(d, dilation=dilation, kernel_size=kernel_size, channels=channels,
            summary_size=summary_size, dev=dev) for _ ∈ 1:n_models],
        [opt_func() for _ ∈ 1:n_models]
    )
end
