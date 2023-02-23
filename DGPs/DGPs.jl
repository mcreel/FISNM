abstract type DGP end

error_msg(t, f) = error("$f not implemented for ::$t")

# Generic functions for DGPs
priordraw(d::DGP, args...) = error_msg(typeof(d), "priordraw")
generate(d::DGP, args...) = error_msg(typeof(d), "generate")
nfeatures(d::DGP, args...) = error_msg(typeof(d), "nfeatures")
nparams(d::DGP, args...) = error_msg(typeof(d), "nparams")

# Data transform for a particular DGP
data_transform(d::DGP, S::Int; dev=cpu) = fit(ZScoreTransform, dev(priordraw(d, S)))

# Generate to specific device directly
generate(d::DGP, S::Int; dev=cpu) = map(dev, generate(d, S))

# Generate prior parameters according to uniform with lower and upper bounds
function uniformpriordraw(d::DGP, S::Int)
    lb, ub = θbounds(d)
    (ub .- lb) .* rand(Float32, length(lb), S) .+ lb 
end



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
            Dense(d.N ÷ summary_size => d.N ÷ summary_size, hardtanh), # this is a new layer
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

function build_lstm(
    d::DGP; hidden_nodes=32, hidden_layers=2, activation=tanh, dev=cpu
)
    dim_in, dim_out = n_features(d), n_params(d)
    dev(
        Chain(
            Dense(dim_in => hidden_nodes, activation),
            [LSTM(hidden_nodes => hidden_nodes) for _ ∈ 1:hidden_layers]...,
            Dense(hidden_nodes => dim_out)
        )
    )
end