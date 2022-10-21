# -----------------------------------------
# Temporal Convolutional Network in Flux.jl
# Author: Jonathan Chassot, May 17, 2022
# -----------------------------------------
# Reference:
#       Shaojie Bai, J. Zico Kolter, Vladlen Koltun. (2018)
#       An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
#       https://arxiv.org/abs/1803.01271
# -----------------------------------------
# Note that this gist uses batch normalization instead of weight normalization as in the original paper
# -----------------------------------------
# Temporal blocks which compose the layers of the TCN
function TemporalBlock(
    chan_in::Int, chan_out::Int; 
    dilation::Int, kernel_size::Int,
    residual::Bool = true, pad = SamePad()
)
    # Causal convolutions
    causal_conv = Chain(
        Conv((1, kernel_size), chan_in => chan_out, dilation = dilation, 
            pad = pad),
        BatchNorm(chan_out, relu),
        Conv((1, kernel_size), chan_out => chan_out, dilation = dilation, 
            pad = pad),
        BatchNorm(chan_out, relu),
    )
    residual || return causal_conv
    # Skip connection (residual net)
    residual_conv = Conv((1, 1), chan_in => chan_out)
    Chain(
        Parallel(+, causal_conv, residual_conv),
        x -> relu.(x)
    )
end

# Temporal Convolutional Network with `length(channels) - 1` layers
# e.g., `TCN([1, 8, 8, 1], kernel_size = 3)` constructs a TCN with 3 TemporalBlock layers:
#   1.) 1 => 8, dilation = 2⁰ = 1
#   2.) 8 => 8, dilation = 2¹ = 2
#   3.) 8 => 1, dilation = 2² = 4
# each of them with `kernel_size = 3` 
function TCN(
    channels::AbstractVector{Int}; 
    kernel_size::Int, dilation_factor::Int = 2, 
    residual::Bool = true, pad = SamePad()
)
    Chain([TemporalBlock(chan_in, chan_out, dilation = dilation_factor ^ (i - 1), 
        kernel_size = kernel_size, residual = residual,
        pad = pad) 
        for (i, (chan_in, chan_out)) ∈ enumerate(zip(channels[1:end-1], channels[2:end]))]...)
end

# Computes the receptive field size for a specified dilation, kernel size, and number of layers
receptive_field_size(dilation::Int, kernel_size::Int, layers::Int) = 
    1 + (kernel_size - 1) * (dilation ^ layers - 1) / (dilation - 1)

# Minimum number of layers necessary to achieve a specified receptive field size
# (take ceil(Int, necessary_layers(...)) for final number of layers)
necessary_layers(dilation::Int, kernel_size::Int, receptive_field::Int) =
    log(dilation, (receptive_field - 1) * (dilation - 1) / (kernel_size - 1)) + 1