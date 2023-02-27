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
        BatchNorm(chan_out, leakyrelu),
        Conv((1, kernel_size), chan_out => chan_out, dilation = dilation, 
            pad = pad),
        BatchNorm(chan_out, leakyrelu),
    )
    residual || return causal_conv
    # Skip connection (residual net)
    residual_conv = Conv((1, 1), chan_in => chan_out)
    Chain(
        Parallel(+, causal_conv, residual_conv),
        x -> leakyrelu.(x)
    )
end

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