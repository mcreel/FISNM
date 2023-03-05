# Transform (K × S × T) arrays to RNN or CNN format
tabular2rnn(X) = [view(X, :, :, i) for i ∈ axes(X, 3)]
@views tabular2conv(X) = permutedims(reshape(X, size(X)..., 1), (4, 3, 1, 2))

# Note that when doing RMSE, the root is taken on each parameter individually


# In the following losses, Ŷ is always the sequence of predictions (RNN style prediction)
rmse_full(Ŷ, Y) = mean(sqrt, mean(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ), dims=2))
mse_full(Ŷ, Y) = mean(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ))
rmse_last(Ŷ, Y) = mean(sqrt, mean(abs2.(Ŷ[end] - Y), dims=2))
mse_last(Ŷ, Y) = mean(mean(abs2.(Ŷ[end] - Y)))

# In the following losses, Ŷ is the same dimension as Y (CNN style prediction)
rmse_conv(Ŷ, Y) = mean(sqrt.(mean(abs2.(Ŷ - Y), dims=2)))
mse_conv(Ŷ, Y) = mean(abs2, Ŷ - Y)