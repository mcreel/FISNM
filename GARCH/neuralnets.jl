using Flux
using StatsBase

# Some helpers
tabular2rnn(X) = [view(X, i:i, :) for i ∈ 1:size(X, 1)]

# In the following losses, Ŷ is always the sequence of predictions
# RMSE on last item only
rmse_loss(Ŷ, Y) = sqrt(mean(abs2.(Ŷ[end] - Y)))
# MSE on full sequence predicton
mse_full(Ŷ, Y) = sum(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ))
# ------------------------------------------------------------------------------------------

# Create a neural network according to chosen specification
function build_net(;
    input_nodes=1, output_nodes=3, hidden_nodes=32, dropout_rate=.2, 
    hidden_layers=2, rnn_cell=Flux.LSTM, activation=tanh, add_dropout=true,
    dev=cpu
)
    # Input layer
    layers = Any[Dense(input_nodes => hidden_nodes, activation)] 
    # Hidden layers
    for i ∈ 1:hidden_layers 
        add_dropout && push!(layers, Dropout(dropout_rate))
        push!(layers, rnn_cell(hidden_nodes => hidden_nodes))
    end
    # Output layer
    push!(layers, Dense(hidden_nodes => output_nodes))
    Chain(layers...) |> dev
end

# Bidirectional RNN, see:
#   - https://github.com/maetshju/flux-blstm-implementation/blob/master/01-blstm.jl
#   - https://github.com/FluxML/model-zoo/blob/master/contrib/audio/speech-blstm/01-speech-blstm.jl
struct BiRNN
    encoder
    forward
    backward
    output
end
Flux.@functor BiRNN

function (m::BiRNN)(x)
    x = m.encoder.(x)
    x = vcat.(m.forward.(x), Flux.flip(m.backward, x))
    m.output.(x)[end]
end

# Create a Bidirectional RNN
function build_bidirectional_net(;
    input_nodes=1, output_nodes=3, hidden_nodes=32, dropout_rate=.2, 
    hidden_layers=2, rnn_cell=Flux.LSTM, activation=tanh, add_dropout=true,
    dev=cpu
)
    # Encoder / input layer
    enc = Dense(input_nodes => hidden_nodes, activation)
    # RNN layers
    layers = Any[]
    for i ∈ 1:hidden_layers
        add_dropout && push!(layers, Dropout(dropout_rate))
        push!(layers, rnn_cell(hidden_nodes => hidden_nodes))
    end
    rnn₁ = Chain(layers...)
    rnn₂ = deepcopy(rnn₁)
    # Output layer
    out = Dense(2hidden_nodes => output_nodes)
    # Bidirectional RNN
    BiRNN(enc, rnn₁, rnn₂, out) |> dev
end

# Alias for LSTM and BiLSTM net construction
lstm_net(n_hidden, dev=cpu) = 
    build_net(hidden_nodes=n_hidden, add_dropout=false, dev=dev)
bilstm_net(n_hidden, dev=cpu) = 
    build_bidirectional_net(hidden_nodes=n_hidden, add_dropout=False, dev=dev)   

# ------------------------------------------------------------------------------------------

# Trains a recurrent neural network
function train_rnn!(
    m, opt, dgp, n, S, dtY; 
    epochs=100, batchsize=32, dev=cpu, loss=mse_full,
    validation_loss=true, verbosity=1, bidirectional=false
)
    Flux.trainmode!(m) # In case we have dropout / batchnorm
    θ = Flux.params(m) # Extract parameters
    
    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = map(dev, dgp(n, S))
        StatsBase.transform!(dtY, Yv)
        Xv = tabular2rnn(Xv)
        losses = zeros(epochs)
    end
    
    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = map(dev, dgp(n, S)) # Generate a new batch
        # Standardize targets for MSE scaling
        # no need to do this for every sample, use a high accuracy
        # transform from large draw from prior
        StatsBase.transform!(dtY, Y)

        # ----- Minibatch training ---------------------------------------------
        for idx ∈ Iterators.partition(1:S, batchsize)
            Flux.reset!(m)
            # Extract batch, transform features to format for RNN
            Xb, Yb = tabular2rnn(X[:, idx]), Y[:, idx]
            # Compute loss and gradients
            if bidirectional # Special case for bidirectional RNN
                ∇ = gradient(θ) do
                    Ŷ = [m(Xb)]
                    loss(Ŷ, Yb)
                end
                Flux.update!(opt, θ, ∇) # Take gradient descent step
            else
                ∇ = gradient(θ) do
                    m(Xb[1]) # don't use first, to warm up state
                    Ŷ = [m(x) for x ∈ Xb[2:end]]
                    loss(Ŷ, Yb)
                end
                Flux.update!(opt, θ, ∇) # Take gradient descent step
            end
        end

        # Compute validation loss and print status if verbose
        if validation_loss
            Flux.reset!(m)
            Flux.testmode!(m)
            if bidirectional # Special case for bidirectional RNN
                Ŷ = [m(Xv)]
            else
                m(Xv[1]) # Warm up state on first observation
                Ŷ = [m(x) for x ∈ Xv[2:end]]
            end
            current_loss = loss(Ŷ, Yv)
            losses[epoch] = current_loss
            Flux.trainmode!(m)
            epochs % verbosity == 0 && @info "$epoch / $epochs" current_loss
        else
            epochs % verbosity == 0 && @info "$epoch / $epochs"
        end
    end
    # Return losses if tracked
    if validation_loss
        losses
    else
        nothing
    end
end

# Trains a bidirectional neural network
function train_birnn!(

)

end