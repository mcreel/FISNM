using Flux
using StatsBase

# Transform (K × S × T) arrays to RNN or CNN format
tabular2rnn(X) = [view(X, :, :, i) for i ∈ axes(X, 3)]
@views tabular2conv(X) = permutedims(reshape(X, size(X)..., 1), (4, 3, 1, 2))

# In the following losses, Ŷ is always the sequence of predictions
# RMSE on last item only
rmse_full(Ŷ, Y) = mean(sqrt.(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ, dims=2)))
rmse_loss(Ŷ, Y) = mean(sqrt.(mean(abs2.(Ŷ[end] - Y),dims=2)))
#rmse_conv(Ŷ, Y) = sqrt(mean(abs2, Ŷ - Y))
rmse_conv(Ŷ, Y) = mean(sqrt.(mean(abs2.(Ŷ - Y), dims=2)))
# MSE on full sequence predicton
mse_full(Ŷ, Y) = sum(mean(abs2.(ŷᵢ - Y) for ŷᵢ ∈ Ŷ))
mse_conv(Ŷ, Y ) = sum(abs2, Ŷ - Y)
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
lstm_net(n_hidden, n_out, n_in, dev=cpu) = 
    build_net(hidden_nodes=n_hidden, output_nodes=n_out, input_nodes=n_in, 
        add_dropout=false, dev=dev)
bilstm_net(n_hidden, n_out, n_in, dev=cpu) = 
    build_bidirectional_net(hidden_nodes=n_hidden, output_nodes=n_out, input_nodes=n_in, 
        add_dropout=false, dev=dev)   

# Creates a 'bias-corrected' version of a pre-trained TCN
function bias_corrected_tcn(tcn, X, Y)
    # Compute estimate of a TCN on some data X, Y. Uses this estimate to compute the bias
    # of the net. Adds bias correction accordingly and returns the bias corrected net.
    Flux.testmode!(tcn) # In case of dropout / layer normalization
    bias = mean(tcn(X) - Y, dims=2)
    # Beware of memory issues with the deepcopy, ideally always overwrite, 
    # i.e., tcn = bias_corrected_tcn(tcn, ...)
    Chain(deepcopy(tcn), x -> x .- bias) 
end


# Trains a recurrent neural network
function train_rnn!(
    m, opt, dgp, n, dtY; 
    epochs=1000, batchsize=32, passes_per_batch=10, dev=cpu, loss=mse_full,
    validation_loss=true, validation_frequency=10, validation_size=2000, verbosity=1, bidirectional=false
)
    Flux.trainmode!(m) # In case we have dropout / batchnorm
    θ = Flux.params(m) # Extract parameters
    best_model = deepcopy(m)
    best_loss = Inf
    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = map(dev, dgp(n, validation_size))
        #StatsBase.transform!(dtY, Yv) # NOTE: want to see loss for real scale
        Xv = tabular2rnn(Xv)
        losses = zeros(epochs)
    end
    
    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = map(dev, dgp(n, batchsize)) # Generate a new batch
        X = tabular2rnn(X) 
        # Standardize targets for MSE scaling
        # no need to do this for every sample, use a high accuracy
        # transform from large draw from prior
        StatsBase.transform!(dtY, Y)

        # ----- training ---------------------------------------------
        for i = 1:passes_per_batch
            Flux.reset!(m)
            # Compute loss and gradients
            if bidirectional # Special case for bidirectional RNN
                ∇ = gradient(θ) do
                    Ŷ = [m(X)]
                    loss(Ŷ, Y)
                end
                Flux.update!(opt, θ, ∇) # Take gradient descent step
            else
                ∇ = gradient(θ) do
                    m(X[1]) # don't use first, to warm up state
                    Ŷ = [m(x) for x ∈ X[2:end]]
                    loss(Ŷ, Y)
                end
                Flux.update!(opt, θ, ∇) # Take gradient descent step
            end
        end

        # Compute validation loss and print status if verbose
        if validation_loss && mod(epoch, validation_frequency)==0
            Flux.reset!(m)
            Flux.testmode!(m)
            if bidirectional # Special case for bidirectional RNN
                Ŷ = [m(Xv)]
            else
            m(Xv[1]) # Warm up state on first observation
            Ŷ = [StatsBase.reconstruct(dtY, m(x)) for x ∈ Xv]
            end
            current_loss = loss(Ŷ, Yv)
            if current_loss < best_loss
                best_loss = current_loss
                best_model = deepcopy(m)
            end
            losses[epoch] = current_loss
            Flux.trainmode!(m)
            epoch % verbosity == 0 && @info "$epoch / $epochs"   best_loss  current_loss
        else
            epoch % verbosity == 0 && @info "$epoch / $epochs"
        end
    end
    # Return losses if tracked
    if validation_loss
        losses, best_model
    else
        nothing, nothing
    end
end

# Train a convolutional neural network
function train_cnn!(
    m, opt, dgp, dtY;
    epochs=1000, batchsize=32, passes_per_batch=10, dev=cpu, loss=rmse_conv,
    validation_loss=true, validation_frequency=10, validation_size=2_000, verbosity=1, 
    transform=true
)
    Flux.trainmode!(m) # In case we have dropout / layer normalization
    θ = Flux.params(m) # Extract parameters
    best_model = deepcopy(m) 
    best_loss = Inf

    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = generate(dgp, validation_size, dev=dev)
        # Want to see validation RMSE on original scale => no rescaling
        Xv = tabular2conv(Xv)
        losses = zeros(epochs)
    end

    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = generate(dgp, batchsize, dev=dev) # Generate a new batch
        transform && StatsBase.transform!(dtY, Y)
        # Transform features to format for CNN
        X = tabular2conv(X)
        # ----- Training ---------------------------------------------
        for _ ∈ 1:passes_per_batch
            # Compute loss and gradients
            ∇ = gradient(θ) do
                Ŷ = m(X)
                loss(Ŷ, Y)
            end
            Flux.update!(opt, θ, ∇) # Take gradient descent step
        end
        # Compute validation loss and print status if verbose
        # Do this for the last 100 epochs, too, in case frequency is low
        if validation_loss && (mod(epoch, validation_frequency)==0 || epoch > epochs - 1000)
            Flux.testmode!(m)
            Ŷ = transform ? StatsBase.reconstruct(dtY, m(Xv)) : m(Xv)
            current_loss = loss(Ŷ, Yv)
            if current_loss < best_loss
                best_loss = current_loss
                best_model = deepcopy(m)
            end
            losses[epoch] = current_loss
            Flux.trainmode!(m)
            epoch % verbosity == 0 && @info "$epoch / $epochs"   best_loss  current_loss
        else
            epoch % verbosity == 0 && @info "$epoch / $epochs"
        end
    end
    # Return losses if tracked
    if validation_loss
        losses, best_model
    else
        nothing
    end
end
