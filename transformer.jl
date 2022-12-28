using Flux
using Transformers
using Transformers: PositionEmbedding

# Helper to clone neural networks
clone(net, N::Int) = [deepcopy(net) for _ ∈ 1:N]


# ----- Input layer, pre-encoder -----------------------------------------------------------
struct InputLayer
    embedding::Dense
    position_encoding::PositionEmbedding
    dropout::Dropout
end
function InputLayer(n_features::Int; 
    d_model::Int=512, dropout::AbstractFloat=.1
)
    InputLayer(
        Dense(n_features => d_model),
        PositionEmbedding(d_model),
        Dropout(dropout)
    )
end
@Flux.functor InputLayer
function (layer::InputLayer)(x)
    x = layer.embedding(x)
    x = x .+ layer.position_encoding(x)
    layer.dropout(x)
end


# ----- Encoder ----------------------------------------------------------------------------
struct Encoder
    input_layer::InputLayer
    layers::Vector{<:Transformer}
end
function Encoder(n_features::Int; 
    d_model::Int=512, n_heads::Int=8, d_ffnn::Int=2048, 
    dropout::AbstractFloat=.1, n_layers::Int=6, future::Bool=false
)
    # Number of features per attention head
    @assert d_model % n_heads == 0 "d_model should be divisible by n_heads"
    k = d_model ÷ n_heads
    Encoder(
        InputLayer(n_features, d_model=d_model, dropout=dropout),
        clone(
            Transformer(d_model, n_heads, k, d_ffnn, future=future, 
                pdrop=dropout), 
            n_layers
        )
    )
end
@Flux.functor Encoder
function (enc::Encoder)(x)
    x = enc.input_layer(x)
    for layer ∈ enc.layers
        x = layer(x)
    end
    x
end

# ----- Decoder ----------------------------------------------------------------------------
struct Decoder
    input_layer::InputLayer
    layers::Vector{<:TransformerDecoder}
    output_layer::Dense
end
function Decoder(n_features::Int, seqlen::Int, n_outputs::Int;
    d_model::Int=512, n_heads=8, d_ffnn::Int=2048,
    dropout::AbstractFloat=.1, n_layers::Int=6
)
    # Number of features per attention head
    @assert d_model % n_heads == 0 "d_model should be divisible by n_heads"
    k = d_model ÷ n_heads
    Decoder(
        InputLayer(n_features, d_model=d_model, dropout=dropout), # TODO: Is this input layer or dense??
        clone(
            TransformerDecoder(d_model, n_heads, k, d_ffnn, pdrop=dropout), 
            n_layers
        ),
        Dense(d_model * seqlen => n_outputs)
    )
end
@Flux.functor Decoder
function (dec::Decoder)(x, x̃)
    # x̃ is the 'encoded' x, coming from the Encoder
    x = dec.input_layer(x)
    for layer ∈ dec.layers
        x̃ = layer(x, x̃)
    end
    dec.output_layer(Flux.flatten(x̃))
end

# ----- Transformer ------------------------------------------------------------------------
struct CustomTransformer
    encoder::Encoder
    decoder::Decoder
end
function CustomTransformer(n_features::Int, seqlen::Int, n_outputs::Int;
    d_model::Int=512, n_heads::Int=8, d_ffnn=2048, dropout::AbstractFloat=0.1, 
    n_layers::Int=6, future::Bool=true
)
    CustomTransformer(
        Encoder(n_features, d_model=d_model, n_heads=n_heads, 
            d_ffnn=d_ffnn, dropout=dropout, n_layers=n_layers, future=future),
        Decoder(n_features, seqlen, n_outputs, d_model=d_model, 
            n_heads=n_heads, d_ffnn=d_ffnn, dropout=dropout, n_layers=n_layers)
    )
end
@Flux.functor CustomTransformer
(t::CustomTransformer)(x) = t.decoder(x, t.encoder(x))





# ----- Utilities --------------------------------------------------------------------------
function train_transformer!(
    m::CustomTransformer, opt, dgp, dtY;
    epochs=1_000, batchsize=32, passes_per_batch=10, dev=cpu, loss=rmse_conv, # TODO: rmse_conv works here right? => make sure
    validation_loss=true, validation_frequency=10, validation_size=2_000, verbosity=1,
    transform=true
)
    Flux.trainmode!(m)
    θ = Flux.params(m)
    best_model = deepcopy(m)
    best_loss = Inf

    # Create a validation set to compute and keep track of losses
    if validation_loss
        Xv, Yv = generate(dgp, validation_size, dev=dev)
        # Want to see validation RMSE on original scale => no rescaling
        Xv = tabular2trans(Xv)
        losses = zeros(epochs)
    end

    # Iterate over training epochs
    for epoch ∈ 1:epochs
        X, Y = generate(dgp, batchsize, dev=dev) # Generate new batch
        transform && StatsBase.transform!(dtY, Y)
        # Transform features for Transformer format
        X = tabular2trans(X)
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
        if validation_loss && (mod(epoch, validation_frequency)==0 || epoch > epochs - 100)
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