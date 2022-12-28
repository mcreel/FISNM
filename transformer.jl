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

    function Encoder(n_features::Int; 
        d_model::Int=512, n_heads::Int=8, d_ffnn::Int=2048, 
        dropout::AbstractFloat=.1, n_layers::Int=6
    )
        # Number of features per attention head
        @assert d_model % n_heads == 0 "d_model should be divisible by n_heads"
        k = d_model ÷ n_heads
        new(
            InputLayer(n_features, d_model=d_model, dropout=dropout),
            clone(
                Transformer(d_model, n_heads, k, d_ffnn, future=false, 
                    pdrop=dropout), 
                n_layers
            )
        )
    end
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
    function Decoder(n_features::Int, seqlen::Int, n_outputs::Int;
        d_model::Int=512, n_heads=8, d_ffnn::Int=2048,
        dropout::AbstractFloat=.1, n_layers::Int=6
    )
        # Number of features per attention head
        @assert d_model % n_heads == 0 "d_model should be divisible by n_heads"
        new(
            InputLayer(n_features, d_model=d_model, dropout=dropout), # TODO: Is this input layer or dense??
            clone(
                TransformerDecoder(d_model, n_heads, k, d_ffnn, pdrop=dropout), 
                n_layers
            ),
            Dense(d_model * seqlen => n_outputs)
        )
    end
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
    function CustomTransformer(n_features::Int, seqlen::Int, n_outputs::Int;
        d_model::Int=512, n_heads::Int=8, d_ffnn=2048, dropout::AbstractFloat=0.1, 
        n_layers::Int=6
    )
        encoder = Encoder(n_features, d_model=d_model, n_heads=n_heads, 
            d_ffnn=d_ffnn, dropout=dropout, n_layers=n_layers)
        decoder = Decoder(n_features, seqlen, n_outputs, d_model=d_model, 
            n_heads=n_heads, d_ffnn=d_ffnn, dropout=dropout, n_layers=n_layers)
        new(encoder, decoder)
    end
end
@Flux.functor CustomTransformer
(t::CustomTransformer)(x) = t.decoder(x, t.encoder(x))
