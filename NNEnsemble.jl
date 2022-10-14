include("neuralnets.jl")

struct TCNEnsemble
    models
    optimizers
end

function train_ensemble!(
    ensemble::TCNEnsemble, dgp, n, dtY;
    epochs=1_000, batchsize=32, passes_per_batch=2, dev=cpu, loss=rmse_conv,
    validation_loss=true, validation_frequency=10, validation_size=2_000, verbosity=1,
    transform=true
)
    Flux.trainmode!(ensemble)
    for (i, (m, o)) ∈ enumerate(zip(ensemble.models, ensemble.optimizers))
        if validation_loss
            _, bm = train_cnn!(m, o, dgp, n, dtY, epochs=epochs, batchsize=batchsize, 
                passes_per_batch=passes_per_batch, dev=dev, loss=loss, 
                validation_loss=validation_loss, validation_frequency=validation_frequency,
                validation_size=validation_size, verbosity=verbosity, transform=transform)
            ensemble.models[i] = bm
        else
            train_cnn!(m, o, dgp, n, dtY, epochs=epochs, batchsize=batchsize, 
                passes_per_batch=passes_per_batch, dev=dev, loss=loss, 
                validation_loss=validation_loss, validation_frequency=validation_frequency,
                validation_size=validation_size, verbosity=verbosity, transform=transform)
        end
    end
end

Flux.trainmode!(e::TCNEnsemble) = [Flux.trainmode!(m) for m ∈ e.models]
Flux.testmode!(e::TCNEnsemble) = [Flux.testmode!(m) for m ∈ e.models]

(e::TCNEnsemble)(X) = mean(m(X) for m ∈ e.models)