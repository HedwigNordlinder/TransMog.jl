using Pkg
Pkg.activate(".")

using ForwardBackward, Flowfusion, Flux, Onion, RandomFeatureMaps, Optimisers, Plots

struct TransMogModel{A}
    layers::A
end

Flux.@layer TransMogModel
function TransMogModel(; embeddim = 128, spacedim = 2, layers = 3, n_gaussians = 20)
    embed_time_t = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_time_s = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state = Chain(RandomFourierFeatures(spacedim => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    projection_layer = Dense(3 * embeddim + spacedim + 2 => embeddim, swish)
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode_means = Chain(Dense(embeddim => embeddim, swish),Dense(embeddim => spacedim * n_gaussians))
    decode_log_variances = Chain(Dense(embeddim => embeddim, swish), Dense(embeddim => spacedim * n_gaussians))
    decode_log_weights = Chain(Dense(embeddim => embeddim, swish),Dense(embeddim => n_gaussians))
    layers = (; embed_time_t, embed_time_s, embed_state, projection_layer, ffs, decode_means, decode_log_variances, decode_log_weights)
    TransMogModel(layers)
end

function (f::TransMogModel)(s, t, Xs)
    l = f.layers
    sXs = tensor(Xs)
    tv = zero(sXs[1:1, :]) .+ expand(t, ndims(sXs))
    sv = zero(sXs[1:1, :]) .+ expand(s, ndims(sXs))
    x = l.projection_layer([l.embed_time_t(tv);l.embed_time_s(sv);l.embed_state(sXs);tv;sv;sXs])
    for ff in l.ffs
        x = x .+ ff(x)
    end
    return l.decode_means(x), l.decode_log_variances(x), l.decode_log_weights(x)
end

function negative_log_likelihood(means, log_variances, log_weights, Xt)
    x = tensor(Xt)
    D = size(x, 1)
    K = size(log_weights, 1)
    μ  = reshape(means,          D, K, :)
    lv = reshape(log_variances,  D, K, :)
    xr = reshape(x,              D, 1, :)
    diff = xr .- μ
    log2π = Float32(log(2π))
    log_gauss = dropdims(
        sum(-0.5f0 .* (log2π .+ lv .+ diff .^ 2 .* exp.(-lv)); dims = 1);
        dims = 1,
    )
    logp = Flux.logsumexp(log_weights .+ log_gauss; dims = 1) .-
           Flux.logsumexp(log_weights;              dims = 1)
    return -sum(logp) / length(logp)
end