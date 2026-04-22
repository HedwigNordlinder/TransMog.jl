using Pkg
Pkg.activate(".")

using ForwardBackward, Flowfusion, Flux, Onion, RandomFeatureMaps, Optimisers, Plots
using CUDA, Random, cuDNN
using ProgressMeter
using Zygote
using Statistics: mean

CUDA.functional() || error("CUDA is not functional — no usable GPU detected")
CUDA.device!(0)
@info "Using GPU 0: $(CUDA.name(CUDA.device()))"

# ============================================================
# Kernel network (unchanged — learns K^θ_{s,t}(·|x_s) as a diagonal MoG)
# ============================================================
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

# ============================================================
# Endpoint distributions and reference bridge sampler (§3, §10 step 3)
# ============================================================
# `sampleP` samples the prior X_0 ~ P, `sampleQ` samples the target X_1 ~ Q.
sampleP(n) = rand(Float32, 2, n) .+ 2f0
sampleQ(n) = Flowfusion.random_literal_cat(n, sigma = 0.05f0)

# Reference bridge: Brownian bridge on [0,1] pinned at Y_0 and Y_1 with diffusion σ.
# Y_t = (1-t) Y_0 + t Y_1 + ε,  ε ~ N(0, σ² t(1-t) I).
# Only Y_0 and Y_t are fed to the model; Y_1 is used solely to simulate the bridge.
function sample_bridge_batch(batch_size; σ = 0.15f0)
    Y0 = sampleP(batch_size)
    Y1 = sampleQ(batch_size)
    t  = rand(Float32, batch_size)
    tr = reshape(t, 1, :)
    mean_bridge = (1f0 .- tr) .* Y0 .+ tr .* Y1
    std_bridge  = σ .* sqrt.(max.(tr .* (1f0 .- tr), 0f0))
    Yt = mean_bridge .+ std_bridge .* randn(Float32, size(Y0)...)
    return Y0, Y1, t, Yt
end

# ============================================================
# MoG utilities
# ============================================================

# Stable diagonal-Gaussian mixture log-density (§11.1). Returns (B,).
function mog_logpdf_diag(Xt, means, logvars, logweights)
    K, B = size(logweights)
    D    = size(means, 1) ÷ K
    μ  = reshape(means,   D, K, B)
    lv = clamp.(reshape(logvars, D, K, B), -15f0, 15f0)
    x  = reshape(Xt, D, 1, B)
    invσ2 = exp.(-lv)
    quad   = sum((x .- μ) .^ 2 .* invσ2; dims = 1)   # (1, K, B)
    logdet = sum(lv; dims = 1)                        # (1, K, B)
    c = Float32(D) * log(2f0 * Float32(π))
    logcomp = -0.5f0 .* dropdims(quad .+ logdet .+ c; dims = 1)  # (K, B)
    logπ    = Flux.logsoftmax(logweights; dims = 1)              # (K, B)
    return vec(Flux.logsumexp(logπ .+ logcomp; dims = 1))        # (B,)
end

# Stable log-mean-exp over a Vector of (B,) log-density vectors (§11.2).
function logmeanexp_stack(logvals)
    cols = [reshape(v, :, 1) for v in logvals]       # each (B, 1)
    A    = reduce(hcat, cols)                        # (B, M)
    m    = maximum(A; dims = 2)                      # (B, 1)
    vec(m .+ log.(mean(exp.(A .- m); dims = 2)))     # (B,)
end

# Device-agnostic MoG sampler (unchanged).
function sample_mog(means, log_variances, log_weights)
    K, B = size(log_weights)
    D = size(means, 1) ÷ K
    μ_r  = reshape(means, D, K, B)
    lv_r = clamp.(reshape(log_variances, D, K, B), -15f0, 15f0)
    u = similar(log_weights); rand!(u)
    u = clamp.(u, 1f-7, 1f0 - 1f-7)
    g = -log.(-log.(u))
    scores = log_weights .+ g
    max_vals = maximum(scores; dims = 1)
    mask = Float32.(scores .>= max_vals)
    mask = mask ./ sum(mask; dims = 1)
    mask_r = reshape(mask, 1, K, B)
    μ_sel  = dropdims(sum(μ_r  .* mask_r; dims = 2); dims = 2)
    lv_sel = dropdims(sum(lv_r .* mask_r; dims = 2); dims = 2)
    ε = similar(μ_sel); randn!(ε)
    μ_sel .+ exp.(0.5f0 .* lv_sel) .* ε
end

# Detached kernel sampler, no gradient flow.
function sample_kernel_detached(model, s, t, Xs)
    Zygote.ignore_derivatives() do
        μ, lv, lw = model(s, t, ContinuousState(Xs))
        sample_mog(μ, lv, lw)
    end
end

# ============================================================
# Losses (§5, §8)
# ============================================================

# Bridge-sampled maximum likelihood: L_MLE = -E[log K^θ_{0,t}(Y_t | Y_0)] (§5).
function bridge_mle_loss(model, Y0, t, Yt)
    s = Zygote.ignore_derivatives(() -> fill!(similar(t), 0f0))
    μ, lv, lw = model(s, t, ContinuousState(Y0))
    -mean(mog_logpdf_diag(Yt, μ, lv, lw))
end

# Chapman–Kolmogorov log-gap regularizer (§6–§8).
#   teacher  : direct kernel  K^θ_{s,t}( · | x_s) — sampled x_t detached
#   student  : composed kernel ∫ K^θ_{u,t}(·|x_u) K^{θ,sg}_{s,u}(x_u|x_s) dx_u
#              estimated by log( (1/M) Σ_m K^θ_{u,t}(x_t|x_u^{(m)}) ), first leg detached.
# Loss is E[ log K_dir(x_t|x_s) - log K̂_cmp(x_t|x_s) ], a plug-in KL estimator.
function ck_loggap_loss(model, Xs, s, u, t; n_middle = 4)
    # (1) Detached teacher sample x_t ~ K_{s,t}(·|x_s).
    Xt = Zygote.ignore_derivatives() do
        μ_dir, lv_dir, lw_dir = model(s, t, ContinuousState(Xs))
        sample_mog(μ_dir, lv_dir, lw_dir)
    end
    # (2) Tracked direct log-density.
    μ_dir, lv_dir, lw_dir = model(s, t, ContinuousState(Xs))
    logp_dir = mog_logpdf_diag(Xt, μ_dir, lv_dir, lw_dir)
    # (3) Monte Carlo composition estimate — first leg detached, second leg tracked.
    log_terms = [begin
        Xu = sample_kernel_detached(model, s, u, Xs)
        μ_c, lv_c, lw_c = model(u, t, ContinuousState(Xu))
        mog_logpdf_diag(Xt, μ_c, lv_c, lw_c)
    end for _ in 1:n_middle]
    logp_cmp = logmeanexp_stack(log_terms)
    mean(logp_dir .- logp_cmp)
end

# ============================================================
# Training
# ============================================================
function sample_times(batch_size)
    s  = rand(Float32, batch_size)
    τ1 = rand(Float32, batch_size)
    τ2 = rand(Float32, batch_size)
    u  = s .+ (1f0 .- s) .* min.(τ1, τ2)
    t  = s .+ (1f0 .- s) .* max.(τ1, τ2)
    return s, u, t
end

function train_kernel!(model;
        n_epochs = 5,
        iters_per_epoch = 4000,
        batch_size = 400,
        eta = 1f-3,
        λ_mle = 1f0,
        λ_ck_max = 0.1f0,
        warmup_epochs = 1,
        n_middle = 4,
        σ_bridge = 0.15f0,
    )
    opt_state = Flux.setup(AdamW(eta = eta), model)
    losses = Float32[]
    mle_losses = Float32[]
    ck_losses  = Float32[]
    prog = Progress(n_epochs * iters_per_epoch; desc = "Training kernel", showspeed = true)
    for epoch in 1:n_epochs
        # Warmup on MLE only (§9), then linearly ramp λ_ck.
        λ_ck = if epoch <= warmup_epochs
            0f0
        else
            Float32(λ_ck_max * min(1f0, (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)))
        end
        for i in 1:iters_per_epoch
            # --- MLE batch: bridge samples (Y_0, t, Y_t)
            Y0_cpu, _Y1, t_mle_cpu, Yt_cpu = sample_bridge_batch(batch_size; σ = σ_bridge)
            Y0    = Y0_cpu    |> gpu
            t_mle = t_mle_cpu |> gpu
            Yt    = Yt_cpu    |> gpu

            # --- CK batch: times 0 ≤ s ≤ u ≤ t ≤ 1, state X_s from a detached K_{0,s}.
            s_cpu, u_cpu, t_ck_cpu = sample_times(batch_size)
            s    = s_cpu    |> gpu
            u    = u_cpu    |> gpu
            t_ck = t_ck_cpu |> gpu
            Xs = if λ_ck > 0
                zeros_B = fill!(similar(Y0, batch_size), 0f0)
                sample_kernel_detached(model, zeros_B, s, Y0)
            else
                Y0
            end

            l, g = Flux.withgradient(model) do m
                L_mle = bridge_mle_loss(m, Y0, t_mle, Yt)
                L_ck  = λ_ck > 0 ? ck_loggap_loss(m, Xs, s, u, t_ck; n_middle = n_middle) : 0f0
                λ_mle * L_mle + λ_ck * L_ck
            end
            Flux.update!(opt_state, model, g[1])

            # Track components separately for diagnostics.
            L_mle_val = bridge_mle_loss(model, Y0, t_mle, Yt) |> cpu
            push!(mle_losses, Float32(L_mle_val))
            push!(ck_losses,  λ_ck > 0 ? Float32(l - λ_mle * L_mle_val) / max(λ_ck, 1f-8) : 0f0)
            push!(losses, l)

            recent = length(losses) >= 100 ? sum(@view losses[end-99:end]) / 100 : l
            next!(prog; showvalues = [
                (:epoch, "$epoch/$n_epochs"),
                (:λ_ck,  round(λ_ck; digits = 4)),
                (:iter,  "$i/$iters_per_epoch"),
                (:total, round(l;          digits = 4)),
                (:mle,   round(L_mle_val;  digits = 4)),
                (Symbol("total (ma100)"), round(recent; digits = 4)),
            ])
        end
    end
    finish!(prog)
    return (; total = losses, mle = mle_losses, ck = ck_losses)
end

# ============================================================
# Inference
# ============================================================

# Real objective: one-shot pushforward through K^θ_{0,1}.
function generate_oneshot(model, X0)
    B = size(X0, 2)
    s = fill!(similar(X0, B), 0f0)
    t = fill!(similar(X0, B), 1f0)
    μ, lv, lw = model(s, t, ContinuousState(X0))
    sample_mog(μ, lv, lw)
end

# Multi-step rollout: semigroup-consistency diagnostic only.
function generate(model, X0; n_steps = 100)
    ts = Float32.(range(0f0, 1f0, length = n_steps + 1))
    X = X0
    for i in 1:n_steps
        B = size(X, 2)
        s = fill(ts[i],   B)
        t = fill(ts[i+1], B)
        μ, lv, lw = model(s, t, ContinuousState(X))
        X = sample_mog(μ, lv, lw)
    end
    return X
end

# ============================================================
# Run it
# ============================================================
model = TransMogModel() |> gpu
loss_hist = train_kernel!(model)
model = model |> cpu

loss_plot = plot(loss_hist.total; label = "total",
                 xlabel = "iteration", ylabel = "loss",
                 title = "Training loss", size = (700, 400))
plot!(loss_plot, loss_hist.mle; label = "MLE (−log K₀,ₜ)", alpha = 0.7)
savefig(loss_plot, "training_loss.png")
display(loss_plot)

n_inference = 5000
n_rollout_steps = 5
X0_eval   = sampleP(n_inference)
X1_true   = sampleQ(n_inference)
X_oneshot = generate_oneshot(model, X0_eval)
X_rollout = generate(model, X0_eval; n_steps = n_rollout_steps)

function endpoint_scatter(X_gen, gen_label)
    p = scatter(X0_eval[1, :], X0_eval[2, :];
                ms = 1, msw = 0, color = :blue, alpha = 0.5,
                label = "X0", size = (500, 500), legend = :topleft)
    scatter!(p, X1_true[1, :], X1_true[2, :]; ms = 1, msw = 0,
             color = :orange, alpha = 0.5, label = "X1 (true)")
    scatter!(p, X_gen[1, :], X_gen[2, :]; ms = 1, msw = 0,
             color = :green, alpha = 0.5, label = gen_label)
    return p
end

oneshot_plot = endpoint_scatter(X_oneshot, "X1 (one-shot K₀,₁)")
title!(oneshot_plot, "One-shot")
rollout_plot = endpoint_scatter(X_rollout, "X1 ($(n_rollout_steps)-step rollout)")
title!(rollout_plot, "$(n_rollout_steps)-step rollout (CK diagnostic)")

sample_plot = plot(oneshot_plot, rollout_plot; layout = (1, 2), size = (1000, 500))
savefig(sample_plot, "generated_samples.png")
display(sample_plot)
