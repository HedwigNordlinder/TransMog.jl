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
# Kernel network — learns K^θ_{s,t}(·|x_s) as a MoG with FULL diagonal covariance
# parameterized through the lower-triangular Cholesky factor L  (Σ = L Lᵀ).
# ============================================================
struct TransMogModel{A}
    layers::A
end

Flux.@layer TransMogModel
function TransMogModel(; embeddim = 128, spacedim = 2, layers = 3, n_gaussians = 20)
    D     = spacedim
    n_off = D * (D - 1) ÷ 2                      # strict lower-tri entries per component
    embed_time_t = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_time_s = Chain(RandomFourierFeatures(1 => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    embed_state  = Chain(RandomFourierFeatures(spacedim => embeddim, 1f0), Dense(embeddim => embeddim, swish))
    projection_layer = Dense(3 * embeddim + spacedim + 2 => embeddim, swish)
    ffs = [Dense(embeddim => embeddim, swish) for _ in 1:layers]
    decode_means       = Chain(Dense(embeddim => embeddim, swish), Dense(embeddim => D * n_gaussians))
    decode_log_diag    = Chain(Dense(embeddim => embeddim, swish), Dense(embeddim => D * n_gaussians))
    decode_tril        = Chain(Dense(embeddim => embeddim, swish), Dense(embeddim => n_off * n_gaussians))
    decode_log_weights = Chain(Dense(embeddim => embeddim, swish), Dense(embeddim => n_gaussians))
    layers = (; embed_time_t, embed_time_s, embed_state, projection_layer, ffs,
              decode_means, decode_log_diag, decode_tril, decode_log_weights)
    TransMogModel(layers)
end

# Forward returns (means, log_diag, tril_off, log_weights).
#   means     : (D·K, B)  — component means
#   log_diag  : (D·K, B)  — log of Cholesky diagonal, i.e. log L_ii
#   tril_off  : (D(D-1)/2 · K, B) — strict lower-triangular entries, row-major packed
#   log_weights : (K, B)  — mixture logits
function (f::TransMogModel)(s, t, Xs)
    l = f.layers
    sXs = tensor(Xs)
    tv = zero(sXs[1:1, :]) .+ expand(t, ndims(sXs))
    sv = zero(sXs[1:1, :]) .+ expand(s, ndims(sXs))
    x = l.projection_layer([l.embed_time_t(tv);l.embed_time_s(sv);l.embed_state(sXs);tv;sv;sXs])
    for ff in l.ffs
        x = x .+ ff(x)
    end
    return l.decode_means(x), l.decode_log_diag(x), l.decode_tril(x), l.decode_log_weights(x)
end

# ============================================================
# Endpoint distributions and reference bridge sampler
# ============================================================
sampleP(n) = rand(Float32, 2, n) .+ 2f0
sampleQ(n) = Flowfusion.random_literal_cat(n, sigma = 0.05f0)

# Joint Brownian-bridge sample of (Y_s, Y_t) pinned at (Y_0, Y_1), with 0 ≤ s ≤ t ≤ 1.
# Under the bridge, (Y_s, Y_t) | (Y_0, Y_1) is Gaussian with
#   mean_u     = (1-u) Y_0 + u Y_1                  for u ∈ {s, t}
#   Var(Y_s)   = σ² s(1-s)
#   Cov(Y_s,Y_t) = σ² s(1-t)
# Sampled factorized: Y_s from its marginal, then Y_t | Y_s.
# The conditional is Gaussian with
#   mean_{t|s} = mean_t + ((1-t)/(1-s)) (Y_s - mean_s)
#   var_{t|s}  = σ² (t-s)(1-t) / (1-s).
function sample_bridge_batch(batch_size; σ = 0.15f0, w_max = 1f0)
    Y0 = sampleP(batch_size)
    Y1 = sampleQ(batch_size)
    # Curriculum: restrict t - s ≤ w_max. Sample s uniform, then t uniform in
    # [s, min(1, s + w_max)].
    s   = rand(Float32, batch_size)
    gap = min.(w_max, 1f0 .- s)
    t   = s .+ rand(Float32, batch_size) .* gap
    sr = reshape(s, 1, :)
    tr = reshape(t, 1, :)
    mean_s = (1f0 .- sr) .* Y0 .+ sr .* Y1
    mean_t = (1f0 .- tr) .* Y0 .+ tr .* Y1
    std_s  = σ .* sqrt.(max.(sr .* (1f0 .- sr), 0f0))
    Y_s = mean_s .+ std_s .* randn(Float32, size(Y0)...)
    one_minus_s = max.(1f0 .- sr, 1f-6)                        # guard s → 1
    scale    = (1f0 .- tr) ./ one_minus_s
    std_cond = σ .* sqrt.(max.((tr .- sr) .* (1f0 .- tr) ./ one_minus_s, 0f0))
    mean_cond = mean_t .+ scale .* (Y_s .- mean_s)
    Y_t = mean_cond .+ std_cond .* randn(Float32, size(Y0)...)
    return Y0, Y1, s, t, Y_s, Y_t
end

# ============================================================
# Cholesky / lower-triangular helpers — generic in D.
# Off-diagonal entries are packed row-major: tril_off[pack(i,j)] stores L[i,j] for j < i.
# ============================================================
@inline pack_tril_idx(i, j) = (i - 1) * (i - 2) ÷ 2 + j

# Clamp on log L_ii: L_ii ∈ [e^-7, e^3] ≈ [9e-4, 20] — sane range for the 2D toy.
const LOG_DIAG_CLAMP_LO = -7f0
const LOG_DIAG_CLAMP_HI =  3f0

# y = L⁻¹ * b via forward substitution, where L is lower triangular with L_ii = exp(log_diag_i).
# Shapes: log_diag (D, K, B), tril_off (n_off, K, B), b (D, K, B) → y (D, K, B).
function chol_solve_tril(log_diag, tril_off, b)
    D = size(b, 1)
    diag_L = exp.(log_diag)
    ybuf = Zygote.Buffer(b)
    for i in 1:D
        acc = b[i, :, :]
        for j in 1:i-1
            idx = pack_tril_idx(i, j)
            acc = acc .- tril_off[idx, :, :] .* ybuf[j, :, :]
        end
        ybuf[i, :, :] = acc ./ diag_L[i, :, :]
    end
    return copy(ybuf)
end

# z = L * ε for a chosen component — per-batch L parameterized by log_diag (D, B) and tril_off (n_off, B).
function chol_mul_tril(log_diag, tril_off, ε)
    D = size(ε, 1)
    diag_L = exp.(log_diag)
    zbuf = Zygote.Buffer(ε)
    for i in 1:D
        acc = diag_L[i, :] .* ε[i, :]
        for j in 1:i-1
            idx = pack_tril_idx(i, j)
            acc = acc .+ tril_off[idx, :] .* ε[j, :]
        end
        zbuf[i, :] = acc
    end
    return copy(zbuf)
end

# ============================================================
# MoG utilities (full-covariance via Cholesky)
# ============================================================

# Stable log-density of a diagonal-Cholesky-covariance MoG. Returns (B,).
function mog_logpdf_chol(Xt, means, log_diag, tril_off, logweights)
    K, B = size(logweights)
    D    = size(means, 1) ÷ K
    n_off = size(tril_off, 1) ÷ K
    μ  = reshape(means,    D, K, B)
    ld = clamp.(reshape(log_diag, D, K, B), LOG_DIAG_CLAMP_LO, LOG_DIAG_CLAMP_HI)
    tr = reshape(tril_off, n_off, K, B)
    x  = reshape(Xt, D, 1, B)
    diff = x .- μ                                       # (D, K, B) via broadcast
    y = chol_solve_tril(ld, tr, diff)                   # (D, K, B)
    mahal   = dropdims(sum(y .^ 2; dims = 1); dims = 1) # (K, B)
    log_det = 2f0 .* dropdims(sum(ld; dims = 1); dims = 1)  # (K, B)  — Σ log|Σ_k| = 2 Σ log L_ii
    c = Float32(D) * log(2f0 * Float32(π))
    logcomp = -0.5f0 .* (mahal .+ log_det .+ c)         # (K, B)
    logπ    = Flux.logsoftmax(logweights; dims = 1)     # (K, B)
    return vec(Flux.logsumexp(logπ .+ logcomp; dims = 1))  # (B,)
end

# Stable log-mean-exp across a Vector of (B,) log-density vectors.
function logmeanexp_stack(logvals)
    cols = [reshape(v, :, 1) for v in logvals]
    A    = reduce(hcat, cols)
    m    = maximum(A; dims = 2)
    vec(m .+ log.(mean(exp.(A .- m); dims = 2)))
end

# Device-agnostic MoG sampler with Cholesky covariance.
function sample_mog(means, log_diag, tril_off, log_weights)
    K, B = size(log_weights)
    D    = size(means, 1) ÷ K
    n_off = size(tril_off, 1) ÷ K
    μ_r  = reshape(means, D, K, B)
    ld_r = clamp.(reshape(log_diag, D, K, B), LOG_DIAG_CLAMP_LO, LOG_DIAG_CLAMP_HI)
    tr_r = reshape(tril_off, n_off, K, B)

    # Gumbel-max component selection
    u = similar(log_weights); rand!(u)
    u = clamp.(u, 1f-7, 1f0 - 1f-7)
    g = -log.(-log.(u))
    scores = log_weights .+ g
    max_vals = maximum(scores; dims = 1)
    mask = Float32.(scores .>= max_vals)
    mask = mask ./ sum(mask; dims = 1)
    mask_r = reshape(mask, 1, K, B)

    μ_sel  = dropdims(sum(μ_r  .* mask_r; dims = 2); dims = 2)      # (D, B)
    ld_sel = dropdims(sum(ld_r .* mask_r; dims = 2); dims = 2)      # (D, B)
    tr_sel = dropdims(sum(tr_r .* mask_r; dims = 2); dims = 2)      # (n_off, B)

    ε = similar(μ_sel); randn!(ε)
    z = chol_mul_tril(ld_sel, tr_sel, ε)                            # (D, B)
    return μ_sel .+ z
end

# Detached kernel sampler, no gradient flow.
function sample_kernel_detached(model, s, t, Xs)
    Zygote.ignore_derivatives() do
        μ, ld, tr, lw = model(s, t, ContinuousState(Xs))
        sample_mog(μ, ld, tr, lw)
    end
end

# ============================================================
# Losses
# ============================================================

function bridge_mle_loss(model, Y_s, s, t, Y_t)
    μ, ld, tr, lw = model(s, t, ContinuousState(Y_s))
    -mean(mog_logpdf_chol(Y_t, μ, ld, tr, lw))
end

function ck_loggap_loss(model, Xs, s, u, t; n_middle = 4)
    Xt = Zygote.ignore_derivatives() do
        μ_dir, ld_dir, tr_dir, lw_dir = model(s, t, ContinuousState(Xs))
        sample_mog(μ_dir, ld_dir, tr_dir, lw_dir)
    end
    μ_dir, ld_dir, tr_dir, lw_dir = model(s, t, ContinuousState(Xs))
    logp_dir = mog_logpdf_chol(Xt, μ_dir, ld_dir, tr_dir, lw_dir)
    log_terms = [begin
        Xu = sample_kernel_detached(model, s, u, Xs)
        μ_c, ld_c, tr_c, lw_c = model(u, t, ContinuousState(Xu))
        mog_logpdf_chol(Xt, μ_c, ld_c, tr_c, lw_c)
    end for _ in 1:n_middle]
    logp_cmp = logmeanexp_stack(log_terms)
    mean(logp_dir .- logp_cmp)
end

# ============================================================
# Training
# ============================================================
function sample_times(batch_size; w_max = 1f0)
    # Curriculum: t - s ≤ w_max, with u uniform on [s, t].
    s   = rand(Float32, batch_size)
    gap = min.(w_max, 1f0 .- s)
    t   = s .+ rand(Float32, batch_size) .* gap
    u   = s .+ rand(Float32, batch_size) .* (t .- s)
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
        w_start = 0.1f0,
        w_end   = 1f0,
    )
    opt_state = Flux.setup(AdamW(eta = eta), model)
    losses = Float32[]
    mle_losses = Float32[]
    ck_losses  = Float32[]
    prog = Progress(n_epochs * iters_per_epoch; desc = "Training kernel", showspeed = true)
    for epoch in 1:n_epochs
        λ_ck = if epoch <= warmup_epochs
            0f0
        else
            Float32(λ_ck_max * min(1f0, (epoch - warmup_epochs) / max(1, n_epochs - warmup_epochs)))
        end
        # Curriculum: widen the (s, t) interval window linearly from w_start (epoch 1)
        # to w_end (epoch n_epochs). Short intervals are easier kernels to fit — the
        # model builds up shorter transitions first and extends to the full horizon.
        w_max = Float32(w_start + (epoch - 1) * (w_end - w_start) / max(1, n_epochs - 1))
        for i in 1:iters_per_epoch
            Y0_cpu, _Y1, s_mle_cpu, t_mle_cpu, Ys_cpu, Yt_cpu =
                sample_bridge_batch(batch_size; σ = σ_bridge, w_max = w_max)
            Y0    = Y0_cpu    |> gpu
            s_mle = s_mle_cpu |> gpu
            t_mle = t_mle_cpu |> gpu
            Ys    = Ys_cpu    |> gpu
            Yt    = Yt_cpu    |> gpu

            s_cpu, u_cpu, t_ck_cpu = sample_times(batch_size; w_max = w_max)
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
                L_mle = bridge_mle_loss(m, Ys, s_mle, t_mle, Yt)
                L_ck  = λ_ck > 0 ? ck_loggap_loss(m, Xs, s, u, t_ck; n_middle = n_middle) : 0f0
                λ_mle * L_mle + λ_ck * L_ck
            end
            Flux.update!(opt_state, model, g[1])

            L_mle_val = bridge_mle_loss(model, Ys, s_mle, t_mle, Yt) |> cpu
            push!(mle_losses, Float32(L_mle_val))
            push!(ck_losses,  λ_ck > 0 ? Float32(l - λ_mle * L_mle_val) / max(λ_ck, 1f-8) : 0f0)
            push!(losses, l)

            recent = length(losses) >= 100 ? sum(@view losses[end-99:end]) / 100 : l
            next!(prog; showvalues = [
                (:epoch, "$epoch/$n_epochs"),
                (:w_max, round(w_max; digits = 3)),
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
function generate_oneshot(model, X0)
    B = size(X0, 2)
    s = fill!(similar(X0, B), 0f0)
    t = fill!(similar(X0, B), 1f0)
    μ, ld, tr, lw = model(s, t, ContinuousState(X0))
    sample_mog(μ, ld, tr, lw)
end

function generate(model, X0; n_steps = 100)
    ts = Float32.(range(0f0, 1f0, length = n_steps + 1))
    X = X0
    for i in 1:n_steps
        B = size(X, 2)
        s = fill(ts[i],   B)
        t = fill(ts[i+1], B)
        μ, ld, tr, lw = model(s, t, ContinuousState(X))
        X = sample_mog(μ, ld, tr, lw)
    end
    return X
end

# Same rollout, but keeps every intermediate state (including the initial X0 and final X1).
# Returns (states, ts) where states[i] corresponds to time ts[i].
function generate_with_trace(model, X0; n_steps = 5)
    ts = Float32.(range(0f0, 1f0, length = n_steps + 1))
    states = Vector{typeof(X0)}(undef, n_steps + 1)
    states[1] = X0
    X = X0
    for i in 1:n_steps
        B = size(X, 2)
        s = fill(ts[i],   B)
        t = fill(ts[i+1], B)
        μ, ld, tr, lw = model(s, t, ContinuousState(X))
        X = sample_mog(μ, ld, tr, lw)
        states[i+1] = X
    end
    return states, ts
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

# Per-step rollout trace: one subplot per time step, showing X0 & X1_true faintly in the
# background and the current rolled-out state in green.
rollout_states, rollout_ts = generate_with_trace(model, X0_eval; n_steps = n_rollout_steps)
step_plots = [begin
    p = scatter(X0_eval[1, :], X0_eval[2, :]; ms = 1, msw = 0,
                color = :blue,   alpha = 0.2, label = "",
                size = (300, 300), legend = false)
    scatter!(p, X1_true[1, :], X1_true[2, :]; ms = 1, msw = 0,
             color = :orange, alpha = 0.3, label = "")
    scatter!(p, Xi[1, :], Xi[2, :];         ms = 1, msw = 0,
             color = :green,  alpha = 0.6, label = "")
    title!(p, "t = $(round(rollout_ts[i]; digits = 2))")
    p
end for (i, Xi) in enumerate(rollout_states)]

rollout_trace_plot = plot(step_plots...;
                          layout = (1, length(step_plots)),
                          size   = (250 * length(step_plots), 300))
savefig(rollout_trace_plot, "rollout_steps.png")
display(rollout_trace_plot)
