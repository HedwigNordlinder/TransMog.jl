# Rewriting the 2D cat example into bridge-sampled maximum likelihood + Chapman–Kolmogorov KL training

This note describes how to rewrite the current 2D cat example so that it matches the intended large-scale workflow:

- `x_0 \sim P` is easy to sample from.
- `x_1 \sim Q` is the target data distribution and is only available through samples.
- We **do not** train the network to learn a bridge conditioned on both endpoints.
- Instead, we train a network to learn the **Markov projection of the marginal transition kernel**
  \[
  K^{\theta}_{s,t}(x_t \mid x_s), \qquad 0 \le s < t \le 1,
  \]
  using samples from an **easy conditional reference process**.
- The training data are samples
  \[
  y_t \sim \operatorname{Law}(Y_t \mid Y_0 = y_0, Y_1 = y_1),
  \]
  where the bridge law belongs to a reference stochastic process that can be sampled cheaply.
- The kernel network is parameterized as a mixture of Gaussians (MoG), trained by **maximum likelihood** on bridge samples together with a **Chapman–Kolmogorov (CK) self-consistency penalty**.

The emphasis below is on a rewrite that is **generic**, **GPU-friendly**, and **scales to large problems**, without using toy structure from the cat example.

---

## 1. What changes conceptually

The current cat example does two things:

1. It matches the endpoint pushforward `K_{0,1}` to samples from `Q` using RFF-MMD.
2. It enforces semigroup structure using an MMD discrepancy between a direct kernel and a composed kernel.

The rewrite should instead do this:

1. **Supervised kernel fitting from bridge samples**
   \[
   \mathcal L_{\mathrm{MLE}}(\theta)
   = -\mathbb E\big[\log K^{\theta}_{0,t}(Y_t \mid Y_0)\big],
   \]
   where `(Y_0, Y_1, t, Y_t)` are sampled from the reference bridge construction.

2. **Chapman–Kolmogorov regularization**
   for randomly chosen `0 <= s < u < t <= 1`, enforcing that
   \[
   K^{\theta}_{s,t}(x_t \mid x_s)
   \approx
   \widetilde K^{\theta}_{s,t}(x_t \mid x_s)
   :=
   \int K^{\theta}_{s,u}(x_u \mid x_s) K^{\theta}_{u,t}(x_t \mid x_u) \, dx_u.
   \]

This is the correct object for semigroup self-consistency. The quantity to work with is
\[
\log \widetilde K^{\theta}_{s,t}(x_t \mid x_s)
= \log \int K^{\theta}_{s,u}(x_u \mid x_s) K^{\theta}_{u,t}(x_t \mid x_u) \, dx_u,
\]
not an integral of the log over Lebesgue measure.

---

## 2. What should stay from the current script

The following pieces are already aligned with the large-scale goal and should be kept, possibly with minor cleanup:

- `TransMogModel`
- the time/state embeddings
- the residual feedforward stack
- the MoG output heads:
  - `decode_means`
  - `decode_log_variances`
  - `decode_log_weights`
- the device-agnostic MoG sampler `sample_mog`
- the detached sampler `sample_kernel_detached`
- the GPU setup and `Pkg.activate(".")`
- the time-sampling logic `sample_times`
- the one-shot generator `generate_oneshot`
- the multi-step rollout `generate` as a **diagnostic**, not as the primary inference path

What should **not** stay:

- `RFFMMD`
- `empirical_feature_mean`
- `_mog_features_3d`
- `mog_feature_mean_per_batch`
- `mog_feature_mean_batch`
- `endpoint_mmd_loss`
- `ck_mmd_loss`
- the MMD-specific training loop

---

## 3. Data generation should be rewritten around bridge samples

The current example uses

```julia
sampleX0(n)
sampleX1(n)
```

as independent endpoint samplers and then uses endpoint discrepancy to supervise the model. In the rewrite, the central data primitive should instead be something like

```julia
sample_bridge_batch(batch_size) -> (Y0, Y1, t, Yt)
```

where:

- `Y0` has shape `(D, B)` and contains start points from `P`
- `Y1` has shape `(D, B)` and contains terminal points from `Q`
- `t` has shape `(B,)`
- `Yt` has shape `(D, B)` and contains samples from the **reference bridge law**
  \[
  Y_t \mid Y_0 = Y0[:,b],\; Y_1 = Y1[:,b].
  \]

### Important modeling point

The network still learns a **marginal** transition kernel
\[
K^{\theta}_{0,t}(x_t \mid x_0),
\]
so the model input remains `(s, t, x_s)` and **does not** take `x_1` as input.

The role of `Y1` is only to generate reference bridge samples `Yt`. It is not part of the learned Markov kernel.

This is exactly what is meant by learning the **Markov projection of the marginal transition kernel** using conditional bridge samples.

---

## 4. Replace RFF-MMD with MoG log-likelihood

Because the network outputs a diagonal-covariance MoG, the first missing primitive is a stable batched log-density evaluator.

For a single conditioning column `b`, the network outputs
\[
K^{\theta}_{s,t}(\cdot \mid x_s^{(b)})
=
\sum_{k=1}^K \pi_k^{(b)} \, \mathcal N\big(\mu_k^{(b)}, \operatorname{diag}(\sigma_{k}^{2,(b)})\big).
\]

The log-density at target `x_t^{(b)}` is
\[
\log K^{\theta}_{s,t}(x_t^{(b)} \mid x_s^{(b)})
=
\operatorname{logsumexp}_k
\left(
\log \pi_k^{(b)} + \log \phi\big(x_t^{(b)}; \mu_k^{(b)}, \operatorname{diag}(\sigma_k^{2,(b)})\big)
\right).
\]

### Add this utility

Create a function with a signature like

```julia
mog_logpdf_diag(Xt, means, logvars, logweights)
```

returning a vector of length `B`, one log-density per conditioning column.

### Numerical requirements

- clamp `logvars`, for example to `[-15, 15]`
- compute mixture logits with `Flux.logsoftmax(logweights; dims = 1)`
- use `logsumexp(...; dims = 1)` for the mixture reduction
- do **not** materialize large temporary arrays more than needed
- parameterize variances through log-variances or Cholesky factors

### Shape conventions

If `means` has shape `(D*K, B)` and `logweights` has shape `(K, B)`, then reshape once:

```julia
μ  = reshape(means,   D, K, B)
lv = reshape(logvars, D, K, B)
```

Then broadcast the target `Xt` to shape `(D, 1, B)` and evaluate the Gaussian log-densities in parallel.

---

## 5. The supervised loss becomes bridge-sampled maximum likelihood

The endpoint MMD loss should be replaced by

\[
\mathcal L_{\mathrm{MLE}}(\theta)
=
-\mathbb E\Big[ \log K^{\theta}_{0,t}(Y_t \mid Y_0) \Big].
\]

### Batched implementation

Define

```julia
function bridge_mle_loss(model, Y0, t, Yt)
    B = size(Y0, 2)
    s = fill!(similar(t), 0f0)
    μ, lv, lw = model(s, t, ContinuousState(Y0))
    -mean(mog_logpdf_diag(Yt, μ, lv, lw))
end
```

### Why this is the right replacement

This directly trains the network to put mass on bridge-sampled intermediate states. Because the model is only conditioned on `Y0`, this is a fit to the **marginal transition kernel**, not to the full endpoint-conditioned bridge.

---

## 6. The Chapman–Kolmogorov penalty should be written as a KL between direct and composed kernels

The ideal CK regularizer for fixed `x_s` is
\[
D_{\mathrm{KL}}\!
\left(
K^{\theta}_{s,t}(\cdot \mid x_s)
\;\|\;
\widetilde K^{\theta}_{s,t}(\cdot \mid x_s)
\right),
\]
where
\[
\widetilde K^{\theta}_{s,t}(x_t \mid x_s)
=
\int K^{\theta}_{s,u}(x_u \mid x_s)
      K^{\theta}_{u,t}(x_t \mid x_u)
\, dx_u.
\]

In large problems this is not available in closed form, so the implementation should use a **self-normalized Monte Carlo estimator** for the composed term.

### Practical detached-teacher version

The most stable practical version is:

1. Sample `x_s` from a detached earlier-leg kernel, e.g. from `K_{0,s}`.
2. Sample `x_t` from the detached direct kernel `K_{s,t}`.
3. Penalize the gap between the direct log-density and the composed log-density estimate.

This gives a teacher-student style CK penalty in which the teacher is the detached direct kernel and the student is the composition.

This is not the only possible gradient estimator, but it is a very reasonable implementation path and matches the current detached-sampling style of the cat example.

---

## 7. Self-normalized estimator for the composed log-density gradient

For fixed `(x_s, x_t)` define
\[
\widetilde K^{\theta}_{s,t}(x_t \mid x_s)
=
\mathbb E_{X_u \sim K^{\theta}_{s,u}(\cdot \mid x_s)}
\big[ K^{\theta}_{u,t}(x_t \mid X_u) \big].
\]

Let `x_u^{(1)}, \dots, x_u^{(M)} \sim K_{s,u}(\cdot \mid x_s)` be sampled from the first leg, and define
\[
w_m = K^{\theta}_{u,t}(x_t \mid x_u^{(m)}),
\qquad
\bar w_m = \frac{w_m}{\sum_{\ell=1}^M w_\ell}.
\]

Then the self-normalized score identity gives the approximation
\[
\nabla_\theta \log \widetilde K^{\theta}_{s,t}(x_t \mid x_s)
\approx
\sum_{m=1}^M \bar w_m
\Big(
\nabla_\theta \log K^{\theta}_{s,u}(x_u^{(m)} \mid x_s)
+
\nabla_\theta \log K^{\theta}_{u,t}(x_t \mid x_u^{(m)})
\Big).
\]

### Practical simplification for memory and stability

A good first implementation is:

- sample the first-leg states `x_u^{(m)}` **detached**
- differentiate only through the **second-leg** log-density terms
- optionally differentiate through the direct term `log K_{s,t}(x_t|x_s)` if using a direct-vs-composed gap loss

This preserves the current low-variance style of the example and avoids storing large first-leg computation graphs.

### Important warning

The scalar estimator
\[
\log\left(\frac1M \sum_{m=1}^M w_m\right)
\]
is a biased estimator of `log \widetilde K`. That is acceptable in a practical CK regularizer, but it should be documented clearly. The main reason to prefer it is that it is simple and cheap.

---

## 8. Recommended CK loss for the rewrite

A pragmatic large-scale loss is
\[
\mathcal L_{\mathrm{CK}}(\theta)
=
\mathbb E\Big[
\log K^{\theta}_{s,t}(X_t \mid X_s)
-
\log \widehat{\widetilde K}^{\theta}_{s,t}(X_t \mid X_s)
\Big],
\]
where:

- `X_s` is sampled from a detached earlier-leg kernel, usually `K_{0,s}`
- `X_t` is sampled from the detached direct kernel `K_{s,t}(\cdot|X_s)`
- `\widehat{\widetilde K}` is the Monte Carlo composition estimate
  \[
  \widehat{\widetilde K}^{\theta}_{s,t}(x_t \mid x_s)
  = \frac1M \sum_{m=1}^M K^{\theta}_{u,t}(x_t \mid x_u^{(m)}),
  \qquad
  x_u^{(m)} \sim K^{\bar\theta}_{s,u}(\cdot\mid x_s),
  \]
  with the first leg optionally detached.

This is the direct analog of the old CK-MMD term, but with log-densities instead of feature discrepancies.

### Suggested implementation

```julia
function ck_loggap_loss(model, Xs, s, u, t; n_middle = 4)
    Xt = Zygote.ignore_derivatives() do
        μ_dir, lv_dir, lw_dir = model(s, t, ContinuousState(Xs))
        sample_mog(μ_dir, lv_dir, lw_dir)
    end

    logp_dir = begin
        μ_dir, lv_dir, lw_dir = model(s, t, ContinuousState(Xs))
        mog_logpdf_diag(Xt, μ_dir, lv_dir, lw_dir)
    end

    log_terms = map(1:n_middle) do _
        Xu = sample_kernel_detached(model, s, u, Xs)
        μ_c, lv_c, lw_c = model(u, t, ContinuousState(Xu))
        mog_logpdf_diag(Xt, μ_c, lv_c, lw_c)
    end

    logp_cmp = reduce((a,b) -> logaddexp.(a, b), log_terms) .- log(Float32(n_middle))
    mean(logp_dir .- logp_cmp)
end
```

This is the cleanest drop-in replacement for `ck_mmd_loss`.

### Why this version is acceptable

- it is generic
- it uses only MoG likelihood primitives
- it does not rely on toy structure
- it remains GPU-friendly
- it matches the teacher-student structure already present in the MMD version

---

## 9. Full training loss

The training objective should now be
\[
\mathcal L(\theta)
=
\lambda_{\mathrm{MLE}} \, \mathcal L_{\mathrm{MLE}}(\theta)
+
\lambda_{\mathrm{CK}} \, \mathcal L_{\mathrm{CK}}(\theta).
\]

### Recommended curriculum

Exactly as in the current example, use a warmup schedule:

1. first train with `λ_ck = 0`
2. once the one-shot transition kernel is reasonable, ramp up `λ_ck`
3. keep `λ_ck` smaller than the likelihood weight unless diagnostics show under-regularization

The reason is unchanged: the CK term is only meaningful once the base kernel is already approximately sane.

---

## 10. Concrete rewrite plan for the current file

### Step 1: delete the RFF/MMD section entirely

Remove everything from

```julia
struct RFFMMD
```

down to the end of `ck_mmd_loss`.

### Step 2: add MoG density utilities

Add the following new utilities:

- `mog_logpdf_diag(Xt, means, logvars, logweights)`
- optionally `component_logpdf_diag(...)` if you want a more modular implementation
- optionally `logmeanexp_cols(...)` or `logaddexp` helpers for stable composition

### Step 3: add bridge-data sampling

Replace

```julia
sampleX0(n)
sampleX1(n)
```

as the training data source by a single bridge-batch function:

```julia
sample_bridge_batch(batch_size)
```

Internally, that function should:

1. sample `Y0 ~ P`
2. sample `Y1 ~ Q`
3. sample a random time `t ~ Uniform(0,1)`
4. sample `Yt ~ Law(Y_t | Y_0 = Y0, Y_1 = Y1)` from the easy bridge simulator

In the 2D cat example, `P` and `Q` remain the same endpoint samplers, but the supervision now comes from bridge samples rather than endpoint discrepancy.

### Step 4: add bridge MLE loss

Introduce

```julia
bridge_mle_loss(model, Y0, t, Yt)
```

and make it the primary fitting term.

### Step 5: replace CK-MMD by CK log-gap

Introduce

```julia
ck_loggap_loss(model, Xs, s, u, t; n_middle = 4)
```

using detached first-leg samples and a log-mean-exp composition estimate.

### Step 6: rewrite the training loop

Rename

```julia
train_mmd!
```

to something like

```julia
train_kernel!
```

and make the loop look like

```julia
for each iteration
    Y0, Y1, t, Yt = sample_bridge_batch(batch_size)
    s, u, t_ck    = sample_times(batch_size)

    Xs = if λ_ck > 0
        zeros_B = fill!(similar(Y0, batch_size), 0f0)
        sample_kernel_detached(model, zeros_B, s, Y0)
    else
        Y0
    end

    l, g = Flux.withgradient(model) do m
        L_mle = bridge_mle_loss(m, Y0, t, Yt)
        L_ck  = λ_ck > 0 ? ck_loggap_loss(m, Xs, s, u, t_ck; n_middle = n_middle) : 0f0
        λ_mle * L_mle + λ_ck * L_ck
    end

    Flux.update!(opt_state, model, g[1])
end
```

### Step 7: keep one-shot inference unchanged

The real objective is still the one-shot pushforward through `K_{0,1}`:

```julia
generate_oneshot(model, X0)
```

Keep multi-step rollout only as a semigroup diagnostic.

---

## 11. Suggested utility implementations

Below is the level of implementation detail the rewrite should contain.

### 11.1 Stable diagonal-Gaussian mixture log-density

```julia
function mog_logpdf_diag(Xt, means, logvars, logweights)
    K, B = size(logweights)
    D = size(means, 1) ÷ K

    μ  = reshape(means,   D, K, B)
    lv = clamp.(reshape(logvars, D, K, B), -15f0, 15f0)
    x  = reshape(Xt, D, 1, B)

    invσ2 = exp.(-lv)
    quad  = sum((x .- μ).^2 .* invσ2; dims = 1)
    logdet = sum(lv; dims = 1)
    c = D * log(2f0 * Float32(pi))

    logcomp = -0.5f0 .* dropdims(quad .+ logdet .+ c; dims = 1)
    logπ = Flux.logsoftmax(logweights; dims = 1)

    return vec(logsumexp(logπ .+ logcomp; dims = 1))
end
```

The exact helper for `logsumexp` can come from a package or be defined locally.

### 11.2 Stable log-mean-exp across Monte Carlo legs

```julia
function logmeanexp_stack(logvals)
    # logvals: Vector of length M, each element a (B,) vector
    A = reduce(hcat, logvals)                # (B, M)
    m = maximum(A; dims = 2)
    vec(m .+ log.(mean(exp.(A .- m); dims = 2)))
end
```

Then in the CK term use

```julia
logp_cmp = logmeanexp_stack(log_terms)
```

rather than explicit `logaddexp` if that is easier to maintain.

---

## 12. Diagnostics that replace the old MMD plots

Once MMD is removed, diagnostics should become likelihood- and semigroup-based.

### Keep these diagnostics

1. **training curves**
   - total loss
   - bridge MLE
   - CK term

2. **one-shot samples**
   - draw `X1_hat = generate_oneshot(model, X0_eval)`
   - compare visually to samples from `Q`

3. **multi-step rollout as CK diagnostic**
   - draw `X_rollout = generate(model, X0_eval; n_steps = k)`
   - compare one-shot vs rollout mismatch

4. **log-likelihood on held-out bridge samples**
   - evaluate `-mean(log K_{0,t}(Y_t|Y_0))` on validation bridge data

5. **CK gap on held-out states**
   - evaluate the detached teacher-student log-gap on validation batches

### Remove these diagnostics

- RFF bandwidth tuning
- feature-mean discrepancy plots
- MMD-specific reporting

---

## 13. Scaling advice for large problems

The rewrite should explicitly avoid design choices that only work in the 2D toy setting.

### 13.1 Keep the kernel network generic

Do not hard-code:

- dimension `D = 2`
- cat-shaped endpoint distributions
- toy bandwidth heuristics
- toy visual diagnostics as part of the training logic

Everything should be parameterized by `spacedim` and the bridge sampler.

### 13.2 MoG size and covariance structure

For large problems, the first practical bottleneck is usually not dimension alone but the number of mixture components and covariance parameterization.

Recommendations:

- start with diagonal covariance, as in the current script
- keep `n_gaussians` moderate until the optimization is stable
- only move to richer covariance parameterizations after the rest of the pipeline works

### 13.3 CK estimator budget

The cost driver in the CK term is `n_middle` times one forward call of the second leg.

Recommendations:

- start with `n_middle = 1` or `2`
- increase only if the CK signal is too noisy
- use detached first-leg sampling first
- only later consider differentiating through both legs

### 13.4 Memory discipline

To keep the code viable on GPU at larger scale:

- avoid storing all component-wise intermediate tensors longer than necessary
- reuse reshapes and temporary buffers where practical
- keep the first leg detached in the CK term initially
- prefer one tracked forward pass per sampled middle state

---

## 14. Minimal package-level rewrite checklist

The top of the new script can remain close to the current one:

```julia
using Pkg
Pkg.activate(".")

using ForwardBackward, Flowfusion, Flux, Onion, RandomFeatureMaps, Optimisers, Plots
using CUDA, Random, cuDNN
using ProgressMeter
using Zygote
using Statistics: mean
```

However, after the rewrite:

- `RandomFeatureMaps` will no longer be needed unless used elsewhere
- anything imported only for MMD can be removed
- if a stable `logsumexp` helper is needed from another package, add it explicitly

So one of the cleanup tasks should be to prune imports once the RFF block is removed.

---

## 15. Summary of the final intended script structure

The cleaned-up file should have the following structure:

1. package activation and imports
2. GPU setup
3. `TransMogModel`
4. endpoint samplers `sampleP`, `sampleQ`
5. bridge sampler `sample_bridge_batch`
6. MoG utilities
   - `sample_mog`
   - `sample_kernel_detached`
   - `mog_logpdf_diag`
   - optional stable log-mean-exp helpers
7. losses
   - `bridge_mle_loss`
   - `ck_loggap_loss`
8. training utilities
   - `sample_times`
   - `train_kernel!`
9. inference
   - `generate_oneshot`
   - `generate`
10. diagnostics / plotting

This is the right abstraction barrier for moving from the current 2D cat example to larger problems.

---

## 16. Final recommendation

For the rewrite, the implementation target should be:

- **primary fit**: bridge-sampled maximum likelihood
- **regularization**: detached-teacher CK log-gap using self-normalized composition estimates
- **inference target**: one-shot `K_{0,1}` pushforward
- **rollout**: diagnostic only

This keeps the training objective aligned with the Markov-projection interpretation while avoiding toy-model shortcuts.

If a second-pass rewrite is planned later, the next improvement should be to replace the biased log-of-Monte-Carlo CK scalar by a more principled bridge-score estimator, but that should come only after the above pipeline is stable end-to-end.
