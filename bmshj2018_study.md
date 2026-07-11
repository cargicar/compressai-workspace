# Study Notes: "End-to-end Optimized Image Compression" (Ballé, Laparra, Simoncelli, ICLR 2017)

Paper: arXiv:1611.01704, published at ICLR 2017.

## Overview / Big Picture

The paper proposes a full image compression pipeline built from neural networks, trained
end-to-end (analysis transform → quantizer → synthesis transform) to directly optimize the
rate–distortion trade-off, rather than optimizing each piece (transform, quantizer, entropy
code) separately, as JPEG/JPEG2000 do.

Key components:
- **Analysis transform** `g_a`: maps image `x` to a latent code `y` (three stages of conv + downsample + GDN nonlinearity).
- **Quantizer**: rounds `y` to integers, giving discrete code `q`.
- **Synthesis transform** `g_s`: maps the (dequantized) code back to a reconstructed image `x̂`.
- **Entropy code**: compresses `q` losslessly, close to its entropy.

## GDN (Generalized Divisive Normalization) — implementation deep dive

Implemented from scratch in `bmshj2018_scratch.py` (no compressai). GDN is the nonlinearity used
in `g_a`/`g_s` instead of ReLU/BatchNorm.

### Why not just ReLU?

Natural (and physical-field) images have a well-known statistical quirk: nearby pixels'
*magnitudes* are correlated even after linear correlations are removed — a bright edge in one
place tends to co-occur with high contrast nearby, regardless of sign. GDN is a nonlinearity
purpose-built to remove exactly this dependency (originally derived from models of biological
visual cortex neurons). It makes the resulting representation easier for a simple
independent-per-channel entropy model to code efficiently.

### The formula

```
y_i = x_i / sqrt(beta_i + Σ_j gamma_ij * x_j²)
```

Per output channel `i`: divide `x_i` by the square root of `beta_i` plus a *learned weighted sum
of every channel's squared value* at that pixel (`gamma_ij` = how much channel `j`'s energy
suppresses channel `i`). It's "divisive" (dividing, not subtracting) and "generalized" because
`gamma` is a full matrix mixing all channels together, not a per-channel scalar. `inverse=True`
(IGDN, used in the synthesis/decoder transform) does the mirror operation: multiply by that same
`sqrt(...)` instead of dividing, to invert what GDN did in the encoder.

### Computing it efficiently: a 1x1 convolution

```python
norm = F.conv2d(x ** 2, gamma, beta)   # beta_i + sum_j gamma_ij x_j^2, per pixel
norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)
return x * norm
```

Computing `Σ_j gamma_ij x_j²` at *every pixel* is exactly what a 1x1 convolution does: a `Conv2d`
with a `(C_out, C_in, 1, 1)` weight computes, independently at every pixel, a linear combination
of that pixel's input channels. Feeding `gamma` (reshaped to `(C,C,1,1)`) as the conv weight and
`x**2` as the input gives `Σ_j gamma_ij x_j²` everywhere in one fused GPU op; `beta` as the conv's
bias adds `beta_i` for free. `rsqrt` gives `1/sqrt(...)`, multiplied back into `x`.

### Why beta/gamma need special handling: the non-negativity problem

`beta_i`/`gamma_ij` must stay non-negative — negative values under a square root are undefined.
The obvious fix (clamp the raw parameter to `>= 0` every forward pass) has a real problem: a
plain `clamp` has zero gradient at the boundary, so a parameter that dips to the floor can get
**permanently stuck there** — no gradient signal ever pushes it back up.

#### Why plain `clamp` gets permanently stuck (worked example)

`torch.clamp(x, min=bound)`'s gradient rule is fixed and dumb: **inside the clamped region, the
gradient is always zero**, full stop — it doesn't look at anything else.

Say `bound=1.0` and the raw parameter has drifted to `x=0.8` (below the bound, e.g. from a bad
update earlier in training). Suppose the true gradient signal from the rest of the network, over
the next few steps, is `-0.3`, `-0.2`, `-0.1` — consistently negative, meaning (with the standard
update `x_new = x_old - lr*grad`) the optimizer *wants* to push `x` **up**.

With plain `clamp`: step 1 forward → `clamp(0.8, min=1.0) = 1.0`. Backward → since `x<bound`,
clamp's gradient is defined as exactly `0`, regardless of the upstream gradient. Update:
`x_new = 0.8 - lr*0 = 0.8`, **unchanged**. Steps 2, 3, ... are identical — `x` sits at `0.8`
forever, and the effective beta used by the network stays pinned at `1.0` forever. The parameter
is now dead: it doesn't matter that the network's true gradient keeps saying "make beta bigger" —
that signal never reaches `x`, because clamp's backward rule severs it completely and permanently
the moment `x` crosses the boundary. Nothing in the future can ever revive it.

#### What `_LowerBoundFn`'s custom backward changes

```python
@staticmethod
def forward(ctx, x, bound):
    return torch.max(x, bound)

@staticmethod
def backward(ctx, grad_output):
    x, bound = ctx.saved_tensors
    pass_through = (x >= bound) | (grad_output < 0)
    return pass_through.type(grad_output.dtype) * grad_output, None
```

The **forward pass is identical to clamp** — `max(x,bound)` is exactly what `clamp(x,min=bound)`
computes. All the difference is in `backward`. Every case:
- `x >= bound` (not clamped): `pass_through=True` unconditionally — ordinary identity gradient,
  same as clamp here.
- `x < bound` and `grad_output < 0`: `pass_through=True` — **gradient passes through, non-zero.**
- `x < bound` and `grad_output >= 0`: `pass_through=False` — gradient zeroed, same as clamp.

The middle case is the entire fix. `grad_output` is `dLoss/d(this node's output)`; with the
standard update `x_new = x_old - lr*grad_output`, a negative `grad_output` means the update pushes
`x` **up**. Since `x` is currently below the bound, pushing it up is exactly the useful direction
(toward, and eventually past, the boundary). So `_LowerBoundFn` lets that gradient through, even
though `x` is technically in the "clamped" region. It only blocks gradient when `grad_output>=0` —
the optimizer wanting to push `x` further *below* a bound it's already below, which would be
pointless (forward pass already pins the output at `bound` no matter how far below `x` sinks) and
harmful (the deeper `x` sinks, the longer it takes to recover if the gradient direction reverses).

Redoing the same worked example with `_LowerBoundFn`: step 1, `x=0.8`, forward → `max(0.8,1.0)=1.0`
(same as clamp). Backward: `grad=-0.3`. `x>=bound`? No. `grad<0`? **Yes** → passes through, returns
`-0.3`. Update (lr=0.1): `x_new = 0.8 - 0.1*(-0.3) = 0.83`. Step 2: `x=0.83`, `grad=-0.2`, still
negative → passes, `x_new=0.85`. Step 3: `x=0.85→0.86`. `x` keeps creeping upward every step the
gradient wants that, and eventually crosses back above `1.0` — no longer clamped at all, free to
take whatever value training actually needs, instead of being frozen at exactly `1.0` forever. If
at some point the gradient flips sign (say `x=0.95`, still clamped, now `grad=+0.4` — optimizer
wants beta *smaller*): `x>=bound`? No. `grad<0`? No. → **blocked**, `x` stays at `0.95`, correctly
refusing to sink further into already-clamped territory.

So `_LowerBoundFn` alone — even applied directly to beta, with no squaring at all — already fixes
the "stuck forever" pathology. That's the core, load-bearing fix.

#### What `NonNegativeParametrizer`'s squaring adds on top

```python
def init(self, x):
    return torch.sqrt(torch.clamp(x + self.pedestal, min=self.pedestal.item()))

def forward(self, x):
    out = _LowerBoundFn.apply(x, self.bound)
    return out ** 2 - self.pedestal
```

Given `_LowerBoundFn` alone already solves the stuck-gradient problem, why not just apply it
directly to beta and skip the sqrt/square? Here `_LowerBoundFn` is applied to `x` — the **stored,
square-root-domain** parameter — not to beta itself. Two things this adds on top of the same core
fix:
1. **An unconditional non-negativity guarantee that doesn't hinge on the bound being exactly
   right.** `out**2` is `>= 0` for *any* real `out`, no boundary reasoning needed for that part at
   all — the `LowerBound` here is really just keeping `out` away from the numerically awkward
   near-zero region (where `sqrt`'s gradient blows up), not doing the "keep it non-negative" job;
   squaring does that unconditionally.
2. **Smoother gradient scaling as the effective value approaches its floor**, since the
   relationship between the stored parameter and the effective beta is quadratic rather than
   linear (`d(beta)/d(raw) = 2*raw`, shrinking toward zero as `raw` shrinks) — the same general
   idea as parameterizing a variance via `exp(raw)` or a std-dev via `softplus(raw)` elsewhere in
   ML: represent a positive quantity through a smooth, monotonic transform of an unconstrained
   parameter, so gradient dynamics scale with the parameter's own magnitude. This point is a
   secondary refinement, not verified to the same rigor as the core fix above — but it's the
   standard motivation for this style of reparameterization in general.

**Bottom line:** `_LowerBoundFn`'s custom backward is what prevents the permanent-freeze bug —
true whether applied directly to beta or, as here, to the sqrt-domain stored parameter.
`NonNegativeParametrizer` layers squaring on top mainly to get the non-negativity guarantee "for
free" via an unconditional identity (`x² >= 0`), plus somewhat smoother gradient scaling near the
floor.

## AnalysisTransform (`g_a`) — implementation deep dive

Traced with the actual sizes `bmshj2018_scratch.py` uses: `patch_size=128` during training,
`256×256` slices during full-volume eval, `N=64` channels, `M=64` latent channels (all defaults).
`AnalysisTransform(in_channels=1, N=64, M=64)`.

### What `Conv2d(in, out, kernel_size=5, stride=2, padding=2)` does, generically

It slides a learned `5×5` filter across the input, but only every other position (`stride=2`),
producing `out` independent feature maps, each computed as a learned weighted combination of
*all* `in` input channels within each 5×5 neighborhood (plus a bias). Two things happen at once in
one op: feature extraction (the 5×5 receptive field lets each output pixel see local spatial
context) and downsampling (stride 2 halves resolution) — no separate pooling layer needed.

`padding=2` (= `kernel//2`) with `stride=2` is what makes the output size come out to *exactly*
half the input for any even input size: `out = floor((H + 2·pad − kernel)/stride) + 1`; plugging
in `pad=2, kernel=5, stride=2` gives `out = floor((H−1)/2) + 1`, exactly `H/2` whenever `H` is
even. That's why `128→64→32→16→8` and `256→128→64→32→16` work out cleanly, with no cropping or
padding artifacts to worry about when reconstructing later.

### The four layers, traced with real shapes

| Layer | Op | Input → Output (patch_size=128 case) | What it does |
|---|---|---|---|
| 1 | `Conv2d(1, 64, 5, 2, 2)` | `(B,1,128,128) → (B,64,64,64)` | Takes the single grayscale channel and applies 64 *different* learned 5×5 filters to it, each producing its own feature map — since `in_channels=1`, each of the 64 outputs is just a different local pattern detector (edge/blob/gradient-like) on the raw pixel values, sampled at half resolution. |
| — | `GDN(64)` | shape unchanged | Normalizes each of the 64 channels by a learned combination of all 64 channels' local energy. |
| 2 | `Conv2d(64, 64, 5, 2, 2)` | `(B,64,64,64) → (B,64,32,32)` | Now `in_channels=64`, so each of the 64 output channels is a learned mixture of **all 64** input feature maps within a 5×5 window — cross-channel/cross-feature combination starts here, building more abstract, larger-receptive-field features while halving resolution again. |
| — | `GDN(64)` | shape unchanged | Same normalization role, on the new deeper features. |
| 3 | `Conv2d(64, 64, 5, 2, 2)` | `(B,64,32,32) → (B,64,16,16)` | Same idea, one level deeper/coarser — receptive fields compound across stride-2 layers, so this layer combines increasingly large-scale structure. |
| — | `GDN(64)` | shape unchanged | Same. |
| 4 | `Conv2d(64, 64, 5, 2, 2)` | `(B,64,16,16) → (B,64,8,8)` | The final layer, **with no GDN after it**. Its job differs from the previous three: not "extract more features" but "project into exactly the `M`-channel latent space the entropy model expects." Its output *is* `y`, the raw latent passed straight into `FactorizedEntropy`. |

For the `256×256` full-volume eval case, the same four layers give `256→128→64→32→16`, so the
latent is `(B,64,16,16)` instead — same channel count `M=64`, bigger spatial grid since the input
was bigger. Either way, total downsampling is `2⁴ = 16×` in each spatial dimension — matching
CompressAI's own `bmshj2018_factorized` `g_a` (same 4-stage/16× downsample structure, just
`in_channels=1` here instead of the pseudo-RGB `3`).

### Why this whole stack is called "the analysis transform"

`g_a` plays the same conceptual role as the DCT in JPEG — a transform that converts the raw image
into a more compressible representation — except learned rather than fixed, interleaved with GDN
specifically because GDN's job is to decorrelate the joint magnitude statistics that a simple,
per-channel-independent entropy model (`FactorizedEntropy`) needs in order to code efficiently.
The last conv layer having no GDN makes sense given its distinct job: layers 1-3 extract and
normalize features; layer 4 hands off a clean `M`-channel latent to the entropy model, and
normalizing right before quantization+entropy coding isn't what the earlier GDN layers are for.

## FactorizedEntropy — implementation deep dive

The piece that makes the whole thing trainable end-to-end: a *learned, differentiable model of
how probable each latent value is*, which is exactly what's needed to compute "how many bits
would this take" without ever running a real entropy coder during training.

### The problem: bit-cost needs probability

Shannon's result: encoding a symbol with probability `p` costs `-log2(p)` bits in the best case.
To train the network to minimize rate, we need to know — for every latent value `g_a` produces —
how probable that value is, and that probability model has to be **differentiable**, since it
feeds directly into the loss. We don't know the true distribution of latent values ahead of time
(it depends on the trained network itself), so the entropy model has to be a small *learnable*
function, trained jointly with everything else, that gradually shapes itself to match whatever
distribution the latents actually end up having.

### Why "factorized"

This model assumes each of the `M` latent channels has its own **independent** distribution — no
cross-channel or spatial correlation modeled at all: `P(y) = Π_c P_c(y_c)`. A real simplifying
assumption (the follow-up "hyperprior" Ballé paper adds a second small network specifically to
capture the correlations this one ignores) — literally why CompressAI calls this variant
`bmshj2018-factorized`.

### Why a CDF, not a density

To get the probability *mass* in the unit-width bin around an integer value `y`
(`P(round(·) = y)`), integrate the density from `y-0.5` to `y+0.5`, which is just
`CDF(y+0.5) - CDF(y-0.5)`. So rather than model a density and integrate it, the network directly
models the **cumulative distribution function** — a bin's probability is then just two function
evaluations and a subtraction. That's the entire purpose of `_logits_cumulative`.

### The architecture: a tiny per-channel MLP, constrained to always be monotonic

```python
def _logits_cumulative(self, x):
    logits = x
    for i in range(len(self.matrices)):
        matrix = F.softplus(self.matrices[i])
        logits = torch.matmul(matrix, logits) + self.biases[i]
        if i < len(self.factors):
            factor = torch.tanh(self.factors[i])
            logits = logits + factor * torch.tanh(logits)
    return logits
```

A CDF has one hard requirement: it must be **non-decreasing** — probability only accumulates
moving right along the number line, never goes backward. A generic MLP can represent
non-monotonic functions too, which would be an invalid CDF. Every piece here is chosen to
guarantee monotonicity *by construction*, regardless of what the learned parameters end up being:

- **`F.softplus(self.matrices[i])`**: `softplus(x)=log(1+eˣ)` is always `>0`. Using it on the raw
  matrix guarantees every weight in that layer's affine map is non-negative. An affine map with
  non-negative weights, applied to an already-nondecreasing input, produces another nondecreasing
  output — negative weights could flip the direction of increase for some combination, breaking
  monotonicity. This one line keeps the *linear* part of each layer monotonic.
- **`logits + tanh(factor) * tanh(logits)`**: a stack of non-negative-weight affine maps alone is
  too rigid to represent arbitrary CDF shapes — real flexibility needs a genuine nonlinearity, but
  an arbitrary one (e.g. ReLU) could break monotonicity. This specific term is *provably*
  monotonic for any input: let `a = tanh(factor) ∈ (-1,1)` and `f(l) = l + a·tanh(l)`. Its
  derivative is `f'(l) = 1 + a·sech²(l)`. Since `sech²(l) ∈ (0,1]` and `|a| < 1` strictly,
  `a·sech²(l)` can never reach `±1`, so `f'(l) > 0` **always**, for every input and every learned
  `factor`. This nonlinearity adds real representational flexibility while being mathematically
  guaranteed to never violate monotonicity — training never needs to "stay valid," it's valid no
  matter where gradient descent takes the parameters.

### The shapes: genuinely independent tiny MLPs per channel

`filters = (1,3,3,3,3,1)` — 5 layers, input/output width 1 (each latent value is a single scalar;
modeling a 1-D CDF), hidden width 3. `matrices[i]` has shape `(channels, filters[i+1], filters[i])`
— a **separate** matrix per channel, not shared. `_logits_cumulative`'s batched `torch.matmul` over
the `channels` dimension runs `M` completely independent tiny 5-layer MLPs in parallel, one per
channel with its own weights, since different channels can genuinely have differently-shaped
distributions. This is also why `forward()` reshapes before calling it:
```python
y_perm = y_tilde.permute(1, 0, 2, 3).reshape(C, 1, -1)   # (C, 1, B*H*W)
```
`channels` has to become the outer/batch dimension for this matmul (each channel needs its own
weights), and batch/height/width flatten into one long axis, since the *same* per-channel MLP is
applied identically to every spatial position and every image in the batch for that channel.

### `forward()`: quantization switch, then likelihood

```python
if self.training:
    y_tilde = y + torch.empty_like(y).uniform_(-0.5, 0.5)
else:
    y_tilde = torch.round(y)
```
The additive-uniform-noise relaxation from the rate/quantization notes below, concretely
implemented: differentiable noise during training, real rounding at eval — both feed the *same*
likelihood computation, since the density model gives valid bin-probabilities either way (integer
or not).

```python
lower = self._logits_cumulative(y_perm - 0.5)
upper = self._logits_cumulative(y_perm + 0.5)
```
The network outputs a **logit** (unbounded real number), not a probability directly —
`sigmoid(logit)` is the actual CDF value in `(0,1)`. So `sigmoid(upper) - sigmoid(lower)` is
exactly `CDF(y+0.5) - CDF(y-0.5)`: the probability mass currently assigned to this value — the
`likelihood`, and `-log2(likelihood)` is literally the rate loss term `L = -E[log2 P_q] + λ·D`.

**The sign trick** — a real numerical-stability subtlety, not decoration:
```python
sign = -torch.sign(lower + upper).detach()
likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
```
`sigmoid` saturates for large `|x|` — so close to `0` or `1` that floating-point precision runs
out, and subtracting two numbers both `~0.999999999...` loses almost all meaningful precision
(catastrophic cancellation), even if their true difference is a perfectly well-resolved
probability. If both `lower`/`upper` sit deep in the saturated region (e.g. both large positive —
far out in the CDF's right tail), that's exactly the bad regime. The fix uses
`sigmoid(-x) = 1 - sigmoid(x)`: flipping the sign of *both* logits by the same amount doesn't
change their absolute difference (hence the final `abs()`), but lets the code choose, per-element,
to evaluate `sigmoid` on whichever side of the two mirror-image tails is numerically
better-conditioned — sigmoid's precision is best near its midpoint (`x=0`), so this always
evaluates in a saner region rather than blindly computing wherever the raw logits happen to land.

```python
likelihood = torch.clamp(likelihood, min=self.likelihood_bound)
```
Floors the likelihood at `1e-9` so `-log2(likelihood)` can never become infinite — protects the
rate loss from blowing up if the model assigns a value essentially zero probability (more likely
early in training, or for outlier values).

### What this is *not* doing

Only produces a **differentiable rate estimate** — never packs a single real bit. No
`quantiles`/`update()` machinery here to build integer CDF tables for an actual ANS/range coder,
unlike CompressAI's real `EntropyBottleneck`. The genuinely-measured "bytes on disk" number
reported at eval time comes from a completely separate mechanism — `real_compress_bytes()` running
`zlib`/`lzma` on the rounded integer latent — which doesn't touch `FactorizedEntropy`'s learned
parameters at all. `FactorizedEntropy` exists purely to give the training loop something
differentiable to optimize rate against, and to report the paper's own "entropy estimate" bpp at
eval — two different jobs, two different mechanisms in this codebase.

## Section 3 — Optimization of the Nonlinear Transform Coding Model

### Rate and distortion (basic definitions)

- **Distortion (D)**: how much the reconstructed image differs from the original — here,
  mean squared error (MSE) between original and reconstructed pixels. Higher D = worse/blurrier
  reconstruction.
- **Rate (R)**: number of bits needed to store/transmit the compressed representation. Lower R =
  smaller file.
- These trade off against each other: shrinking the file (lower R) generally increases distortion,
  and vice versa. This is the classic **rate–distortion trade-off**.
- The trade-off is controlled by a scalar **λ**: the training objective is `R + λD` (Eq. 7). A
  larger λ weights distortion more heavily → model favors low-distortion/high-rate solutions.
  A smaller λ tolerates more distortion in exchange for a lower rate. Training separate models at
  different λ values traces out the whole rate–distortion curve (the convex hull of achievable
  (R, D) points — see Figure 2, left panel).

### Why entropy is used as the rate proxy

- Directly simulating an entropy coder during training is unnecessary: a well-designed entropy
  code achieves a bit rate only slightly above the true entropy of the discrete distribution
  (Rissanen & Langdon, 1981). So the training objective (Eq. 7) defines rate directly as entropy:
  `L = -E[log2 P_q] + λ E[d(z, ẑ)]`.

### The quantization problem

- The quantizer is just rounding: `ŷ_i = q_i = round(y_i)` (Eq. 8).
- Rounding is a step function: its derivative is zero almost everywhere (or undefined at jumps),
  so gradient descent cannot optimize through it directly — gradients vanish.

### The fix: additive uniform noise relaxation

- During **training only**, replace the quantizer with additive i.i.d. uniform noise of width 1
  (same width as a quantization bin): `ỹ = y + Δy`, `Δy ~ U(-1/2, 1/2)`.
- This relaxation has two useful properties:
  1. The density of `ỹ` exactly equals the probability mass function of the true quantized
     variable at integer points (Eq. 10) — so differential entropy of `ỹ` is a good proxy for the
     discrete entropy of `q`.
  2. Uniform noise is a standard model of quantization error in general, so it's also a reasonable
     stand-in for the distortion term.
- Since `ỹ` is continuous and differentiable in the network parameters, standard SGD (via Adam)
  can now be used to train the model end-to-end (Eq. 11 gives the full continuous loss).
- Figure 4 in the paper empirically verifies this approximation is good: the noisy/relaxed rate
  and distortion track the true discrete-quantization values closely across the tested range of λ.
- Note: the noise relaxation is used **only for training**. At test/deployment time, actual
  rounding + a real entropy coder (CABAC) are used.

## Section 3.1 — Relationship to Variational Generative Image Models

### The mathematical coincidence

- Once the loss is made continuous (via the noise relaxation), it closely resembles the objective
  used to fit **variational autoencoders (VAEs)** (Kingma & Welling, 2014; Rezende et al., 2014).
- VAE setup: given data `x`, approximate the intractable true posterior `p(y|x)` with a simpler
  distribution `q(y|x)`, fit by minimizing `KL[q(y|x) || p(y|x)]` (Eq. 12). This KL divergence
  splits into three terms: a constant, a **reconstruction/distortion-like term**, and a
  **rate-like term** (regularizing `q` toward the prior `p(y)`).
- The authors show that with specific choices, their objective becomes exactly a VAE objective:
  - Treat the noisy code `ỹ` as the VAE's latent variable.
  - "Generative model" `p(x|ỹ)`: a Gaussian centered at the synthesis transform's output,
    variance related to `1/(2λ)` (Eq. 13).
  - Approximate posterior `q(ỹ|x)`: a uniform distribution of width 1 centered at the analysis
    transform's output (Eq. 15).
  - With these substitutions, the KL-divergence objective (Eq. 12) becomes mathematically
    identical to the rate–distortion objective (with MSE as distortion).
- Nice result: compression (rate–distortion theory) and generative modeling (VAEs) — two
  different research motivations — can lead to the same loss function.

### But conceptually, the two frameworks differ in important ways

1. **Continuous vs. discrete domain.** VAEs operate in continuous space; real compression must
   ultimately produce discrete bits. The continuous relaxation here is used strictly for
   *training*; evaluation always uses actual discrete bit rates (not differential entropy), to
   avoid misleading comparisons.
2. **Role of λ.** Generative models effectively want λ → ∞ (perfectly explain the data, i.e., the
   lossless-compression limit) — they don't care about the whole rate–distortion curve, just
   about explaining the data optimally. Compression models, in contrast, are explicitly optimized
   at many different λ values to map out the full rate–distortion trade-off curve — λ is a design
   knob, not something to be pushed to an extreme.
3. **Generality of the distortion term.** The clean correspondence to a genuine generative model
   (a normalizable probability density) only holds for well-behaved metrics like Euclidean
   distance (MSE). If a more general/perceptual distortion metric is used instead (as in the
   authors' earlier work), it may no longer correspond to any valid probability density — but it
   can still be a perfectly valid rate–distortion objective.

**Takeaway:** the VAE connection is a useful mathematical bridge, but compression and generative
modeling are solving different problems, and the equivalence only holds under specific
(non-perceptual, MSE-based) conditions.

## Diagrams created in this chat
- Rate–distortion trade-off curve (convex achievable region, two example λ operating points).

---
*(This document will be updated as we continue through the paper.)*