# From-scratch PyTorch reimplementation of Ballé et al. 2017 (no CompressAI)

## Context

We've been using CompressAI's pretrained `bmshj2018_factorized` as a black-box baseline
(`bmshj2018_compression.py`) and studying the paper's theory (`bmshj2018_study.md`). The user
wants to actually *build* the model — analysis/synthesis transforms, GDN, the factorized entropy
model, the noise-relaxation training trick, and a training/eval loop — ourselves in raw PyTorch,
without importing `compressai`, to understand it at the implementation level and (since this
project's whole focus is compressing PDEBench scalar-field data) train a model that's actually
native to 1-channel scientific data instead of a repurposed RGB natural-image model.

Confirmed scope (via user's earlier answers): train on this project's PDEBench data (not natural
images), start with a small proof-of-concept (fast to verify correctness, not a long high-fidelity
run), and a single λ first (not a rate-distortion sweep — that can be a follow-up once this works).

## Reference math (confirmed from reading CompressAI's actual source, to reimplement faithfully)

**GDN/IGDN** (`compressai/layers/gdn.py`):
```
GDN:  y[i] = x[i] / sqrt(beta[i] + Σ_j gamma[j,i] * x[j]^2)     (IGDN: sqrt instead of rsqrt)
```
Implemented via a 1x1 conv2d (`weight=gamma` reshaped to `(C,C,1,1)`, `bias=beta`) over `x**2`.
Non-negativity of beta/gamma via the reparameterization + lower-bound trick from Ballé's original
GDN paper: store `sqrt(value + pedestal)`, clamp via a custom autograd `LowerBound` (forward =
`max(x,bound)`; backward passes gradient through even when clamped, *unless* the gradient would
push further below the bound), then square and subtract the pedestal on every forward pass. This
keeps beta/gamma ≥ ~0 while remaining differentiable (a plain `clamp` would zero gradients at the
boundary). Defaults: `beta_min=1e-6`, `gamma_init=0.1` (diagonal), `reparam_offset=2**-18`.

**Factorized entropy model** (`compressai/entropy_models/entropy_models.py`): a small per-channel
"monotonic MLP" models each latent channel's CDF (Appendix 6.1 of the paper):
```python
filters = (1, 3, 3, 3, 3, 1)   # 5 layers
logits = inputs
for i in range(5):
    logits = softplus(matrices[i]) @ logits + biases[i]   # softplus -> non-negative weights -> monotonic
    if i < 4:
        logits = logits + tanh(factors[i]) * tanh(logits)  # monotonic nonlinearity (extra flexibility)
likelihood = sigmoid(cumulative(x+0.5)) - sigmoid(cumulative(x-0.5))
likelihood = max(likelihood, 1e-9)   # avoid log(0)
```
We do **not** need CompressAI's `quantiles`/`aux_loss`/`update()` machinery — that exists only to
build integer CDF tables for real ANS coding, which is out of scope here (see "Not building" below).

**Training-time quantization**: additive uniform noise `y_tilde = y + U(-0.5, 0.5)` (differentiable
proxy). **Eval-time**: real `round()`. Both paths reuse the same likelihood formula (valid since the
density model gives probability mass in a unit bin around any point, integer or not).

**Model**: `g_a`: `conv(1,N,5,2)→GDN→conv(N,N,5,2)→GDN→conv(N,N,5,2)→GDN→conv(N,M,5,2)` (4x
downsample, factor 16 total); `g_s` mirrors with `ConvTranspose2d`/IGDN back to 1 channel — natively
grayscale, no pseudo-RGB replication needed this time.

**Loss**: `loss = λ * MSE(x, x_hat) + bpp`, where `bpp = -log2(likelihood).sum() / num_pixels`
(CompressAI's own convention, dropping the `255²` MSE scaling since our data is normalized to
`[0,1]` physical field values, not 8-bit pixels).

## Not building (explicitly out of scope for this pass)

- Real ANS/range-coded bitstreams (CompressAI's `compress()`/`decompress()` + CDF-table calibration).
  Rate is reported as the differentiable entropy estimate `-log2(likelihood)`, exactly how the
  paper itself reports Figure 2's rate-distortion curve — a well-designed entropy coder gets only
  slightly above this (Rissanen & Langdon 1981, already noted in `bmshj2018_study.md`).
- As a cheap bonus (reusing the `real_compress_bytes()` pattern already written for
  `svd_compression.py`), the eval step will *also* zlib/lzma-compress the rounded integer latent for
  one genuinely measured "bytes on disk" number, round-trip verified — consistent with this
  project's established rigor, at near-zero extra implementation cost.
- Multi-λ sweep — one model, one λ, to get the pipeline verified end-to-end first.

## New file: `bmshj2018_scratch.py`

Self-contained, single-file script (matching the existing `svd_compression.py` /
`bmshj2018_compression.py` convention: module docstring, config-driven CLI, `experiments/..._TIMESTAMP/`
output dir with `plots/`, `run.log` via the existing print-shadowing helper — copy that helper
verbatim, it's the pdb-safety fix from earlier in this session).

**Components:**
1. `LowerBound` — custom `torch.autograd.Function` (forward=`max`, gradient-passthrough rule above).
2. `NonNegativeParametrizer` — the pedestal/bound/square reparameterization for GDN's beta/gamma.
3. `GDN(nn.Module)` — forward/inverse, built from the conv2d-over-squared-input formula above.
4. `FactorizedEntropy(nn.Module)` — the monotonic-MLP density model + likelihood computation +
   noise-vs-round quantization switch (training vs eval mode).
5. `AnalysisTransform` / `SynthesisTransform` (`nn.Sequential`-style stacks of conv/deconv + GDN/IGDN).
6. `Bmshj2018Scratch(nn.Module)` — wires the above into `forward(x) -> (x_hat, likelihoods)`.
7. `PDEBenchSliceDataset` — loads a modest in-RAM cache of 2D slices (axis=0 only, for fast
   contiguous HDF5 reads) from training timesteps, serves random crops (`--patch-size`, default
   128) for training batches; a separate held-out timestep is cached similarly for periodic
   validation, and one *full* held-out volume (all 256 slices) is used for the final full-volume
   eval — mirroring `bmshj2018_compression.py`'s existing eval methodology so results are
   comparable (rel_err/PSNR/comp_ratio/BPV, same formulas).
8. `train()` — Adam optimizer, prints periodic loss/bpp/PSNR, saves a training-curve plot
   (loss/rate/distortion vs iteration) and the final model checkpoint (`torch.save`).
9. `evaluate()` — full-volume reconstruction on the held-out volume, real `round()` quantization,
   entropy-estimate bpp, PSNR/rel_err (same formulas as the other two scripts), plus the bonus real
   zlib/lzma-compressed byte count; saves a results CSV (same schema as the other scripts, one row
   for now) and a full-volume reconstruction plot (same 3-mid-plane style as the existing Plot 4s).

**CLI** (config-driven like the other scripts, plus new args):
`--field` (default from config, e.g. `pressure`), `--channels-n` (default 64), `--channels-m`
(default 64), `--lambda_` (default 0.01), `--patch-size` (default 128), `--batch-size` (default 16),
`--iterations` (default 3000), `--lr` (default 1e-4), `--train-timesteps` (default `0-34`),
`--val-timestep` (default `40`, held out), `--slices-per-timestep` (default 16, for the training
cache), `--output-dir`.

**Known limitation to note in the docstring**: training diversity comes from one simulation's
z-slices/timesteps, not a large natural-image corpus — expected to underperform the pretrained
RGB-trained CompressAI model on absolute quality; the goal here is a correct, working from-scratch
pipeline, not beating the baseline.

## Verification

1. Run end-to-end on the real `Turb_M1.hdf5` data with the small proof-of-concept defaults; confirm
   no errors, finite (non-NaN/Inf) likelihoods and loss throughout training.
2. Confirm loss/bpp/distortion trend downward over the training run (sanity check that gradients
   are actually flowing through GDN's `LowerBound` and the factorized entropy model correctly).
3. Confirm the real zlib/lzma round-trip assertion passes on the rounded latent (same check already
   used in `svd_compression.py`).
4. Visually inspect the saved training-curve and full-volume reconstruction plots.
5. Clean up any test experiment output dirs afterward, as done for the other two scripts.

## Implementation summary (post-verification)

Built as planned in `bmshj2018_scratch.py`: `GDN`/`IGDN` with the non-negative reparameterization +
`LowerBound` trick, `FactorizedEntropy` (the Appendix 6.1 monotonic-MLP density model), the
noise-vs-round quantization switch, full analysis/synthesis conv stacks (natively 1-channel), a
PDEBench slice loader, and train/evaluate loops — including the bonus real zlib/lzma-compressed
byte count on the rounded latent (round-trip verified), extending this project's established
"real bytes" rigor to this model too.

**A real bug caught during verification, not just a smoke-test formality**: the initial default
`λ=0.01` (copied loosely from CompressAI's own convention) caused the model to collapse to a
near-constant reconstruction — visually confirmed, and an absurd ~24,000x "compression ratio" was
the tell (a near-constant signal is trivially compressible). Root cause: CompressAI scales MSE by
`255²` before applying its λ, since it works in 8-bit pixel units; this reimplementation correctly
dropped that scaling for our `[0,1]`-normalized physical field data, but the default λ wasn't scaled
up to compensate — leaving distortion essentially unweighted in the loss, so the optimizer had
almost no incentive to reconstruct anything beyond the cheapest (near-constant) code. Fixed by
raising the default to `λ=1000` (documented in the script's module docstring and `--lambda_` help
text with the reasoning), confirmed by rerunning.

**Verified results** (small proof-of-concept scale: N=M=64, patch_size=128, 3000 iterations,
trained on timesteps 0-34, evaluated on fully held-out timestep 40):
- Training curves are textbook rate-distortion dynamics: loss drops sharply then smooths, bpp
  decreases steadily (1.36 → 0.58), PSNR rises and plateaus (~25 → ~37dB) — no divergence, no
  NaN/Inf, train/val track closely.
- Final held-out full-volume eval: rel_err=0.064, PSNR=34.2dB, bpp (entropy estimate)=0.571
  (comp_ratio_estimate≈56x), real zlib/lzma-measured bytes → comp_ratio≈170x.
- Reconstruction plot visually confirms real turbulent structure is captured (not just blur/mean),
  appropriately soft for a heavily compressed, briefly-trained model.
- Round-trip correctness assertion on the real-compressed rounded latent passed.

Kept the meaningful run's outputs at `experiments/scratch_poc2/` (checkpoint + plots + CSV) since
it's a real, useful result; deleted an earlier 100-iteration/16-channel microtest run that only
existed to catch bugs before scaling up.

**Natural next steps** (not yet done): a λ sweep for a real rate-distortion curve comparable to
`svd_compression.py`/`bmshj2018_compression.py`, and wiring results into `compare_compression.py`.
