"""
From-scratch PyTorch reimplementation of Ballé, Laparra, Simoncelli,
"End-to-end Optimized Image Compression" (ICLR 2017, arXiv:1611.01704) —
the model CompressAI calls `bmshj2018_factorized` — without importing
compressai. See bmshj2018_study.md for the paper's theory notes this
implementation follows.

Reimplemented from first principles (math confirmed by reading CompressAI's
own source, not guessed):
  - GDN / inverse GDN activation, with the non-negative reparameterization +
    gradient-preserving lower-bound trick from Balle's original GDN paper.
  - The flexible per-channel "factorized" density model (a small monotonic
    MLP per latent channel) from Appendix 6.1, used both to estimate rate
    during training and to evaluate it at test time.
  - The additive-uniform-noise relaxation for training (Eq. 8-11 in the
    study notes) vs. real rounding at evaluation.

Trained natively on this project's own PDEBench scalar-field data (1-channel
grayscale slices of Turb_M1.hdf5), not natural RGB images — so no pseudo-RGB
replication hack is needed, unlike bmshj2018_compression.py's pretrained
model. Training diversity comes from one simulation's z-slices/timesteps,
not a large natural-image corpus, so this is expected to underperform the
pretrained RGB-trained CompressAI model in absolute quality — the goal here
is a correct, working from-scratch pipeline, not beating that baseline.

Not implemented (out of scope): real ANS/range-coded bitstreams (that needs
CompressAI's `quantiles`/`update()` machinery to calibrate integer CDF
tables). Rate is reported as the differentiable entropy estimate
-log2(likelihood), exactly how the paper itself reports its rate-distortion
curve (a well-designed entropy coder gets only slightly above this estimate
— Rissanen & Langdon 1981). As a cheap bonus, evaluation also zlib/lzma-
compresses the rounded integer latent for one genuinely measured "bytes on
disk" number, reusing the same real-entropy-coding pattern already used in
svd_compression.py.

Loss: `lambda_ * MSE(x, x_hat) + bpp`, where `bpp = -log2(likelihood).sum() / num_pixels`
(CompressAI's own convention). Note: CompressAI scales MSE by 255^2 before applying its own
lambda, since it works with 8-bit pixel images; our data is normalized to [0,1] physical field
values instead, so `--lambda_` needs to be roughly 255^2 (~65000x) larger than CompressAI's
lambda values for a comparable rate/distortion balance -- e.g. CompressAI's lambda~0.01 -> ours
~650. Too small a lambda starves the distortion term of any real gradient signal and the model
collapses to a near-constant (cheap-to-encode) reconstruction; that failure mode is exactly what
motivated raising this script's default from an initial (buggy) 0.01 to 1000.

Usage
-----
    python bmshj2018_scratch.py config_simmldc.yaml
    python bmshj2018_scratch.py config_simmldc.yaml --iterations 5000 --lambda_ 2000
    python bmshj2018_scratch.py config_simmldc.yaml --channels-n 32 --channels-m 32
    python bmshj2018_scratch.py config_simmldc.yaml --output-dir results/scratch_run1
"""

import argparse
import lzma
import math
import os
import zlib
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


# ---------------------------------------------------------------------------
# GDN / IGDN
# ---------------------------------------------------------------------------

class _LowerBoundFn(torch.autograd.Function):
    """max(x, bound), but gradient still flows through when clamped unless
    it would push x further below the bound (Balle's GDN reparameterization
    trick) — a plain clamp would zero the gradient at the boundary entirely."""

    @staticmethod
    def forward(ctx, x, bound):
        ctx.save_for_backward(x, bound)
        return torch.max(x, bound)

    @staticmethod
    def backward(ctx, grad_output):
        x, bound = ctx.saved_tensors
        pass_through = (x >= bound) | (grad_output < 0)
        return pass_through.type(grad_output.dtype) * grad_output, None


class NonNegativeParametrizer(nn.Module):
    """Stores sqrt(value + pedestal); forward squares it back, guaranteeing
    the effective value stays >= `minimum` while remaining differentiable."""

    def __init__(self, minimum: float = 0.0, reparam_offset: float = 2 ** -18):
        super().__init__()
        self.minimum = float(minimum)
        self.reparam_offset = float(reparam_offset)
        pedestal = self.reparam_offset ** 2
        self.register_buffer("pedestal", torch.tensor([pedestal]))
        bound = (self.minimum + self.reparam_offset ** 2) ** 0.5
        self.register_buffer("bound", torch.tensor([bound]))

    def init(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(torch.clamp(x + self.pedestal, min=self.pedestal.item()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = _LowerBoundFn.apply(x, self.bound)
        return out ** 2 - self.pedestal


class GDN(nn.Module):
    """Generalized Divisive Normalization: y_i = x_i / sqrt(beta_i + sum_j gamma_ij x_j^2).
    inverse=True computes IGDN: y_i = x_i * sqrt(...), used in the synthesis transform."""

    def __init__(self, channels: int, inverse: bool = False,
                 beta_min: float = 1e-6, gamma_init: float = 0.1,
                 reparam_offset: float = 2 ** -18):
        super().__init__()
        self.inverse = inverse

        self.beta_reparam = NonNegativeParametrizer(minimum=beta_min, reparam_offset=reparam_offset)
        beta = self.beta_reparam.init(torch.ones(channels))
        self.beta = nn.Parameter(beta)

        self.gamma_reparam = NonNegativeParametrizer(reparam_offset=reparam_offset)
        gamma = self.gamma_reparam.init(gamma_init * torch.eye(channels))
        self.gamma = nn.Parameter(gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.size(1)
        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma).reshape(C, C, 1, 1)
        norm = F.conv2d(x ** 2, gamma, beta)   # beta_i + sum_j gamma_ij x_j^2, per pixel
        norm = torch.sqrt(norm) if self.inverse else torch.rsqrt(norm)
        return x * norm


# ---------------------------------------------------------------------------
# Factorized entropy model (Appendix 6.1 density model)
# ---------------------------------------------------------------------------

class FactorizedEntropy(nn.Module):
    """Per-channel flexible density model built from a small monotonic MLP,
    used both as the training-time rate estimate and the eval-time one.
    Not implemented: quantiles/update()/real ANS coding (see module docstring)."""

    def __init__(self, channels: int, filters=(3, 3, 3, 3),
                 init_scale: float = 10.0, likelihood_bound: float = 1e-9):
        super().__init__()
        self.channels = channels
        self.filters = (1,) + tuple(filters) + (1,)
        self.likelihood_bound = likelihood_bound

        scale = init_scale ** (1.0 / (len(self.filters) - 1))
        self.matrices = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.factors = nn.ParameterList()
        for i in range(len(self.filters) - 1):
            init_val = math.log(math.expm1(1.0 / scale / self.filters[i + 1]))
            matrix = torch.full((channels, self.filters[i + 1], self.filters[i]), init_val)
            self.matrices.append(nn.Parameter(matrix))
            bias = torch.empty(channels, self.filters[i + 1], 1).uniform_(-0.5, 0.5)
            self.biases.append(nn.Parameter(bias))
            if i < len(self.filters) - 2:
                factor = torch.zeros(channels, self.filters[i + 1], 1)
                self.factors.append(nn.Parameter(factor))

    def _logits_cumulative(self, x: torch.Tensor) -> torch.Tensor:
        """x: (channels, 1, N) -> logits of the learned per-channel CDF at x."""
        logits = x
        for i in range(len(self.matrices)):
            matrix = F.softplus(self.matrices[i])          # non-negative weights -> monotonic
            logits = torch.matmul(matrix, logits) + self.biases[i]
            breakpoint()
            if i < len(self.factors):
                factor = torch.tanh(self.factors[i])
                logits = logits + factor * torch.tanh(logits)   # monotonic nonlinearity
        return logits

    def forward(self, y: torch.Tensor):
        """y: (B, C, H, W). Returns (y_tilde, likelihood), same shape as y.
        Training: y_tilde = y + U(-0.5, 0.5) (differentiable proxy for rounding).
        Eval:     y_tilde = round(y) (real quantization)."""
        B, C, H, W = y.shape
        if self.training:
            noise = torch.empty_like(y).uniform_(-0.5, 0.5)
            y_tilde = y + noise
        else:
            y_tilde = torch.round(y)

        y_perm = y_tilde.permute(1, 0, 2, 3).reshape(C, 1, -1)   # (C, 1, B*H*W)
        lower = self._logits_cumulative(y_perm - 0.5)
        upper = self._logits_cumulative(y_perm + 0.5)
        # Numerically stable c(y+0.5) - c(y-0.5): flip sign so sigmoid stays away from
        # its saturated tails regardless of how large the logits get.
        sign = -torch.sign(lower + upper).detach()
        likelihood = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))
        likelihood = torch.clamp(likelihood, min=self.likelihood_bound)

        likelihood = likelihood.reshape(C, B, H, W).permute(1, 0, 2, 3)
        return y_tilde, likelihood


# ---------------------------------------------------------------------------
# Analysis / synthesis transforms + full model
# ---------------------------------------------------------------------------

class AnalysisTransform(nn.Module):
    def __init__(self, in_channels: int, N: int, M: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, N, 5, stride=2, padding=2),
            GDN(N),
            nn.Conv2d(N, M, 5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.net(x)


class SynthesisTransform(nn.Module):
    def __init__(self, out_channels: int, N: int, M: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(M, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, N, 5, stride=2, padding=2, output_padding=1),
            GDN(N, inverse=True),
            nn.ConvTranspose2d(N, out_channels, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        return self.net(x)


class Bmshj2018Scratch(nn.Module):
    def __init__(self, in_channels: int = 1, N: int = 64, M: int = 64):
        super().__init__()
        self.g_a = AnalysisTransform(in_channels, N, M)
        self.g_s = SynthesisTransform(in_channels, N, M)
        self.entropy = FactorizedEntropy(M)

    def forward(self, x):
        y = self.g_a(x)
        y_tilde, likelihood = self.entropy(y)
        x_hat = self.g_s(y_tilde)
        return x_hat, likelihood, y_tilde


# ---------------------------------------------------------------------------
# PDEBench slice data
# ---------------------------------------------------------------------------

def load_slice_cache(h5_path: str, field: str, timesteps: list, slices_per_timestep: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Read a modest in-RAM cache of 2D z-slices (axis 0, contiguous HDF5 reads)."""
    slices = []
    with h5py.File(h5_path, "r") as f:
        dset = f[field]
        n_slices_avail = dset.shape[1]
        for t in timesteps:
            idxs = rng.choice(n_slices_avail, size=min(slices_per_timestep, n_slices_avail), replace=False)
            for idx in sorted(idxs.tolist()):
                slices.append(dset[t, idx].astype(np.float32))
    return np.stack(slices, axis=0)   # (n_slices, H, W)


def sample_batch(cache: np.ndarray, patch_size: int, batch_size: int,
                  rng: np.random.Generator, device: str) -> torch.Tensor:
    n, H, W = cache.shape
    batch = np.empty((batch_size, 1, patch_size, patch_size), dtype=np.float32)
    for i in range(batch_size):
        idx = rng.integers(n)
        y0 = rng.integers(0, H - patch_size + 1)
        x0 = rng.integers(0, W - patch_size + 1)
        batch[i, 0] = cache[idx, y0:y0 + patch_size, x0:x0 + patch_size]
    return torch.from_numpy(batch).to(device)


def load_full_volume(h5_path: str, field: str, timestep: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f[field][timestep].astype(np.float32)


# ---------------------------------------------------------------------------
# Real entropy coding of the rounded latent (bonus "bytes on disk" metric)
# ---------------------------------------------------------------------------

def real_compress_bytes(q: np.ndarray) -> int:
    """Real, lossless, round-trip-verified compressed size (best of zlib/lzma)."""
    raw = q.tobytes()
    zlib_bytes = zlib.compress(raw, level=9)
    lzma_bytes = lzma.compress(raw, preset=9)
    if len(zlib_bytes) <= len(lzma_bytes):
        assert zlib.decompress(zlib_bytes) == raw, "real entropy coding round-trip failed"
        return len(zlib_bytes)
    assert lzma.decompress(lzma_bytes) == raw, "real entropy coding round-trip failed"
    return len(lzma_bytes)


def compute_metrics(input_vol: np.ndarray, recon_vol: np.ndarray) -> dict:
    error = input_vol - recon_vol
    mse = float((error ** 2).mean())
    rmse = float(np.sqrt(mse))
    rel_err = float(np.sqrt(mse) / (np.sqrt((input_vol ** 2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr = float(20 * np.log10(sig_range / (rmse + 1e-12)))
    return dict(rel_err=rel_err, rmse=rmse, psnr=psnr)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

def train(model, train_cache, val_cache, args, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(0)

    history = {"iter": [], "loss": [], "bpp": [], "mse": [], "psnr": []}
    val_history = {"iter": [], "loss": [], "bpp": [], "psnr": []}

    model.train()
    for it in range(1, args.iterations + 1):
        x = sample_batch(train_cache, args.patch_size, args.batch_size, rng, device)

        x_hat, likelihood, _ = model(x)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = (-torch.log2(likelihood).sum()) / num_pixels
        mse = F.mse_loss(x_hat, x)
        loss = args.lambda_ * mse + bpp

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 50 == 0 or it == 1:
            psnr = 10 * math.log10(1.0 / max(mse.item(), 1e-12))
            history["iter"].append(it)
            history["loss"].append(loss.item())
            history["bpp"].append(bpp.item())
            history["mse"].append(mse.item())
            history["psnr"].append(psnr)
            print(f"iter {it:>6}/{args.iterations}  loss={loss.item():.4f}  "
                  f"bpp={bpp.item():.4f}  mse={mse.item():.6f}  psnr={psnr:.2f}dB")

        if it % 500 == 0:
            model.eval()
            with torch.no_grad():
                xv = sample_batch(val_cache, args.patch_size, args.batch_size, rng, device)
                xv_hat, v_like, _ = model(xv)
                v_pixels = xv.size(0) * xv.size(2) * xv.size(3)
                v_bpp = (-torch.log2(v_like).sum()) / v_pixels
                v_mse = F.mse_loss(xv_hat, xv)
                v_loss = args.lambda_ * v_mse + v_bpp
                v_psnr = 10 * math.log10(1.0 / max(v_mse.item(), 1e-12))
            val_history["iter"].append(it)
            val_history["loss"].append(v_loss.item())
            val_history["bpp"].append(v_bpp.item())
            val_history["psnr"].append(v_psnr)
            print(f"  [val] iter {it:>6}  loss={v_loss.item():.4f}  "
                  f"bpp={v_bpp.item():.4f}  psnr={v_psnr:.2f}dB")
            model.train()

    return history, val_history


@torch.no_grad()
def evaluate(model, vol01: np.ndarray, vmin: float, vmax: float, batch_size: int, device: str):
    """Full-volume reconstruction on a held-out volume, real rounding, entropy-estimate
    bpp, plus a real zlib/lzma-compressed byte count of the rounded latent."""
    model.eval()
    D, H, W = vol01.shape
    recon01 = np.empty_like(vol01)
    total_neg_log2_likelihood = 0.0
    latent_chunks = []

    for start in range(0, D, batch_size):
        batch = vol01[start:start + batch_size]
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        x_hat, likelihood, y_tilde = model(x)
        total_neg_log2_likelihood += (-torch.log2(likelihood)).sum().item()
        recon01[start:start + batch.shape[0]] = x_hat.squeeze(1).clamp(0, 1).cpu().numpy()
        latent_chunks.append(y_tilde.round().to(torch.int32).cpu().numpy())

    bpp_estimate = total_neg_log2_likelihood / (D * H * W)

    latent = np.concatenate(latent_chunks, axis=0)
    real_bytes = real_compress_bytes(latent)

    recon_vol = recon01 * (vmax - vmin) + vmin
    input_vol = vol01 * (vmax - vmin) + vmin
    metrics = compute_metrics(input_vol, recon_vol)

    n_voxels = D * H * W
    bytes_in = n_voxels * 4
    metrics.update(
        bpp_estimate=bpp_estimate,
        comp_ratio_estimate=32.0 / bpp_estimate,
        real_bytes=real_bytes,
        real_bpv=(real_bytes * 8) / n_voxels,
        real_comp_ratio=bytes_in / real_bytes if real_bytes > 0 else float("inf"),
    )
    return metrics, recon_vol, input_vol


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_timestep_range(s: str) -> list:
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s)]


def main():
    parser = argparse.ArgumentParser(description="From-scratch bmshj2018-factorized (no compressai)")
    parser.add_argument("config", help="path to config_simmldc.yaml")
    parser.add_argument("--field", default=None, help="override config's field_key")
    parser.add_argument("--channels-n", type=int, default=64, help="main conv channel count (N)")
    parser.add_argument("--channels-m", type=int, default=64, help="latent channel count (M)")
    parser.add_argument("--lambda_", type=float, default=1000.0,
                        help="rate-distortion trade-off weight (loss = lambda*MSE + bpp; "
                             "note this data is normalized to [0,1] not 255-level pixels, "
                             "so lambda needs to be ~255^2 larger than compressai's own "
                             "lambda convention for a comparable rate/distortion balance)")
    parser.add_argument("--patch-size", type=int, default=128, help="training crop size")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-timesteps", default="0-34", help="e.g. '0-34'")
    parser.add_argument("--val-timestep", type=int, default=40, help="held out of training entirely")
    parser.add_argument("--slices-per-timestep", type=int, default=16,
                        help="random z-slices cached per timestep for training/quick-val batches")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--output-dir", default=None,
                        help="output directory (default: experiments/scratch_TIMESTAMP)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    field = args.field or dcfg["field_key"]
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = args.output_dir or os.path.join(
        "experiments", "scratch_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    import shutil
    shutil.copy(args.config, os.path.join(out_dir, "config_simmldc.yaml"))

    log_path = os.path.join(out_dir, "run.log")
    import builtins
    _log = open(log_path, "w")
    _builtin_print = builtins.print

    def print(*a, **kw):
        _builtin_print(*a, **kw)
        kw.pop("file", None)
        _builtin_print(*a, file=_log, **kw)
        _log.flush()

    print(f"Output dir : {out_dir}")
    print(f"Config     : {args.config}")
    print(f"Device     : {device}")
    print(f"Field      : {field}")
    print(f"N, M       : {args.channels_n}, {args.channels_m}")
    print(f"Lambda     : {args.lambda_}")
    print(f"Patch size : {args.patch_size}")
    print(f"Iterations : {args.iterations}\n")

    train_timesteps = parse_timestep_range(args.train_timesteps)
    print(f"Train timesteps : {train_timesteps[0]}-{train_timesteps[-1]} ({len(train_timesteps)} steps)")
    print(f"Val timestep    : {args.val_timestep} (held out of training)\n")

    # ------------------------------------------------------------------ #
    # Load training/validation slice caches
    # ------------------------------------------------------------------ #
    rng = np.random.default_rng(42)
    print("Loading training slice cache ...")
    train_cache_raw = load_slice_cache(dcfg["h5_path"], field, train_timesteps,
                                        args.slices_per_timestep, rng)
    print(f"Train cache: {train_cache_raw.shape}  ({train_cache_raw.nbytes/1024/1024:.1f} MB)")

    print("Loading validation slice cache (quick, held-out timestep) ...")
    val_cache_raw = load_slice_cache(dcfg["h5_path"], field, [args.val_timestep],
                                      args.slices_per_timestep, rng)
    print(f"Val cache  : {val_cache_raw.shape}  ({val_cache_raw.nbytes/1024/1024:.1f} MB)\n")

    vmin, vmax = float(train_cache_raw.min()), float(train_cache_raw.max())
    print(f"Normalization (from training cache): vmin={vmin:.4f}  vmax={vmax:.4f}\n")
    train_cache = (train_cache_raw - vmin) / (vmax - vmin + 1e-8)
    val_cache = (val_cache_raw - vmin) / (vmax - vmin + 1e-8)

    # ------------------------------------------------------------------ #
    # Build + train model
    # ------------------------------------------------------------------ #
    model = Bmshj2018Scratch(in_channels=1, N=args.channels_n, M=args.channels_m).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    print("Training ...")
    history, val_history = train(model, train_cache, val_cache, args, device)
    print("\nTraining done.\n")

    # ------------------------------------------------------------------ #
    # Plot 1 — training curves
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["iter"], history["loss"], color="steelblue", label="train")
    axes[0].plot(val_history["iter"], val_history["loss"], color="darkorange", label="val")
    axes[0].set_xlabel("iteration"); axes[0].set_ylabel("loss"); axes[0].set_title("Loss")
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["iter"], history["bpp"], color="steelblue", label="train")
    axes[1].plot(val_history["iter"], val_history["bpp"], color="darkorange", label="val")
    axes[1].set_xlabel("iteration"); axes[1].set_ylabel("bpp (entropy estimate)"); axes[1].set_title("Rate")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    axes[2].plot(history["iter"], history["psnr"], color="steelblue", label="train")
    axes[2].plot(val_history["iter"], val_history["psnr"], color="darkorange", label="val")
    axes[2].set_xlabel("iteration"); axes[2].set_ylabel("PSNR (dB)"); axes[2].set_title("Distortion")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Full-volume evaluation on the held-out validation timestep
    # ------------------------------------------------------------------ #
    print(f"\nEvaluating on full held-out volume (t={args.val_timestep}) ...")
    full_vol = load_full_volume(dcfg["h5_path"], field, args.val_timestep)
    full_vol01 = (full_vol - vmin) / (vmax - vmin + 1e-8)

    metrics, recon_vol, input_vol = evaluate(model, full_vol01, vmin, vmax, args.batch_size, device)

    print(f"rel_err              : {metrics['rel_err']:.6f}")
    print(f"PSNR                 : {metrics['psnr']:.2f} dB")
    print(f"bpp (entropy est.)   : {metrics['bpp_estimate']:.4f}  "
          f"(comp_ratio_est. {metrics['comp_ratio_estimate']:.2f}x)")
    print(f"Real bytes (zlib/lzma): {metrics['real_bytes']:,}  "
          f"(BPV={metrics['real_bpv']:.4f}, comp_ratio={metrics['real_comp_ratio']:.2f}x)")

    # ------------------------------------------------------------------ #
    # Plot 2 — full-volume reconstruction
    # ------------------------------------------------------------------ #
    D, H, W = input_vol.shape
    mD, mH, mW = D // 2, H // 2, W // 2
    plane_defs = [
        ("XY (z=mid)", input_vol[:, :, mW], recon_vol[:, :, mW]),
        ("XZ (y=mid)", input_vol[:, mH, :], recon_vol[:, mH, :]),
        ("YZ (x=mid)", input_vol[mD, :, :], recon_vol[mD, :, :]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"From-scratch bmshj2018  |  N={args.channels_n} M={args.channels_m} lambda={args.lambda_}  "
        f"rel_err={metrics['rel_err']:.4f}  bpp_est={metrics['bpp_estimate']:.3f}  "
        f"real_comp={metrics['real_comp_ratio']:.2f}x",
        fontsize=10,
    )
    for col, (lbl, inp_p, rec_p) in enumerate(plane_defs):
        vmax_p = np.percentile(np.abs(inp_p), 99)
        for row, (data, row_lbl) in enumerate([(inp_p, "Input"), (rec_p, "Reconstruction")]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap="RdBu_r", vmin=-vmax_p, vmax=vmax_p, origin="lower", aspect="equal")
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if col == 0:
                ax.set_ylabel(row_lbl, fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.colorbar(im, ax=ax, shrink=0.85)
    plt.tight_layout()
    out = os.path.join(plots_dir, "full_volume_reconstruction.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Save results CSV + checkpoint
    # ------------------------------------------------------------------ #
    import csv
    csv_path = os.path.join(out_dir, "scratch_results.csv")
    row = dict(
        channels_n=args.channels_n, channels_m=args.channels_m, lambda_=args.lambda_,
        iterations=args.iterations, **metrics,
    )
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"\nResults saved to {csv_path}")

    ckpt_path = os.path.join(out_dir, "model.pt")
    torch.save({"model_state": model.state_dict(),
                "vmin": vmin, "vmax": vmax,
                "args": vars(args)}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    print("\nDone.")
    _log.close()


if __name__ == "__main__":
    main()
