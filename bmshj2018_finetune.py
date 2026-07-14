"""
Fine-tunes CompressAI's pretrained bmshj2018-factorized model (the same
zoo checkpoint bmshj2018_compression.py runs inference-only) on this
project's own PDEBench scalar-field data.

Unlike bmshj2018_compression.py (frozen pretrained weights, quality sweep,
no gradients) and bmshj2018_scratch.py (a from-scratch reimplementation,
trained from random init), this script starts from the pretrained
ImageNet/CLIC weights and continues training with backprop on
pseudo-RGB slices of Turb_M1.hdf5, using CompressAI's own real
entropy-coding machinery (model.compress/decompress) for evaluation.

Loss follows CompressAI's own training convention (examples/train.py):
    loss = lambda_ * 255**2 * MSE(x, x_hat) + bpp
The 255**2 factor is CompressAI's convention regardless of the data's
native scale — our slices are already normalised to [0, 1] like natural
images, so no extra rescaling is needed (unlike bmshj2018_scratch.py's
lambda, which needs an explicit ~255**2 correction because that script's
loss has no such factor built in).

The entropy bottleneck's CDF parameters (`entropy_bottleneck.quantiles`)
need a separate optimizer at a higher learning rate — mixing them into the
main optimizer either starves them of signal or destabilizes the main
rate-distortion loss (CompressAI's own examples/train.py convention).

Config
------
Use `config_compresai.yaml`'s `finetune:` section (quality/metric/lambda_/
batch_size/iterations/lr/aux_lr/patch_size/axis/train_timesteps/
val_timestep/slices_per_timestep). CLI flags override the config value
they correspond to.

Usage
-----
    python bmshj2018_finetune.py config_compresai.yaml
    python bmshj2018_finetune.py config_compresai.yaml --quality 6 --iterations 5000
    python bmshj2018_finetune.py config_compresai.yaml --lambda_ 0.02 --lr 5e-5
    python bmshj2018_finetune.py config_compresai.yaml --output-dir results/finetune_run1
"""

import argparse
import builtins
import csv
import os
import shutil
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from compressai.ops import compute_padding
from compressai.zoo import bmshj2018_factorized

# CompressAI's commonly-cited per-quality lambda values for the "mse" metric.
# ms-ssim uses a different, unpublished-here scale — pass --lambda_ explicitly for it.
STANDARD_MSE_LAMBDAS = {
    1: 0.0018, 2: 0.0035, 3: 0.0067, 4: 0.0130,
    5: 0.0250, 6: 0.0483, 7: 0.0932, 8: 0.1800,
}


# ---------------------------------------------------------------------------
# PDEBench slice data (same convention as bmshj2018_scratch.py)
# ---------------------------------------------------------------------------

def load_slice_cache(h5_path: str, field: str, timesteps: list, slices_per_timestep: int,
                      rng: np.random.Generator) -> np.ndarray:
    """Random (Y,Z) slices (axis 0) for each t in timesteps -> (n_slices, H, W)."""
    slices = []
    with h5py.File(h5_path, "r") as f:
        dset = f[field]
        n_slices_avail = dset.shape[1]
        for t in timesteps:
            idxs = rng.choice(n_slices_avail, size=min(slices_per_timestep, n_slices_avail), replace=False)
            for idx in sorted(idxs.tolist()):
                slices.append(dset[t, idx].astype(np.float32))
    return np.stack(slices, axis=0)


def sample_batch_rgb(cache: np.ndarray, patch_size: int, batch_size: int,
                      rng: np.random.Generator, device: str) -> torch.Tensor:
    """Random patch_size x patch_size crops, replicated to pseudo-RGB -> (B, 3, P, P)."""
    n, H, W = cache.shape
    batch = np.empty((batch_size, patch_size, patch_size), dtype=np.float32)
    for i in range(batch_size):
        idx = rng.integers(n)
        y0 = rng.integers(0, H - patch_size + 1)
        x0 = rng.integers(0, W - patch_size + 1)
        batch[i] = cache[idx, y0:y0 + patch_size, x0:x0 + patch_size]
    x = torch.from_numpy(batch).unsqueeze(1).to(device)
    return x.repeat(1, 3, 1, 1)


def load_full_volume(h5_path: str, field: str, timestep: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        return f[field][timestep].astype(np.float32)


def parse_timestep_range(s) -> list:
    s = str(s)
    if "-" in s:
        lo, hi = s.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s)]


# ---------------------------------------------------------------------------
# Metrics (same formulas as bmshj2018_compression.py / svd_compression.py)
# ---------------------------------------------------------------------------

def compute_metrics(input_vol: np.ndarray, recon_vol: np.ndarray, n_bytes: int, n_voxels: int) -> dict:
    error = input_vol - recon_vol
    mse = float((error ** 2).mean())
    rmse = float(np.sqrt(mse))
    rel_err = float(np.sqrt(mse) / (np.sqrt((input_vol ** 2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    bytes_in = n_voxels * 4
    comp_ratio = bytes_in / n_bytes if n_bytes > 0 else float("inf")
    bpv = (n_bytes * 8) / n_voxels
    return dict(rel_err=rel_err, rmse=rmse, psnr=psnr, comp_ratio=comp_ratio, bpv=bpv, n_bytes=n_bytes)


# ---------------------------------------------------------------------------
# Real entropy-coded full-volume compress/decompress (same as bmshj2018_compression.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compress_volume(vol01: np.ndarray, model, device: str, axis: int, batch_size: int):
    slices = np.moveaxis(vol01, axis, 0)
    N, H, W = slices.shape
    pad, unpad = compute_padding(H, W, min_div=64)

    recon_slices = np.empty_like(slices)
    total_bytes = 0
    for start in range(0, N, batch_size):
        batch = slices[start:start + batch_size]
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        x = x.repeat(1, 3, 1, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        out_enc = model.compress(x)
        total_bytes += sum(len(s) for s in out_enc["strings"][0])
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        x_hat = F.pad(out_dec["x_hat"], unpad)
        rec = x_hat.mean(dim=1).clamp(0, 1).cpu().numpy()
        recon_slices[start:start + batch.shape[0]] = rec

    return np.moveaxis(recon_slices, 0, axis), total_bytes


# ---------------------------------------------------------------------------
# Optimizers — main params vs. entropy_bottleneck.quantiles (CompressAI convention)
# ---------------------------------------------------------------------------

def configure_optimizers(model, lr: float, aux_lr: float):
    params_dict = dict(model.named_parameters())
    main_names = sorted(n for n, p in params_dict.items() if p.requires_grad and not n.endswith(".quantiles"))
    aux_names = sorted(n for n, p in params_dict.items() if p.requires_grad and n.endswith(".quantiles"))
    optimizer = torch.optim.Adam((params_dict[n] for n in main_names), lr=lr)
    aux_optimizer = torch.optim.Adam((params_dict[n] for n in aux_names), lr=aux_lr)
    return optimizer, aux_optimizer


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(model, train_cache, val_cache, args, device):
    optimizer, aux_optimizer = configure_optimizers(model, args.lr, args.aux_lr)
    rng = np.random.default_rng(0)

    history = {"iter": [], "loss": [], "bpp": [], "mse": [], "psnr": []}
    val_history = {"iter": [], "loss": [], "bpp": [], "psnr": []}

    model.train()
    for it in range(1, args.iterations + 1):
        x = sample_batch_rgb(train_cache, args.patch_size, args.batch_size, rng, device)

        out = model(x)
        num_pixels = x.size(0) * x.size(2) * x.size(3)
        bpp = sum(-torch.log2(lk).sum() for lk in out["likelihoods"].values()) / num_pixels
        mse = F.mse_loss(out["x_hat"], x)
        loss = args.lambda_ * 255 ** 2 * mse + bpp

        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if it % 50 == 0 or it == 1:
            psnr = float(20 * np.log10(1.0 / (np.sqrt(mse.item()) + 1e-12)))
            history["iter"].append(it)
            history["loss"].append(loss.item())
            history["bpp"].append(bpp.item())
            history["mse"].append(mse.item())
            history["psnr"].append(psnr)
            print(f"iter {it:>6}/{args.iterations}  loss={loss.item():.4f}  "
                  f"bpp={bpp.item():.4f}  mse={mse.item():.6f}  psnr={psnr:.2f}dB  "
                  f"aux_loss={aux_loss.item():.2f}")

        if it % 500 == 0:
            model.eval()
            with torch.no_grad():
                xv = sample_batch_rgb(val_cache, args.patch_size, args.batch_size, rng, device)
                out_v = model(xv)
                v_pixels = xv.size(0) * xv.size(2) * xv.size(3)
                v_bpp = sum(-torch.log2(lk).sum() for lk in out_v["likelihoods"].values()) / v_pixels
                v_mse = F.mse_loss(out_v["x_hat"], xv)
                v_loss = args.lambda_ * 255 ** 2 * v_mse + v_bpp
                v_psnr = float(20 * np.log10(1.0 / (np.sqrt(v_mse.item()) + 1e-12)))
            val_history["iter"].append(it)
            val_history["loss"].append(v_loss.item())
            val_history["bpp"].append(v_bpp.item())
            val_history["psnr"].append(v_psnr)
            print(f"  [val] iter {it:>6}  loss={v_loss.item():.4f}  "
                  f"bpp={v_bpp.item():.4f}  psnr={v_psnr:.2f}dB")
            model.train()

    return history, val_history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune pretrained bmshj2018-factorized on PDEBench data")
    parser.add_argument("config", help="path to config_compresai.yaml (or another config with a finetune: section)")
    parser.add_argument("--quality", type=int, default=None, choices=range(1, 9),
                        help="pretrained checkpoint (1-8) to initialize from (default: config's finetune.quality)")
    parser.add_argument("--metric", choices=["mse", "ms-ssim"], default=None,
                        help="which pretrained checkpoint family (default: config's finetune.metric)")
    parser.add_argument("--lambda_", type=float, default=None,
                        help="rate-distortion weight (default: config's finetune.lambda_, falling back to "
                             "CompressAI's standard mse table by quality; required for ms-ssim)")
    parser.add_argument("--patch-size", type=int, default=None,
                        help="training crop size, must be a multiple of 16 (default: config's finetune.patch_size)")
    parser.add_argument("--batch-size", type=int, default=None, help="default: config's finetune.batch_size")
    parser.add_argument("--iterations", type=int, default=None, help="default: config's finetune.iterations")
    parser.add_argument("--lr", type=float, default=None, help="main optimizer lr (default: config's finetune.lr)")
    parser.add_argument("--aux-lr", type=float, default=None,
                        help="entropy_bottleneck.quantiles optimizer lr (default: config's finetune.aux_lr)")
    parser.add_argument("--axis", type=int, default=None, choices=[0, 1, 2],
                        help="volume axis sliced into 2D planes (default: config's finetune.axis)")
    parser.add_argument("--train-timesteps", default=None, help="default: config's finetune.train_timesteps")
    parser.add_argument("--val-timestep", type=int, default=None, help="default: config's finetune.val_timestep")
    parser.add_argument("--slices-per-timestep", type=int, default=None,
                        help="default: config's finetune.slices_per_timestep")
    parser.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    parser.add_argument("--output-dir", default=None, help="default: experiments/finetune_TIMESTAMP")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg["data"]
    fcfg = cfg.get("finetune", {})

    if args.quality is None:
        args.quality = fcfg.get("quality", 4)
    if args.metric is None:
        args.metric = fcfg.get("metric", "mse")
    if args.lambda_ is None:
        args.lambda_ = fcfg.get("lambda_")
    if args.lambda_ is None:
        if args.metric != "mse":
            raise ValueError("No default lambda_ table for ms-ssim — pass --lambda_ explicitly "
                              "(or set finetune.lambda_ in the config).")
        args.lambda_ = STANDARD_MSE_LAMBDAS[args.quality]
    if args.patch_size is None:
        args.patch_size = fcfg.get("patch_size", 128)
    if args.batch_size is None:
        args.batch_size = fcfg.get("batch_size", 8)
    if args.iterations is None:
        args.iterations = fcfg.get("iterations", 2000)
    if args.lr is None:
        args.lr = fcfg.get("lr", 1e-4)
    if args.aux_lr is None:
        args.aux_lr = fcfg.get("aux_lr", 1e-3)
    if args.axis is None:
        args.axis = fcfg.get("axis", 0)
    if args.train_timesteps is None:
        args.train_timesteps = fcfg.get("train_timesteps", "0-34")
    if args.val_timestep is None:
        args.val_timestep = fcfg.get("val_timestep", 40)
    if args.slices_per_timestep is None:
        args.slices_per_timestep = fcfg.get("slices_per_timestep", 16)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.patch_size % 16 != 0:
        raise ValueError(f"patch_size={args.patch_size} must be a multiple of 16 "
                          f"(the model's 4 stride-2 downsampling stages).")

    out_dir = args.output_dir or os.path.join(
        "experiments", "finetune_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, os.path.basename(args.config)))

    log_path = os.path.join(out_dir, "run.log")
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
    print(f"Init from  : bmshj2018-factorized quality={args.quality} metric={args.metric} (pretrained)")
    print(f"Lambda     : {args.lambda_:g}")
    print(f"Patch size : {args.patch_size}")
    print(f"Batch size : {args.batch_size}")
    print(f"Iterations : {args.iterations}")
    print(f"LR / aux LR: {args.lr:g} / {args.aux_lr:g}")
    print(f"Axis       : {args.axis}\n")

    train_timesteps = parse_timestep_range(args.train_timesteps)
    print(f"Train timesteps : {train_timesteps[0]}-{train_timesteps[-1]} ({len(train_timesteps)} steps)")
    print(f"Val timestep    : {args.val_timestep} (held out of training)\n")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    field = dcfg["field_key"]
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

    full_vol = load_full_volume(dcfg["h5_path"], field, args.val_timestep)
    full_vol01 = (full_vol - vmin) / (vmax - vmin + 1e-8)
    n_voxels = full_vol.size

    # ------------------------------------------------------------------ #
    # Baseline — pretrained model, no fine-tuning, on the held-out volume
    # ------------------------------------------------------------------ #
    print("Evaluating pretrained baseline (no fine-tuning) on held-out volume ...")
    baseline_model = bmshj2018_factorized(quality=args.quality, metric=args.metric, pretrained=True)
    baseline_model = baseline_model.to(device).eval()
    baseline_model.update(force=True)
    base_recon01, base_bytes = compress_volume(full_vol01, baseline_model, device, args.axis, args.batch_size)
    base_recon = base_recon01 * (vmax - vmin) + vmin
    base_metrics = compute_metrics(full_vol, base_recon, base_bytes, n_voxels)
    print(f"  baseline  rel_err={base_metrics['rel_err']:.6f}  PSNR={base_metrics['psnr']:.2f}dB  "
          f"comp={base_metrics['comp_ratio']:.2f}x  BPV={base_metrics['bpv']:.4f}\n")
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # Fine-tune
    # ------------------------------------------------------------------ #
    print(f"Loading pretrained bmshj2018-factorized quality={args.quality} metric={args.metric} ...")
    model = bmshj2018_factorized(quality=args.quality, metric=args.metric, pretrained=True)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    print("Fine-tuning ...")
    history, val_history = train(model, train_cache, val_cache, args, device)
    print("\nFine-tuning done.\n")

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

    fig.suptitle(f"bmshj2018-factorized fine-tune  quality={args.quality} lambda={args.lambda_:g}", fontsize=10)
    plt.tight_layout()
    out = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Real entropy-coded eval on the held-out volume
    # ------------------------------------------------------------------ #
    model.update(force=True)
    model.eval()
    print(f"\nEvaluating fine-tuned model on full held-out volume (t={args.val_timestep}) ...")
    ft_recon01, ft_bytes = compress_volume(full_vol01, model, device, args.axis, args.batch_size)
    ft_recon = ft_recon01 * (vmax - vmin) + vmin
    ft_metrics = compute_metrics(full_vol, ft_recon, ft_bytes, n_voxels)

    print(f"  pretrained  rel_err={base_metrics['rel_err']:.6f}  PSNR={base_metrics['psnr']:.2f}dB  "
          f"comp={base_metrics['comp_ratio']:.2f}x  BPV={base_metrics['bpv']:.4f}")
    print(f"  fine-tuned  rel_err={ft_metrics['rel_err']:.6f}  PSNR={ft_metrics['psnr']:.2f}dB  "
          f"comp={ft_metrics['comp_ratio']:.2f}x  BPV={ft_metrics['bpv']:.4f}")

    # ------------------------------------------------------------------ #
    # Plot 2 — full-volume reconstruction: input vs pretrained vs fine-tuned
    # ------------------------------------------------------------------ #
    D, H, W = full_vol.shape
    mD, mH, mW = D // 2, H // 2, W // 2
    plane_defs = [
        ("XY (z=mid)", full_vol[:, :, mW], base_recon[:, :, mW], ft_recon[:, :, mW]),
        ("XZ (y=mid)", full_vol[:, mH, :], base_recon[:, mH, :], ft_recon[:, mH, :]),
        ("YZ (x=mid)", full_vol[mD, :, :], base_recon[mD, :, :], ft_recon[mD, :, :]),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    fig.suptitle(
        f"bmshj2018-factorized  quality={args.quality}  lambda={args.lambda_:g}  |  "
        f"pretrained rel_err={base_metrics['rel_err']:.4f} BPV={base_metrics['bpv']:.4f}  vs.  "
        f"fine-tuned rel_err={ft_metrics['rel_err']:.4f} BPV={ft_metrics['bpv']:.4f}",
        fontsize=10,
    )
    for col, (lbl, inp_p, base_p, ft_p) in enumerate(plane_defs):
        vmax_p = np.percentile(np.abs(inp_p), 99)
        for row, (data, row_lbl) in enumerate([(inp_p, "Input"), (base_p, "Pretrained"), (ft_p, "Fine-tuned")]):
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
    # Results CSV + checkpoint
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(out_dir, "finetune_results.csv")
    rows = [
        dict(label="pretrained", quality=args.quality, lambda_=args.lambda_, iterations=0, **base_metrics),
        dict(label="finetuned", quality=args.quality, lambda_=args.lambda_, iterations=args.iterations, **ft_metrics),
    ]
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults table saved to {csv_path}")

    ckpt_path = os.path.join(out_dir, "model_finetuned.pt")
    torch.save({"model_state": model.state_dict(), "vmin": vmin, "vmax": vmax,
                "quality": args.quality, "metric": args.metric, "args": vars(args)}, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    print(f"\n{'='*60}")
    print("bmshj2018-factorized FINE-TUNE SUMMARY")
    print(f"{'='*60}")
    print(f"Volume:      {D}×{H}×{W}")
    print(f"Pretrained:  rel_err={base_metrics['rel_err']:.4f}  comp={base_metrics['comp_ratio']:.2f}x  "
          f"BPV={base_metrics['bpv']:.4f}")
    print(f"Fine-tuned:  rel_err={ft_metrics['rel_err']:.4f}  comp={ft_metrics['comp_ratio']:.2f}x  "
          f"BPV={ft_metrics['bpv']:.4f}")
    print("\nDone.")
    _log.close()


if __name__ == "__main__":
    main()
