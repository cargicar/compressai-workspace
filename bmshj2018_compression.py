"""
bmshj2018-factorized (CompressAI zoo) compression baseline for 3D simulation data.

Applies a *pretrained* learned image codec — Balle et al. 2018, "Variational
Image Compression with a Scale Hyperprior" (factorized-prior variant) — to a
single HDF5 snapshot, reusing the same config (`config_simmldc.yaml`) and the
same metric set (rel_err, comp_ratio, BPV, PSNR) as `svd_compression.py`, so
results are directly comparable.

The network is a 2D, fixed 3-channel codec (`conv(3, N)` / `deconv(N, 3)`), so
it cannot ingest a 3D volume directly. We slice the volume into 2D planes
along one axis, replicate the single scalar channel three times ("pseudo-RGB"),
and run the real CompressAI entropy coder (`model.compress` /
`model.decompress`) to get an actual arithmetic-coded bitstream — not an
estimated bit-rate.

Compression model (per volume)
-------------------------------
  - Normalise : global min-max scaling of the whole snapshot to [0, 1]
                (matches the range the model was trained on)
  - Slicing   : volume (D,H,W) -> D slices of shape (H,W), taken along --axis
  - Encoder   : replicate each slice to 3 channels -> g_a -> entropy_bottleneck.compress
  - Decoder   : entropy_bottleneck.decompress -> g_s -> average the 3 output
                channels back to a scalar slice -> undo normalisation
  - Storage   : real coded bytes, summed over all slice bitstreams
                (pretrained model weights are shared/fixed, like a codec
                definition, and are excluded — same convention as JPEG/PNG)

Compression metrics reported (bits per voxel, comp_ratio, rel_err, PSNR) use
the exact same formulas as svd_compression.py so the two baselines line up.

Usage
-----
    python bmshj2018_compression.py config_simmldc.yaml
    python bmshj2018_compression.py config_simmldc.yaml --qualities 1 4 8
    python bmshj2018_compression.py config_simmldc.yaml --metric ms-ssim
    python bmshj2018_compression.py config_simmldc.yaml --axis 2 --batch-size 32
    python bmshj2018_compression.py config_simmldc.yaml --output-dir results/bmshj_run1
"""

import argparse
import csv
import os
import shutil
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from compressai.ops import compute_padding
from compressai.zoo import bmshj2018_factorized


# ---------------------------------------------------------------------------
# Codec utilities
# ---------------------------------------------------------------------------

@torch.no_grad()
def compress_volume(vol01: np.ndarray, model, device: str, axis: int, batch_size: int):
    """
    Compress every 2D slice of a normalised (0..1) volume with a pretrained
    bmshj2018-factorized model and reconstruct the full volume.

    Parameters
    ----------
    vol01 : (D, H, W) float32, values in [0, 1]
    axis  : which axis to slice along (0, 1, or 2)

    Returns
    -------
    recon01      : (D, H, W) float32 reconstruction, values in [0, 1]
    total_bytes  : int, real entropy-coded bitstream size (all slices)
    """
    slices = np.moveaxis(vol01, axis, 0)          # (N, H, W)
    N, H, W = slices.shape
    pad, unpad = compute_padding(H, W, min_div=64)

    recon_slices = np.empty_like(slices)
    total_bytes = 0

    for start in range(0, N, batch_size):
        batch = slices[start:start + batch_size]                    # (b, H, W)
        x = torch.from_numpy(batch).unsqueeze(1).to(device)          # (b, 1, H, W)
        x = x.repeat(1, 3, 1, 1)                                     # pseudo-RGB
        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        out_enc = model.compress(x)
        total_bytes += sum(len(s) for s in out_enc["strings"][0])
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        x_hat = torch.nn.functional.pad(out_dec["x_hat"], unpad)     # crop back
        rec = x_hat.mean(dim=1).clamp(0, 1).cpu().numpy()            # (b, H, W)
        recon_slices[start:start + batch.shape[0]] = rec

    recon01 = np.moveaxis(recon_slices, 0, axis)
    return recon01, total_bytes


def compute_metrics(input_vol: np.ndarray, recon_vol: np.ndarray, quality: int,
                     n_bytes: int, n_voxels: int) -> dict:
    """Same formulas as svd_compression.py's compute_metrics, for direct comparison."""
    error     = input_vol - recon_vol
    mse       = float((error**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = float(np.sqrt(mse) / (np.sqrt((input_vol**2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    bytes_in   = n_voxels * 4                 # float32 baseline, same convention as SVD script
    comp_ratio = bytes_in / n_bytes if n_bytes > 0 else float('inf')
    bpv        = (n_bytes * 8) / n_voxels

    return dict(
        quality=quality,
        rel_err=rel_err,
        rmse=rmse,
        psnr=psnr,
        comp_ratio=comp_ratio,
        bpv=bpv,
        n_bytes=n_bytes,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='bmshj2018-factorized compression baseline')
    parser.add_argument('config', help='path to config_simmldc.yaml')
    parser.add_argument('--qualities', type=int, nargs='+', default=list(range(1, 9)),
                        help='CompressAI quality levels to sweep, 1 (lowest) - 8 (highest)')
    parser.add_argument('--metric', choices=['mse', 'ms-ssim'], default='mse',
                        help='pretrained model optimised for this metric')
    parser.add_argument('--axis', type=int, default=0, choices=[0, 1, 2],
                        help='volume axis to slice into 2D planes fed to the codec')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='number of slices per forward pass')
    parser.add_argument('--device', default=None,
                        help='cuda / cpu (default: cuda if available)')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: experiments/bmshj2018_TIMESTAMP)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    dcfg = cfg['data']

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Output directory
    out_dir = args.output_dir or os.path.join(
        'experiments', 'bmshj2018_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(out_dir, 'config_simmldc.yaml'))

    log_path = os.path.join(out_dir, 'run.log')

    # Mirror print() output to run.log without touching sys.stdout/sys.stderr: CPython's
    # input() only uses real GNU readline (arrow-key history) when sys.stdout/sys.stdin are
    # still the exact original objects, so a sys.stdout swap would break pdb for the rest
    # of the process even after being restored to look identical.
    import builtins
    _log = open(log_path, 'w')
    _builtin_print = builtins.print

    def print(*args, **kwargs):
        _builtin_print(*args, **kwargs)
        kwargs.pop('file', None)
        _builtin_print(*args, file=_log, **kwargs)
        _log.flush()

    print(f"Output dir : {out_dir}")
    print(f"Config     : {args.config}")
    print(f"Device     : {device}")
    print(f"Metric     : {args.metric}")
    print(f"Axis       : {args.axis}")
    print(f"Qualities  : {args.qualities}\n")

    # ------------------------------------------------------------------ #
    # Load volume (same convention as svd_compression.py)
    # ------------------------------------------------------------------ #
    print(f"Loading {dcfg['h5_path']} — field={dcfg['field_key']} t={dcfg['timestep']} ...")
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)

    D, H, W = vol.shape
    n_voxels = D * H * W
    print(f"Volume     : {D}×{H}×{W}\n")

    vmin, vmax = float(vol.min()), float(vol.max())
    vol01 = (vol - vmin) / (vmax - vmin + 1e-8)   # global [0,1] scaling for the pretrained codec

    # ------------------------------------------------------------------ #
    # Sweep quality levels
    # ------------------------------------------------------------------ #
    results = {}
    recon_by_quality = {}
    print(f"{'q':>3}  {'rel_err':>10}  {'PSNR(dB)':>10}  {'Comp':>10}  {'BPV':>10}  {'Bytes':>12}")
    print('-' * 62)

    for q in args.qualities:
        model = bmshj2018_factorized(quality=q, metric=args.metric, pretrained=True)
        model = model.to(device).eval()
        model.update(force=True)

        recon01, n_bytes = compress_volume(vol01, model, device, args.axis, args.batch_size)
        recon_vol = recon01 * (vmax - vmin) + vmin

        m = compute_metrics(vol, recon_vol, q, n_bytes, n_voxels)
        results[q] = m
        recon_by_quality[q] = recon_vol
        print(f"{q:>3}  {m['rel_err']:>10.6f}  {m['psnr']:>10.2f}  "
              f"{m['comp_ratio']:>9.2f}x  {m['bpv']:>10.4f}  {m['n_bytes']:>12,}")

        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    print()

    # Best quality at rel_err <= 1%
    under_1pct = [r for r in results.values() if r['rel_err'] <= 0.01]
    if under_1pct:
        best = max(under_1pct, key=lambda r: r['comp_ratio'])
        print(f"Best (rel_err ≤ 1%): quality={best['quality']}  rel_err={best['rel_err']:.4f}  "
              f"comp={best['comp_ratio']:.2f}x  BPV={best['bpv']:.4f}")
    else:
        best = min(results.values(), key=lambda r: r['rel_err'])
        print(f"Note: rel_err never reaches 1% — "
              f"best is quality={best['quality']}  rel_err={best['rel_err']:.4f}  "
              f"comp={best['comp_ratio']:.2f}x  BPV={best['bpv']:.4f}")

    best_q = best['quality']
    best_recon = recon_by_quality[best_q]

    # ------------------------------------------------------------------ #
    # Plot 1 — rate-distortion curve (rel_err vs BPV)
    # ------------------------------------------------------------------ #
    qs       = sorted(results.keys())
    rel_errs = [results[q]['rel_err'] for q in qs]
    bpvs     = [results[q]['bpv']     for q in qs]
    comps    = [results[q]['comp_ratio'] for q in qs]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.semilogy(bpvs, rel_errs, 'o-', color='steelblue', markersize=5, label='rel_err')
    ax1.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')
    for q, x, y in zip(qs, bpvs, rel_errs):
        ax1.annotate(f"q={q}", (x, y), textcoords='offset points', xytext=(4, 4), fontsize=7)
    ax2.plot(bpvs, comps, 's--', color='darkorange', markersize=4, alpha=0.7, label='comp_ratio')

    ax1.set_xlabel('Bits per voxel (BPV)')
    ax1.set_ylabel('Relative reconstruction error (log)', color='steelblue')
    ax2.set_ylabel('Compression ratio', color='darkorange')
    ax1.set_title(f'bmshj2018-factorized ({args.metric})  Rate–Distortion  |  {D}×{H}×{W}')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, 'rate_distortion.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 2 — Full-volume reconstruction at best quality
    # ------------------------------------------------------------------ #
    mD, mH, mW = D // 2, H // 2, W // 2
    plane_defs = [
        ('XY (z=mid)', vol[:, :, mW], best_recon[:, :, mW]),
        ('XZ (y=mid)', vol[:, mH, :], best_recon[:, mH, :]),
        ('YZ (x=mid)', vol[mD, :, :], best_recon[mD, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'bmshj2018-factorized ({args.metric})  Full-volume reconstruction  |  quality={best_q}  '
        f'rel_err={best["rel_err"]:.4f}  comp_ratio={best["comp_ratio"]:.1f}x  BPV={best["bpv"]:.4f}',
        fontsize=10
    )
    for col, (lbl, inp_p, rec_p) in enumerate(plane_defs):
        vmax_p = np.percentile(np.abs(inp_p), 99)
        for row, (data, row_lbl) in enumerate([(inp_p, 'Input'), (rec_p, 'Reconstruction')]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax_p, vmax=vmax_p,
                           origin='lower', aspect='equal')
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if col == 0:
                ax.set_ylabel(row_lbl, fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    out = os.path.join(plots_dir, f'full_volume_q{best_q}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Save results table
    # ------------------------------------------------------------------ #
    csv_path = os.path.join(out_dir, 'bmshj2018_results.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=list(results[qs[0]].keys()))
        writer.writeheader()
        for q in qs:
            writer.writerow(results[q])
    print(f"\nResults table saved to {csv_path}")

    print(f"\n{'='*60}")
    print(f"bmshj2018-factorized SUMMARY  (t={dcfg['timestep']}, metric={args.metric})")
    print(f"{'='*60}")
    print(f"Volume:                     {D}×{H}×{W}")
    if under_1pct:
        print(f"Best (rel_err ≤ 1%):        quality={best_q}  "
              f"comp={best['comp_ratio']:.2f}x  BPV={best['bpv']:.4f}")
    print("\nDone.")
    _log.close()


if __name__ == '__main__':
    main()
