"""
SVD compression baseline for 3D simulation data.

Computes a truncated SVD on non-overlapping 3D patches extracted from a single
HDF5 snapshot.  Sweeps over a range of k (number of SVD components) and
reports rel_err, comp_ratio, BPV, and PSNR for each.

No gradient descent — SVD is computed analytically from the data in one pass.

Compression model (per patch)
------------------------------
  - Encoder  : project normalised patch onto top-k right singular vectors
               coefficients = x_norm @ Vt.T   shape: (k,)
  - Decoder  : reconstruct from coefficients + undo normalisation
               x_norm_hat = coefficients @ Vt  → x_hat = x_norm_hat * std + mean
  - Storage  : k × float32 = k × 4 bytes
  - Basis Vt : k × P³ × 4 bytes total, amortised over all patches in the volume

Compression metrics reported (first sweep, exact float32 coefficients):
  comp_coeff = (P³ × 4) / (k × 4) = P³ / k             ← coefficients only, 4 bytes each
  comp_total = (P³ × 4) / (k×4 + k×P³×4/n_patches)     ← amortised basis included

These never touch a real compressor — they're a byte count of "if we kept k raw
float32 numbers per patch." A second sweep makes the comparison to a real codec
(e.g. bmshj2018_compression.py's actual arithmetic-coded bytes) fair: for every
(k, bits) pair, coefficients are quantized to `bits`-bit integers (per-component
scale) and the whole volume's coefficient matrix is losslessly entropy-coded with
a real compressor (best of zlib/lzma, round-trip verified) — real_comp_ratio and
real_bpv come from that measured byte count, not an analytical formula. rel_err
here reflects both truncation (k) and quantization (bits) error.

Usage
-----
    python svd_compression.py config_svd.yaml
    python svd_compression.py config_svd.yaml --k-max 60
    python svd_compression.py config_svd.yaml --k-values 5 10 20 50
    python svd_compression.py config_svd.yaml --quant-bits 4 8 12 16
    python svd_compression.py config_svd.yaml --output-dir results/svd_run1
"""

import argparse
import lzma
import os
import zlib
from datetime import datetime

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

try:
    from sklearn.utils.extmath import randomized_svd
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


# ---------------------------------------------------------------------------
# Patch utilities
# ---------------------------------------------------------------------------

def extract_tiled_patches(vol: np.ndarray, patch_size: int):
    """
    Extract all non-overlapping patch_size³ tiles from a 3D volume.

    Tiles that do not fit are discarded (same behaviour as the LCA inference
    script).  Each patch is normalised to zero mean, unit variance before
    being returned — matching LCA's internal normalisation.

    Returns
    -------
    X        : (n_patches, P³) float32 — normalised, flattened patches
    means    : (n_patches,) float32 — per-patch mean (to undo normalisation)
    stds     : (n_patches,) float32 — per-patch std  (to undo normalisation)
    positions: list of (x,y,z) top-left corners
    grid     : (nD, nH, nW) tile counts per dimension
    """
    D, H, W = vol.shape
    P = patch_size
    nD, nH, nW = D // P, H // P, W // P
    positions = [(i*P, j*P, k*P)
                 for i in range(nD)
                 for j in range(nH)
                 for k in range(nW)]

    n = len(positions)
    X     = np.empty((n, P**3), dtype=np.float32)
    means = np.empty(n, dtype=np.float32)
    stds  = np.empty(n, dtype=np.float32)

    for idx, (x, y, z) in enumerate(positions):
        patch = vol[x:x+P, y:y+P, z:z+P].ravel().astype(np.float32)
        m = patch.mean()
        s = patch.std() + 1e-8
        means[idx] = m
        stds[idx]  = s
        X[idx]     = (patch - m) / s

    return X, means, stds, positions, (nD, nH, nW)


def reconstruct_volume(
    coeffs: np.ndarray,      # (n_patches, k) — projection of each patch onto Vt[:k]
    Vt_k: np.ndarray,        # (k, P³) — top-k right singular vectors
    means: np.ndarray,       # (n_patches,)
    stds: np.ndarray,        # (n_patches,)
    positions: list,
    vol_shape: tuple,        # (D_out, H_out, W_out) — tiled region only
    patch_size: int,
) -> np.ndarray:
    """Reconstruct the tiled region from SVD coefficients."""
    D, H, W = vol_shape
    P = patch_size
    recon_vol = np.zeros((D, H, W), dtype=np.float32)

    for idx, (x, y, z) in enumerate(positions):
        patch_norm = coeffs[idx] @ Vt_k          # (P³,)
        patch = patch_norm * stds[idx] + means[idx]
        recon_vol[x:x+P, y:y+P, z:z+P] = patch.reshape(P, P, P)

    return recon_vol


def compute_metrics(
    input_vol: np.ndarray,
    recon_vol: np.ndarray,
    k: int,
    patch_size: int,
    n_patches: int,
) -> dict:
    """Compute reconstruction quality and compression metrics."""
    error     = input_vol - recon_vol
    mse       = float((error**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = float(np.sqrt(mse) / (np.sqrt((input_vol**2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    P3 = patch_size**3
    # Coefficients only (basis amortised / excluded):
    bytes_coeff   = k * 4                         # k × float32
    bytes_in      = P3 * 4                        # P³ × float32
    comp_coeff    = P3 / k                        # = bytes_in / bytes_coeff
    bpv_coeff     = (bytes_coeff * 8) / P3        # bits per voxel, coeff only

    # Amortised basis cost per patch:
    bytes_basis_pp = (k * P3 * 4) / n_patches
    bytes_total    = bytes_coeff + bytes_basis_pp
    comp_total     = bytes_in / bytes_total if bytes_total > 0 else float('inf')
    bpv_total      = (bytes_total * 8) / P3

    return dict(
        k=k,
        rel_err=rel_err,
        rmse=rmse,
        psnr=psnr,
        comp_coeff=comp_coeff,
        bpv_coeff=bpv_coeff,
        comp_total=comp_total,
        bpv_total=bpv_total,
    )


# ---------------------------------------------------------------------------
# Real quantization + entropy coding (comparable to a real "bytes on disk" codec)
# ---------------------------------------------------------------------------

def quantize_coeffs(coeffs_k: np.ndarray, col_scale_k: np.ndarray, bits: int):
    """
    Per-column uniform scalar quantization of SVD coefficients.

    Each retained component (column) gets its own scale, since singular values
    (and hence coefficient magnitudes) decay sharply across components —
    matching standard practice (e.g. JPEG's per-frequency quantization tables).

    Returns
    -------
    q       : integer symbols, dtype sized to `bits` (int8/int16/int32)
    dequant : float32 reconstruction of `coeffs_k` after quantization (lossy)
    """
    levels = 2 ** bits
    qmax = levels // 2 - 1
    dtype = np.int8 if bits <= 8 else (np.int16 if bits <= 16 else np.int32)
    scale = np.where(col_scale_k > 0, col_scale_k, 1.0)
    q = np.round(coeffs_k / scale * qmax).clip(-qmax, qmax).astype(dtype)
    dequant = q.astype(np.float32) * scale / qmax
    return q, dequant


def real_compress_bytes(q: np.ndarray) -> int:
    """
    Real, lossless entropy coding of quantized integer symbols — the best of
    zlib and lzma — with a round-trip check, so this is a genuinely measured
    "bytes on disk" size, not an estimate.
    """
    raw = q.tobytes()
    zlib_bytes = zlib.compress(raw, level=9)
    lzma_bytes = lzma.compress(raw, preset=9)
    if len(zlib_bytes) <= len(lzma_bytes):
        assert zlib.decompress(zlib_bytes) == raw, "real entropy coding round-trip failed"
        return len(zlib_bytes)
    assert lzma.decompress(lzma_bytes) == raw, "real entropy coding round-trip failed"
    return len(lzma_bytes)


def compute_real_metrics(
    input_vol: np.ndarray,
    recon_vol: np.ndarray,
    k: int,
    bits: int,
    real_bytes: int,
    n_patches: int,
    patch_size: int,
) -> dict:
    """Same rel_err/PSNR formulas as compute_metrics, but comp_ratio/BPV come from a
    real measured compressed byte count for the whole volume's coefficient matrix,
    not an analytical byte-counting formula."""
    error     = input_vol - recon_vol
    mse       = float((error**2).mean())
    rmse      = float(np.sqrt(mse))
    rel_err   = float(np.sqrt(mse) / (np.sqrt((input_vol**2).mean()) + 1e-8))
    sig_range = float(input_vol.max() - input_vol.min())
    psnr      = float(20 * np.log10(sig_range / (rmse + 1e-12)))

    bytes_in        = n_patches * patch_size**3 * 4   # whole covered volume, float32 baseline
    real_comp_ratio = bytes_in / real_bytes if real_bytes > 0 else float('inf')
    real_bpv        = (real_bytes * 8) / (n_patches * patch_size**3)

    return dict(
        k=k,
        bits=bits,
        rel_err=rel_err,
        psnr=psnr,
        real_bytes=real_bytes,
        real_comp_ratio=real_comp_ratio,
        real_bpv=real_bpv,
    )


def pareto_frontier(points: list) -> list:
    """Given (rel_err, bpv, ...) tuples, return the rate-distortion frontier:
    for increasing rel_err, the entries where bpv reaches a new minimum."""
    frontier = []
    best_bpv = float('inf')
    for p in sorted(points, key=lambda p: p[0]):
        if p[1] < best_bpv:
            frontier.append(p)
            best_bpv = p[1]
    return frontier


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='SVD compression baseline')
    parser.add_argument('config', help='path to config_simmldc.yaml')
    parser.add_argument('--k-values', type=int, nargs='+', default=None,
                        help='specific k values to evaluate (default: auto)')
    parser.add_argument('--k-max', type=int, default=None,
                        help='maximum k to sweep (default: n_patches)')
    parser.add_argument('--quant-bits', type=int, nargs='+', default=[4, 6, 8, 10, 12, 16],
                        help='bit-depths to sweep for the real quantization + entropy-coding pass')
    parser.add_argument('--output-dir', default=None,
                        help='output directory (default: experiments/svd_TIMESTAMP)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dcfg = cfg['data']
    P    = dcfg['patch_size']

    _svd_backend = cfg.get('svd', {}).get('backend', 'sklearn')
    if _svd_backend == 'sklearn' and not _SKLEARN_AVAILABLE:
        print("Warning: sklearn not installed — falling back to numpy SVD")
        _svd_backend = 'numpy'
    _USE_SKLEARN = (_svd_backend == 'sklearn')

    # Output directory
    out_dir = args.output_dir or os.path.join(
        'experiments', 'svd_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    import shutil
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
    print(f"Patch size : {P}³ = {P**3:,} voxels")
    print(f"SVD backend: {'sklearn randomized_svd' if _USE_SKLEARN else 'numpy linalg.svd'}\n")

    # ------------------------------------------------------------------ #
    # Load volume
    # ------------------------------------------------------------------ #
    print(f"Loading {dcfg['h5_path']} — field={dcfg['field_key']} t={dcfg['timestep']} ...")
    with h5py.File(dcfg['h5_path'], 'r') as f:
        vol = f[dcfg['field_key']][dcfg['timestep']].astype(np.float32)

    vol = (vol - vol.mean()) / (vol.std() + 1e-8)   # global normalisation
    D, H, W = vol.shape
    print(f"Volume     : {D}×{H}×{W}\n")

    # ------------------------------------------------------------------ #
    # Extract non-overlapping patches
    # ------------------------------------------------------------------ #
    X, means, stds, positions, (nD, nH, nW) = extract_tiled_patches(vol, P)
    n_patches = len(positions)
    D_out, H_out, W_out = nD * P, nH * P, nW * P
    input_vol = vol[:D_out, :H_out, :W_out].copy()

    print(f"Tiles      : {nD}×{nH}×{nW} = {n_patches} patches "
          f"(covering {D_out}×{H_out}×{W_out} of {D}×{H}×{W})")
    print(f"Patch matrix X : {X.shape}  ({X.nbytes/1024/1024:.1f} MB)\n")

    # ------------------------------------------------------------------ #
    # Compute SVD
    # ------------------------------------------------------------------ #
    k_max = min(args.k_max or n_patches, n_patches, P**3)
    print(f"Computing truncated SVD (k_max={k_max}) ...")

    if _USE_SKLEARN:
        U, s, Vt = randomized_svd(X, n_components=k_max, random_state=42)
    else:
        # Full SVD then truncate — works since n_patches << P³
        U_full, s_full, Vt_full = np.linalg.svd(X, full_matrices=False)
        U, s, Vt = U_full[:, :k_max], s_full[:k_max], Vt_full[:k_max]

    # Pre-compute all projections (n_patches, k_max) — cheap since k_max is small
    # coefficients[i, :k] = U[i, :k] * s[:k]  for any k
    coeffs_full = U * s[np.newaxis, :]   # (n_patches, k_max)

    print(f"SVD done.  Singular values: max={s[0]:.3f}  min={s[-1]:.4f}\n")

    # ------------------------------------------------------------------ #
    # Sweep k values
    # ------------------------------------------------------------------ #
    if args.k_values:
        k_values = sorted(v for v in args.k_values if 1 <= v <= k_max)
    else:
        # Auto: logarithmically spaced + exact k_max
        k_log = np.unique(np.round(np.geomspace(1, k_max, 30)).astype(int))
        k_values = sorted(set(k_log.tolist() + [k_max]))

    results = []
    print(f"{'k':>6}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
          f"{'Comp(coeff)':>13}  {'BPV(coeff)':>12}  "
          f"{'Comp(+basis)':>14}  {'BPV(+basis)':>13}")
    print('-' * 90)

    for k in k_values:
        coeffs_k = coeffs_full[:, :k]           # (n_patches, k)
        Vt_k     = Vt[:k]                       # (k, P³)
        recon_vol = reconstruct_volume(
            coeffs_k, Vt_k, means, stds, positions,
            (D_out, H_out, W_out), P
        )
        m = compute_metrics(input_vol, recon_vol, k, P, n_patches)
        results.append(m)
        print(f"{k:>6}  {m['rel_err']:>10.6f}  {m['psnr']:>10.2f}  "
              f"{m['comp_coeff']:>13.2f}x  {m['bpv_coeff']:>12.3f}  "
              f"{m['comp_total']:>14.2f}x  {m['bpv_total']:>13.3f}")

    print()

    # Best k at rel_err <= 1%
    under_1pct = [r for r in results if r['rel_err'] <= 0.01]
    if under_1pct:
        best = max(under_1pct, key=lambda r: r['comp_coeff'])
        print(f"Best (rel_err ≤ 1%): k={best['k']}  rel_err={best['rel_err']:.4f}  "
              f"comp(coeff)={best['comp_coeff']:.2f}x  BPV(coeff)={best['bpv_coeff']:.3f}")
    else:
        best = min(results, key=lambda r: r['rel_err'])
        print(f"Note: rel_err never reaches 1% — "
              f"best is k={best['k']}  rel_err={best['rel_err']:.4f}  "
              f"comp(coeff)={best['comp_coeff']:.2f}x  BPV(coeff)={best['bpv_coeff']:.3f}")

    # ------------------------------------------------------------------ #
    # Real quantization + entropy-coding sweep (k × bits)
    #
    # The sweep above counts coefficients as raw, unquantized float32 — never
    # actually compressed. This sweep quantizes each retained component
    # (per-column scale) to `bits`-bit integers and entropy-codes the whole
    # volume's coefficient matrix with a real, round-trip-verified compressor
    # (zlib/lzma), giving a genuinely measured "bytes on disk" size —
    # comparable to bmshj2018_compression.py's real coded-byte count.
    # ------------------------------------------------------------------ #
    col_scale = np.abs(coeffs_full).max(axis=0)   # (k_max,) per-component scale

    n_combos = len(k_values) * len(args.quant_bits)
    print(f"\nReal quantization + entropy coding sweep "
          f"(k × bits, {len(k_values)}×{len(args.quant_bits)} = {n_combos} combos) ...")
    print(f"{'k':>6}  {'bits':>5}  {'rel_err':>10}  {'PSNR(dB)':>10}  "
          f"{'RealComp':>10}  {'RealBPV':>9}  {'RealBytes':>12}")
    print('-' * 72)

    real_results = []
    for k in k_values:
        col_scale_k = col_scale[:k]
        Vt_k = Vt[:k]
        for bits in args.quant_bits:
            q, dequant = quantize_coeffs(coeffs_full[:, :k], col_scale_k, bits)
            real_bytes = real_compress_bytes(q)
            recon_vol_q = reconstruct_volume(
                dequant, Vt_k, means, stds, positions,
                (D_out, H_out, W_out), P
            )
            rm = compute_real_metrics(input_vol, recon_vol_q, k, bits, real_bytes, n_patches, P)
            real_results.append(rm)
            print(f"{k:>6}  {bits:>5}  {rm['rel_err']:>10.6f}  {rm['psnr']:>10.2f}  "
                  f"{rm['real_comp_ratio']:>9.2f}x  {rm['real_bpv']:>9.4f}  {rm['real_bytes']:>12,}")

    print()

    frontier = pareto_frontier([(r['rel_err'], r['real_bpv'], r) for r in real_results])
    frontier_rel_errs = [p[0] for p in frontier]
    frontier_bpvs     = [p[1] for p in frontier]
    frontier_best     = min(frontier, key=lambda p: p[0])[2]  # frontier point closest to 0 rel_err

    real_under_1pct = [p for p in frontier if p[0] <= 0.01]
    if real_under_1pct:
        real_best = min(real_under_1pct, key=lambda p: p[1])[2]  # lowest bpv among those under 1%
        print(f"Best real (rel_err ≤ 1%): k={real_best['k']}  bits={real_best['bits']}  "
              f"rel_err={real_best['rel_err']:.4f}  "
              f"comp={real_best['real_comp_ratio']:.2f}x  BPV={real_best['real_bpv']:.4f}")
    else:
        print(f"Note: real rel_err never reaches 1% — "
              f"best is k={frontier_best['k']}  bits={frontier_best['bits']}  "
              f"rel_err={frontier_best['rel_err']:.4f}  "
              f"comp={frontier_best['real_comp_ratio']:.2f}x  BPV={frontier_best['real_bpv']:.4f}")

    # ------------------------------------------------------------------ #
    # Plot 1 — Singular value spectrum
    # ------------------------------------------------------------------ #
    cumvar = np.cumsum(s**2) / np.sum(s**2) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.semilogy(np.arange(1, len(s)+1), s, color='steelblue')
    ax1.set_xlabel('Component index')
    ax1.set_ylabel('Singular value (log scale)')
    ax1.set_title(f'Singular value spectrum  (patch {P}³, {n_patches} patches)')
    ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(1, len(s)+1), cumvar, color='darkorange')
    ax2.axhline(99, color='red', linestyle='--', linewidth=0.8, label='99%')
    ax2.axhline(95, color='gray', linestyle=':', linewidth=0.8, label='95%')
    ax2.set_xlabel('k (number of components)')
    ax2.set_ylabel('Cumulative explained variance (%)')
    ax2.set_title('Cumulative explained variance')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # Mark where 99% and 95% variance is reached
    for pct, color in [(99, 'red'), (95, 'gray')]:
        idx = np.searchsorted(cumvar, pct)
        if idx < len(s):
            ax2.axvline(idx + 1, color=color, linestyle='--', linewidth=0.8)
            ax2.text(idx + 1.5, pct - 3, f'k={idx+1}', fontsize=8, color=color)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'singular_values.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 2 — rel_err vs k
    # ------------------------------------------------------------------ #
    ks       = [r['k']        for r in results]
    rel_errs = [r['rel_err']  for r in results]
    comps    = [r['comp_coeff'] for r in results]
    bpvs     = [r['bpv_coeff']  for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.semilogy(ks, rel_errs, 'o-', color='steelblue', markersize=4, label='rel_err')
    ax2.plot(ks, comps, 's--', color='darkorange', markersize=4, label='comp_ratio (coeff)')

    ax1.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% target')

    ax1.set_xlabel('k (SVD components)')
    ax1.set_ylabel('Relative reconstruction error (log)', color='steelblue')
    ax2.set_ylabel('Compression ratio (coeff only)', color='darkorange')
    ax1.set_title(f'SVD: reconstruction quality vs k  (patch {P}³, {n_patches} patches)')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'rel_err_vs_k.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 3 — rel_err vs BPV (rate-distortion curve)
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(bpvs, rel_errs, 'o-', color='steelblue', markersize=5,
                label='SVD (coeff only, 4 B/coeff)')
    ax.semilogy([r['bpv_total'] for r in results], rel_errs, 's--',
                color='teal', markersize=4, alpha=0.7, label='SVD (+ amortised basis)')

    ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% error target')

    # Annotate some k values
    for r in results[::max(1, len(results)//8)]:
        ax.annotate(f"k={r['k']}", (r['bpv_coeff'], r['rel_err']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7)

    ax.set_xlabel('Bits per voxel (BPV)')
    ax.set_ylabel('Relative reconstruction error (log scale)')
    ax.set_title(f'SVD Rate–Distortion  |  patch {P}³  |  {n_patches} patches')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, 'rate_distortion.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 3b — Real quantization + entropy-coding rate-distortion
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(bpvs, rel_errs, 'o-', color='steelblue', markersize=4, alpha=0.5,
                label='SVD (raw float32 coeffs, no quantization)')
    ax.scatter([r['real_bpv'] for r in real_results], [r['rel_err'] for r in real_results],
               s=12, color='lightgray', alpha=0.6, zorder=2,
               label='SVD (quantized + real-compressed, all k×bits)')
    ax.semilogy(frontier_bpvs, frontier_rel_errs, 's-', color='darkorange', markersize=5,
                zorder=3, label='SVD (quantized + real-compressed, Pareto frontier)')

    ax.axhline(0.01, color='red', linestyle=':', linewidth=1, label='1% error target')

    ax.set_xlabel('Bits per voxel (BPV)')
    ax.set_ylabel('Relative reconstruction error (log scale)')
    ax.set_title(f'SVD Real Rate–Distortion  |  patch {P}³  |  {n_patches} patches')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(plots_dir, 'real_rate_distortion.png')
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Plot 4 — Full-volume reconstruction at best k
    # ------------------------------------------------------------------ #
    best_k = best['k']
    coeffs_best = coeffs_full[:, :best_k]
    Vt_best     = Vt[:best_k]
    recon_vol   = reconstruct_volume(
        coeffs_best, Vt_best, means, stds, positions,
        (D_out, H_out, W_out), P
    )

    mD, mH, mW = D_out // 2, H_out // 2, W_out // 2
    plane_defs = [
        ('XY (z=mid)', input_vol[:, :, mW], recon_vol[:, :, mW]),
        ('XZ (y=mid)', input_vol[:, mH, :], recon_vol[:, mH, :]),
        ('YZ (x=mid)', input_vol[mD, :, :], recon_vol[mD, :, :]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f'SVD Full-volume reconstruction  |  k={best_k}  '
        f'rel_err={best["rel_err"]:.4f}  '
        f'comp_ratio={best["comp_coeff"]:.1f}x  BPV={best["bpv_coeff"]:.3f}',
        fontsize=10
    )
    for col, (lbl, inp_p, rec_p) in enumerate(plane_defs):
        vmax = np.percentile(np.abs(inp_p), 99)
        for row, (data, row_lbl) in enumerate([(inp_p, 'Input'), (rec_p, 'Reconstruction')]):
            ax = axes[row, col]
            im = ax.imshow(data, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           origin='lower', aspect='equal')
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if col == 0:
                ax.set_ylabel(row_lbl, fontsize=9)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            plt.colorbar(im, ax=ax, shrink=0.85)

    plt.tight_layout()
    out = os.path.join(plots_dir, f'full_volume_k{best_k}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {out}")

    # ------------------------------------------------------------------ #
    # Save results table
    # ------------------------------------------------------------------ #
    import csv
    csv_path = os.path.join(out_dir, 'svd_results.csv')
    with open(csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults table saved to {csv_path}")

    real_csv_path = os.path.join(out_dir, 'svd_real_results.csv')
    with open(real_csv_path, 'w', newline='') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=real_results[0].keys())
        writer.writeheader()
        writer.writerows(real_results)
    print(f"Real-compression results table saved to {real_csv_path}")

    print(f"\n{'='*60}")
    print(f"SVD SUMMARY  (patch {P}³, {n_patches} tiles, t={dcfg['timestep']})")
    print(f"{'='*60}")
    print(f"Max components (n_patches): {n_patches}")
    print(f"Patch volume:               {P**3:,} voxels")
    print(f"Covered volume:             {D_out}×{H_out}×{W_out}")
    k99 = int(np.searchsorted(cumvar, 99)) + 1
    k95 = int(np.searchsorted(cumvar, 95)) + 1
    print(f"k for 95% variance:         {k95}  →  comp={P**3/k95:.1f}x  BPV={k95*32/P**3:.3f}")
    print(f"k for 99% variance:         {k99}  →  comp={P**3/k99:.1f}x  BPV={k99*32/P**3:.3f}")
    if under_1pct:
        print(f"Best (rel_err ≤ 1%):        k={best['k']}  "
              f"comp(coeff)={best['comp_coeff']:.2f}x  BPV={best['bpv_coeff']:.3f}")
    if real_under_1pct:
        print(f"Best real (rel_err ≤ 1%):   k={real_best['k']}  bits={real_best['bits']}  "
              f"comp={real_best['real_comp_ratio']:.2f}x  BPV={real_best['real_bpv']:.4f}")

    print("\nDone.")
    _log.close()


if __name__ == '__main__':
    main()
