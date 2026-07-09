"""
Compare two compression baselines (SVD and bmshj2018-factorized) on the
same rel_err / BPV metrics.

Both `svd_compression.py` and `bmshj2018_compression.py` already sweep a
rate-controlling parameter (k / quality) and save a results CSV with
comparable metrics (rel_err, PSNR, comp_ratio, BPV). This script just loads
both CSVs, overlays their rate-distortion and compression-ratio curves, and
prints the BPV and comp_ratio each method needs/achieves at a set of target
relative errors so the two are easy to read off side by side.

Usage
-----
    python compare_compression.py experiments/svd_2026-07-08_10-00-00 experiments/bmshj2018_2026-07-08_10-05-00
    python compare_compression.py path/to/svd_results.csv path/to/bmshj2018_results.csv
    python compare_compression.py <svd_run_dir> <bmshj_run_dir> --targets 0.01 0.02 0.05 --output compare.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def resolve_csv(path: str, expected_name: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, expected_name)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"Expected {candidate} inside {path}")
        return candidate
    return path


def load_csv(path: str) -> list:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in row.items()} for row in rows]


def value_at_target(rel_errs: np.ndarray, values: np.ndarray, target: float):
    """Interpolate `values` (BPV or comp_ratio) at `target` rel_err (None if out of range)."""
    order = np.argsort(rel_errs)
    re, val = rel_errs[order], values[order]
    if target < re.min() or target > re.max():
        return None
    return float(np.interp(target, re, val))


def main():
    parser = argparse.ArgumentParser(description='Compare SVD vs bmshj2018-factorized results')
    parser.add_argument('svd', help='svd_results.csv or its run directory')
    parser.add_argument('bmshj', help='bmshj2018_results.csv or its run directory')
    parser.add_argument('--targets', type=float, nargs='+',
                        default=[0.01, 0.02, 0.05, 0.1],
                        help='rel_err targets to compare BPV at')
    parser.add_argument('--output', default='compare_rate_distortion.png',
                        help='output plot path')
    args = parser.parse_args()

    svd_path   = resolve_csv(args.svd, 'svd_results.csv')
    bmshj_path = resolve_csv(args.bmshj, 'bmshj2018_results.csv')

    svd_rows   = sorted(load_csv(svd_path), key=lambda r: r['k'])
    bmshj_rows = sorted(load_csv(bmshj_path), key=lambda r: r['quality'])

    svd_relerr   = np.array([r['rel_err']   for r in svd_rows])
    svd_bpv      = np.array([r['bpv_coeff'] for r in svd_rows])       # raw float32 coeffs, no entropy coding
    svd_bpv_lca  = np.array([r['bpv_lca_equiv'] for r in svd_rows])   # COO-style, for reference
    svd_comp     = np.array([r['comp_coeff'] for r in svd_rows])      # = 32 / bpv_coeff, same baseline

    bmshj_relerr = np.array([r['rel_err']    for r in bmshj_rows])
    bmshj_bpv    = np.array([r['bpv']        for r in bmshj_rows])    # real arithmetic-coded bytes
    bmshj_comp   = np.array([r['comp_ratio'] for r in bmshj_rows])

    # ------------------------------------------------------------------ #
    # Console comparison table — BPV and compression ratio side by side
    # ------------------------------------------------------------------ #
    print(f"{'target rel_err':>15}  {'SVD BPV':>9}  {'SVD Comp':>10}  "
          f"{'bmshj BPV':>10}  {'bmshj Comp':>11}  {'winner (comp)':>14}")
    print('-' * 82)
    for t in sorted(args.targets):
        svd_b, bmshj_b = value_at_target(svd_relerr, svd_bpv, t), value_at_target(bmshj_relerr, bmshj_bpv, t)
        svd_c, bmshj_c = value_at_target(svd_relerr, svd_comp, t), value_at_target(bmshj_relerr, bmshj_comp, t)

        svd_b_s   = f"{svd_b:.4f}"   if svd_b   is not None else "n/a"
        bmshj_b_s = f"{bmshj_b:.4f}" if bmshj_b is not None else "n/a"
        svd_c_s   = f"{svd_c:.2f}x"  if svd_c   is not None else "n/a"
        bmshj_c_s = f"{bmshj_c:.2f}x" if bmshj_c is not None else "n/a"

        if svd_c is not None and bmshj_c is not None:
            winner = "SVD" if svd_c > bmshj_c else "bmshj2018"
        else:
            winner = "n/a"
        print(f"{t:>15.3f}  {svd_b_s:>9}  {svd_c_s:>10}  "
              f"{bmshj_b_s:>10}  {bmshj_c_s:>11}  {winner:>14}")

    svd_best   = min(svd_rows, key=lambda r: r['rel_err'])
    bmshj_best = min(bmshj_rows, key=lambda r: r['rel_err'])
    print(f"\nSVD best achievable:       rel_err={svd_best['rel_err']:.4f}  "
          f"comp_ratio={svd_best['comp_coeff']:.2f}x  BPV(coeff)={svd_best['bpv_coeff']:.4f}  "
          f"k={int(svd_best['k'])}")
    print(f"bmshj2018 best achievable: rel_err={bmshj_best['rel_err']:.4f}  "
          f"comp_ratio={bmshj_best['comp_ratio']:.2f}x  BPV={bmshj_best['bpv']:.4f}  "
          f"quality={int(bmshj_best['quality'])}")

    # ------------------------------------------------------------------ #
    # Overlay plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(svd_bpv, svd_relerr, 'o-', color='steelblue', markersize=5,
                label='SVD (raw float32 coeffs)')
    ax.semilogy(svd_bpv_lca, svd_relerr, '^:', color='teal', markersize=4, alpha=0.7,
                label='SVD (COO-style estimate)')
    ax.semilogy(bmshj_bpv, bmshj_relerr, 's-', color='darkorange', markersize=5,
                label='bmshj2018-factorized (real coded bytes)')

    for t in args.targets:
        ax.axhline(t, color='gray', linestyle=':', linewidth=0.6)

    for r in svd_rows[::max(1, len(svd_rows)//8)]:
        ax.annotate(f"k={int(r['k'])}", (r['bpv_coeff'], r['rel_err']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7, color='steelblue')
    for r in bmshj_rows:
        ax.annotate(f"q={int(r['quality'])}", (r['bpv'], r['rel_err']),
                    textcoords='offset points', xytext=(4, -8), fontsize=7, color='darkorange')

    ax.set_xlabel('Bits per voxel (BPV)')
    ax.set_ylabel('Relative reconstruction error (log scale)')
    ax.set_title('SVD vs bmshj2018-factorized — Rate–Distortion comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"\nSaved {args.output}")

    # ------------------------------------------------------------------ #
    # Compression-ratio comparison plot: comp_ratio vs rel_err
    # ------------------------------------------------------------------ #
    svd_comp_lca = 32.0 / svd_bpv_lca  # comp_ratio = 32 / bpv, same float32-baseline convention

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(svd_relerr, svd_comp, 'o-', color='steelblue', markersize=5,
              label='SVD (raw float32 coeffs)')
    ax.loglog(svd_relerr, svd_comp_lca, '^:', color='teal', markersize=4, alpha=0.7,
              label='SVD (COO-style estimate)')
    ax.loglog(bmshj_relerr, bmshj_comp, 's-', color='darkorange', markersize=5,
              label='bmshj2018-factorized (real coded bytes)')

    for t in args.targets:
        ax.axvline(t, color='gray', linestyle=':', linewidth=0.6)

    for r in svd_rows[::max(1, len(svd_rows)//8)]:
        ax.annotate(f"k={int(r['k'])}", (r['rel_err'], r['comp_coeff']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7, color='steelblue')
    for r in bmshj_rows:
        ax.annotate(f"q={int(r['quality'])}", (r['rel_err'], r['comp_ratio']),
                    textcoords='offset points', xytext=(4, -8), fontsize=7, color='darkorange')

    ax.set_xlabel('Relative reconstruction error (log scale)')
    ax.set_ylabel('Compression ratio (log scale)')
    ax.set_title('SVD vs bmshj2018-factorized — Compression ratio comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    plt.tight_layout()
    base, ext = os.path.splitext(args.output)
    comp_output = (base[:-len('rate_distortion')] + 'comp_ratio' if base.endswith('rate_distortion')
                   else base + '_comp_ratio') + ext
    plt.savefig(comp_output, dpi=150)
    plt.close()
    print(f"Saved {comp_output}")


if __name__ == '__main__':
    main()
