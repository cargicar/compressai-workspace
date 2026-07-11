"""
Compare two compression baselines (SVD and a bmshj2018-factorized model) on
the same rel_err / BPV metrics.

The bmshj2018 side can be either `bmshj2018_compression.py`'s pretrained-model
sweep (`bmshj2018_results.csv`, swept over `quality`) or
`bmshj2018_scratch.py`'s from-scratch run (`scratch_results.csv`, currently a
single λ, not a sweep) — this script auto-detects which one is present in the
given directory and normalizes both into the same rel_err/bpv/comp_ratio
fields so the rest of the script doesn't care which one it's looking at.

Both `svd_compression.py` and `bmshj2018_compression.py`/`bmshj2018_scratch.py`
already report comparable metrics (rel_err, PSNR, comp_ratio, BPV). This script
just loads the CSVs, overlays their rate-distortion and compression-ratio
curves, and prints the BPV and comp_ratio each method needs/achieves at a set
of target relative errors so they're easy to read off side by side.

If the SVD run directory also has a `svd_real_results.csv` (from
svd_compression.py's quantization + real entropy-coding sweep), its Pareto
frontier is loaded too and is the fairest comparison against bmshj2018's real
coded bytes — the plain SVD curves never touch a real compressor, they're a
raw float32 byte count.

Usage
-----
    python compare_compression.py experiments/svd_2026-07-08_10-00-00 experiments/bmshj2018_2026-07-08_10-05-00
    python compare_compression.py experiments/svd_2026-07-08_10-00-00 experiments/scratch_poc2
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


def resolve_bmshj_csv(path: str):
    """Auto-detect a pretrained (bmshj2018_results.csv) or from-scratch
    (scratch_results.csv) results file. Returns (kind, csv_path)."""
    if os.path.isdir(path):
        pretrained = os.path.join(path, 'bmshj2018_results.csv')
        scratch = os.path.join(path, 'scratch_results.csv')
        if os.path.isfile(pretrained):
            return 'pretrained', pretrained
        if os.path.isfile(scratch):
            return 'scratch', scratch
        raise FileNotFoundError(
            f"Expected bmshj2018_results.csv or scratch_results.csv inside {path}")
    basename = os.path.basename(path)
    kind = 'scratch' if basename == 'scratch_results.csv' else 'pretrained'
    return kind, path


def load_csv(path: str) -> list:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in row.items()} for row in rows]


def normalize_bmshj_rows(kind: str, rows: list) -> list:
    """Add unified 'bpv' / 'comp_ratio' / 'label' fields regardless of source,
    so the rest of the script doesn't need to know pretrained vs scratch."""
    normalized = []
    for r in rows:
        row = dict(r)
        if kind == 'pretrained':
            row['bpv'] = r['bpv']
            row['comp_ratio'] = r['comp_ratio']
            row['label'] = f"q={int(r['quality'])}"
        else:
            row['bpv'] = r['real_bpv']
            row['comp_ratio'] = r['real_comp_ratio']
            row['label'] = f"λ={r['lambda_']:.0f}"
        normalized.append(row)
    return normalized


def load_optional_sibling_csv(reference_path: str, filename: str):
    """Look for `filename` next to `reference_path`; return rows or None if absent."""
    candidate = os.path.join(os.path.dirname(reference_path), filename)
    if not os.path.isfile(candidate):
        return None
    return load_csv(candidate)


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


def value_at_target(rel_errs: np.ndarray, values: np.ndarray, target: float):
    """Interpolate `values` (BPV or comp_ratio) at `target` rel_err (None if out of range)."""
    order = np.argsort(rel_errs)
    re, val = rel_errs[order], values[order]
    if target < re.min() or target > re.max():
        return None
    return float(np.interp(target, re, val))


def main():
    parser = argparse.ArgumentParser(description='Compare SVD vs a bmshj2018-factorized model')
    parser.add_argument('svd', help='svd_results.csv or its run directory')
    parser.add_argument('bmshj', help='bmshj2018_results.csv / scratch_results.csv or its run directory '
                                       '(pretrained vs from-scratch auto-detected)')
    parser.add_argument('--targets', type=float, nargs='+',
                        default=[0.01, 0.02, 0.05, 0.1],
                        help='rel_err targets to compare BPV at')
    parser.add_argument('--output', default='compare_rate_distortion.png',
                        help='output plot path')
    args = parser.parse_args()

    svd_path = resolve_csv(args.svd, 'svd_results.csv')
    bmshj_kind, bmshj_path = resolve_bmshj_csv(args.bmshj)
    bmshj_label_prefix = 'bmshj2018-pretrained' if bmshj_kind == 'pretrained' else 'bmshj2018-scratch'
    print(f"bmshj2018 source: {bmshj_kind} ({bmshj_path})")
    if bmshj_kind == 'scratch':
        print("Note: from-scratch results are a single λ, not a sweep — plotted as one point.\n")
    else:
        print()

    svd_rows   = sorted(load_csv(svd_path), key=lambda r: r['k'])
    bmshj_rows = sorted(normalize_bmshj_rows(bmshj_kind, load_csv(bmshj_path)), key=lambda r: r['rel_err'])

    svd_relerr   = np.array([r['rel_err']    for r in svd_rows])
    svd_bpv      = np.array([r['bpv_coeff']  for r in svd_rows])   # raw float32 coeffs, no entropy coding
    svd_bpv_tot  = np.array([r['bpv_total']  for r in svd_rows])   # + amortised basis cost
    svd_comp     = np.array([r['comp_coeff'] for r in svd_rows])   # = 32 / bpv_coeff, same baseline

    bmshj_relerr = np.array([r['rel_err']    for r in bmshj_rows])
    bmshj_bpv    = np.array([r['bpv']        for r in bmshj_rows])    # real coded bytes either way
    bmshj_comp   = np.array([r['comp_ratio'] for r in bmshj_rows])

    # Optional: SVD real quantization + entropy-coding sweep (fairest comparison, if present)
    svd_real_rows = load_optional_sibling_csv(svd_path, 'svd_real_results.csv')
    if svd_real_rows:
        real_frontier  = pareto_frontier([(r['rel_err'], r['real_bpv'], r) for r in svd_real_rows])
        svd_real_relerr = np.array([p[0] for p in real_frontier])
        svd_real_bpv    = np.array([p[1] for p in real_frontier])
        svd_real_comp   = np.array([p[2]['real_comp_ratio'] for p in real_frontier])
        print("Found svd_real_results.csv — including SVD's real quantized+entropy-coded frontier.\n")
    else:
        svd_real_relerr = svd_real_bpv = svd_real_comp = None
        print("No svd_real_results.csv found next to the SVD CSV — only the raw-coefficient SVD "
              "curve is available (run svd_compression.py to get the real-compression sweep).\n")

    # For the console table / winner, prefer SVD's real numbers when available — they're the
    # fair comparison against bmshj2018's real coded bytes; the raw-coeff SVD curve never
    # touches a compressor.
    svd_table_relerr = svd_real_relerr if svd_real_rows else svd_relerr
    svd_table_bpv     = svd_real_bpv    if svd_real_rows else svd_bpv
    svd_table_comp    = svd_real_comp   if svd_real_rows else svd_comp
    svd_table_label   = "SVD-real" if svd_real_rows else "SVD"

    # ------------------------------------------------------------------ #
    # Console comparison table — BPV and compression ratio side by side
    # ------------------------------------------------------------------ #
    print(f"{'target rel_err':>15}  {svd_table_label+' BPV':>10}  {svd_table_label+' Comp':>11}  "
          f"{'bmshj BPV':>10}  {'bmshj Comp':>11}  {'winner (comp)':>14}")
    print('-' * 85)
    for t in sorted(args.targets):
        svd_b   = value_at_target(svd_table_relerr, svd_table_bpv, t)
        bmshj_b = value_at_target(bmshj_relerr, bmshj_bpv, t)
        svd_c   = value_at_target(svd_table_relerr, svd_table_comp, t)
        bmshj_c = value_at_target(bmshj_relerr, bmshj_comp, t)

        svd_b_s   = f"{svd_b:.4f}"   if svd_b   is not None else "n/a"
        bmshj_b_s = f"{bmshj_b:.4f}" if bmshj_b is not None else "n/a"
        svd_c_s   = f"{svd_c:.2f}x"  if svd_c   is not None else "n/a"
        bmshj_c_s = f"{bmshj_c:.2f}x" if bmshj_c is not None else "n/a"

        if svd_c is not None and bmshj_c is not None:
            winner = svd_table_label if svd_c > bmshj_c else "bmshj2018"
        else:
            winner = "n/a"
        print(f"{t:>15.3f}  {svd_b_s:>10}  {svd_c_s:>11}  "
              f"{bmshj_b_s:>10}  {bmshj_c_s:>11}  {winner:>14}")

    svd_best   = min(svd_rows, key=lambda r: r['rel_err'])
    bmshj_best = min(bmshj_rows, key=lambda r: r['rel_err'])
    print(f"\nSVD best achievable (raw coeffs): rel_err={svd_best['rel_err']:.4f}  "
          f"comp_ratio={svd_best['comp_coeff']:.2f}x  BPV(coeff)={svd_best['bpv_coeff']:.4f}  "
          f"k={int(svd_best['k'])}")
    if svd_real_rows:
        svd_real_best = min(svd_real_rows, key=lambda r: r['rel_err'])
        print(f"SVD best achievable (real):        rel_err={svd_real_best['rel_err']:.4f}  "
              f"comp_ratio={svd_real_best['real_comp_ratio']:.2f}x  BPV={svd_real_best['real_bpv']:.4f}  "
              f"k={int(svd_real_best['k'])}  bits={int(svd_real_best['bits'])}")
    print(f"{bmshj_label_prefix} best achievable: rel_err={bmshj_best['rel_err']:.4f}  "
          f"comp_ratio={bmshj_best['comp_ratio']:.2f}x  BPV={bmshj_best['bpv']:.4f}  "
          f"({bmshj_best['label']})")

    # ------------------------------------------------------------------ #
    # Overlay plot
    # ------------------------------------------------------------------ #
    is_sweep = len(bmshj_rows) > 1
    bmshj_style = dict(marker='s', linestyle='-', markersize=5) if is_sweep \
        else dict(marker='*', linestyle='none', markersize=5)
    bmshj_plot_label = f"{bmshj_label_prefix} (real coded bytes)" + ("" if is_sweep else ", single λ")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.semilogy(svd_bpv, svd_relerr, 'o-', color='steelblue', markersize=5, alpha=0.5,
                label='SVD (raw float32 coeffs, not compressed)')
    ax.semilogy(svd_bpv_tot, svd_relerr, '^:', color='teal', markersize=4, alpha=0.4,
                label='SVD (+ amortised basis, not compressed)')
    if svd_real_rows:
        ax.semilogy(svd_real_bpv, svd_real_relerr, 'D-', color='seagreen', markersize=5,
                    label='SVD (quantized + real-compressed, Pareto frontier)')
    ax.semilogy(bmshj_bpv, bmshj_relerr, color='darkorange', label=bmshj_plot_label, **bmshj_style)

    for t in args.targets:
        ax.axhline(t, color='gray', linestyle=':', linewidth=0.6)

    for r in svd_rows[::max(1, len(svd_rows)//8)]:
        ax.annotate(f"k={int(r['k'])}", (r['bpv_coeff'], r['rel_err']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7, color='steelblue')
    for r in bmshj_rows:
        ax.annotate(r['label'], (r['bpv'], r['rel_err']),
                    textcoords='offset points', xytext=(4, -8), fontsize=7, color='darkorange')

    ax.set_xlabel('Bits per voxel (BPV)')
    ax.set_ylabel('Relative reconstruction error (log scale)')
    ax.set_title(f'SVD vs {bmshj_label_prefix} — Rate–Distortion comparison')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()
    print(f"\nSaved {args.output}")

    # ------------------------------------------------------------------ #
    # Compression-ratio comparison plot: comp_ratio vs rel_err
    # ------------------------------------------------------------------ #
    svd_comp_tot = 32.0 / svd_bpv_tot  # comp_ratio = 32 / bpv, same float32-baseline convention

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.loglog(svd_relerr, svd_comp, 'o-', color='steelblue', markersize=5, alpha=0.5,
              label='SVD (raw float32 coeffs, not compressed)')
    ax.loglog(svd_relerr, svd_comp_tot, '^:', color='teal', markersize=4, alpha=0.4,
              label='SVD (+ amortised basis, not compressed)')
    if svd_real_rows:
        ax.loglog(svd_real_relerr, svd_real_comp, 'D-', color='seagreen', markersize=5,
                  label='SVD (quantized + real-compressed, Pareto frontier)')
    ax.loglog(bmshj_relerr, bmshj_comp, color='darkorange', label=bmshj_plot_label, **bmshj_style)

    for t in args.targets:
        ax.axvline(t, color='gray', linestyle=':', linewidth=0.6)

    for r in svd_rows[::max(1, len(svd_rows)//8)]:
        ax.annotate(f"k={int(r['k'])}", (r['rel_err'], r['comp_coeff']),
                    textcoords='offset points', xytext=(4, 4), fontsize=7, color='steelblue')
    for r in bmshj_rows:
        ax.annotate(r['label'], (r['rel_err'], r['comp_ratio']),
                    textcoords='offset points', xytext=(4, -8), fontsize=7, color='darkorange')

    ax.set_xlabel('Relative reconstruction error (log scale)')
    ax.set_ylabel('Compression ratio (log scale)')
    ax.set_title(f'SVD vs {bmshj_label_prefix} — Compression ratio comparison')
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
