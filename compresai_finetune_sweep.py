"""
Runs compresai_finetune.py once per architecture (one model fine-tuned at a
time, not concurrently), all from the same config file, each into its own
experiments/finetune_<architecture>_<timestamp>/ directory so logs, plots,
CSVs, and checkpoints never collide or overwrite each other.

Each architecture is fine-tuned in its own subprocess rather than in-process:
that gives every run a clean CUDA context (no leftover allocations from the
previous architecture) and keeps this driver itself independent of
torch/compressai import weight beyond the one shared architecture list.

Config
------
Same config_compresai.yaml (or any config with a finetune: section) used by
compresai_finetune.py — finetune.architecture in the config is ignored here;
--architectures (or its default, all four) picks the sweep instead.

Usage
-----
    python compresai_finetune_sweep.py config_compresai.yaml
    python compresai_finetune_sweep.py config_compresai.yaml --architectures mbt2018-mean mbt2018
    python compresai_finetune_sweep.py config_compresai.yaml --iterations 5000 --quality 6
"""

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime

from compresai_finetune import ARCHITECTURES

ARCH_NAMES = sorted(ARCHITECTURES)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune each compresai_finetune.py architecture in turn, one experiments/ folder each")
    parser.add_argument("config", help="path to config_compresai.yaml (or another config with a finetune: section)")
    parser.add_argument("--architectures", nargs="+", choices=ARCH_NAMES, default=ARCH_NAMES,
                        help="which architectures to sweep, in order (default: all four)")
    parser.add_argument("--output-root", default="experiments",
                        help="parent directory for the per-architecture finetune_<arch>_<timestamp> folders")
    # Passthrough overrides — forwarded to compresai_finetune.py only if set, else its own
    # config-driven defaults apply (same flags it defines, see its --help for details).
    parser.add_argument("--quality", type=int, default=None, choices=range(1, 9))
    parser.add_argument("--metric", choices=["mse", "ms-ssim"], default=None)
    parser.add_argument("--lambda_", type=float, default=None)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--aux-lr", type=float, default=None)
    parser.add_argument("--axis", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--train-timesteps", default=None)
    parser.add_argument("--val-timestep", type=int, default=None)
    parser.add_argument("--slices-per-timestep", type=int, default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    sweep_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.output_root, exist_ok=True)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compresai_finetune.py")

    passthrough = [
        ("--quality", args.quality), ("--metric", args.metric), ("--lambda_", args.lambda_),
        ("--patch-size", args.patch_size), ("--batch-size", args.batch_size),
        ("--iterations", args.iterations), ("--lr", args.lr), ("--aux-lr", args.aux_lr),
        ("--axis", args.axis), ("--train-timesteps", args.train_timesteps),
        ("--val-timestep", args.val_timestep), ("--slices-per-timestep", args.slices_per_timestep),
        ("--device", args.device),
    ]

    run_dirs = {}
    failed = []
    for i, arch in enumerate(args.architectures, 1):
        out_dir = os.path.join(args.output_root, f"finetune_{arch}_{sweep_tag}")
        cmd = [sys.executable, script_path, args.config, "--architecture", arch, "--output-dir", out_dir]
        for flag, val in passthrough:
            if val is not None:
                cmd += [flag, str(val)]

        print(f"\n{'='*70}")
        print(f"[{i}/{len(args.architectures)}] {arch}  ->  {out_dir}")
        print(f"{'='*70}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nWARNING: {arch} exited with code {result.returncode} — excluded from the summary below.")
            failed.append(arch)
            continue
        run_dirs[arch] = out_dir

    # ------------------------------------------------------------------ #
    # Cross-architecture summary — pull the two rows (pretrained/finetuned)
    # each compresai_finetune.py run already wrote to its own finetune_results.csv
    # ------------------------------------------------------------------ #
    summary_rows = []
    for arch, out_dir in run_dirs.items():
        csv_path = os.path.join(out_dir, "finetune_results.csv")
        with open(csv_path, newline="") as f:
            rows = {row["label"]: row for row in csv.DictReader(f)}
        pre, fin = rows["pretrained"], rows["finetuned"]
        summary_rows.append(dict(
            architecture=arch,
            pretrained_rel_err=float(pre["rel_err"]), pretrained_comp_ratio=float(pre["comp_ratio"]),
            pretrained_bpv=float(pre["bpv"]),
            finetuned_rel_err=float(fin["rel_err"]), finetuned_comp_ratio=float(fin["comp_ratio"]),
            finetuned_bpv=float(fin["bpv"]),
            output_dir=out_dir,
        ))

    print(f"\n{'='*90}")
    print("SWEEP SUMMARY")
    print(f"{'='*90}")
    if failed:
        print(f"Failed (excluded above): {failed}\n")
    print(f"{'architecture':>22}  {'pretrained rel_err':>18}  {'finetuned rel_err':>18}  "
          f"{'pretrained comp':>15}  {'finetuned comp':>15}")
    print('-' * 92)
    for r in summary_rows:
        print(f"{r['architecture']:>22}  {r['pretrained_rel_err']:>18.6f}  {r['finetuned_rel_err']:>18.6f}  "
              f"{r['pretrained_comp_ratio']:>14.2f}x  {r['finetuned_comp_ratio']:>14.2f}x")

    if summary_rows:
        summary_path = os.path.join(args.output_root, f"finetune_sweep_summary_{sweep_tag}.csv")
        with open(summary_path, "w", newline="") as csvf:
            writer = csv.DictWriter(csvf, fieldnames=summary_rows[0].keys())
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nSweep summary saved to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
