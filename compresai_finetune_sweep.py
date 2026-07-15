"""
Runs compresai_finetune.py once per architecture, all from the same config
file, each into its own experiments/finetune_<architecture>_<timestamp>/
directory so logs, plots, CSVs, and checkpoints never collide.

On a multi-GPU machine, architectures run CONCURRENTLY, one per GPU, via a
small job-pool scheduler: each subprocess gets CUDA_VISIBLE_DEVICES pinned to
exactly one physical GPU (so inside that subprocess it simply sees "cuda:0"
and never contends with the others), and up to len(gpus) run at once. This is
the right kind of parallelism here — these models are small (3-7M params)
and batches are small, so splitting a *single* fine-tune run across GPUs with
DistributedDataParallel would add real complexity (CompressAI's entropy
bottleneck has known friction with DDP's gradient sync) for little benefit;
running the 4 already-independent architecture fine-tunes concurrently on 4
separate GPUs gets the same ~4x wall-clock win with none of that complexity.

On a single-GPU or CPU-only machine this degrades to the original sequential
behavior automatically (one GPU slot, or one CPU slot if no CUDA at all).

Since concurrent subprocesses' stdout would otherwise interleave into an
unreadable mess, each run's full output is redirected to
<out_dir>/stdout.log (in addition to the per-run run.log
compresai_finetune.py already writes once its own logging is set up) — the
console just gets concise start/finish status lines.

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
    python compresai_finetune_sweep.py config_compresai.yaml --gpus 1 2 3   # skip GPU 0
    python compresai_finetune_sweep.py config_compresai.yaml --max-concurrent 2
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime

import torch

from compresai_finetune import ARCHITECTURES

ARCH_NAMES = sorted(ARCHITECTURES)
POLL_INTERVAL_S = 5


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune each compresai_finetune.py architecture, one per GPU, concurrently")
    parser.add_argument("config", help="path to config_compresai.yaml (or another config with a finetune: section)")
    parser.add_argument("--architectures", nargs="+", choices=ARCH_NAMES, default=ARCH_NAMES,
                        help="which architectures to sweep (default: all four)")
    parser.add_argument("--output-root", default="experiments",
                        help="parent directory for the per-architecture finetune_<arch>_<timestamp> folders")
    parser.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="physical GPU ids to use (default: all visible GPUs, e.g. 0 1 2 3; "
                             "CPU-only if none available)")
    parser.add_argument("--max-concurrent", type=int, default=None,
                        help="cap on simultaneous runs (default: number of GPU slots — more "
                             "architectures than slots just queue for the next free GPU)")
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
    args = parser.parse_args()

    if args.gpus is not None:
        gpu_ids = list(args.gpus)
    elif torch.cuda.is_available():
        gpu_ids = list(range(torch.cuda.device_count()))
    else:
        gpu_ids = [None]  # single CPU slot — sequential, no CUDA_VISIBLE_DEVICES pinning

    max_concurrent = min(args.max_concurrent or len(gpu_ids), len(gpu_ids))
    print(f"GPU slots  : {gpu_ids if gpu_ids != [None] else 'none (CPU)'}")
    print(f"Concurrency: {max_concurrent} run(s) at a time\n")

    sweep_tag = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(args.output_root, exist_ok=True)
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compresai_finetune.py")

    passthrough = [
        ("--quality", args.quality), ("--metric", args.metric), ("--lambda_", args.lambda_),
        ("--patch-size", args.patch_size), ("--batch-size", args.batch_size),
        ("--iterations", args.iterations), ("--lr", args.lr), ("--aux-lr", args.aux_lr),
        ("--axis", args.axis), ("--train-timesteps", args.train_timesteps),
        ("--val-timestep", args.val_timestep), ("--slices-per-timestep", args.slices_per_timestep),
    ]

    # ------------------------------------------------------------------ #
    # Job-pool scheduler: pop a pending architecture onto any free GPU slot,
    # poll running processes, free a slot (and reuse it) as each finishes.
    # ------------------------------------------------------------------ #
    import subprocess

    pending = deque(args.architectures)
    free_gpus = deque(gpu_ids[:max_concurrent])
    running = {}     # arch -> (Popen, gpu_id, out_dir, log_file_handle)
    run_dirs = {}    # arch -> out_dir, for successful runs only
    failed = []

    def launch(arch, gpu_id):
        out_dir = os.path.join(args.output_root, f"finetune_{arch}_{sweep_tag}")
        os.makedirs(out_dir, exist_ok=True)
        cmd = [sys.executable, script_path, args.config, "--architecture", arch, "--output-dir", out_dir]
        for flag, val in passthrough:
            if val is not None:
                cmd += [flag, str(val)]

        env = os.environ.copy()
        gpu_label = "CPU"
        if gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            gpu_label = f"GPU {gpu_id}"
        else:
            cmd += ["--device", "cpu"]

        log_path = os.path.join(out_dir, "stdout.log")
        log_fh = open(log_path, "w")
        print(f"[{arch}] started on {gpu_label}  ->  {out_dir}  (full output: {log_path})")
        proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
        return proc, out_dir, log_fh

    while pending or running:
        while pending and free_gpus:
            arch = pending.popleft()
            gpu_id = free_gpus.popleft()
            proc, out_dir, log_fh = launch(arch, gpu_id)
            running[arch] = (proc, gpu_id, out_dir, log_fh)

        finished = [arch for arch, (proc, _, _, _) in running.items() if proc.poll() is not None]
        for arch in finished:
            proc, gpu_id, out_dir, log_fh = running.pop(arch)
            log_fh.close()
            free_gpus.append(gpu_id)
            if proc.returncode == 0:
                run_dirs[arch] = out_dir
                print(f"[{arch}] finished OK")
            else:
                failed.append(arch)
                print(f"[{arch}] FAILED (exit {proc.returncode}) — see {out_dir}/stdout.log")

        if running:
            time.sleep(POLL_INTERVAL_S)

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
        print(f"Failed (excluded below): {failed}\n")
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
