# Development Log

## bmshj2018-factorized compression baseline

Wrote [bmshj2018_compression.py](bmshj2018_compression.py), a script mirroring `svd_compression.py`'s structure/metrics but using the pretrained `bmshj2018_factorized` codec from `compressai.zoo`.

### How it adapts a 2D RGB codec to a 3D scalar field

- Loads the same `t=15` `pressure` snapshot (256×256×256) via the config's `data` section.
- Globally min-max normalizes the volume to [0,1] (the range the model was trained on).
- Since the network is hardcoded for 3-channel 2D input (`conv(3,N)`), it slices the volume into 256 2D planes along `--axis` (default 0) and replicates each grayscale slice to pseudo-RGB.
- Runs the *real* entropy coder (`model.compress`/`model.decompress`, after `model.update(force=True)`) to get actual arithmetic-coded bytes — not an estimated rate.
- Decoded 3-channel output is averaged back to a scalar slice and reassembled into the volume.
- Reports `rel_err`, `PSNR`, `comp_ratio`, `BPV` with the same formulas as `svd_compression.py`, sweeping CompressAI's quality levels 1–8, plus a rate-distortion plot and a full-volume reconstruction plot at the best quality.

### Setup notes

- Added `h5py`, `pyyaml`, `matplotlib` to the project via `uv add` — the `.venv` had CompressAI/CUDA but not the data-loading/plotting deps `svd_compression.py` needs; both scripts now run from the same `.venv`.
- Pretrained weights download from CompressAI's S3 bucket on first use per quality (cached in `~/.cache/torch/hub/checkpoints/`) — confirmed working in this sandbox.

### Smoke test results

Tested on qualities 1/4/8 against the real `Turb_M1.hdf5` file on GPU: rel_err ranged 0.115 → 0.022, comp_ratio 460x → 45x, and the reconstructed mid-plane slices visually track the turbulent structure well at quality 8.

Run the full sweep with:

```
.venv/bin/python bmshj2018_compression.py config_simmldc.yaml
```

### Caveat

Since the model is trained on natural RGB images and this is a "pseudo-RGB" replication trick on a turbulence field, don't expect it to beat a domain-specific codec (LCA/SVD) at the same bitrate — it's a useful baseline/reference point, not necessarily the best fit for this data.

## Comparing SVD vs bmshj2018-factorized

Both `svd_compression.py` and `bmshj2018_compression.py` already sweep a rate-controlling parameter (k / quality) and save a results CSV with comparable metrics (rel_err, PSNR, comp_ratio, BPV), so rather than duplicating any compression logic, added [compare_compression.py](compare_compression.py) that just loads both CSVs and compares them directly.

### What it does

- Takes two paths — either a run directory (`experiments/svd_.../`) or a CSV file directly — for the SVD and bmshj2018 results.
- Prints a console table of the BPV **and compression ratio** each method needs/achieves to hit target `rel_err` values (0.01/0.02/0.05/0.1 by default), declaring a winner (by comp_ratio) at each.
- Prints each method's best achievable rel_err with its comp_ratio and BPV.
- Saves two overlay plots: `compare_rate_distortion.png` (BPV vs rel_err, log y) and `compare_comp_ratio.png` (comp_ratio vs rel_err, log-log) — both annotated by `k` (SVD) / quality (bmshj2018).
- Uses SVD's `bpv_coeff`/`comp_coeff` (raw float32 storage — SVD doesn't entropy-code anything) against bmshj2018's `bpv`/`comp_ratio` (real arithmetic-coded bytes from `model.compress`), plus SVD's `bpv_lca_equiv`-derived comp ratio as a secondary reference curve.

Usage:

```
.venv/bin/python compare_compression.py experiments/svd_run experiments/bmshj_run
```

### Smoke test results

Ran quick sweeps (SVD k=5..200, bmshj2018 quality=1..8) on the real `Turb_M1.hdf5` snapshot:

```
target rel_err    SVD BPV    SVD Comp   bmshj BPV   bmshj Comp   winner (comp)
----------------------------------------------------------------------------------
          0.050     4.3267       7.50x      0.2897      122.13x       bmshj2018
          0.100     1.6936      22.90x      0.0926      355.14x       bmshj2018

SVD best achievable:       rel_err=0.0279  comp_ratio=3.65x  BPV(coeff)=8.7791  k=200
bmshj2018 best achievable: rel_err=0.0221  comp_ratio=45.32x  BPV=0.7061  quality=8
```

bmshj2018-factorized dominates at every matched error level here — at rel_err≈0.05 it reaches ~122x compression vs SVD's ~7.5x. This is somewhat apples-to-oranges though: SVD's numbers are raw uncompressed float32 coefficients with no entropy coding, while bmshj2018's are real coded bytes after arithmetic coding. The `bpv_lca_equiv`/COO-style curve is SVD's attempt at a fairer comparison (estimating what a sparse/quantized coder would achieve), and it barely moves the needle here.
