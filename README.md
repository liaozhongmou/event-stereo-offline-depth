# Training-free event stereo for offline sparse depth imaging

Reference implementation, frozen experiment records, and figure-generation scripts for
the manuscript *Training-free event stereo for offline sparse depth imaging* by
Zhongmou Liao, Tengfei Lu, and Zhongli Wang, submitted to the *Journal of Electronic
Imaging* (under review).

The pipeline is deterministic and training-free:

1. **LIF event selection** — signed, non-reset leaky integrate-and-fire filtering with a
   rate-adaptive threshold.
2. **Bounded spatiotemporal matching** — adaptive temporal window, weighted line fit with
   Huber IRLS, local evidence voting, and a reverse-consistency check at the common
   reference event time. Disparity is bounded to 4–28 px.
3. **Centered local completion** — accepted correspondences are retained and tagged
   interpolants are added by occupancy-guided local averaging.

The same code evaluates MVSEC Indoor Flying 1–3 over complete ground-truth timelines and
a frozen-parameter DSEC `thun_00_a` test.

## Repository layout

| Path | Contents |
|---|---|
| `src/` | Pipeline, evaluation, controls, timing, activity and DSEC scripts |
| `tests/` | Unit and regression tests (pytest) |
| `diagnostics/` | Temporal-basis, ground-truth audit and stage diagnostics |
| `original/` | Frozen earlier implementation used by the current code |
| `protocol/` | Evaluation contract and input semantics |
| `runs/` | Reference run records: contracts, metrics, timing, output hashes |
| `manuscript/` | Figure/table generation and manuscript verification scripts |
| `requirements_revision3.txt` | Pinned Python dependencies |

## Environment

Python 3.13, CPU only, one numerical thread per process. `numba` is required; the
pure-Python fallback is much slower and must not be used for timing comparisons.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest -q
```

The suite contains 31 tests and does not require the benchmark data.

## Data

No benchmark data are redistributed in this repository.

- **MVSEC** (Zhu et al., *IEEE RA-L* 3(3), 2032–2039, 2018; CC BY-SA 4.0). The converted
  NPZ arrays used in this study are expected next to the repository root:
  `../indoor{1,2,3}_data/indoor{N}_left1.npz`, `indoor{N}_right1.npz`, and
  `indoor{N}_gt_disp.npz`. These arrays predate this revision; their rectification and
  clock alignment are inherited and documented in the manuscript supplement.
- **DSEC** (Gehrig et al., *IEEE RA-L*, 2021; CC BY-SA 4.0), sequence `thun_00_a`, is
  expected at `datasets/dsec/thun_00_a/` with `events_left/`, `events_right/`,
  `disparity_timestamps.txt` and the ground-truth PNG frames.

Ground-truth disparity uses the archived conversion `disp x 0.8819476`. DSEC events use
the official rectification maps and `(t + t_offset) * 1e-6` seconds; DSEC ground-truth
PNG values are divided by 256. See `DATA_LICENSE_NOTICE.md` for attribution and terms.

## Reproducing the experiments

Reference records are committed under `runs/`. Scripts refuse to overwrite an existing
run directory, so rename `runs/` to `reference_runs/` before a full replay.

For MVSEC sequences N = 1, 2, 3:

```bash
python -m src.run_revision3 --mode infer --sequence N --out runs/C3_unified_indoorN_20261021
python -m src.run_controls3 --sequence N --out runs/C3_controls_v2_indoorN_20261021
python -m src.ablate_current3 --sequence N --out runs/C3_current_ablation_indoorN_20261021
python -m src.analyze_revision3 --sequence N --out runs/C3_full_gt_v2_indoorN_20261021
```

DSEC, where inference must finish before scoring so that ground truth is not loaded
during inference:

```bash
python -m src.dsec_revision3 --mode infer --out runs/D3_dsec_thun00a_v2_20261021
python -m src.dsec_revision3 --mode score --out runs/D3_dsec_thun00a_v2_20261021
```

Sensitivity, temporal stress, completion sweep, activity strata and joint timing:

```bash
python -m src.temporal_stress3
python -m src.sensitivity3
python -m src.sweep_completion3 --out runs/C3_completion_sweep_v2_20261021
python -m src.activity_revision3
python -m src.timing_revision3 --mode all --out runs/C3_joint_timing_20261021
python -m src.score_timing3
```

## Figures and tables

```bash
python manuscript/draw_revision3_figures.py
python manuscript/redraw_fig2.py
python manuscript/build_revision3_results.py
```

`manuscript/verify_draft.py` and `manuscript/package_draft.py` additionally require the
LaTeX source package and a TeX installation.

## Scope and caveats

- Timing is machine- and load-dependent; compare same-session runs only.
- The converted MVSEC arrays predate this revision; the original extraction procedure was
  not recovered. Their hashes and interpretation are recorded with the run data.
- This is research software, not a production depth system.

## Citation

If you use this code, please cite the manuscript (under review):

```bibtex
@article{liao2026trainingfree,
  title   = {Training-free event stereo for offline sparse depth imaging},
  author  = {Liao, Zhongmou and Lu, Tengfei and Wang, Zhongli},
  journal = {Journal of Electronic Imaging},
  year    = {2026},
  note    = {Under review}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Benchmark data and derived data
remain under their original licenses; see `DATA_LICENSE_NOTICE.md`.
