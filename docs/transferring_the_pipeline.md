# Transferring the pipeline to another run / machine

Two questions answered here: what you need to carry to process a NEW recording,
and what is already processed and ready to hand off.

---

# Part 1 — what a new run needs

## 1.1 Per-recording inputs (you must supply these)

Under one `<session_dir>`:

| file | required | notes |
|---|---|---|
| `Cam*.mp4` (7) | **yes** | one per camera; names must match the calibration |
| `calibration/Cam*.yaml` (7) | **yes** | DLT matrices; the camera NAMES here define ordering |
| `Cam*_meta.csv` (7) | recommended | frame_id + timestamp; without them dropped frames CANNOT be detected or corrected |
| `<dataset>_bout_summary.csv` | **yes** | start/end frame per bout; the pipeline processes bouts, not whole recordings |
| `Cam*_keyframe.csv` | no | not read by this pipeline |

Without `Cam*_meta.csv` there is no `sync_plan.json`, so every camera is read
positionally and a dropped frame silently misaligns that camera from then on.
Session0 has no meta CSVs at all; 2 of 10 Session1 recordings genuinely
reindex, affecting 25 bouts.

## 1.2 Model artifacts (carry these, they are not in git)

| artifact | size | path |
|---|---|---|
| 2D detector checkpoint | 326 MB | `jax_vitpose_runs/v4_8gpu_20260808/final` |
| SAM3 weights | 6.5 GB | `$HF_HOME` = `<data>/Johnson_lab/sam3` |
| fly body model (MuJoCo XML + mesh npz) | 999 MB | `paths.body_model_dir` |
| JARVIS project `unified_V3_masked` | 1.9 GB | `<JARVIS-HybridNet>/projects/` |

The JARVIS project is loaded only so `ProjectManager` can hand a cfg to
`get_repro_tool`; the session's own `calibration/` dir takes precedence for the
actual geometry. It is a heavy dependency for a small purpose.

## 1.3 Code

- this repo, plus submodules `stac-mjx` and `third_party/JARVIS-HybridNet`
- `third_party/jarvis_jax` (274 MB) — vendored, **not** a submodule, so it
  travels with the repo checkout
- conda env `3d_tracking`

## 1.4 Environment (SAM3 stage only)

```bash
export HF_HOME=<data>/Johnson_lab/sam3
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
unset JAX_PLATFORMS          # never inherit JAX_PLATFORMS=cpu
module load cuda/12.9.1      # batch nodes do not expose libcuda otherwise
```

The pose stage wants the opposite: `unset LD_LIBRARY_PATH` so JAX uses its
bundled CUDA wheels, and `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85`.

## 1.5 Configs to add for a new session

- `configs/recording/<name>.yaml` — cameras, `num_animals`, `predictions_dir`,
  `bouts_csv`, `calib_dir`
- `paths=hyak` (or a new paths group if the data root differs)

## 1.6 Order of operations

```bash
# 1. masks + identity  (writes <processed>/<sess>/<rec>/sam3_masks/)
python third_party/jarvis_jax/scripts/sam3_masks.py paths=hyak sam3=default \
  sam3.session_dir=<session_dir> sam3.bouts_csv=<csv> sam3.out=<out> \
  sam3.project=unified_V3_masked sam3.jarvis_root=<JARVIS-HybridNet>

# 2. pose  (writes <processed>/<sess>/<rec>/pose/)
python scripts/run_bout.py paths=hyak recording=<name> \
  recording.session_dir=<session_dir> recording.bouts_csv=<csv>
```

## 1.7 Gotchas that cost real time here

- **`bout_ids` with commas** must be quoted for hydra: `bout_ids=\'4,19,25\'`.
  Unquoted is "ambiguous value" and fails instantly.
- **`recording.bouts_csv`**: Session0's configured CSV is a dangling symlink;
  override to the processed-side `courtship_bout_summary.csv`.
- **Camera order**: a newly written `sam3_masks.npz` carries a `cameras` array
  whose order is NOT the calibration's lexicographic one. Always reorder
  through it (`load_bout_masks(expected_cameras=...)`) before comparing or
  indexing, or you silently compare different cameras.
- **`DONE` markers**: `run_bout` skips any bout that has one. After changing
  masks or the detector, write to a NEW output root — otherwise the run
  "succeeds" in seconds having done nothing.
- **`predictions_dir` must point where the masks were actually written.** This
  cost two full pipeline runs. It is now derived the same way as `outputs.out`
  (`${paths.processed_root}/.../${basename:${recording.session_dir}}/sam3_masks`)
  so the two cannot drift. Consequence: any process that composes the `pipeline`
  config must register the `basename` OmegaConf resolver — `viz` does so by
  importing `utils.path_utils`.
- **Audit assignment after any mask or detector change**:
  `python scripts/qc/audit_fly_assignment.py`. It is the only check that caught
  the above; per-bout QC, reprojection error and within-bone CV were all clean
  while 19.7% of fly-frames tracked the wrong animal.
- **Shared per-recording artifacts**: `scale.json` and `offsets.h5` are fit
  once per recording on a "whoever gets there first" basis, so a recording's
  bouts cannot be split across workers until those exist. Seed one bout per
  recording first, then fan out.
- **Concurrency is workload-dependent, and measured beats assumed.** SAM3 is
  CPU-bound (JPEG decode): 8 workers drove load to 89 on 32 cores, GPUs to 0%,
  and aggregate throughput DOWN 3x versus 2 workers; use ~6 with
  `OMP_NUM_THREADS` capped. The pose stage with overlays off is GPU-bound and
  scales fine to 8 (69% idle CPU at 8 workers).

## 1.8 Measured throughput (per 8x L40S node, 32 cores)

| stage | rate | 119k-frame courtship set |
|---|---|---|
| SAM3 masks | ~40 bouts/h @ 6 workers | 4.5 h |
| pose through IK (no overlays) | ~0.95 s/fly-frame @ 8 workers | 7.8 h |
| overlay videos | +9 min per bout-fly | ~+30 GPU-h if enabled |

---

# Part 2 — what is processed and ready

## 2.1 Courtship: complete

11 recordings (Session0 + Session1), **160 bouts, 119,279 frames**, all from
one consistent stack: regenerated SAM3 masks + the V4-retrained detector.

| product | path | size |
|---|---|---|
| SAM3 masks + identity | `<processed>/courtship/<sess>/<rec>/sam3_masks/` | 1.9 GB |
| pose through IK | `.../pose/` | 4.6 GB |
| combined IK for analysis | `courtship/Data_analysis/analysis/v2_2026-08-10/` | 1.6 GB |
| previous masks (Session0 only) | `.../sam3_masks_old/` | 380 MB |

`pose/` is the promoted, corrected tree (2026-08-10). The pose outputs it
replaced were built while `recording.predictions_dir` pointed at a hand-named
`Predictions_3D_*` directory in the VIDEO tree while mask regeneration wrote to
the PROCESSED tree, so pose consumed a different mask set than identity was
assigned on -- and every stage still reported success. Measured with
`scripts/qc/audit_fly_assignment.py` over all 160 bouts:

| tree | fly-frames where keypoints sit on the OTHER fly's mask | bouts >10% wrong |
|---|---|---|
| old (deleted 2026-08-10) | **19.7%** | 57 of 160 |
| current `pose/` | **2.3%** | 12 of 160 |

The residual 2.3% is concentrated in cameras already flagged in
`suspect_cameras` (reflection tracking); no bout has a whole-bout identity swap.

Per bout-fly in `pose/bouts/bout_<NNNNN>/fly<N>/`:

| file | contents |
|---|---|
| `outputs.h5` | `kp3d_mm (T,50,3)`, `qpos (T,93)`, `root_se3 (T,7)`, `scale (T,)`, `mesh_mm (T,600,3)`, `kp_names` |
| `kp2d.npz` | `kp2d (T,7,50,2)`, `conf (T,7,50)` |
| `kp3d.npz` / `kp3d_filt.npz` | triangulated 3D + temporally filtered |
| `stac_ik.h5` | raw STAC IK solution |
| `qc.json`, `qc_perframe.npz` | per-bout and per-frame QC |

`sam3_masks.npz` per bout carries `packed`, `valid`, `centroids`, `cameras`,
`in_frame` (out-of-FOV vs in-frame-and-missed), `gap_repair`, `suspect_cameras`,
`sex_meta`, `sync`.

## 2.2 Caveats a recipient must know

- **10 of 160 bouts fail reconstructability** (a fly observable by <4 cameras
  for >20% of frames) and should be excluded:
  S0 #13,15,22,26,27; S1/12_11_50 #2,5; S1/15_25_51 #6,30; S1/17_28_34 #7.
  `scripts/qc/bout_reconstructable.py --apply` marks them; `combine_ik_outputs.py
  --skip-failing` drops them. No `EXCLUDED.json` has been written yet.
- **2 bouts carry a reflection-tracking camera** (S0 #8, #26, both Cam2012631);
  flagged in `suspect_cameras`.
- **Identity is human-verified for every bout**: 132 confirmed + 27 swapped +
  1 marked bad, in `<processed>/courtship/id_review.json`, with a `sex.json` per
  bout. 99 were carried over from the review done against the old tree
  (`scripts/qc/remap_review_to_new_masks.py` explains why a remap rather than a
  copy was required -- the GUI crops to keypoints, so the reviewer judged the
  animal the KEYPOINTS were on, not the animal in mask slot k); the other 61
  were re-reviewed. In the combined h5 an unverified bout would report
  `male_fly = -1` rather than the mask-area heuristic's guess.
- **Tracking quality is per FLY, and the female is the failure mode.** Human
  review of all 160 bouts reported bad female tracking in several bouts with the
  male fine throughout; `scripts/qc/per_fly_quality.py` quantifies it and gates
  on LOO reprojection > 30 px (or a NaN reproj median): **35 of 160 females fail,
  2 of 160 males**. In all 35 the male is clean, so `combine_ik_outputs.py
  --skip-unusable` drops the FLY, not the bout — which keeps 26 good male fits
  that a per-bout exclusion would have discarded. The gate was set against
  rendered frames (`figures/2026-08-10-per-fly-quality/`), not the distribution:
  a 25 px per-camera-reproj cut looked defensible from the male distribution
  alone (male max is 22.6 px) but rejected a 28 px female whose keypoints sit
  correctly on the animal. LOO separates the real failures because it measures
  cross-camera disagreement, which is what a keypoint stranded on a wall
  reflection produces.
- **Known detector weakness**: on frames where the two flies overlap it is ~6%
  worse than the old checkpoint (and 41% better when apart). JARVIS's
  distractor-fly RGB gray-fill is now wired in (`predict_bout_2d(distractor_masks=...)`),
  but measurement showed it was NOT the cause of the mis-assignment — with
  correct masks, fly-to-mask assignment is right with gray-fill both on and off.

## 2.3 Not processed

Free-running has **not** been run through this stack. The `run_bout` driver is
shared (`num_animals=1` is the only branch point) but no free-running recording
has been processed with the regenerated masks or the new detector.
