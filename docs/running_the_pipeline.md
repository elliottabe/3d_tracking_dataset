# Submitting the courtship / free-running SAM3 → keypoint → IK pipeline

This is the per-recording pipeline that turns raw multi-camera video into 3D
keypoints and an articulated IK fit:

```
SAM3 masks (--array over bouts)  ->  precompute (offsets fit-once)  ->  jax array (--array over bouts)  ->  aggregate
   (PyTorch, per-bout id masks)       (run_bout, bout 1)      (ViTPose 2D -> triangulate ->        (session QC
                                                                        smooth -> STAC IK -> polish ->        dashboard)
                                                                        outputs.h5 + sidebyside viz)
```

**One launcher call processes ONE recording.** Each recording becomes its own
dependency-chained set of SLURM jobs, and the SAM3 and jax stages are `--array`
jobs over that recording's bouts (per-bout parallelism). A whole session (many
recordings) is a loop over recordings — use the helper script.

The two assays differ only in:
- **num_animals** — courtship = 2, free-running = 1 (set by the recording config).
- **bout-summary file name** — `courtship_bouts_unified_summary.csv` vs
  `free_running_bout_summary.csv`.

---

## Prerequisites (per recording)

Each recording dir under `.../Video_recordings/<assay>/<Session>/<timestamp>/`
must contain:
- the 7 camera videos `Cam*.mp4`,
- a `calibration/` dir (`Cam*.yaml`),
- the **bout-summary CSV** with columns `fly_id,bout_idx,start_frame,end_frame,...`
  where **`fly_id` == `<Session>/<timestamp>`** (the "session tag" `parse_bouts`
  filters on). If a summary was generated under a differently-named staging dir,
  its `fly_id` prefix must be rewritten to match (e.g.
  `Session1_bouts_04172026/<ts>` → `Session1/<ts>`).

A **recording config** (`configs/recording/<name>.yaml`) sets the paths + rig +
`num_animals`. Existing ones:
- `session0` — courtship Session0 (single recording, num_animals=2).
- `session1` — courtship Session1 (multi-recording, num_animals=2).
- `free_running_session11` — free-running Session11 (multi-recording, num_animals=1).

To target a new recording, override `recording.session_dir` (a full path — never
misparsed; a bare `recording.timestamp=2026_..._..` is read by OmegaConf as an
**int**, so always quote it or override the whole `session_dir`).

---

## Environment (handled by the sbatch scripts — FYI)

The generated sbatch scripts already do this; listed so you know what matters:
```bash
micromamba activate 3d_tracking
module load cuda/12.9.1                 # batch nodes need this to expose libcuda to JAX
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
unset LD_LIBRARY_PATH                   # JAX uses its bundled CUDA wheels
unset JAX_PLATFORMS                     # else a stray JAX_PLATFORMS=cpu (inherited via
                                        # sbatch --export=ALL) silently runs on CPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```
When you invoke a launcher yourself, prefix with `env -u JAX_PLATFORMS` so a `cpu`
value from your shell doesn't propagate. Each jax job logs `[gpu-check] jax sees N
GPU(s)` at startup and **fails fast** if N=0 (so a bad node requeues instead of
grinding on CPU).

---

## Single recording

`slurm_bout_array.py` processes one recording (SAM3 → precompute → jax →
aggregate). It discovers bouts from `bout_*` dirs under `recording.predictions_dir`,
so **bootstrap those dirs first** from the summary:

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
D=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04
# bootstrap bout dirs so the launcher can discover them
for i in $(tail -n +2 "$D/courtship_bouts_unified_summary.csv" | awk -F, 'NF{print $2}'); do
    mkdir -p "$D/Predictions_3D_sam3/bout_$(printf %05d $i)"
done
env -u JAX_PLATFORMS python scripts/slurm_bout_array.py \
    recording=session0 recording.session_dir="$D" \
    outputs.out=/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts
```

Add `--dry-run` to print the scripts + dependency chain without submitting.
`reuse_masks` + per-bout stage checkpoints make re-running idempotent (only
unfinished work is redone) — safe to resubmit after a partial/preempted run.

---

## Whole session (many recordings) — the helper

`scripts/run_free_running_session.sh` loops every recording in a session dir:
bootstraps its bout dirs and submits its chain. It is assay-agnostic via
`--recording` / `--summary`.

**Free-running Session11 (single fly):**
```bash
scripts/run_free_running_session.sh \
    --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/free_running/Session11
# (defaults: --recording free_running_session11 --summary free_running_bout_summary.csv
#            --out /gscratch/portia/eabe/data/Johnson_lab/free_running)
```

**Courtship Session1 (two flies):**
```bash
scripts/run_free_running_session.sh \
    --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 \
    --recording session1 --summary courtship_bouts_unified_summary.csv \
    --out /gscratch/portia/eabe/data/Johnson_lab/courtship
```

Add `--dry-run` to preview all recordings + per-recording bout counts before
submitting. Recordings without the summary are skipped (logged). Each recording
writes to `<out>/<SessionName>_<timestamp>_bouts/`.

**Test one recording first** (recommended for a new assay): run the single-recording
command above for one timestamp, confirm it looks good, then fire the whole-session
helper (idempotent, so it skips the already-done one).

---

## Outputs (per bout, per fly)

Under `<out>/bouts/bout_<idx:05d>/fly<f>/`:
- `kp2d.npz` (ViTPose 2D), `kp3d.npz` (raw DLT 3D), `kp3d_filt.npz` (smoothed),
- `stac_ik.h5` (STAC fit), `qpos_refined.npz`, `outputs.h5` (qpos, FK'd
  `kp3d_mm` + `mesh_mm`, `root_se3`),
- `qc.json` / `qc_perframe.npz`,
- `overlays/<cam>_reproj.mp4` and `sidebyside.mp4` (H.264; video+mask+2D skeleton |
  MuJoCo IK render).

SAM masks + a stacked `maskvid_bout<N>.mp4` QC overlay land next to
`sam3_masks.npz` in `recording.predictions_dir`. Session QC dashboard: `<out>/qc/`.

---

## Monitoring

```bash
squeue -u $USER                                   # queued/running jobs
grep -A2 gpu-check <out>/slurm-precompute-*.out   # confirm "jax sees 1 GPU"
sacct -j <jobid> --format=JobID,State,ExitCode    # per-array-task status
```

---

## Anatomy: `anatomy=v1` (default) works; `anatomy=v2_muscles` does NOT yet

`anatomy=v1` is the default and is what everything above runs on.

`anatomy=v2_muscles` is **not drop-in** today. The keypoint side is fully
compatible — v1 and v2_muscles have the **same 50 keypoints** (identical names +
order), so the ViTPose detector, triangulation, and STAC (which uses
`anatomy.mjcf_path`) would work. But three things block a full run:

1. **The v2.1 model XML isn't at the configured path.** `anatomy/v2_muscles.yaml`
   points `mjcf_path` at `${body_model_dir}/fruitfly_v2.1/fruitfly_v2.1_muscles.xml`,
   but there is no `models/fruitfly_v2.1/` in the repo (the v2.1 model lives under
   `fruitfly_body_models/`). STAC can't load it until it's present (e.g. symlink
   `models/fruitfly_v2.1` → the real dir).
2. **The silhouette / outputs stages are hardcoded to v1.**
   `configs/silhouette/default.yaml` sets `xml: .../fruitfly_v1/fruitfly_v1_free.xml`
   and `mesh_npz: .../fruitfly_cse/fly_v1_visual_canonical_wings.npz`. Stage D
   (polish) and Stage E (`build_fly_outputs`) FK the STAC `qpos` through
   `cfg.silhouette.xml`. If `anatomy=v2_muscles`, `qpos` has the v2 DOF layout but
   the FK uses the v1 model → shape mismatch / garbage. It must be overridden to the
   v2 model: `silhouette.xml=<v2_muscles xml>`.
3. **No v2 CSE silhouette mesh exists** — only `fly_v1_*` meshes are in
   `models/fruitfly_cse/`. `outputs.h5`'s `mesh_mm` + the overlay/sidebyside renders
   need a mesh matching the v2 model; a `fly_v2_*` CSE mesh would have to be built
   (or mesh outputs disabled).

**To actually run v2_muscles** you would: (a) make the v2.1 XML resolvable under
`models/`, (b) set `silhouette.xml=${anatomy.mjcf_path}` (tie it to the anatomy) or
override it per run, and (c) provide a v2 CSE mesh (or skip the mesh outputs). The
cleanest permanent fix is to default `silhouette.xml` to `${anatomy.mjcf_path}` so
the FK model always matches the STAC model. Until then, run with `anatomy=v1`.
