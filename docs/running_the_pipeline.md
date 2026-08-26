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

## Which partition the jobs go to

Every launcher defaults to `--slurm ckpt_all` (`configs/slurm/ckpt_all.yaml`).
That is `ckpt-all` plus a GPU constraint:

```
#SBATCH --partition=ckpt-all
#SBATCH --constraint=h200|a100|l40s|l40|a40
#SBATCH --requeue
```

**Why ckpt-all rather than ckpt-g2.** `ckpt-g2` only reaches l40/l40s/h200.
`ckpt-all` adds the a40 (32 nodes) and a100 (8) pools, roughly doubling the
eligible nodes. That matters when the QUEUE is the bottleneck, not the GPU:
measured 2026-08-26, both partitions had the same *idle* nodes, but the
ckpt-g2 chain sat `PENDING (Priority)` with a start estimate 34 min out while
the same chain on `ckpt-all` started immediately on an a40.

**Why the constraint is not optional.** Bare `ckpt-all` also matches GPUs too
small for this model — rtx6k 24GB, 2080ti 11GB, p100 16GB — and the CPU-only
`n[...]` nodes. Those OOM or cannot run at all. The constraint admits only
cards that fit, ordered biggest-first for readability (slurm treats an OR
constraint as a set, not a preference), so a job may land on the slowest
usable card (a40, ~0.70 TB/s vs h200 ~4.8 TB/s). Waiting for a faster card is
usually the worse trade for per-bout work.

Use `--slurm ckpt_g2` for the old behaviour (no constraint, l40/l40s/h200
only) or `--slurm gpu_l40s` for the non-preemptible partition. Training
(`slurm_train_vit.py`) still defaults to `ckpt_g2`; pass `--slurm ckpt_all`
if you want the wider pool there too.

### Preemption

`ckpt*` partitions are preemptible. Jobs are submitted with `--requeue`, each
array task writes to a fixed per-task directory, and
`jarvis_jax.tracking.resume` skips stages whose artifact already exists
(`kp2d.npz`, `kp3d.npz`, `kp3d_filt.npz`, `scale.json`, `offsets.h5`,
`stac_ik.h5`, `outputs.h5`, plus a `DONE` marker per bout). So a
preempt+requeue resumes at the last completed stage rather than restarting.
There is no mid-stage checkpoint, so an interrupted stage restarts from its
own beginning — minutes, for per-bout IK.

Note the flip side: those same markers mean a re-run **skips completed work**.
To genuinely refit (e.g. after a `scale.json` change) the downstream
artifacts must be removed first — `DONE`, `stac_ik.h5`, `outputs.h5`,
`qpos_refined.npz`, `qc*.{json,npz}`, and the recording-level `offsets.h5`,
which is scale-dependent. Keep `kp2d/kp3d/kp3d_filt.npz` (triangulation does
not depend on scale).

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

## NewBouts curated bout set (free-running, batch IK route)

`processed/free_running/NewBouts/<ts>/` holds, per recording, the
workstation's 3D keypoint predictions (`data3D.csv`), `info.yaml`, and a
manually curated `running_bouts_summary.csv` (schema
`bout,start_frame,end_frame,...,status`). These run on the OLD batch route —
no video, masks, or 2D detection needed:

```bash
# 1. materialize Predictions_3D_newbouts/ per recording (accepted bouts only,
#    bout -> bout_idx, fly_id from info.yaml recording_path):
python scripts/data/convert_newbouts_summary.py \
    --root /gscratch/portia/eabe/data/Johnson_lab/processed/free_running/NewBouts
# 2. one array task per recording (preprocess -> STAC -> postprocess), then
#    a dependent combine/pack/audit finalize:
python scripts/slurm_dir_array.py \
    --base-dir /gscratch/portia/eabe/data/Johnson_lab/processed/free_running/NewBouts \
    --anatomy v1 --postprocessing default --partition ckpt-g2 \
    --out-h5 <packed-output.h5>   # MUST override: the default targets the
                                  # v2_3 walking reference dataset
```

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

## Anatomy: `v1` (default) and `v2_3`

`anatomy=v1` remains the default for the per-bout SAM3 pipeline.

`anatomy=v2_3` is supported on the **old batch route** (`batch_process_predictions`
→ `batch_run_stac` → `batch_postprocess_predictions` → `combine_data`), which is
being used to build `Fruitfly_v2_3_walk_1000hz_interp_padded.h5`.

`anatomy=v2_3`'s `mjcf_path` points at a **derived, adhesion-free** model,
`models/fruitfly_v2_3_ik/fruitfly_v2_3_ik.xml`, generated by
`scripts/models/build_v2_3_ik_model.py` from the shared
`fruitfly_v2.3/fruitfly_muscles_warp.xml`. That shared model is never edited
in place. The derived model is generated with two properties:

1. Its 8 `<adhesion>` actuators (`mjTRN_BODY` transmissions) are stripped.
   Plain `mjx.put_model` — used by `stac_mjx/utils.py:26` and
   `postprocess_stac_data.py` — does not implement `mjTRN_BODY` and raises
   without this. The actuators stay enabled in the shared model on purpose:
   `mjx.put_model(..., impl="warp")` accepts all eight once
   `noslip_iterations` is 0, which the env always sets on the warp backend
   (`fly_mimic/envs/fruitfly/base.py:578-582`), warp is what this project
   trains and rolls out on, and removing them upstream changes `nu` 272→264
   and breaks every checkpoint trained with adhesion. So the shared model is
   left alone and IK/FK uses the derived copy instead.
2. The 50 `tracking[<KP_NAME>]` sites are (re-)inserted from
   `configs/anatomy/v2_3.yaml`. STAC creates its own marker sites at runtime,
   but preprocessing reads these from the XML to fix keypoint column order
   and the rest-pose reference, and postprocess reads them for the
   egocentric outputs. Without them neither stage errors — they silently
   produce degenerate output.

This is safe because the two models are kinematically identical: same
`nq=101/nv=100/nbody=74/njnt=95`, identical joint/body/site ordering, and FK
from a random qpos agrees to exactly 0.0 — only `nu` differs (272 vs 264).
The dataset stores `qpos/qvel/xpos/xquat`, none of which depend on `nu`, so
output produced with the derived model stays valid for checkpoints trained
against the 272-actuator model. Re-run
`scripts/models/build_v2_3_ik_model.py` whenever the upstream v2.3 model
changes; it is idempotent.

`anatomy=v2_muscles` still does NOT work on the per-bout pipeline: it needs a v2
CSE silhouette mesh (only `fly_v1_*` meshes exist in `models/fruitfly_cse/`) and
`configs/silhouette/default.yaml` hardcodes the v1 XML. The batch route used for
v2_3 has no silhouette stage, which is why it is unaffected.

See `docs/specs/2026-08-03-v2_3-walking-reference-dataset-design.md`.
