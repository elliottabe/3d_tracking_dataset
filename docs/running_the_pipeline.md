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

## Install and run on a new cluster

Everything below was verified against the working `3d_tracking` env on Hyak,
2026-09-06. For the full list of what is Hyak-specific and needs adapting
(Slurm partitions, absolute `/gscratch` paths, GPU sizing, …), see
`docs/portability-checklist.md` — this section only covers what to install
and how.

### 1. Clone and submodules

```bash
git clone --recurse-submodules <repository-url>
cd 3d_tracking_dataset
# if already cloned without --recurse-submodules:
git submodule update --init --recursive
```

This checks out **two** submodules (`.gitmodules`): `stac-mjx` (STAC IK
solver) and `third_party/JARVIS-HybridNet` (loaded only so `ProjectManager`
can hand a cfg to `get_repro_tool`; your session's own `calibration/` dir is
still the actual geometry source). `git submodule status` should show both
initialized (a leading `-` means uninitialized, `+` means checked out at a
commit different from what's pinned).

### 2. Sibling body-model clone

The fly body model is **not** in this repo (`models/` is intentionally
empty). Clone it next to this checkout:

```bash
cd ..
git clone -b elliottabe/3d_tracking_paper <fruitfly_body_models-url> fruitfly_body_models
cd 3d_tracking_dataset
```

`paths.body_model_dir` (in both `configs/paths/<cluster>.yaml` and
`third_party/jarvis_jax/configs/paths/<cluster>.yaml`) must point at it —
the existing `paths=hyak` value is `${paths.project_dir}/fruitfly_body_models/`,
i.e. a sibling of this repo, not a path inside it.

If you will use `anatomy=v2_3`, also build its derived model once per fresh
checkout (it's gitignored in `fruitfly_body_models`, so this doesn't persist
across clones):

```bash
python scripts/models/build_v2_3_ik_model.py
```
Skipping this makes `anatomy=v2_3` die at `ParseXML` — it is not optional for
that anatomy.

### 3. Conda environment

```bash
conda env create -f environment.yml   # python, numpy, mesa/glfw/egl, ffmpeg, rapids, uv
conda activate 3d_tracking
uv pip install -r requirements.txt    # jax/flax/mujoco/torch/... (unpinned in requirements.txt)
uv pip install -e third_party/jarvis_jax   # NOT in requirements.txt; required separately
```

`requirements.txt` installs `stac-mjx` editable itself (`-e ./stac-mjx`), but
**not** `third_party/jarvis_jax` — that install is a separate step this repo
had never documented before this pass. Skipping it makes every
`python -m jarvis_jax....` / `-m jarvis_jax.scripts.train_mvq` invocation
fail at import.

Nothing in this repo pins exact versions of the packages that actually
matter (`environment.yml` only pins `python`/`numpy`/`rapids`;
`requirements.txt` leaves `jax`, `flax`, `mujoco`, `torch`, etc. unpinned).
`environment-3d_tracking.yml` (repo root) records the exact versions verified
working together on Hyak 2026-09-06 — **most importantly, jax 0.11.x requires
flax>=0.12.8; an older flax breaks every import in this repo, not just
mvq/jaxls code.** If your fresh install lands on an incompatible flax, pin to
the versions in that file. Verify with:

```bash
python -c "import jax, flax; print(jax.__version__, flax.__version__)"
```

### 4. SAM3 (separate repo, not vendored here)

SAM3 masking is a **third clone**, entirely outside this repo, and was not
documented anywhere before this pass:

```bash
git clone https://github.com/facebookresearch/sam3 <sam3-clone>
cd <sam3-clone> && uv pip install -e . && cd -
```
`third_party/jarvis_jax/scripts/sam3_masks.py` imports `sam3` — this install
is what satisfies that import; there is no `sam3.jarvis_root`-style config
key pointing at the SAM3 clone itself (only at the JARVIS project dir it
loads for calibration plumbing). Pass `--no-sam3-compile` if the target
machine doesn't support the compile path (see `jax-flax-env-pairing`-style
notes in project memory for the hf_hub/tqdm import-order fix already applied
in `sam3_masks.py`/`sam3_driver.py`).

### 5. Runtime environment (every GPU job)

```bash
module load <your-cluster's-CUDA-module>   # see WHY below; name/version is Hyak's, will differ
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH
unset JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export TF_GPU_ALLOCATOR=cuda_malloc_async
```

Why each line, so you can adapt it rather than copy it blind:
- **`module load cuda/12.9.1` (Hyak name)** — batch/non-interactive compute
  nodes here do not expose `libcuda.so` to JAX without an explicit CUDA
  module load; an interactive node or a cluster whose default environment
  already has the driver on the loader path may not need this at all. Verify
  with `python -c "import jax; print(jax.devices())"` — pipeline jobs log
  `[gpu-check] jax sees N GPU(s)` at startup and fail fast if N=0, which is
  the symptom of this being needed and missing.
- **`LD_PRELOAD=.../libstdc++.so.6`** — a libstdc++ ABI mismatch between the
  conda env's bundled copy and the system/CUDA-module one otherwise breaks
  native extensions (mujoco-warp, torch) at import; preloading the conda
  copy forces the compatible ABI.
- **`unset LD_LIBRARY_PATH`** — the pose stage wants JAX's own bundled CUDA
  wheels; a stale `LD_LIBRARY_PATH` (e.g. left over from the SAM3 stage,
  which sets one to reach `nvidia/cu13/lib`) shadows them. The SAM3 stage is
  the one exception that WANTS `LD_LIBRARY_PATH` set — see
  `docs/transferring_the_pipeline.md` §1.4.
- **`unset JAX_PLATFORMS`** — some shells/launchers (here, `sbatch
  --export=ALL`) inherit a stray `JAX_PLATFORMS=cpu` from the login
  environment, which silently forces JAX onto CPU with no error — just a
  very slow, very quiet failure to use the GPU at all.
- **`XLA_PYTHON_CLIENT_MEM_FRACTION` / `TF_GPU_ALLOCATOR`** — tuned against
  Hyak's L40S (48 GB) / A40 / A100 / H200 memory sizes; re-check for OOM or
  wasted headroom on a different GPU's VRAM.

### 6. Model weights and checkpoints

None of these are in git. See `docs/portability-checklist.md` §5 for the
full table (paths + which config key points at each); in short you need to
copy, per fresh cluster:
- SAM3 weights (~6.5 GB) into a directory you'll set as `$HF_HOME`.
- A Hugging Face token with **DINOv3 access requested and granted**
  (`facebook/dinov3-vitb16-pretrain-lvd1689m` is gated) saved at
  `$HF_HOME/token`. `huggingface_hub` reads that file automatically as long
  as `HF_TOKEN` is unset or empty in the environment — do not print the
  token's contents, just confirm the file exists:
  `test -f "$HF_HOME/token" && echo present`.
- The ViTPose 2D detector checkpoint (326 MB,
  `jax_vitpose_runs/v4_8gpu_20260808/final` on Hyak) and, for the mvq route,
  the `jax_mvq_runs/...` checkpoints named in `configs/mvq/p3a.yaml` /
  `p3b.yaml` and `third_party/jarvis_jax/configs/train/mvq_v2.yaml`.

### 7. Add a `paths=` group for the new cluster (both trees)

There are **two separate** Hydra `paths` groups: `configs/paths/` in this
repo, and `third_party/jarvis_jax/configs/paths/` in the vendored package —
a new cluster needs a new YAML in **both**, following the shape of
`configs/paths/workstation.yaml` (main repo) and
`third_party/jarvis_jax/configs/paths/workstation.yaml` (jarvis_jax) — those
are the non-Hyak templates, not verified end-to-end, but structurally
correct. Point `base_dir` (main repo) / `data_root` + friends (jarvis_jax) at
wherever the new cluster's `Video_recordings/` and `processed/` data root
will live (see `docs/portability-checklist.md` §7 for the exact directory
layout expected under it), and `body_model_dir` / `project_dir` at the
sibling `fruitfly_body_models` clone from step 2. Then pass `paths=<name>`
instead of `paths=hyak` on every command below.

### 8. Sanity checks before running anything real

```bash
python -c "import jax, flax; print(jax.__version__, flax.__version__)"   # expect 0.11.x / 0.12.8+
git submodule status                                                     # both submodules initialized
ls ../fruitfly_body_models                                                # sibling clone present
test -f "$HF_HOME/token" && echo "HF token present"                       # do not print it
```

Once these pass, continue with "Prerequisites (per recording)" below for the
SAM3→pose pipeline, or see `docs/benchmark/2026-09-mvq/v2-notes.md` for the
exact mvq v2 training launch command (same runtime env as step 5, plus
`HF_HOME`/`HF_TOKEN=` and `cd third_party/jarvis_jax && python -u -m
jarvis_jax.scripts.train_mvq model=mvq train=mvq_v2 paths=<name> ...`).

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
(`kp2d.npz`, `kp3d.npz`, `kp3d_filt.npz`, `scale.json`, `offsets_fly<f>.h5`,
`stac_ik.h5`, `outputs.h5`, plus a `DONE` marker per bout). So a
preempt+requeue resumes at the last completed stage rather than restarting.
There is no mid-stage checkpoint, so an interrupted stage restarts from its
own beginning — minutes, for per-bout IK.

Note the flip side: those same markers mean a re-run **skips completed work**.
To genuinely refit (e.g. after a `scale.json` change) the downstream
artifacts must be removed first — `DONE`, `stac_ik.h5`, `outputs.h5`,
`qpos_refined.npz`, `qc*.{json,npz}`, and the recording-level
`offsets_fly0.h5`/`offsets_fly1.h5` (+ their `.json` provenance), which are
scale-dependent. Offsets are per fly, pooled over every triangulated bout of
that fly on high-confidence frames only (`stac.offsets_min_conf`, default 0.7)
and fit with temporal smoothing off; a run with an early, few-bout offsets file
can be refit the same way once more bouts are triangulated (the `.json` says
how many bouts backed it). Keep `kp2d/kp3d/kp3d_filt.npz` (triangulation does
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

## Fly identity: reprocessing the courtship set with male = fly1

The 160 reviewed courtship bouts have a canonicalized mask set at

    /gscratch/portia/eabe/data/Johnson_lab/processed/courtship_canonical/<Session>/<recording>/sam3_masks

built by `scripts/canonicalize_sam_masks.py` from the human ID review
(`processed/courtship/id_review_reviewed_20260829.json`). In it, **mask slot 1
is the male in every bout**, and each npz's `sex_meta` records the review file,
its sha256, the original slot and whether a swap was applied. `run_bout` writes
`fly{slot}`, so pointing the pipeline at this set makes `fly1` the male
throughout with no other change:

```bash
python scripts/run_bout.py recording=session1 \
    recording.timestamp="2026_04_02_16_21_32" \
    'recording.predictions_dir=${paths.processed_root}/courtship_canonical/${recording.name}/${basename:${recording.session_dir}}/sam3_masks'
```

Do NOT point `predictions_dir` at the video tree's `Predictions_3D_sam3*`
instead: measured over all 160 bouts, 25 of them (21 of Session0's 30) have
their two fly slots reversed relative to the reviewed set, so identity would be
inverted on those bouts while every stage reported success.

`jarvis_jax.tracking.sexing.canonicalize_bout` reads that `sex_meta` and treats
it as the authority, falling back to the wing-song CV heuristic only when no
human decision is present; `sex.json` records which path was taken
(`authority`) and whether the heuristic agreed (`heuristic_agrees`).

Regenerate the acceptance figure with:

```bash
python scripts/viz/canonical_sex_check.py \
    --bouts Session1/2026_04_02_16_21_32/bout_00009,Session1/2026_04_02_16_39_56/bout_00018,Session1/2026_04_02_16_56_37/bout_00007,Session0/2025_10_20_13_20_04/bout_00028 \
    --out figures/2026-09-02-mask-canonicalization
```

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

## Session pipeline (mvq route)

`scripts/slurm/mvq_session_pipeline.sh` is the whole-session driver for the
**mvq** route (typed-slot mvq lift instead of ViTPose+DLT). It composes
`scripts/slurm_bout_array.py` once per recording — it does not reimplement any
stage — and adds one session-level `collect` job at the end:

```
lift[ts] -> precompute[ts] -> ik[ts] -> aggregate[ts]     (one chain per recording)
                                           aggregate[*] -> collect
```

```bash
# Session0 (one recording, 30 bouts)
scripts/slurm/mvq_session_pipeline.sh \
    --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0

# Session1 (10 of 14 recordings have masks; 130 bouts)
scripts/slurm/mvq_session_pipeline.sh \
    --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1
```

Options: `--processed <root>` (default
`/gscratch/portia/eabe/data/Johnson_lab/processed/courtship`), `--recording
<cfg>` (default: the session folder name lowercased — `Session0` → `session0`),
`--run-name` (default `pose_mvq_p3b`), `--mvq-config` (default `p3b`),
`--slurm` (default `ckpt_all`), `--only <ts>`, `--local-gpus N
[--max-local-gpus N]`, `--no-collect`, `--dry-run`. Exit status is non-zero if
any recording's chain failed to submit; the others are still submitted.

**What each stage depends on**

| stage | job | gated on | what it does |
|---|---|---|---|
| lift | `mvqlift_<Session>` array, one task per bout | — (or SAM3) | `scripts/mvq_lift_bout.py` → `bouts/bout_*/fly*/{kp2d,kp3d}.npz` + `sex.json`, `mvq_meta.json` |
| precompute | `ctprecomp_<Session>` | `afterok:<the WHOLE lift array>` | per-fly body scale (`scale.json`) + per-fly marker offsets (`offsets_fly<f>.h5`) from a **recording-wide** sample — this is why the lift array runs first: pooling one bout gave a 15.5 %-low scale (the scale-from-first-bout defect) |
| ik | `ctjax_<Session>` array, one task per bout | `afterok:precompute` | `scripts/run_bout.py` — filter → STAC IK → bridge → outputs/QC |
| aggregate | `ctagg_<Session>` | `afterok:<the WHOLE ik array>` | `bouts/*/fly*/qc.json` → `<run>/qc/session_qc.json` + dashboard |
| collect | `mvqcollect_<Session>` (CPU, `--gpus=0`) | `afterok:<every recording's aggregate>` | `scripts/session_collect.py` → `<processed>/<Session>/<run-name>_session_summary.{json,md}` |

**Inputs required** (per recording, all pre-existing — this route creates none
of them):

1. **SAM3 masks** at
   `<processed>/<Session>/<ts>/sam3_masks/bout_*/sam3_masks.npz`. On the mvq
   route the masks are an *input* (the lift places its crops from their
   centroids and never re-segments), so recording discovery is exactly "has
   masks"; a recording without them is skipped with a message.
2. **ID review** in those masks — the human fly0/fly1 assignment. Without it
   `mvq.identity: mask` falls back to the model's sex head for that bout, which
   is recorded in `mvq_meta.json: identity_resolved` and shows up in the collect
   summary.
3. **Bouts CSV** in the video recording dir. When
   `courtship_bouts_unified_summary.csv` is missing or a broken symlink
   (Session0's points into a deleted `Predictions_3D_*`), the driver writes a
   `fly_id`-less copy of `courtship_bouts_fly0_summary.csv` into the run root
   and passes it as `recording.bouts_csv` — the per-fly CSVs carry
   `fly_id = "<Session>/<ts>_fly0"`, which `bout_start_frame` (matching the bare
   `<Session>/<ts>` tag) would never find.

`--local-gpus N` runs the lift as N local workers on the current GPU node (one
GPU each, bouts split round-robin) and then submits the chain with
`--mvq-lift skip`, which *verifies* the on-disk lift (gate string + `sex.json`)
rather than trusting it. N is capped at 4 (`--max-local-gpus` to raise
deliberately): 8 concurrent JAX pipelines took a 128 GB-cgroup node to
`CUDA_ERROR_UNKNOWN`. The worker loop lives in `scripts/slurm/mvq_local_lift.sh`,
shared with `scripts/slurm/mvq_p3a_campaign.sh`. It refuses to start while a
training process matching `--guard-pattern` is alive.

`--dry-run` prints every sbatch script (per-recording ones via the submitter's
own `--dry-run`) plus the dependency graph and submits nothing.

**Session summary.** The collect job writes
`<processed>/<Session>/<run-name>_session_summary.json` and `.md`: one row per
recording plus a TOTAL, with bouts, bout-flies solved, frames, LOO median/p90
px, reproj median px, IoU hard median, female/male missing %,
containment-dropped %, and bouts with unsolvable flies — plus the list of
bout-flies with no `stac_ik.h5`. It can be run by hand at any time:

```bash
python scripts/session_collect.py --session-name Session1 \
    --processed /gscratch/portia/eabe/data/Johnson_lab/processed/courtship \
    --run-name pose_mvq_p3b
```

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
