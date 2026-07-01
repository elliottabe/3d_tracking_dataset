# Parallel V3-masked Session1 Prediction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run full 3D pose predictions on the 10 courtship Session1 recordings that have bout CSVs, via a SLURM `--array` job that runs the original JARVIS-HybridNet SAM3 shard pipeline with the `unified_V3_masked` model.

**Architecture:** A manifest generator produces a stable `recording → bouts_csv` mapping (one line per recording); a SLURM array submitter reads line `$SLURM_ARRAY_TASK_ID` and runs `predict3D_multianimal_shard.py` on that recording, with per-recording tasks running concurrently (throttled) and each task internally sharding its bouts across GPUs.

**Tech Stack:** Bash, SLURM (`sbatch --array`), micromamba (`jarvis` env), JARVIS-HybridNet + SAM3 (PyTorch), `ckpt-g2` partition.

## Global Constraints

- Run from the **repo copy** of JARVIS: `/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet` (this is where `unified_V3_masked` lives; the Github runtime root does not have it).
- Conda env: `jarvis` (micromamba). Env setup must match `submit_courtship_repredict.sh`: `module load cuda/12.9.1 gcc/12`; `source ~/.bashrc`; `micromamba activate jarvis`; `unset LD_LIBRARY_PATH`.
- Model project: `unified_V3_masked`, `--num_animals 2`.
- Exact weight paths (verified 2026-07-01):
  - CenterDetect (loaded-but-unused; from Github root): `/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet/projects/red_data_unified/models/CenterDetect/phase4_center_ft/EfficientTrack-medium_final.pth`
  - KeypointDetect (V3): `<REPO>/projects/unified_V3_masked/models/KeypointDetect/Run_20260619-185121/EfficientTrack-large_final.pth`
  - HybridNet (V3, newer run): `<REPO>/projects/unified_V3_masked/models/HybridNet/Run_20260620-173554/HybridNet-large_final.pth`
- GPUs per task: `--num_gpus 4` (each shard = 2 GPUs → 2 shards/recording). Concurrency throttle: `%5`.
- Bouts root (input CSVs): `/gscratch/portia/eabe/data/Johnson_lab/courtship/Session1_bouts_04172026`.
- Video root (recordings): `/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1`.
- Do NOT run heavy prediction on the login node; submission is via `sbatch` (login-node-safe). All `DRY_RUN=1` and manifest generation are lightweight and login-node-safe.

Throughout this plan, `<REPO>` = `/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet`.

---

## File Structure

- **Create** `<REPO>/gen_session1_manifest.sh` — builds the `recording → bouts_csv` manifest, validating each recording has `calibration/` + `.mp4`s.
- **Create** `<REPO>/submit_session1_array.sh` — the SLURM array submitter; reads the manifest, supports `DRY_RUN`, submits `--array=1-N%K`.
- **Generated** `<REPO>/session1_manifest.tsv` — 10-line TSV artifact (committed for reproducibility / to document the exact mapping used).

Both scripts are co-located with the existing `submit_courtship_repredict.sh`, which they are modeled on.

---

### Task 1: Manifest generator script

**Files:**
- Create: `<REPO>/gen_session1_manifest.sh`
- Test: manual run producing `<REPO>/session1_manifest.tsv`

**Interfaces:**
- Consumes: env vars `BROOT` (default Session1 bouts root), `VID` (default Session1 video root); positional arg `$1` = output path (default `session1_manifest.tsv`).
- Produces: a TSV file, one line per recording: `<recording_dir>\t<bouts_csv>`, sorted by recording name. Exits non-zero if any mapped recording dir lacks `calibration/` or `.mp4`s.

- [ ] **Step 1: Write the script**

Create `<REPO>/gen_session1_manifest.sh`:

```bash
#!/bin/bash
# Build the recording -> bouts_csv manifest for the Session1 parallel
# prediction array job. Each line: <recording_dir><TAB><bouts_csv>.
# Fails fast if a mapped recording is missing calibration/ or videos.
set -euo pipefail

BROOT=${BROOT:-/gscratch/portia/eabe/data/Johnson_lab/courtship/Session1_bouts_04172026}
VID=${VID:-/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1}
OUT=${1:-session1_manifest.tsv}

: > "$OUT"
n=0
while IFS= read -r csv; do
  # recording name = grandparent dir of the CSV (.../<rec>/Predictions_3D_*/csv)
  rec=$(basename "$(dirname "$(dirname "$csv")")")
  recdir="$VID/$rec"
  [[ -d "$recdir" ]]              || { echo "ERROR: no recording dir: $recdir (from $csv)" >&2; exit 1; }
  [[ -d "$recdir/calibration" ]] || { echo "ERROR: no calibration/ in $recdir" >&2; exit 1; }
  ls "$recdir"/*.mp4 >/dev/null 2>&1 || { echo "ERROR: no .mp4 in $recdir" >&2; exit 1; }
  printf '%s\t%s\n' "$recdir" "$csv" >> "$OUT"
  n=$((n+1))
done < <(find "$BROOT" -name courtship_bouts_unified_summary.csv | sort)

echo "wrote $n manifest lines to $OUT" >&2
```

- [ ] **Step 2: Make it executable and run it**

Run:
```bash
cd <REPO>
chmod +x gen_session1_manifest.sh
./gen_session1_manifest.sh session1_manifest.tsv
```
Expected stderr: `wrote 10 manifest lines to session1_manifest.tsv`

- [ ] **Step 3: Verify the manifest is correct (10 lines, all valid)**

Run:
```bash
wc -l < session1_manifest.tsv          # expect: 10
while IFS=$'\t' read -r rec csv; do
  [[ -d "$rec/calibration" && -f "$csv" ]] && echo "OK  $(basename $rec)" || echo "BAD $rec"
done < session1_manifest.tsv
```
Expected: 10 lines, all prefixed `OK`, recording names:
`2026_04_02_12_11_50, 14_54_28, 15_25_51, 15_44_42, 16_03_48, 16_21_32, 16_39_56, 16_56_37, 17_28_34, 17_52_50`.

- [ ] **Step 4: Commit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/JARVIS-HybridNet/gen_session1_manifest.sh third_party/JARVIS-HybridNet/session1_manifest.tsv
git commit -m "feat(cse): Session1 prediction manifest generator + manifest"
```

---

### Task 2: SLURM array submitter script

**Files:**
- Create: `<REPO>/submit_session1_array.sh`
- Test: `DRY_RUN=1` run (Task 3)

**Interfaces:**
- Consumes: `<REPO>/session1_manifest.tsv` (from Task 1); env overrides `PROJECT, NUM_GPUS, THROTTLE, PARTITION, MEM, CPUS, TIME_LIMIT, CONDA_ENV, DRY_RUN, MANIFEST, ARRAY_SPEC`.
- Produces: an `sbatch --array` submission (or, under `DRY_RUN=1`, a printed plan + the resolved task-1 command). Output dirs land as `<recording>/Predictions_3D_<arrayjob>_<task>/`.

- [ ] **Step 1: Write the script**

Create `<REPO>/submit_session1_array.sh`:

```bash
#!/bin/bash
# Submit a SLURM array job to predict all in-scope Session1 recordings with the
# unified_V3_masked model via the SAM3 shard pipeline. One array task per
# recording (manifest line); each task shards its bouts across NUM_GPUS/2 GPUs.
#
# Usage:
#   DRY_RUN=1 ./submit_session1_array.sh     # print the plan (default)
#   DRY_RUN=0 ./submit_session1_array.sh     # actually submit
# Override a subset, e.g. re-run one recording:
#   DRY_RUN=0 ARRAY_SPEC=7 ./submit_session1_array.sh
set -euo pipefail

JARVIS_ROOT=/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
PROJECT=${PROJECT:-unified_V3_masked}
NUM_GPUS=${NUM_GPUS:-4}
THROTTLE=${THROTTLE:-5}
PARTITION=${PARTITION:-ckpt-g2}
MEM=${MEM:-128}
CPUS=${CPUS:-16}
TIME_LIMIT=${TIME_LIMIT:-1-00:00:00}
CONDA_ENV=${CONDA_ENV:-jarvis}
DRY_RUN=${DRY_RUN:-1}
MANIFEST=${MANIFEST:-$JARVIS_ROOT/session1_manifest.tsv}

CENTER_W=/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet/projects/red_data_unified/models/CenterDetect/phase4_center_ft/EfficientTrack-medium_final.pth
KP_W=$JARVIS_ROOT/projects/unified_V3_masked/models/KeypointDetect/Run_20260619-185121/EfficientTrack-large_final.pth
HYBRID_W=$JARVIS_ROOT/projects/unified_V3_masked/models/HybridNet/Run_20260620-173554/HybridNet-large_final.pth
PREDICT_SCRIPT=$JARVIS_ROOT/tools/predict3D_multianimal_shard.py
NODELIST="g[3090-3137]"

# --- validate inputs ---
[[ -f "$MANIFEST" ]] || { echo "ERROR: manifest not found: $MANIFEST (run gen_session1_manifest.sh)" >&2; exit 1; }
for w in "$CENTER_W" "$KP_W" "$HYBRID_W"; do
  [[ -f "$w" ]] || { echo "ERROR: missing weights: $w" >&2; exit 1; }
done
[[ -f "$PREDICT_SCRIPT" ]] || { echo "ERROR: missing predict script: $PREDICT_SCRIPT" >&2; exit 1; }
(( NUM_GPUS % 2 == 0 )) || { echo "ERROR: NUM_GPUS must be even (each shard uses 2)" >&2; exit 1; }

N=$(wc -l < "$MANIFEST")
ARRAY_SPEC=${ARRAY_SPEC:-1-${N}%${THROTTLE}}

mkdir -p "$JARVIS_ROOT/OutFiles"

echo "Project: $PROJECT | partition: $PARTITION | GPUs/task: $NUM_GPUS | array: $ARRAY_SPEC | DRY_RUN=$DRY_RUN"
echo "Manifest: $MANIFEST ($N recordings)"

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "=== task -> recording map ==="
  nl -w2 -s'  ' "$MANIFEST" | sed 's#\t# | bouts=#'
  echo
  echo "=== resolved command for array task 1 ==="
  line=$(sed -n '1p' "$MANIFEST"); rec_dir=$(cut -f1 <<<"$line"); bouts_csv=$(cut -f2 <<<"$line")
  cat <<CMD
python -u $PREDICT_SCRIPT \\
    --project $PROJECT --video_folder $rec_dir --calib_folder $rec_dir/calibration \\
    --num_animals 2 --num_gpus $NUM_GPUS --bouts_csv $bouts_csv \\
    --output_name <arrayjob>_1 \\
    --center-weights $CENTER_W --kp-weights $KP_W --hybridnet-weights $HYBRID_W \\
    --save-masks --save-clips
CMD
  echo
  echo "DRY_RUN — nothing submitted. Re-run with DRY_RUN=0 to submit."
  exit 0
fi

sbatch --array=${ARRAY_SPEC} <<EOF
#!/bin/bash
#SBATCH --job-name=s1pred
#SBATCH --partition=${PARTITION}
#SBATCH --account=portia
#SBATCH --time=${TIME_LIMIT}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gpus=${NUM_GPUS}
#SBATCH --mem=${MEM}G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o ${JARVIS_ROOT}/OutFiles/slurm-session1-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
#SBATCH --nodelist=${NODELIST}
#SBATCH --exclude=g[3107,3115,3109]
module load cuda/12.9.1
module load gcc/12
set -x
source ~/.bashrc
nvidia-smi
micromamba activate ${CONDA_ENV}
unset LD_LIBRARY_PATH
echo \$SLURMD_NODENAME

line=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${MANIFEST}")
rec_dir=\$(cut -f1 <<<"\$line")
bouts_csv=\$(cut -f2 <<<"\$line")
echo "[task \${SLURM_ARRAY_TASK_ID}] rec=\$rec_dir bouts=\$bouts_csv"

cd ${JARVIS_ROOT}
python -u ${PREDICT_SCRIPT} \\
    --project ${PROJECT} \\
    --video_folder \$rec_dir \\
    --calib_folder \$rec_dir/calibration \\
    --num_animals 2 \\
    --num_gpus ${NUM_GPUS} \\
    --bouts_csv \$bouts_csv \\
    --output_name \${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID} \\
    --center-weights ${CENTER_W} \\
    --kp-weights ${KP_W} \\
    --hybridnet-weights ${HYBRID_W} \\
    --save-masks \\
    --save-clips
EOF

echo "Submitted array ${ARRAY_SPEC}. Monitor: squeue -u \$USER | Logs: ${JARVIS_ROOT}/OutFiles/slurm-session1-*.out"
```

> **Heredoc quoting note:** submit-time variables (`${PARTITION}`, `${MANIFEST}`, `${PREDICT_SCRIPT}`, weight paths, `${NUM_GPUS}`, `${PROJECT}`, `${CONDA_ENV}`) expand now; runtime SLURM variables (`\${SLURM_ARRAY_TASK_ID}`, `\${SLURM_ARRAY_JOB_ID}`, `\${SLURMD_NODENAME}`) and the per-task locals (`\$line`, `\$rec_dir`, `\$bouts_csv`, `\$USER`) are escaped so they evaluate on the compute node.

- [ ] **Step 2: Make it executable and syntax-check**

Run:
```bash
cd <REPO>
chmod +x submit_session1_array.sh
bash -n submit_session1_array.sh && echo "syntax OK"
```
Expected: `syntax OK`

- [ ] **Step 3: Commit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/JARVIS-HybridNet/submit_session1_array.sh
git commit -m "feat(cse): SLURM array submitter for Session1 V3-masked prediction"
```

---

### Task 3: Dry-run validation

**Files:** (no new files — validates Tasks 1–2)

- [ ] **Step 1: Run the dry run**

Run:
```bash
cd <REPO>
DRY_RUN=1 ./submit_session1_array.sh
```

- [ ] **Step 2: Verify the plan output**

Expected:
- Header shows `array: 1-10%5`, `GPUs/task: 4`, `Manifest: ... (10 recordings)`.
- The `task -> recording map` lists 10 numbered recordings.
- The resolved task-1 command references `2026_04_02_12_11_50`, `--project unified_V3_masked`, `--num_animals 2`, `--num_gpus 4`, and all three verified weight paths.
- Ends with `DRY_RUN — nothing submitted.`

- [ ] **Step 3: Confirm weight-path guard works**

Run (should fail fast, proving validation is live):
```bash
KP_W=/does/not/exist DRY_RUN=1 ./submit_session1_array.sh 2>&1 | head -1
```
Wait — this won't trigger because `KP_W` is set inside the script. Instead verify the guard by temporarily pointing the manifest env at a missing file:
```bash
MANIFEST=/does/not/exist DRY_RUN=1 ./submit_session1_array.sh; echo "exit=$?"
```
Expected: `ERROR: manifest not found: /does/not/exist ...` and `exit=1`.

---

### Task 4: Single-recording smoke test

**Files:** (no new files — first real GPU submission)

**Interfaces:**
- Consumes: committed scripts + manifest.
- Produces: `2026_04_02_16_39_56/Predictions_3D_<job>_7/bout_*/{fly0.csv,fly1.csv,sam3_masks.npz}`.

Uses `--array=7` (recording `2026_04_02_16_39_56`, which previously completed with 18 bouts) as a low-risk canary that the CenterDetect-from-red_data_unified wiring and V3 weights load without error.

- [ ] **Step 1: Submit the single-index array**

Run:
```bash
cd <REPO>
DRY_RUN=0 ARRAY_SPEC=7 ./submit_session1_array.sh
squeue -u $USER
```
Expected: one array task `<jobid>_7` queued/running.

- [ ] **Step 2: Watch the log for a clean startup**

Run (replace with the actual job id):
```bash
tail -f OutFiles/slurm-session1-<jobid>_7.out
```
Expected (no crash): SAM3 model loads, `Successfully loaded project unified_V3_masked`, HybridNet/KeypointDetect weights load, CenterDetect loads from `red_data_unified/phase4_center_ft` **without a missing-file error**, then `[bout 1] frames ...` and `N flies detected`.

- [ ] **Step 3: Verify outputs once the task finishes**

Run:
```bash
d=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1/2026_04_02_16_39_56
newest=$(ls -dt $d/Predictions_3D_* | head -1); echo "$newest"
ls "$newest"/bout_*/fly0.csv "$newest"/bout_*/fly1.csv | head
python -c "import pandas as pd,glob,sys; f=sorted(glob.glob('$newest/bout_*/fly0.csv'))[0]; df=pd.read_csv(f,skiprows=2); print(f, df.shape, 'nan_frac=%.3f'%df.isna().mean().mean())"
```
Expected: both `fly0.csv` and `fly1.csv` present per bout; `fly0.csv` has many rows and a low NaN fraction (sane 3D track). Two distinct flies (fly0 vs fly1) — identity not collapsed.

- [ ] **Step 4: GATE — do not proceed to Task 5 until the smoke test produces valid per-fly CSVs.** If it fails, debug (env, weights, GPU count) before the full submit.

---

### Task 5: Full Session1 submission

**Files:** (no new files — the production run)

- [ ] **Step 1: Submit the full array**

Run:
```bash
cd <REPO>
DRY_RUN=0 ./submit_session1_array.sh
```
Expected: `Submitted array 1-10%5.` and 10 tasks visible (≤5 running at once) in `squeue -u $USER`.

- [ ] **Step 2: Monitor progress**

Run periodically:
```bash
squeue -u $USER -o "%.18i %.9P %.8T %.10M %R"
grep -l "predict complete\|DONE\|error\|Traceback" OutFiles/slurm-session1-*_*.out 2>/dev/null
```
Expected: tasks transition PENDING → RUNNING → COMPLETED; no `Traceback` in logs.

- [ ] **Step 3: Spot-check outputs across recordings**

Run:
```bash
VID=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1
while IFS=$'\t' read -r rec csv; do
  newest=$(ls -dt "$rec"/Predictions_3D_*_* 2>/dev/null | head -1)
  nb=$(ls -d "$newest"/bout_* 2>/dev/null | wc -l)
  echo "$(basename $rec): $nb bouts -> ${newest:-NONE}"
done < session1_manifest.tsv
```
Expected: each of the 10 recordings has a new `Predictions_3D_<job>_<task>/` with a non-zero bout count and per-fly CSVs.

- [ ] **Step 4: Re-run any failed indices individually**

For any task index `i` that failed or produced 0 bouts:
```bash
DRY_RUN=0 ARRAY_SPEC=<i> ./submit_session1_array.sh
```

---

## Self-Review

**1. Spec coverage:**
- SLURM array, one task per recording → Task 2 (`--array`, `sed -n "${SLURM_ARRAY_TASK_ID}p"`). ✓
- Scope = 10 recordings with bout CSVs → Task 1 manifest (find + map). ✓
- `unified_V3_masked`, repo copy, `jarvis` env → Global Constraints + Task 2. ✓
- SAM3 shard pipeline, 2 animals → Task 2 command. ✓
- CenterDetect from red_data_unified (zero-code) → Global Constraints + Task 2 `CENTER_W` + Task 4 log check. ✓
- HybridNet newer run, single KeypointDetect run → Global Constraints + Task 2. ✓
- 4 GPUs/task, `%5` throttle → Task 2 defaults. ✓
- Stable index→recording mapping across requeue → Task 1 sorted manifest + Task 2 reads by line number; `--requeue`, `--open-mode=append`. ✓
- Output layout `Predictions_3D_<job>_<task>/bout_*/...` → Task 2 `--output_name`, Task 4/5 verification. ✓
- Fresh dirs, per-task isolation, re-run failed indices → Task 5 Step 4 (`ARRAY_SPEC=<i>`). ✓
- Dry-run validation → Task 3. ✓
- Smoke test before full run → Task 4 (GATE). ✓

**2. Placeholder scan:** No TBD/TODO. All commands and script bodies are complete and literal. `<REPO>` and `<jobid>` are explicitly defined stand-ins the engineer substitutes, not missing content.

**3. Type/name consistency:** Env var names (`NUM_GPUS`, `THROTTLE`, `MANIFEST`, `ARRAY_SPEC`, `CONDA_ENV`) are used identically across Tasks 2–5. Manifest format (`<recording_dir>\t<bouts_csv>`) produced in Task 1 is consumed identically in Task 2 (`cut -f1`/`cut -f2`) and Task 5. Weight-path variables (`CENTER_W`, `KP_W`, `HYBRID_W`) match the Global Constraints values.

Note (fixed inline): Task 3 Step 3 originally tried to override `KP_W` from the environment, which the script sets internally and would ignore; replaced with a `MANIFEST=/does/not/exist` check that actually exercises the guard.
