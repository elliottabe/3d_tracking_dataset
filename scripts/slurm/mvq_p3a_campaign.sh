#!/bin/bash
# Run the WHOLE courtship set (160 bouts, 11 recordings) through the mvq P3a
# route: typed-slot mvq keypoints -> keypoint filter -> body-scale precompute
# -> STAC IK -> polish -> viz, one dependency-chained SLURM chain per
# recording, each writing to its own `pose_mvq_p3a` run root.
#
#   Session0/2025_10_20_13_20_04   30 bouts
#   Session1/2026_04_02_*          130 bouts across 10 recordings
#
# NEVER writes into the existing `pose/` runs (the ViTPose+DLT baseline), nor
# `pose_mvq/` (the bout-28 render's npz), nor `pose_mvq_ik/` (the 1500-frame
# bout-28 IK test). Everything lands in a fresh
# `<processed>/courtship/<Session>/<recording>/pose_mvq_p3a/`.
#
# Usage:
#   scripts/slurm/mvq_p3a_campaign.sh --dry-run          # print the sbatch commands
#   scripts/slurm/mvq_p3a_campaign.sh                    # submit all 11 chains
#   scripts/slurm/mvq_p3a_campaign.sh --only 2026_04_02_16_03_48
#   scripts/slurm/mvq_p3a_campaign.sh --local-gpus 4     # lift HERE, queue the rest
#
# --local-gpus N: run the mvq LIFT stage as N parallel local workers on this
#   node (one GPU each, the bout list split round-robin) instead of queueing it
#   as a GPU array, then submit the unchanged precompute -> IK -> aggregate
#   chain with `--mvq-lift skip`. The ckpt-all queue has been running a day
#   deep; an interactive allocation lifts the whole set in a fraction of that.
#   N is capped at 4 by default: 8 concurrent JAX pipelines took a 128 GB-cgroup
#   node to CUDA_ERROR_UNKNOWN (see the GPU-node-parallel-limits note).
#   REFUSES to start while a training process matching --guard-pattern is alive.
#
# Recording configs: Session0 uses `recording=session0`; each Session1
# recording is selected with `recording=session1 recording.session_dir=<path>`
# -- the session_dir override, not `recording.timestamp`, because session1.yaml
# derives session_dir from the timestamp and an unquoted 2026_04_02_16_03_48 is
# parsed by Hydra as the INT 20260402160348 (underscores are digit separators),
# which sends every job to .../Session1/20260402160348/calibration. Overriding
# the path string cannot be misparsed, and it is what
# scripts/run_free_running_session.sh already does.
#
# Session0's bouts CSV: its `courtship_bouts_unified_summary.csv` is a BROKEN
# symlink (into a deleted Predictions_3D_* dir) and the per-fly CSVs that
# survive carry `fly_id = "Session0/<ts>_fly0"`, which run_bout.py's
# `bout_start_frame` (matching on the bare `Session0/<ts>` tag) would never
# find. So a fly_id-LESS copy is written once into the run root and passed as
# `recording.bouts_csv` -- `parse_bouts` keeps every row of a CSV with no
# fly_id column, which is exactly right for a per-recording file. The two
# per-fly CSVs are byte-identical apart from that column (checked
# 2026-09-04), so which one it is built from does not matter.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VID=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship
PROC=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship
RUN_NAME=pose_mvq_p3a
MVQ_CONFIG=p3a
SLURM_CFG=ckpt_all
DRY=""
ONLY=""
LOCAL_GPUS=0
MAX_LOCAL_GPUS=4
GUARD_PATTERN='run_id=mvq_t1_b16_p4_jitter10'
LIFT_BATCH=8

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run)     DRY="--dry-run"; shift ;;
        --only)        ONLY="$2"; shift 2 ;;
        --slurm)       SLURM_CFG="$2"; shift 2 ;;
        --mvq-config)  MVQ_CONFIG="$2"; shift 2 ;;
        --run-name)    RUN_NAME="$2"; shift 2 ;;
        --local-gpus)  LOCAL_GPUS="$2"; shift 2 ;;
        --max-local-gpus) MAX_LOCAL_GPUS="$2"; shift 2 ;;
        --guard-pattern)  GUARD_PATTERN="$2"; shift 2 ;;
        --lift-batch)  LIFT_BATCH="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

RECORDINGS=(
    "Session0 2025_10_20_13_20_04"
    "Session1 2026_04_02_12_11_50"
    "Session1 2026_04_02_14_54_28"
    "Session1 2026_04_02_15_25_51"
    "Session1 2026_04_02_15_44_42"
    "Session1 2026_04_02_16_03_48"
    "Session1 2026_04_02_16_21_32"
    "Session1 2026_04_02_16_39_56"
    "Session1 2026_04_02_16_56_37"
    "Session1 2026_04_02_17_28_34"
    "Session1 2026_04_02_17_52_50"
)

# The checkpoint the lift workers use in --local-gpus mode; read from the same
# config group the array stage is given, so there is ONE source of truth for
# which weights produced a kp3d.npz.
read_mvq_cfg() {   # $1 = key
    python3 - "$REPO/configs/mvq/${MVQ_CONFIG}.yaml" "$1" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
v = d.get(sys.argv[2])
print("" if v is None else v)
PY
}

# PIDs of live PYTHON processes matching --guard-pattern, excluding this
# script and its own ancestors. `pgrep -f` matches OUR argv too -- the pattern
# is literally one of our arguments -- and so does the `timeout`/`bash` wrapper
# that launched us, so a naive `pgrep -f "$GUARD_PATTERN"` always "finds" the
# training run and the guard never lets anything through (measured
# 2026-09-04; the same pkill self-match that has bitten this repo before).
guard_pids() {
    local self_tree=" " p=$$ pid
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        self_tree="$self_tree$p "
        p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
    done
    for pid in $(pgrep -f "$GUARD_PATTERN" 2>/dev/null); do
        case "$self_tree" in *" $pid "*) continue ;; esac
        case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *python*) echo "$pid" ;; esac
    done
}

if [ "$LOCAL_GPUS" -gt 0 ]; then
    if [ "$LOCAL_GPUS" -gt "$MAX_LOCAL_GPUS" ]; then
        echo "refusing --local-gpus $LOCAL_GPUS > $MAX_LOCAL_GPUS: 8 concurrent JAX" >&2
        echo "pipelines took a 128 GB-cgroup node to CUDA_ERROR_UNKNOWN. Raise" >&2
        echo "--max-local-gpus deliberately if this node is bigger." >&2
        exit 2
    fi
    GUARD_HITS="$(guard_pids)"
    if [ -n "$GUARD_HITS" ]; then
        echo "refusing to start local GPU work: python process(es) $GUARD_HITS match" >&2
        echo "  $GUARD_PATTERN" >&2
        echo "and are still alive (this node's GPUs are held by the user's training run)." >&2
        exit 3
    fi
    MVQ_CKPT="$(read_mvq_cfg checkpoint)"
    MVQ_STEP="$(read_mvq_cfg step)"
    MVQ_THRESH="$(read_mvq_cfg exist_thresh)"
    MVQ_MERGE="$(read_mvq_cfg merge_dist_units)"
    [ -n "$MVQ_CKPT" ] || { echo "configs/mvq/${MVQ_CONFIG}.yaml has no checkpoint" >&2; exit 2; }
    echo "local lift: $LOCAL_GPUS worker(s), checkpoint $MVQ_CKPT step ${MVQ_STEP:-final}"
fi

cd "$REPO"
n=0
for rec in "${RECORDINGS[@]}"; do
    set -- $rec
    SESSION="$1"; TS="$2"
    if [ -n "$ONLY" ] && [ "$TS" != "$ONLY" ] && [ "$SESSION" != "$ONLY" ]; then
        continue
    fi
    SESSION_DIR="$VID/$SESSION/$TS"
    PRED="$PROC/$SESSION/$TS/sam3_masks"
    OUT="$PROC/$SESSION/$TS/$RUN_NAME"
    if [ ! -d "$PRED" ]; then
        echo "skip $SESSION/$TS (no $PRED)"
        continue
    fi
    NB=$(ls -d "$PRED"/bout_*/sam3_masks.npz 2>/dev/null | wc -l)
    echo "=== $SESSION / $TS : $NB bouts with masks -> $OUT ==="

    # recording-config selection + the CSV fixups described in the header
    OVERRIDES=(recording="$(echo "$SESSION" | tr 'A-Z' 'a-z')")
    OVERRIDES+=("recording.session_dir=$SESSION_DIR")
    if [ "$SESSION" = "Session0" ]; then
        CSV="$OUT/bouts_unified_summary.csv"
        if [ -z "$DRY" ]; then
            mkdir -p "$OUT"
            awk -F, 'NR==1{print "bout_idx,start_frame,end_frame"; next} NF{print $2","$3","$4}' \
                "$SESSION_DIR/courtship_bouts_fly0_summary.csv" > "$CSV"
        fi
        OVERRIDES+=("recording.bouts_csv=$CSV")
    fi
    OVERRIDES+=("outputs.out=$OUT")

    LIFT_MODE=array
    if [ "$LOCAL_GPUS" -gt 0 ]; then
        LIFT_MODE=skip
        BOUTS=$(ls -d "$PRED"/bout_*/sam3_masks.npz 2>/dev/null \
                | sed -E 's#.*/bout_0*([0-9]+)/sam3_masks.npz#\1#' | sort -n)
        if [ -z "$DRY" ]; then
            w=0
            PIDS=()
            for g in $(seq 0 $((LOCAL_GPUS - 1))); do
                # round-robin slice, so a long bout does not land every time on
                # the same worker
                ARGS=""
                i=0
                for b in $BOUTS; do
                    if [ $((i % LOCAL_GPUS)) -eq "$g" ]; then ARGS="$ARGS --bout $b"; fi
                    i=$((i + 1))
                done
                [ -n "$ARGS" ] || continue
                LOG="$OUT/local-lift-gpu$g.log"
                mkdir -p "$OUT"
                echo "  worker $g -> $LOG"
                # shellcheck disable=SC2086
                CUDA_VISIBLE_DEVICES=$g \
                XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
                HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 \
                PYTHONPATH=third_party/jarvis_jax:. \
                python -u scripts/mvq_lift_bout.py \
                    --session-dir "$SESSION_DIR" --predictions-dir "$PRED" \
                    --out "$OUT" $ARGS \
                    --run "$MVQ_CKPT" ${MVQ_STEP:+--step "$MVQ_STEP"} \
                    --exist-thresh "$MVQ_THRESH" --batch "$LIFT_BATCH" \
                    --merge-dist-units "$MVQ_MERGE" \
                    --anatomy configs/anatomy/v1.yaml \
                    --recording-cfg "configs/recording/$(echo "$SESSION" | tr 'A-Z' 'a-z').yaml" \
                    ${CSV:+--bouts-csv "$CSV"} > "$LOG" 2>&1 &
                PIDS+=("$!")
                w=$((w + 1))
            done
            echo "  waiting for $w local lift worker(s)..."
            # `wait` with no args always returns 0, so a crashed worker went
            # undetected and the recording's precompute/IK chain got queued
            # anyway (the on-disk kp3d.npz gate then refused it one step
            # later, after other recordings had already been submitted).
            # wait on each PID individually and check its own exit status.
            LIFT_FAILED=0
            for pid in "${PIDS[@]}"; do
                if ! wait "$pid"; then
                    echo "  ERROR: local lift worker pid $pid ($SESSION/$TS) exited non-zero" >&2
                    LIFT_FAILED=1
                fi
            done
            N_BOUTS_ATTEMPTED=$(echo "$BOUTS" | wc -w)
            N_EXPECTED_KP3D=$((N_BOUTS_ATTEMPTED * 2))   # fly0 + fly1 per bout
            # `|| true`: under `set -o pipefail`, a glob that matches nothing
            # makes `ls` exit non-zero even though `wc -l` correctly reports 0,
            # which would otherwise abort the script here (via `set -e`)
            # instead of falling through to the error message below.
            N_LIFTED_KP3D=$(ls "$OUT"/bouts/bout_*/fly*/kp3d.npz 2>/dev/null | wc -l) || true
            if [ "$LIFT_FAILED" -ne 0 ] || [ "$N_LIFTED_KP3D" -lt "$N_EXPECTED_KP3D" ]; then
                echo "ERROR: local lift for $SESSION/$TS incomplete (worker failure=$LIFT_FAILED," \
                     "kp3d.npz written=$N_LIFTED_KP3D/$N_EXPECTED_KP3D expected) --" \
                     "skipping this recording's chain submission" >&2
                ANY_LIFT_FAILED=1
                unset CSV
                continue
            fi
        else
            echo "  (dry-run) would lift locally on $LOCAL_GPUS GPU(s): $(echo $BOUTS | tr '\n' ' ')"
        fi
    fi

    echo "+ python scripts/slurm_bout_array.py --lifter mvq --mvq-lift $LIFT_MODE" \
         "--mvq-config $MVQ_CONFIG --slurm $SLURM_CFG $DRY ${OVERRIDES[*]}"
    env -u JAX_PLATFORMS python scripts/slurm_bout_array.py \
        --lifter mvq --mvq-lift "$LIFT_MODE" --mvq-config "$MVQ_CONFIG" \
        --slurm "$SLURM_CFG" $DRY "${OVERRIDES[@]}"
    n=$((n + 1))
    unset CSV
done
echo "submitted $n recording chain(s) ($RUN_NAME, mvq=$MVQ_CONFIG)"
if [ "${ANY_LIFT_FAILED:-0}" -ne 0 ]; then
    echo "FAILED: one or more recordings' local lift crashed or under-produced" \
         "kp3d.npz -- see ERROR lines above; that recording's chain was NOT submitted" >&2
    exit 1
fi
