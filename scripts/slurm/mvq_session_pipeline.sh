#!/bin/bash
# Run a WHOLE courtship session through the mvq route: one dependency-chained
# SLURM chain per recording (mvq lift -> body-scale/offsets precompute -> STAC
# IK array -> aggregate), then ONE session-level `collect` job gated on every
# recording's aggregate.
#
#     lift[ts] -> precompute[ts] -> ik[ts] -> aggregate[ts]     (per recording)
#                                                aggregate[*] -> collect
#
# This COMPOSES scripts/slurm_bout_array.py (which handles exactly one
# recording); it does not reimplement any stage. The DLT-route equivalent is
# scripts/run_free_running_session.sh, whose CLI style this mirrors.
#
# Usage:
#   scripts/slurm/mvq_session_pipeline.sh --session <video session dir> \
#       [--processed <root>] [--recording <cfg>] [--run-name pose_mvq_p3b] \
#       [--mvq-config p3b] [--slurm ckpt_all] [--only <ts>] \
#       [--local-gpus N [--max-local-gpus N]] [--no-collect] [--dry-run]
#
#   # Session0 (one recording, 30 bouts):
#   scripts/slurm/mvq_session_pipeline.sh \
#       --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0
#
#   # Session1 (10 recordings with masks, 130 bouts), lifting locally first:
#   scripts/slurm/mvq_session_pipeline.sh \
#       --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 \
#       --local-gpus 4
#
# Recording discovery: a subfolder of --session is processed only when
# `<processed>/<Session>/<ts>/sam3_masks/bout_*/sam3_masks.npz` exists. On the
# mvq route the SAM3 masks are an INPUT (the lift places its crops from their
# centroids and never re-segments), so a recording without them is skipped with
# a message rather than queued -- slurm_bout_array.py would refuse it anyway.
#
# Recording config: `recording=<cfg>` (default: the session folder name
# lowercased -- Session0 -> session0, Session1 -> session1) plus
# `recording.session_dir=<path>`. The session_dir override, not
# `recording.timestamp`, because session1.yaml derives session_dir from the
# timestamp and an unquoted 2026_04_02_16_03_48 is parsed by Hydra as the INT
# 20260402160348 (underscores are digit separators), which sends every job to
# .../Session1/20260402160348/calibration.
#
# Bouts CSV: when a recording's `courtship_bouts_unified_summary.csv` is
# missing or a BROKEN SYMLINK (Session0's points into a deleted
# Predictions_3D_* dir), a fly_id-LESS copy is written once into the run root
# and passed as `recording.bouts_csv`. The surviving per-fly CSVs carry
# `fly_id = "<Session>/<ts>_fly0"`, which run_bout.py's `bout_start_frame`
# (matching on the bare `<Session>/<ts>` tag) would never find; `parse_bouts`
# keeps every row of a CSV with no fly_id column, which is exactly right for a
# per-recording file. The two per-fly CSVs are byte-identical apart from that
# column (checked 2026-09-04).
#
# --local-gpus N: run the LIFT stage as N parallel local workers on this node
#   (one GPU each) instead of queueing it as a GPU array, then submit the
#   unchanged precompute -> IK -> aggregate chain with `--mvq-lift skip`. N is
#   capped at 4 by default: 8 concurrent JAX pipelines took a 128 GB-cgroup node
#   to CUDA_ERROR_UNKNOWN (see the GPU-node-parallel-limits note). REFUSES to
#   start while a training process matching --guard-pattern is alive. The worker
#   loop itself lives in scripts/slurm/mvq_local_lift.sh, shared with
#   scripts/slurm/mvq_p3a_campaign.sh.
#
# --dry-run prints every sbatch command (the per-recording ones via the
#   submitter's own --dry-run) plus the dependency graph, and submits nothing.
#
# Exit status: non-zero if ANY recording's chain failed to submit (the other
# recordings are still submitted -- a queue slot is worth more than symmetry --
# and the collect job then gates only on the aggregates that exist).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=scripts/slurm/mvq_local_lift.sh
source "$REPO/scripts/slurm/mvq_local_lift.sh"

SESSION_DIR=""
PROC=/gscratch/portia/eabe/data/Johnson_lab/processed/courtship
RECORDING_CFG=""          # default derived from the session folder name
RUN_NAME=pose_mvq_p3b
MVQ_CONFIG=p3b
SLURM_CFG=ckpt_all
ONLY=""
LOCAL_GPUS=0
MAX_LOCAL_GPUS=4
GUARD_PATTERN='run_id=mvq_t1_b16_p4_jitter10'
LIFT_BATCH=8
COLLECT=1
DRY=""

while [ $# -gt 0 ]; do
    case "$1" in
        --session)        SESSION_DIR="${2%/}"; shift 2 ;;
        --processed)      PROC="${2%/}"; shift 2 ;;
        --recording)      RECORDING_CFG="$2"; shift 2 ;;
        --run-name)       RUN_NAME="$2"; shift 2 ;;
        --mvq-config)     MVQ_CONFIG="$2"; shift 2 ;;
        --slurm)          SLURM_CFG="$2"; shift 2 ;;
        --only)           ONLY="$2"; shift 2 ;;
        --local-gpus)     LOCAL_GPUS="$2"; shift 2 ;;
        --max-local-gpus) MAX_LOCAL_GPUS="$2"; shift 2 ;;
        --guard-pattern)  GUARD_PATTERN="$2"; shift 2 ;;
        --lift-batch)     LIFT_BATCH="$2"; shift 2 ;;
        --no-collect)     COLLECT=0; shift ;;
        --dry-run)        DRY="--dry-run"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

[ -n "$SESSION_DIR" ] || {
    echo "usage: $0 --session <video session dir> [--processed root] [--recording cfg]" >&2
    echo "          [--run-name name] [--mvq-config name] [--slurm cfg] [--only ts]" >&2
    echo "          [--local-gpus N [--max-local-gpus N]] [--no-collect] [--dry-run]" >&2
    exit 2
}
[ -d "$SESSION_DIR" ] || { echo "no such session dir: $SESSION_DIR" >&2; exit 2; }

SESSION="$(basename "$SESSION_DIR")"
: "${RECORDING_CFG:=$(echo "$SESSION" | tr 'A-Z' 'a-z')}"
SLURM_YAML="$REPO/configs/slurm/${SLURM_CFG}.yaml"
[ -f "$SLURM_YAML" ] || { echo "no such slurm config: $SLURM_YAML" >&2; exit 2; }

# The per-recording submitter. Overridable so the dry-run tests can exercise
# THIS script's discovery/CSV/graph logic against a fake processed tree without
# composing the real Hydra config (which points at absolute cluster paths).
read -r -a BOUT_ARRAY_CMD <<< "${MVQ_BOUT_ARRAY_CMD:-env -u JAX_PLATFORMS python scripts/slurm_bout_array.py}"

cd "$REPO"

if [ "$LOCAL_GPUS" -gt 0 ]; then
    if [ "$LOCAL_GPUS" -gt "$MAX_LOCAL_GPUS" ]; then
        echo "refusing --local-gpus $LOCAL_GPUS > $MAX_LOCAL_GPUS: 8 concurrent JAX" >&2
        echo "pipelines took a 128 GB-cgroup node to CUDA_ERROR_UNKNOWN. Raise" >&2
        echo "--max-local-gpus deliberately if this node is bigger." >&2
        exit 2
    fi
    if [ -z "$DRY" ]; then
        GUARD_HITS="$(mvq_guard_pids "$GUARD_PATTERN")"
        if [ -n "$GUARD_HITS" ]; then
            echo "refusing to start local GPU work: python process(es) $GUARD_HITS match" >&2
            echo "  $GUARD_PATTERN" >&2
            echo "and are still alive (this node's GPUs are held by the user's training run)." >&2
            exit 3
        fi
    fi
    # The checkpoint the local workers use, read from the SAME config group the
    # array stage is given, so there is ONE source of truth for which weights
    # produced a kp3d.npz.
    MVQ_YAML="$REPO/configs/mvq/${MVQ_CONFIG}.yaml"
    [ -f "$MVQ_YAML" ] || { echo "no such mvq config: $MVQ_YAML" >&2; exit 2; }
    export MVQ_CKPT MVQ_STEP MVQ_THRESH MVQ_MERGE MVQ_LIFT_BATCH
    MVQ_CKPT="$(mvq_read_yaml "$MVQ_YAML" checkpoint)"
    MVQ_STEP="$(mvq_read_yaml "$MVQ_YAML" step)"
    MVQ_THRESH="$(mvq_read_yaml "$MVQ_YAML" exist_thresh)"
    MVQ_MERGE="$(mvq_read_yaml "$MVQ_YAML" merge_dist_units)"
    MVQ_LIFT_BATCH="$LIFT_BATCH"
    [ -n "$MVQ_CKPT" ] || { echo "$MVQ_YAML has no checkpoint" >&2; exit 2; }
    echo "local lift: $LOCAL_GPUS worker(s), checkpoint $MVQ_CKPT step ${MVQ_STEP:-final}"
fi

echo "Session   : $SESSION  ($SESSION_DIR)"
echo "Processed : $PROC/$SESSION"
echo "Run name  : $RUN_NAME   (mvq=$MVQ_CONFIG, recording=$RECORDING_CFG, slurm=$SLURM_CFG)"
echo

# One stage's job id out of a captured submitter log ("" when absent, e.g.
# under --dry-run, which prints the scripts but submits nothing).
_jid() {   # $1 = log path, $2 = stage name
    # `|| true`: no match makes grep exit 1, and under `set -o pipefail` that
    # would abort a `set -e` caller on `ID="$(_jid ...)"` -- which is the NORMAL
    # case under --dry-run, where the submitter prints scripts and no job ids.
    { grep -m1 -E "^Submitted $2: " "$1" | awk '{print $3}'; } || true
}

TS_LIST=()          # recordings whose chain was submitted (or dry-run printed)
LIFT_IDS=()
PRECOMP_IDS=()
IK_IDS=()
AGG_IDS=()
ANY_FAILED=0
n=0

shopt -s nullglob
for d in "$SESSION_DIR"/*/; do
    TS="$(basename "${d%/}")"
    if [ -n "$ONLY" ] && [ "$TS" != "$ONLY" ]; then
        continue
    fi
    PRED="$PROC/$SESSION/$TS/sam3_masks"
    BOUTS=()
    while IFS= read -r b; do [ -n "$b" ] && BOUTS+=("$b"); done < <(mvq_bout_ids "$PRED")
    if [ "${#BOUTS[@]}" -eq 0 ]; then
        echo "skip $SESSION/$TS (no bout_*/sam3_masks.npz under $PRED)"
        continue
    fi
    OUT="$PROC/$SESSION/$TS/$RUN_NAME"
    echo "=== $SESSION / $TS : ${#BOUTS[@]} bouts with masks -> $OUT ==="

    OVERRIDES=("recording=$RECORDING_CFG" "recording.session_dir=$SESSION_DIR/$TS")
    # fly_id-less bouts CSV when the unified one is missing / a broken symlink
    CSV=""
    if [ ! -f "$SESSION_DIR/$TS/courtship_bouts_unified_summary.csv" ]; then
        SRC=""
        for cand in courtship_bouts_fly0_summary.csv courtship_bouts_fly1_summary.csv; do
            if [ -f "$SESSION_DIR/$TS/$cand" ]; then SRC="$SESSION_DIR/$TS/$cand"; break; fi
        done
        if [ -z "$SRC" ]; then
            echo "  ERROR: no usable bouts CSV for $SESSION/$TS (unified missing/broken," \
                 "no per-fly summary) -- skipping" >&2
            ANY_FAILED=1
            continue
        fi
        CSV="$OUT/bouts_unified_summary.csv"
        echo "  bouts CSV: unified missing/broken -> fly_id-less copy of $(basename "$SRC") at $CSV"
        if [ -z "$DRY" ]; then
            mkdir -p "$OUT"
            awk -F, 'NR==1{print "bout_idx,start_frame,end_frame"; next} NF{print $2","$3","$4}' \
                "$SRC" > "$CSV"
        fi
        OVERRIDES+=("recording.bouts_csv=$CSV")
    fi
    OVERRIDES+=("outputs.out=$OUT")

    LIFT_MODE=array
    LIFT_ID="<lift_JOBID>"
    if [ "$LOCAL_GPUS" -gt 0 ]; then
        LIFT_MODE=skip
        LIFT_ID="local"
        if [ -z "$DRY" ]; then
            export MVQ_RECORDING_CFG="configs/recording/${RECORDING_CFG}.yaml"
            export MVQ_BOUTS_CSV="$CSV"
            if ! mvq_local_lift "$SESSION_DIR/$TS" "$PRED" "$OUT" "$LOCAL_GPUS" "${BOUTS[@]}"; then
                echo "ERROR: local lift for $SESSION/$TS failed -- NOT submitting its chain" >&2
                ANY_FAILED=1
                continue
            fi
        else
            echo "  (dry-run) would lift locally on $LOCAL_GPUS GPU(s): ${BOUTS[*]}"
        fi
    fi

    echo "+ ${BOUT_ARRAY_CMD[*]} --lifter mvq --mvq-lift $LIFT_MODE" \
         "--mvq-config $MVQ_CONFIG --slurm $SLURM_CFG $DRY ${OVERRIDES[*]}"
    CHAIN_LOG="$(mktemp)"
    rc=0
    "${BOUT_ARRAY_CMD[@]}" \
        --lifter mvq --mvq-lift "$LIFT_MODE" --mvq-config "$MVQ_CONFIG" \
        --slurm "$SLURM_CFG" $DRY "${OVERRIDES[@]}" > "$CHAIN_LOG" 2>&1 || rc=$?
    cat "$CHAIN_LOG"
    if [ "$rc" -ne 0 ]; then
        echo "ERROR: chain submission for $SESSION/$TS failed (exit $rc)" >&2
        rm -f "$CHAIN_LOG"
        ANY_FAILED=1
        continue
    fi

    # Job ids come from the submitter's own "Submitted <stage>: <id>" lines
    # (slurm_bout_array.py::_run). Under --dry-run it prints none, so the graph
    # below carries placeholders instead.
    P_ID="$(_jid "$CHAIN_LOG" precompute)"
    K_ID="$(_jid "$CHAIN_LOG" jax_array)"
    A_ID="$(_jid "$CHAIN_LOG" aggregate)"
    L_ID="$(_jid "$CHAIN_LOG" mvq_lift)"
    [ -n "$L_ID" ] && LIFT_ID="$L_ID"
    rm -f "$CHAIN_LOG"
    if [ -z "$DRY" ] && [ -z "$A_ID" ]; then
        echo "ERROR: chain for $SESSION/$TS submitted no aggregate job id" >&2
        ANY_FAILED=1
        continue
    fi

    TS_LIST+=("$TS")
    LIFT_IDS+=("$LIFT_ID")
    PRECOMP_IDS+=("${P_ID:-<precompute_JOBID>}")
    IK_IDS+=("${K_ID:-<ik_JOBID>}")
    AGG_IDS+=("${A_ID:-<aggregate_JOBID>}")
    n=$((n + 1))
    echo
done

echo "submitted $n recording chain(s) ($SESSION, $RUN_NAME, mvq=$MVQ_CONFIG)"

# --------------------------------------------------------------------------
# Session collect job -- one per session, gated on EVERY aggregate.
# --------------------------------------------------------------------------
COLLECT_ID=""
if [ "$COLLECT" -eq 1 ] && [ "${#AGG_IDS[@]}" -gt 0 ]; then
    DEP="afterok:$(IFS=:; echo "${AGG_IDS[*]}")"
    PARTITION="$(mvq_read_yaml "$SLURM_YAML" partition)"
    ACCOUNT="$(mvq_read_yaml "$SLURM_YAML" account)"
    CONDA_ENV="$(mvq_read_yaml "$SLURM_YAML" conda_env)"
    LOGDIR="$PROC/$SESSION"
    # CPU-only: --gpus=0 and NO --constraint. ckpt_all's constraint
    # (h200|a100|l40s|l40|a40) exists to keep the GPU stages off nodes whose
    # cards are too small; carrying it here would pin a pure-python summary job
    # to a GPU node for no reason. session_collect.py only reads JSON.
    COLLECT_SCRIPT="#!/bin/bash
#SBATCH --job-name=mvqcollect_${SESSION}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#SBATCH --mem=16G
#SBATCH --open-mode=append
#SBATCH -o ${LOGDIR}/slurm-mvqcollect-%j.out
#SBATCH --dependency=${DEP}
#SBATCH --requeue
set -x
source ~/.bashrc
micromamba activate ${CONDA_ENV}
cd ${REPO}
mkdir -p ${LOGDIR}
python -u scripts/session_collect.py --session-name ${SESSION} \\
    --processed ${PROC} --run-name ${RUN_NAME}
"
    echo
    echo "--- collect script${DRY:+ (dry-run)} ---"
    echo "+ sbatch --parsable  # (script on stdin, dependency=$DEP)"
    echo "$COLLECT_SCRIPT"
    if [ -z "$DRY" ]; then
        mkdir -p "$LOGDIR"
        COLLECT_ID="$(printf '%s' "$COLLECT_SCRIPT" | sbatch --parsable)"
        COLLECT_ID="${COLLECT_ID%%;*}"
        echo "Submitted collect: $COLLECT_ID  (dependency=$DEP)"
    else
        COLLECT_ID="<collect_JOBID>"
    fi
elif [ "$COLLECT" -eq 0 ]; then
    echo "collect: skipped (--no-collect)"
else
    echo "collect: skipped (no recording chain was submitted)"
fi

# --------------------------------------------------------------------------
# Dependency graph
# --------------------------------------------------------------------------
if [ "${#TS_LIST[@]}" -gt 0 ]; then
    echo
    echo "Dependency graph${DRY:+ (dry-run, nothing submitted)}:"
    for i in "${!TS_LIST[@]}"; do
        t="${TS_LIST[$i]}"
        echo "  lift[$t] -> precompute[$t] -> ik[$t] -> aggregate[$t]" \
             "  (jobs: lift=${LIFT_IDS[$i]} precompute=${PRECOMP_IDS[$i]}" \
             "ik=${IK_IDS[$i]} aggregate=${AGG_IDS[$i]})"
    done
    if [ -n "$COLLECT_ID" ]; then
        echo "  aggregate[*] -> collect   (jobs: aggregate=$(IFS=:; echo "${AGG_IDS[*]}")" \
             "collect=$COLLECT_ID)"
    fi
    echo "Monitor : squeue -u \$USER"
fi

if [ "$ANY_FAILED" -ne 0 ]; then
    echo "FAILED: one or more recordings' chains were not submitted -- see ERROR lines above" >&2
    exit 1
fi
