# shellcheck shell=bash
# Shared helpers for running the mvq LIFT stage as local GPU workers instead of
# queueing it as a SLURM array. SOURCE this file; it defines functions only and
# runs nothing at source time.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/mvq_local_lift.sh"
#
# Callers: scripts/slurm/mvq_session_pipeline.sh (the session driver) and --
# once the P3b campaign that is holding it open finishes --
# scripts/slurm/mvq_p3a_campaign.sh, whose worker loop this was factored out of
# verbatim so the two routes cannot drift.
#
# Functions
#   mvq_read_yaml <file> <key>      one top-level scalar out of a YAML file
#   mvq_guard_pids <pattern>        live python PIDs matching, excluding US
#   mvq_bout_ids <predictions_dir>  bout ids that have a sam3_masks.npz
#   mvq_local_lift <sess> <pred> <out> <n_gpus> <bouts...>
#                                   run the lift on N local GPUs; 0 iff every
#                                   expected kp3d.npz landed and no worker died
#
# `mvq_local_lift` reads its non-positional knobs from the environment
# (MVQ_CKPT / MVQ_STEP / MVQ_THRESH / MVQ_MERGE / MVQ_LIFT_BATCH /
# MVQ_ANATOMY_CFG / MVQ_RECORDING_CFG / MVQ_BOUTS_CSV) rather than from a
# 12-positional signature -- a mis-ordered positional here silently lifts with
# the wrong checkpoint, which the gate string would then bake into kp3d.npz.

# One top-level scalar from a YAML file (empty string when absent).
mvq_read_yaml() {   # $1 = yaml path, $2 = key
    python3 - "$1" "$2" <<'PY'
import sys, yaml
d = yaml.safe_load(open(sys.argv[1])) or {}
v = d.get(sys.argv[2])
print("" if v is None else v)
PY
}

# PIDs of live PYTHON processes matching $1, excluding this script and its own
# ancestors. `pgrep -f` matches OUR argv too -- the pattern is literally one of
# our arguments -- and so does the `timeout`/`bash` wrapper that launched us, so
# a naive `pgrep -f "$pattern"` always "finds" the training run and the guard
# never lets anything through (measured 2026-09-04; the same pkill self-match
# that has bitten this repo before).
mvq_guard_pids() {   # $1 = pattern
    local pattern="$1" self_tree=" " p=$$ pid
    while [ -n "$p" ] && [ "$p" -gt 1 ] 2>/dev/null; do
        self_tree="$self_tree$p "
        p="$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')"
    done
    for pid in $(pgrep -f "$pattern" 2>/dev/null); do
        case "$self_tree" in *" $pid "*) continue ;; esac
        case "$(ps -o comm= -p "$pid" 2>/dev/null)" in *python*) echo "$pid" ;; esac
    done
}

# Bout ids (unpadded, numerically sorted) that actually have a sam3_masks.npz
# under $1. This -- not the bout_* dir listing -- is the mvq route's bout set:
# with `--lifter mvq` the masks are an INPUT, never work to queue.
mvq_bout_ids() {   # $1 = predictions_dir
    # A shell loop with an `-e` guard, NOT `ls <glob> | sed`: under `nullglob`
    # (which the session driver sets) an unmatched glob leaves `ls -d` with no
    # operand at all, and `ls -d` then prints ".", which sed happily passes
    # through as a "bout". Measured: a recording with zero sam3_masks.npz
    # reported "1 bouts with masks" and got a chain submitted.
    local f b
    for f in "$1"/bout_*/sam3_masks.npz; do
        [ -e "$f" ] || continue
        b="${f%/sam3_masks.npz}"; b="${b##*/bout_}"
        printf '%d\n' "$((10#$b))"
    done | sort -n
}

# Run the mvq lift for ONE recording on `n_gpus` local GPUs (one worker per
# GPU, the bout list split round-robin so a long bout does not land every time
# on the same worker). Returns 0 only when every worker exited 0 AND both
# flies' kp3d.npz landed for every attempted bout -- `wait` with no args always
# returns 0, so an earlier version let a crashed worker queue the rest of the
# chain anyway and the failure only surfaced one step later in
# slurm_bout_array.py's --mvq-lift skip gate, after other recordings had been
# submitted.
mvq_local_lift() {   # $1 sess_dir  $2 predictions_dir  $3 out_dir  $4 n_gpus  $5.. bout ids
    local sess_dir="$1" pred="$2" out="$3" n_gpus="$4"; shift 4
    local bouts=("$@")
    local g i b args log w=0 pid rc=0
    local pids=()

    [ "${#bouts[@]}" -gt 0 ] || { echo "  no bouts to lift" >&2; return 1; }
    mkdir -p "$out"
    for g in $(seq 0 $((n_gpus - 1))); do
        args=""
        i=0
        for b in "${bouts[@]}"; do
            if [ $((i % n_gpus)) -eq "$g" ]; then args="$args --bout $b"; fi
            i=$((i + 1))
        done
        [ -n "$args" ] || continue
        log="$out/local-lift-gpu$g.log"
        echo "  worker $g -> $log"
        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=$g \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
        HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 \
        PYTHONPATH=third_party/jarvis_jax:. \
        python -u scripts/mvq_lift_bout.py \
            --session-dir "$sess_dir" --predictions-dir "$pred" \
            --out "$out" $args \
            --run "$MVQ_CKPT" ${MVQ_STEP:+--step "$MVQ_STEP"} \
            --exist-thresh "$MVQ_THRESH" --batch "${MVQ_LIFT_BATCH:-8}" \
            --merge-dist-units "$MVQ_MERGE" \
            --anatomy "${MVQ_ANATOMY_CFG:-configs/anatomy/v1.yaml}" \
            --recording-cfg "$MVQ_RECORDING_CFG" \
            ${MVQ_BOUTS_CSV:+--bouts-csv "$MVQ_BOUTS_CSV"} > "$log" 2>&1 &
        pids+=("$!")
        w=$((w + 1))
    done
    echo "  waiting for $w local lift worker(s)..."
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "  ERROR: local lift worker pid $pid exited non-zero" >&2
            rc=1
        fi
    done

    local n_expected n_written f
    n_expected=$(( ${#bouts[@]} * 2 ))    # fly0 + fly1 per bout
    # Counted with a guarded shell loop rather than `ls <glob> | wc -l`: an
    # unmatched glob either survives literally (ls errors, exit 1 -> `set -e`
    # abort under pipefail) or, under `nullglob`, leaves `ls` with no operand
    # so it lists the CWD and reports a fake non-zero count. Both were live
    # hazards on this path; the loop has neither.
    n_written=0
    for f in "$out"/bouts/bout_*/fly*/kp3d.npz; do
        # `if`, not `[ -e ] && …`: a false test on the LAST iteration makes the
        # whole loop exit 1 and `set -e` would abort the caller right here.
        if [ -e "$f" ]; then n_written=$((n_written + 1)); fi
    done
    if [ "$rc" -ne 0 ] || [ "$n_written" -lt "$n_expected" ]; then
        echo "  ERROR: local lift incomplete (worker failure=$rc," \
             "kp3d.npz written=$n_written/$n_expected expected)" >&2
        return 1
    fi
    echo "  local lift complete: $n_written/$n_expected kp3d.npz"
    return 0
}
