#!/bin/bash
# Submit the SAM3 mask + keypoint->IK pipeline for EVERY recording in a
# multi-recording session -- one dependency-chained SLURM chain PER RECORDING
# (SAM3 --array over that recording's bouts -> precompute -> jax --array over the
# same bouts -> aggregate), each writing to its own output dir. This is the
# whole-session wrapper around scripts/slurm_courtship_array.py, which processes
# ONE recording per call.
#
# Works for any assay -- the only per-assay differences are the recording config
# (which sets num_animals + camera rig) and the per-recording bout-summary file
# name. Each recording dir must contain that summary, whose `fly_id` column ==
# "<SessionName>/<timestamp>" (== the session tag parse_bouts matches on).
#
# Usage:
#   scripts/run_free_running_session.sh --session <session_dir> \
#       [--recording <cfg>] [--summary <name.csv>] [--out <base>] [--dry-run]
#
# Free-running single fly (defaults):
#   scripts/run_free_running_session.sh \
#     --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/free_running/Session11
#
# Courtship (two flies), e.g. Session1 -- needs a Session1 recording config
# (num_animals=2) and the courtship summary name:
#   scripts/run_free_running_session.sh \
#     --session /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 \
#     --recording session1 --summary courtship_bouts_unified_summary.csv \
#     --out /gscratch/portia/eabe/data/Johnson_lab/courtship
#
# Notes:
#   * Bootstraps <recording>/Predictions_3D_sam3/bout_<idx> dirs (from the summary)
#     so slurm_courtship_array can discover the bouts.
#   * `env -u JAX_PLATFORMS` strips a stray JAX_PLATFORMS=cpu from the submit shell
#     (the sbatch scripts also unset it) so jobs use the GPU.
#   * Idempotent: SAM3 reuse_masks + per-bout stage checkpoints -> re-running only
#     redoes unfinished work.
set -euo pipefail

RECORDING_CFG="free_running_session11"
SUMMARY="free_running_bout_summary.csv"
OUT_BASE="/gscratch/portia/eabe/data/Johnson_lab/free_running"
SESSION_DIR=""
DRY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --session)   SESSION_DIR="$2"; shift 2 ;;
        --recording) RECORDING_CFG="$2"; shift 2 ;;
        --summary)   SUMMARY="$2"; shift 2 ;;
        --out)       OUT_BASE="$2"; shift 2 ;;
        --dry-run)   DRY="--dry-run"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[ -n "$SESSION_DIR" ] || { echo "usage: $0 --session <session_dir> [--recording cfg] [--summary name.csv] [--out base] [--dry-run]" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SESSION_NAME="$(basename "$SESSION_DIR")"
cd "$REPO"

shopt -s nullglob
n=0
for d in "$SESSION_DIR"/*/; do
    d="${d%/}"
    ts="$(basename "$d")"
    summary="$d/$SUMMARY"
    if [ ! -f "$summary" ]; then
        echo "skip $ts (no $SUMMARY)"
        continue
    fi
    # Bootstrap bout dirs so slurm_courtship_array can discover the bouts.
    pred="$d/Predictions_3D_sam3"
    for i in $(tail -n +2 "$summary" | awk -F, 'NF{print $2}'); do
        mkdir -p "$pred/$(printf 'bout_%05d' "$i")"
    done
    nb=$(tail -n +2 "$summary" | awk 'NF' | wc -l)
    echo "=== $SESSION_NAME / $ts : $nb bouts -> submitting chain ($RECORDING_CFG) ==="
    env -u JAX_PLATFORMS python scripts/slurm_courtship_array.py $DRY \
        recording="$RECORDING_CFG" \
        recording.session_dir="$d" \
        outputs.out="$OUT_BASE/${SESSION_NAME}_${ts}_bouts"
    n=$((n + 1))
done
echo "submitted $n recording chain(s) for $SESSION_NAME"



