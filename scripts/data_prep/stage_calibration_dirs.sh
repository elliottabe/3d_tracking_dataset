#!/usr/bin/env bash
# Stage per-timestamp calibration/ dirs and per-bout clips/ dirs from a source
# session into an already-staged bouts directory tree.
#
# For each <timestamp>/ present in the destination:
#   - copies <timestamp>/calibration/ from the source
#   - for each <timestamp>/Predictions_3D_*/bout_*/ in the destination, copies
#     the matching bout_*/clips/ from the source
#
# Safe to re-run (rsync only transfers what's missing or changed; existing,
# non-empty calibration/ or clips/ dirs are left alone).
#
# Usage:
#   scripts/stage_calibration_dirs.sh <SRC_SESSION_DIR> <DST_SESSION_DIR>
#
# Example (Session1 → Session1_bouts_04172026):
#   scripts/stage_calibration_dirs.sh \
#       /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1 \
#       /gscratch/portia/eabe/data/Johnson_lab/courtship/Session1_bouts_04172026

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <SRC_SESSION_DIR> <DST_SESSION_DIR>" >&2
    exit 2
fi

SRC="$1"
DST="$2"

if [[ ! -d "$SRC" ]]; then
    echo "ERROR: source session dir does not exist: $SRC" >&2
    exit 1
fi
if [[ ! -d "$DST" ]]; then
    echo "ERROR: destination session dir does not exist: $DST" >&2
    exit 1
fi

n_copied=0
n_skipped_no_src=0
n_already_staged=0

n_clips_copied=0
n_clips_skipped_no_src=0
n_clips_already_staged=0

for dst_ts in "$DST"/*/; do
    [[ -d "$dst_ts" ]] || continue
    ts=$(basename "$dst_ts")
    src_calib="$SRC/$ts/calibration"
    dst_calib="$dst_ts/calibration"

    if [[ ! -d "$src_calib" ]]; then
        echo "  [skip] $ts: no calibration/ in source ($src_calib)"
        n_skipped_no_src=$((n_skipped_no_src + 1))
    elif [[ -d "$dst_calib" ]] && [[ -n "$(ls -A "$dst_calib" 2>/dev/null)" ]]; then
        echo "  [exists] $ts: calibration/ already staged"
        n_already_staged=$((n_already_staged + 1))
    else
        echo "  [copy]  $ts: $src_calib -> $dst_calib"
        mkdir -p "$dst_calib"
        rsync -aL "$src_calib/" "$dst_calib/"
        n_copied=$((n_copied + 1))
    fi

    # Stage bout clips/ for every staged bout under this timestamp.
    shopt -s nullglob
    for dst_bout in "$dst_ts"Predictions_3D_*/bout_*/; do
        [[ -d "$dst_bout" ]] || continue
        rel=${dst_bout#"$dst_ts"}              # e.g. Predictions_3D_123/bout_00001/
        src_clips="$SRC/$ts/${rel}clips"
        dst_clips="${dst_bout}clips"
        tag="$ts/${rel%/}"

        if [[ ! -d "$src_clips" ]]; then
            echo "    [clips skip]   $tag: no clips/ in source ($src_clips)"
            n_clips_skipped_no_src=$((n_clips_skipped_no_src + 1))
            continue
        fi

        if [[ -d "$dst_clips" ]] && [[ -n "$(ls -A "$dst_clips" 2>/dev/null)" ]]; then
            echo "    [clips exists] $tag: clips/ already staged"
            n_clips_already_staged=$((n_clips_already_staged + 1))
            continue
        fi

        echo "    [clips copy]   $tag: $src_clips -> $dst_clips"
        mkdir -p "$dst_clips"
        rsync -aL "$src_clips/" "$dst_clips/"
        n_clips_copied=$((n_clips_copied + 1))
    done
    shopt -u nullglob
done

echo
echo "Calibration: copied=$n_copied already_staged=$n_already_staged skipped_no_src=$n_skipped_no_src"
echo "Clips:       copied=$n_clips_copied already_staged=$n_clips_already_staged skipped_no_src=$n_clips_skipped_no_src"
