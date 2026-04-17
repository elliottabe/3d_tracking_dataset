#!/usr/bin/env bash
# Stage per-timestamp calibration/ dirs from a source session into an already-
# staged bouts directory tree. For each <timestamp>/ present in the destination,
# copies the matching <timestamp>/calibration/ from the source. Safe to re-run
# (rsync only transfers what's missing or changed).
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

for dst_ts in "$DST"/*/; do
    [[ -d "$dst_ts" ]] || continue
    ts=$(basename "$dst_ts")
    src_calib="$SRC/$ts/calibration"
    dst_calib="$dst_ts/calibration"

    if [[ ! -d "$src_calib" ]]; then
        echo "  [skip] $ts: no calibration/ in source ($src_calib)"
        n_skipped_no_src=$((n_skipped_no_src + 1))
        continue
    fi

    if [[ -d "$dst_calib" ]] && [[ -n "$(ls -A "$dst_calib" 2>/dev/null)" ]]; then
        echo "  [exists] $ts: calibration/ already staged"
        n_already_staged=$((n_already_staged + 1))
        continue
    fi

    echo "  [copy]  $ts: $src_calib -> $dst_calib"
    mkdir -p "$dst_calib"
    rsync -aL "$src_calib/" "$dst_calib/"
    n_copied=$((n_copied + 1))
done

echo
echo "Done. copied=$n_copied already_staged=$n_already_staged skipped_no_src=$n_skipped_no_src"
