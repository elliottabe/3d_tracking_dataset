#!/bin/bash
# Re-run Stage B onward for every courtship recording whose artifacts were moved
# aside by stage_b_restage.py (they predate the reproj_resid_px outlier gate,
# c4bdaa8 2026-08-14).
#
# One process per RECORDING with bout_ids='' (all bouts); run_bout skips any
# bout-fly that still has its DONE marker, so this is resumable and safe to
# re-run. Concurrency is capped at 4: measured on these 128G-cgroup nodes,
# 8 concurrent JAX pipelines hit CUDA_ERROR_UNKNOWN. Each worker is pinned to
# its own GPU so four processes do not pile onto device 0.
#
# offsets.h5 / scale.json are per-RECORDING, so bouts within one recording must
# be processed by one process (the first bout fits them, the rest reuse) --
# which is why the unit of parallelism here is the recording, not the bout.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

VID=/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship
LOGS=slurm_logs/restage_$(date +%Y%m%d-%H%M)
mkdir -p "$LOGS"

export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export LD_PRELOAD=/mmfs1/gscratch/portia/eabe/miniconda3/envs/3d_tracking/lib/libstdc++.so.6
unset JAX_PLATFORMS

RECS=(
  "session0|$VID/Session0/2025_10_20_13_20_04"
  "session1|$VID/Session1/2026_04_02_12_11_50"
  "session1|$VID/Session1/2026_04_02_14_54_28"
  "session1|$VID/Session1/2026_04_02_15_25_51"
  "session1|$VID/Session1/2026_04_02_15_44_42"
  "session1|$VID/Session1/2026_04_02_16_03_48"
  "session1|$VID/Session1/2026_04_02_16_21_32"
  "session1|$VID/Session1/2026_04_02_16_39_56"
  "session1|$VID/Session1/2026_04_02_16_56_37"
  "session1|$VID/Session1/2026_04_02_17_28_34"
  "session1|$VID/Session1/2026_04_02_17_52_50"
)

MAXJOBS=4
i=0
for entry in "${RECS[@]}"; do
  cfg="${entry%%|*}"; dir="${entry##*|}"; name=$(basename "$dir")
  while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 20; done
  gpu=$(( i % 4 )); i=$(( i + 1 ))
  echo "[launch] $name -> GPU $gpu   log $LOGS/$name.log"
  (
    CUDA_VISIBLE_DEVICES=$gpu python scripts/run_bout.py \
      recording="$cfg" recording.session_dir="$dir" \
      > "$LOGS/$name.log" 2>&1
    echo "[done] $name rc=$?" >> "$LOGS/_status.txt"
  ) &
  sleep 5
done
wait
echo "=== ALL RECORDINGS FINISHED ==="
cat "$LOGS/_status.txt" 2>/dev/null
echo
echo "bout-flies still missing stac_ik.h5:"
find /gscratch/portia/eabe/data/Johnson_lab/processed/courtship \
     -path "*/pose/bouts/bout_*/fly*" -maxdepth 7 -type d 2>/dev/null \
  | while read -r d; do [ -f "$d/stac_ik.h5" ] || echo "  $d"; done | head -20
