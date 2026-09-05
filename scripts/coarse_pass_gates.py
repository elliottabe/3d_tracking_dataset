#!/usr/bin/env python3
"""Phase 1 coarse pass, stage 2: gates + bout-summary CSV from coarse tracks.

Thin CLI wrapper: all the gate logic now lives in
`jarvis_jax.tracking.bout_gates` (moved there in P4b task 5 so the mvq-aware
gates are importable without a `scripts/`-relative `sys.path` hack -- this
script had no package-level dependency on anything else under `scripts/`).
Every name is re-exported here so existing `import coarse_pass_gates as
gates` callers (tests, `scripts/viz/coarse_pass_timeline.py`) are unaffected.

Reads the `coarse_tracks.npz` written by EITHER `scripts/coarse_pass.py`
(SAM3, GPU, queued) or `jarvis_jax.tracking.coarse_track.write_coarse_tracks`
(mvq, mask-free) and applies gates -- SAM3's area/border/wing-ratio gates or
mvq's existence/wing-angle gates, dispatched automatically on the file's own
schema (see `bout_gates.is_mvq_schema`) -- to emit a bout-summary CSV in
TODAY'S EXACT SCHEMA (fly_id, bout_idx, start_frame, end_frame, source_fly)
-- the drop-in `run_bout.py` already reads via `recording.bouts_csv` +
`bout_ids`. This stage is pure CPU/numpy/pandas: no GPU, no SAM3, safe to run
locally and iterate on gate thresholds without re-running the coarse pass.

See `jarvis_jax.tracking.bout_gates` for the full gate documentation
(docs/specs/2026-08-31-pipeline-schematic-and-inventory-design.md s2 for
SAM3, docs/specs/2026-09-04-mvq-maskfree-frontend-design.md s5.1 for mvq).

Usage (same PYTHONPATH convention as `scripts/coarse_pass_mvq.py`, needed so
this repo-root script's `jarvis_jax` import resolves to the checkout you are
actually editing rather than whatever `pip install -e` last pointed at):

    PYTHONPATH=third_party/jarvis_jax:. python scripts/coarse_pass_gates.py \\
        --tracks .../coarse_tracks.npz --out-csv .../bouts.csv \\
        --session-tag Session0/2025_10_20_13_20_04
"""
from jarvis_jax.tracking.bout_gates import (  # noqa: F401 -- re-exported for callers
    AREA_RATIO_MIN, BASELINE_WINDOW, BORDER_MIN_PX, EXIST_MIN, MIN_CAMS,
    PROXIMITY_MAX_PX_DEFAULT, PROXIMITY_MAX_UNITS_DEFAULT, SEP_MIN_PX,
    WING_ANGLE_MIN_DEFAULT, WING_RATIO_MIN_DEFAULT, apply_gates,
    build_arg_parser, coarse_to_real, compute_gate_signals, is_mvq_schema,
    load_ground_truth, load_tracks, main, overlap, rolling_median, run,
    segment_runs, validate, write_bout_csv)

if __name__ == "__main__":
    main()
