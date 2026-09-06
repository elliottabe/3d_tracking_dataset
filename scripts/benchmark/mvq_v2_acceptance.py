#!/usr/bin/env python
"""mvq v2 acceptance suite: one scorecard for spec §5
(`docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md`).

THRESHOLD TABLE (copied verbatim from spec §5 -- do not edit one without the
other):

| check | threshold |
|---|---|
| validation (153 human framesets, unprompted, oracle and policy) | mpjpe <= 0.085 mm; policy misses 0; sex_acc >= 0.99; contact_pair <= 0.146 mm; cross_fly_frac <= 0.036 overall and <= 0.055 contact pairs; single_fly cohort misses 0 |
| centre-shift rule (spec P4 §6) | error at 1 mm within +10 %; misses < 2 % |
| masked field checks, bouts 1/4/28 of 20_04 (containment OFF) | pose-jump frames 0 %; straddle frames <= P3b; female-missing <= P3b |
| mask-free field checks, bout 117 stride 1 | pose-jump frames <= 5 % (P3b 28 %); straddle <= 10 % (P3b 23 %); no identity swap in the contact sheet |
| mask-free coarse pass, 20_04 | female trackable >= 0.85 of coarse frames (jitter-10: 0.48); no existence >= 0.5 on the stale-window bout 69 frames; gate bout 1 false peaks not asserted |
| single fly (free_running 2 bouts, Clip Session6) | exactly one fly asserted on >= 99 % of frames; keypoint stability as bouts 28; agreement with the DLT/JARVIS reference within the courtship LOO band |
| existence calibration | reliability curve within 0.05 of the diagonal on validation |
| figures read back | contact sheets for each field check; validation overlay of the female cohort |

This script runs the check groups below, EACH independently selectable
(`--val --shift --masked-bouts --maskfree-bout117 --coarse-20_04
--single-fly --calibration`, or `--all`) and each writing/merging its own
rows into ONE `scorecard.json` (+ `scorecard.md`) under `--out` -- running
one group later only replaces THAT group's rows, every other group's rows
already on disk are kept untouched (`merge_rows`). `--dry-run` prints the
exact commands each selected group would run and validates the static
config (paths exist, run dir looks like an mvq run) without executing
anything or writing the scorecard.

GPU GATE. Every check group except `calibration` (a pure JSON read) needs a
real mvq checkpoint plus GPU-scale compute (a val pass, a mask-free coarse
pass, `mvq_lift_bout.py`, ...) that this repo's CLAUDE.md says must never
run on the login node. They are therefore SKIPPED (not FAILED) unless
`--gpu` is also passed -- `--dry-run` still shows what each WOULD run. As of
this script's authoring (2026-09-05) no v2 checkpoint exists yet; do not
pass `--gpu` until one does.

BASELINES (P3b, the checkpoint v2 must match or beat) are hardcoded below,
sourced from `docs/benchmark/2026-09-mvq/p3b-notes.md` ("re-lift bouts 1, 4,
28" / "P3b on the mask-free route" tables) and
`figures/2026-09-mvq/p4_maskfree/gate_bouts_probe/gate_bouts_stats.json`
(bout 1 / bout 69 frame ranges, `gate-bouts-probe-2026-09-05.md`'s
recommendation section) -- see the P3B_* constants.

CALIBRATION. Reads `<run>/mvq_run.json`'s `"calibration"` block, which Task
6's `scripts/benchmark/mvq_calibrate.py` writes:
`{"exist_temperature", "vis_temperature", "reliability_exist": {"edges",
"acc", "conf", "n", "max_gap", "ece"}, "reliability_vis": {...}, "n_val"}`.
A missing block (or missing file) is SKIPPED, never FAILED -- a checkpoint
that has not been calibrated yet is not a calibration FAILURE, it is a step
not yet run.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "third_party", "jarvis_jax"))
sys.path.insert(0, ROOT)

# Pure numpy -- safe to import eagerly even in --dry-run / no-jax contexts.
from scripts.benchmark.mvq_perkp_bouts import bout_dir_metrics, tracks_npz_metrics  # noqa: E402

MM_PER_UNIT = 0.1              # mirrors jarvis_jax.train.train_mvq.MM_PER_UNIT
CONTACT_PAIR_UNITS = 30.0      # mirrors jarvis_jax.train.train_mvq.CONTACT_UNITS

VAL_ROOT_DEFAULT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902"

SESSION0_20_04 = {
    "session_dir": "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04",
    "calib_dir": "/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/calibration",
    "predictions_dir": "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/sam3_masks",
    "centerdetect": "/gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004",
}

# P3b baselines (docs/benchmark/2026-09-mvq/p3b-notes.md, "re-lift bouts 1, 4, 28
# of Session0/2025_10_20_13_20_04", containment-off column (b)).
P3B_MASKED_BOUTS = {
    1: {"straddle_frac": 0.1793, "female_missing_frac": 0.1092},
    4: {"straddle_frac": 0.0000, "female_missing_frac": 0.2563},
    28: {"straddle_frac": 0.0000, "female_missing_frac": 0.1545},
}
# P3b baseline, mask-free bout 117 (frames 111488-114688, stride 1) -- p3b-notes.md
# "P3b on the mask-free route" comparison table (straddle_frac 0.2334) plus the
# pose-jump number reported alongside it for this task's brief.
P3B_MASKFREE_BOUT117 = {"pose_jump_frac": 0.28, "straddle_frac": 0.2334}
MASKFREE_BOUT117_FRAMES = (111488, 114688)   # [start, end) -- 3200 frames @ stride 1

# jitter-10 baseline (the mask-free coarse pass's OWN prior checkpoint), from
# figures/2026-09-mvq/p4_maskfree/gate_bouts_probe/gate_bouts_stats.json's
# "female_frac_trackable_whole_recording": 0.4775 (rounded 0.478 in the task brief).
P3B_COARSE_FEMALE_TRACKABLE = 0.478
# Stale-window false-positive bout, gate_idx 69 -- gate-bouts-probe-2026-09-05.md's
# recommendation section ("69 was a bare-substrate/lost-track false positive");
# exact frame range from gate_bouts_stats.json's per_bout row for bout_idx=69.
GATE_BOUT_69_FRAMES = (61456, 63376)
# "wall" false-positive bout (no fly visible under either marker) -- same doc,
# contact-sheet table, bout 1: frames 16-2592.
GATE_BOUT_1_FRAMES = (16, 2592)

# courtship LOO reprojection band, docs/benchmark/2026-09-mvq/session-driver-2026-09-05.md
# ("Session0/pose_mvq_p3a_r2 ... LOO median 0.77 px, p90 3.66 px").
LOO_BAND_PX = {"median": 0.77, "p90": 3.66}

FIG_DIR_DEFAULT = "figures/2026-09-mvq/v2_train"
OUT_DIR_DEFAULT = "docs/benchmark/2026-09-mvq/2026-09-05-mvq-v2"


# --------------------------------------------------------------------------
# row / scorecard plumbing (pure, no jax/GPU -- this is what the unit tests
# exercise with fabricated numbers, per the task-7 brief's step 4/step-1 test).
# --------------------------------------------------------------------------
def make_row(group, check, metric, value, threshold, op, evidence, *, baseline=None, note=None):
    """One scorecard row. `value=None` (or NaN) means the check group could
    not produce a number (GPU step skipped, an upstream artifact missing,
    ...) -- status SKIP, never FAIL, per this script's own SKIP philosophy
    (a step not yet run is not a regression)."""
    status, passed = "SKIP", None
    if value is not None and not (isinstance(value, float) and np.isnan(value)):
        if op == "le":
            passed = bool(value <= threshold)
        elif op == "ge":
            passed = bool(value >= threshold)
        elif op == "eq":
            passed = bool(value == threshold)
        elif op == "lt":
            passed = bool(value < threshold)
        else:
            raise ValueError(f"unknown op {op!r}")
        status = "PASS" if passed else "FAIL"
    return {"group": group, "check": check, "metric": metric, "value": value,
            "threshold": threshold, "op": op, "baseline": baseline, "status": status,
            "pass": passed, "evidence": evidence, "note": note}


def skip_row(group, check, metric, threshold, op, evidence, note, *, baseline=None):
    return make_row(group, check, metric, None, threshold, op, evidence, baseline=baseline, note=note)


def build_scorecard(rows, *, run=None, generated_at=None):
    accepted = not any(r["status"] == "FAIL" for r in rows)
    return {"run": run, "generated_at": generated_at, "rows": rows, "accepted": accepted}


def merge_rows(existing_rows, new_rows, groups_run):
    """Existing rows whose group is NOT in `groups_run` survive untouched;
    every row for a group in `groups_run` is replaced by `new_rows` (a group
    that produced zero rows this time -- e.g. GPU steps not run -- still
    drops its stale old rows rather than leaving a FAIL/PASS from a
    different checkpoint on disk)."""
    kept = [r for r in existing_rows if r["group"] not in groups_run]
    return kept + new_rows


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return "nan" if np.isnan(v) else f"{v:.4g}"
    return str(v)


def scorecard_markdown(scorecard):
    lines = ["# mvq v2 acceptance scorecard", ""]
    if scorecard.get("run"):
        lines.append(f"run: `{scorecard['run']}`  ")
    lines.append(f"generated: {scorecard.get('generated_at')}  ")
    lines.append(f"**ACCEPTED: {scorecard['accepted']}**")
    lines.append("")
    lines += ["| group | check | metric | value | op | threshold | baseline | status | evidence |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in scorecard["rows"]:
        lines.append(f"| {r['group']} | {r['check']} | {r['metric']} | {_fmt(r['value'])} | "
                     f"{r['op']} | {_fmt(r['threshold'])} | {_fmt(r.get('baseline'))} | "
                     f"{r['status']} | {r['evidence']} |")
    notes = [r for r in scorecard["rows"] if r.get("note")]
    if notes:
        lines += ["", "## notes", ""]
        for r in notes:
            lines.append(f"- **{r['group']}/{r['check']}**: {r['note']}")
    return "\n".join(lines) + "\n"


def _jsonable(obj):
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = obj.item()
        return None if isinstance(v, float) and not np.isfinite(v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


def write_scorecard(out_dir, scorecard):
    os.makedirs(out_dir, exist_ok=True)
    jpath = os.path.join(out_dir, "scorecard.json")
    mpath = os.path.join(out_dir, "scorecard.md")
    with open(jpath, "w") as f:
        json.dump(_jsonable(scorecard), f, indent=1)
    with open(mpath, "w") as f:
        f.write(scorecard_markdown(scorecard))
    return jpath, mpath


def load_existing_rows(out_dir):
    jpath = os.path.join(out_dir, "scorecard.json")
    if not os.path.isfile(jpath):
        return []
    with open(jpath) as f:
        d = json.load(f)
    return d.get("rows", [])


# --------------------------------------------------------------------------
# shell-command construction (for --dry-run printing AND for --gpu execution)
# --------------------------------------------------------------------------
def _run_cmd(cmd, *, log_path, dry_run):
    """`cmd`: list[str]. Runs via `bash -lc <joined>` (needs `module load` /
    shell functions the way a real GPU job invokes these scripts -- see
    CLAUDE.md's "rendering practicalities"). Returns (returncode, log_path)
    or (None, None) under --dry-run (prints the command instead)."""
    joined = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] would run:\n    {joined}")
        return None, None
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"$ {joined}\n\n")
        f.flush()
        proc = subprocess.run(["bash", "-lc", joined], stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode, log_path


def _final_dir(run):
    """`run` may already be a `final/` dir, or the parent run dir -- both
    conventions appear across this codebase's mvq scripts; normalise to the
    `final/` dir every check group needs."""
    if os.path.basename(run.rstrip("/")) == "final":
        return run.rstrip("/")
    cand = os.path.join(run, "final")
    return cand if os.path.isdir(cand) else run


# --------------------------------------------------------------------------
# check groups
# --------------------------------------------------------------------------
def _run_val_evaluate(run, root, batch_size, num_workers, attn_impl):
    """The actual GPU/jax work for the "val" group -- imported lazily so
    importing this module (and every pure-python helper above) never
    requires jax/flax to be importable, let alone a GPU."""
    from flax import nnx
    from jarvis_jax.data.distractor import build_part_index
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.models.mvq.checkpoint import load_mvq_model
    from jarvis_jax.sharding import data_parallel_mesh, replicate
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.train.train_mvq import _cohorts, evaluate

    model, meta = load_mvq_model(run, attn_impl=attn_impl)
    val_ds = V12WindowDataset(root, "val", T=1, train=False)
    cohorts = _cohorts(val_ds)
    part_of_k, _ = build_part_index(val_ds.keypoint_names)
    # `LossWeights` is not persisted in `mvq_run.json` as of this writing (only
    # `MVQTrainConfig`/`MVQConfig` are) -- default weights only affect the
    # reproj_px/uv2d_px/head_vs_reproj_px diagnostics and the loss's matching
    # step, none of which feed the spec §5 acceptance rows below (mpjpe3d_mm,
    # policy_miss_frac, sex_acc, cohort_contact_pair, cross_fly_frac*,
    # policy_miss_frac_single_fly all come from the per-sample/per-slot
    # bookkeeping in `evaluate`, independent of the loss weight values).
    weights = LossWeights()
    mesh = data_parallel_mesh()
    gm, sm = nnx.split(model)
    model = nnx.merge(gm, replicate(sm, mesh))
    return evaluate(model, val_ds, batch_size, cohorts=cohorts, part_of_k=part_of_k,
                    weights=weights, mesh=mesh, num_workers=num_workers)


def check_val(run, *, root=VAL_ROOT_DEFAULT, batch_size=32, num_workers=8,
             attn_impl=None, gpu=False):
    group = "val"
    evidence = (f"train_mvq.evaluate(load_mvq_model({run!r}), V12WindowDataset({root!r}, 'val'), "
               f"cohorts=_cohorts(val_ds))['unprompted']")
    if not gpu:
        return [skip_row(group, "val", "n/a", None, "le", evidence,
                         "GPU step skipped (pass --gpu once a v2 checkpoint exists)")]
    try:
        result = _run_val_evaluate(run, root, batch_size, num_workers, attn_impl)
        u = result["unprompted"]
    except Exception as e:  # noqa: BLE001 -- surfaced as a SKIP row, not a crash
        return [skip_row(group, "val", "n/a", None, "le", evidence, f"val evaluate() raised: {e!r}")]

    contact_units = u.get("cohort_contact_pair")
    contact_mm = (contact_units * MM_PER_UNIT if contact_units is not None
                 and np.isfinite(contact_units) else contact_units)
    return [
        make_row(group, "mpjpe", "mpjpe3d_mm", u.get("mpjpe3d_mm"), 0.085, "le", evidence),
        make_row(group, "policy_miss", "policy_miss_frac", u.get("policy_miss_frac"), 0.0, "eq", evidence),
        make_row(group, "sex_acc", "sex_acc", u.get("sex_acc"), 0.99, "ge", evidence),
        make_row(group, "contact_pair_mpjpe", "cohort_contact_pair_mm", contact_mm, 0.146, "le", evidence),
        make_row(group, "cross_fly", "cross_fly_frac", u.get("cross_fly_frac"), 0.036, "le", evidence),
        make_row(group, "cross_fly_contact_pair", "cross_fly_frac_contact_pair",
                 u.get("cross_fly_frac_contact_pair"), 0.055, "le", evidence),
        make_row(group, "single_fly_miss", "policy_miss_frac_single_fly",
                 u.get("policy_miss_frac_single_fly"), 0.0, "eq", evidence),
    ]


def check_shift(run, *, root=VAL_ROOT_DEFAULT, out_dir=FIG_DIR_DEFAULT, gpu=False):
    group = "shift"
    evidence = f"scripts/benchmark/mvq_centre_shift.py --run {run} --root {root} --out {out_dir}"
    if not gpu:
        return [skip_row(group, "centre_shift", "n/a", None, "le", evidence,
                         "GPU step skipped (pass --gpu once a v2 checkpoint exists)")]
    try:
        from scripts.benchmark import mvq_centre_shift

        class _A:
            pass
        a = _A()
        a.run, a.step, a.attn_impl, a.root, a.out, a.batch = run, None, None, root, out_dir, 16
        result = mvq_centre_shift.run(a)
        rows = result["rows"]
    except Exception as e:  # noqa: BLE001
        return [skip_row(group, "centre_shift", "n/a", None, "le", evidence,
                         f"mvq_centre_shift.run() raised: {e!r}")]

    by_shift = {r["shift_mm"]: r for r in rows}
    base = by_shift.get(0.0, {}).get("mpjpe_policy_mm")
    at1 = by_shift.get(1.0, {})
    err1 = at1.get("mpjpe_policy_mm")
    miss1 = at1.get("miss_frac")
    err_thresh = 1.1 * base if base is not None and np.isfinite(base) else float("nan")
    return [
        make_row(group, "error_at_1mm", "mpjpe_policy_mm@1mm", err1, err_thresh, "le", evidence,
                baseline=base, note="threshold = 1.1x the shift=0mm value"),
        make_row(group, "miss_at_1mm", "miss_frac@1mm", miss1, 0.02, "lt", evidence),
    ]


def check_masked_bouts(run, *, session=SESSION0_20_04, out_root="OutFiles/v2_accept/off", gpu=False):
    group = "masked_bouts"
    bouts = (1, 4, 28)
    cmd = (f"python scripts/mvq_lift_bout.py --session-dir {session['session_dir']} "
          f"--predictions-dir {session['predictions_dir']} --calib-dir {session['calib_dir']} "
          f"--run {run} --identity mask --containment off "
          + " ".join(f"--bout {b}" for b in bouts) + f" --out {out_root}")
    if not gpu:
        return [skip_row(group, f"bout_{b}", "n/a", None, "le", cmd,
                         "GPU step skipped (pass --gpu once a v2 checkpoint exists)")
               for b in bouts]

    rc, log = _run_cmd(cmd.split(), log_path=os.path.join(out_root, "mvq_lift_bout.log"), dry_run=False)
    rows = []
    for b in bouts:
        baseline = P3B_MASKED_BOUTS[b]
        bout_dir = os.path.join(out_root, "bouts", f"bout_{b:05d}")
        evidence = f"{cmd} ; bout_dir_metrics({bout_dir!r}) ; log={log}"
        if rc != 0 or not os.path.isdir(bout_dir):
            rows.append(skip_row(group, f"bout_{b}", "n/a", None, "le", evidence,
                                 f"mvq_lift_bout.py did not produce {bout_dir} (rc={rc})"))
            continue
        try:
            m = bout_dir_metrics(bout_dir, male_fly=1)
        except Exception as e:  # noqa: BLE001
            rows.append(skip_row(group, f"bout_{b}", "n/a", None, "le", evidence,
                                 f"bout_dir_metrics raised: {e!r}"))
            continue
        rows.append(make_row(group, f"bout_{b}_pose_jump", "pose_jump_frac",
                             m["pose_jump_frac"], 0.0, "eq", evidence))
        rows.append(make_row(group, f"bout_{b}_straddle", "straddle_frac",
                             m["straddle_frac"], baseline["straddle_frac"], "le", evidence,
                             baseline=baseline["straddle_frac"]))
        rows.append(make_row(group, f"bout_{b}_female_missing", "female_missing_frac",
                             m["female_missing_frac"], baseline["female_missing_frac"], "le", evidence,
                             baseline=baseline["female_missing_frac"]))
    return rows


def check_maskfree_bout117(run, *, session=SESSION0_20_04, out_path="OutFiles/v2_accept/fine_bout117/fine_tracks.npz",
                           gpu=False):
    group = "maskfree_bout117"
    start, end = MASKFREE_BOUT117_FRAMES
    cmd = (f"python scripts/coarse_pass_mvq.py --session-dir {session['session_dir']} "
          f"--calib-dir {session['calib_dir']} --run {run} --centerdetect {session['centerdetect']} "
          f"--start {start} --end {end} --stride 1 --num-animals 2 --out {out_path}")
    baseline = P3B_MASKFREE_BOUT117
    if not gpu:
        return [skip_row(group, "pose_jump", "n/a", None, "le", cmd,
                         "GPU step skipped (pass --gpu once a v2 checkpoint exists)"),
               skip_row(group, "straddle", "n/a", None, "le", cmd,
                        "GPU step skipped (pass --gpu once a v2 checkpoint exists)")]

    rc, log = _run_cmd(cmd.split(), log_path=os.path.join(os.path.dirname(out_path), "coarse_pass_mvq.log"),
                       dry_run=False)
    evidence = (f"{cmd} ; tracks_npz_metrics({out_path!r}) ; contact sheet: "
               f"scripts/viz/contact_sheet.py --tracks {out_path} "
               f"--out {FIG_DIR_DEFAULT}/bout117_contact_sheet.png ; log={log}")
    if rc != 0 or not os.path.isfile(out_path):
        return [skip_row(group, "pose_jump", "n/a", None, "le", evidence,
                         f"coarse_pass_mvq.py did not produce {out_path} (rc={rc})"),
               skip_row(group, "straddle", "n/a", None, "le", evidence,
                        f"coarse_pass_mvq.py did not produce {out_path} (rc={rc})")]
    try:
        m = tracks_npz_metrics(out_path, male_fly=1)
    except Exception as e:  # noqa: BLE001
        return [skip_row(group, "pose_jump", "n/a", None, "le", evidence, f"tracks_npz_metrics raised: {e!r}"),
               skip_row(group, "straddle", "n/a", None, "le", evidence, f"tracks_npz_metrics raised: {e!r}")]
    return [
        make_row(group, "pose_jump", "pose_jump_frac", m["pose_jump_frac"], 0.05, "le", evidence,
                baseline=baseline["pose_jump_frac"]),
        make_row(group, "straddle", "straddle_frac", m["straddle_frac"], 0.10, "le", evidence,
                baseline=baseline["straddle_frac"]),
    ]


def check_coarse_20_04(run, *, session=SESSION0_20_04,
                      out_path="OutFiles/v2_accept/coarse_20_04/coarse_tracks.npz", gpu=False):
    group = "coarse_20_04"
    cmd = (f"python scripts/coarse_pass_mvq.py --session-dir {session['session_dir']} "
          f"--calib-dir {session['calib_dir']} --run {run} --centerdetect {session['centerdetect']} "
          f"--stride 16 --resume --out {out_path}")
    if not gpu:
        return [skip_row(group, c, "n/a", None, "le", cmd,
                         "GPU step skipped (pass --gpu once a v2 checkpoint exists)")
               for c in ("female_trackable", "bout69_stale_window", "bout1_false_peak")]

    rc, log = _run_cmd(cmd.split(), log_path=os.path.join(os.path.dirname(out_path), "coarse_pass_mvq.log"),
                       dry_run=False)
    evidence = f"{cmd} ; coarse_pass_gates.load_tracks({out_path!r}) ; log={log}"
    if rc != 0 or not os.path.isfile(out_path):
        return [skip_row(group, c, "n/a", None, "le", evidence,
                         f"coarse_pass_mvq.py did not produce {out_path} (rc={rc})")
               for c in ("female_trackable", "bout69_stale_window", "bout1_false_peak")]
    try:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        import coarse_pass_gates as gates  # noqa: E402
        z, _meta = gates.load_tracks(out_path)
        exist = np.asarray(z["exist"])          # (F=2,T), fly0=female, fly1=male
        coarse_frame = np.asarray(z["coarse_frame"])
    except Exception as e:  # noqa: BLE001
        return [skip_row(group, c, "n/a", None, "le", evidence, f"load_tracks raised: {e!r}")
               for c in ("female_trackable", "bout69_stale_window", "bout1_false_peak")]

    female_trackable = float(np.mean(exist[0] >= 0.5))

    def _window_max_exist(frames):
        s, e = frames
        mask = (coarse_frame >= s) & (coarse_frame < e)
        return float(np.max(exist[:, mask])) if mask.any() else float("nan")

    bout69_max = _window_max_exist(GATE_BOUT_69_FRAMES)
    bout1_max = _window_max_exist(GATE_BOUT_1_FRAMES)
    return [
        make_row(group, "female_trackable", "frac_trackable_female", female_trackable, 0.85, "ge", evidence,
                baseline=P3B_COARSE_FEMALE_TRACKABLE),
        make_row(group, "bout69_stale_window", f"max(exist) frames {GATE_BOUT_69_FRAMES}", bout69_max,
                0.5, "lt", evidence, note="stale-window false positive must not read as trackable"),
        make_row(group, "bout1_false_peak", f"max(exist) frames {GATE_BOUT_1_FRAMES}", bout1_max,
                0.5, "lt", evidence, note="'wall' false positive (no fly visible) must not be asserted"),
    ]


def check_single_fly(run, *, gpu=False):
    group = "single_fly"
    script = "scripts/pseudo_labels/singlefly_p3b_pass.py"
    evidence = (f"{script} --run {run} (free_running Session11 bouts 2,4 + Clip Session6) "
               f"-- see docs/benchmark/2026-09-mvq/p3b-notes.md 'P3b on single-fly recordings' "
               f"and session-driver-2026-09-05.md for the LOO band")
    checks = ("one_slot_asserted", "keypoint_stability", "reference_agreement_px")
    if not gpu or not os.path.isfile(os.path.join(ROOT, script)):
        note = ("GPU step skipped (pass --gpu once a v2 checkpoint exists)" if gpu is False else
               f"{script} does not exist yet (owned by the pseudo-labels/single-fly task) -- SKIP, not FAIL")
        return [skip_row(group, c, "n/a", None, "ge" if c == "one_slot_asserted" else "le", evidence, note)
               for c in checks]
    # Real invocation deferred until `singlefly_p3b_pass.py` exists and a v2
    # checkpoint is available -- this is the wiring point future work fills in
    # (its expected JSON schema is not yet fixed anywhere this script can read).
    return [skip_row(group, c, "n/a", None, "ge" if c == "one_slot_asserted" else "le", evidence,
                     f"{script} output schema not yet defined -- see this function's docstring")
           for c in checks]


def check_calibration(run):
    group = "calibration"
    final = _final_dir(run)
    meta_path = os.path.join(final, "mvq_run.json")
    evidence = f"{meta_path}#calibration.reliability_exist.max_gap"
    if not os.path.isfile(meta_path):
        return [skip_row(group, "existence_reliability", "reliability_exist.max_gap", 0.05, "le", evidence,
                         f"{meta_path} does not exist")]
    with open(meta_path) as f:
        meta = json.load(f)
    calib = meta.get("calibration")
    if not calib:
        return [skip_row(group, "existence_reliability", "reliability_exist.max_gap", 0.05, "le", evidence,
                         "no 'calibration' block yet (Task 6's mvq_calibrate.py has not run on this "
                         "checkpoint) -- SKIPPED, not FAILED: an uncalibrated checkpoint is a step not "
                         "yet run, not a regression")]
    max_gap = calib.get("reliability_exist", {}).get("max_gap")
    return [make_row(group, "existence_reliability", "reliability_exist.max_gap", max_gap, 0.05, "le", evidence)]


GROUPS = ("val", "shift", "masked_bouts", "maskfree_bout117", "coarse_20_04", "single_fly", "calibration")


def run_groups(run, groups, *, args):
    rows = []
    if "val" in groups:
        rows += check_val(run, root=args.val_root, batch_size=args.val_batch,
                          num_workers=args.val_workers, attn_impl=args.attn_impl, gpu=args.gpu)
    if "shift" in groups:
        rows += check_shift(run, root=args.val_root, out_dir=args.fig_dir, gpu=args.gpu)
    if "masked_bouts" in groups:
        rows += check_masked_bouts(run, out_root=os.path.join(args.work_dir, "masked_off"), gpu=args.gpu)
    if "maskfree_bout117" in groups:
        rows += check_maskfree_bout117(
            run, out_path=os.path.join(args.work_dir, "fine_bout117", "fine_tracks.npz"), gpu=args.gpu)
    if "coarse_20_04" in groups:
        rows += check_coarse_20_04(
            run, out_path=os.path.join(args.work_dir, "coarse_20_04", "coarse_tracks.npz"), gpu=args.gpu)
    if "single_fly" in groups:
        rows += check_single_fly(run, gpu=args.gpu)
    if "calibration" in groups:
        rows += check_calibration(run)
    return rows


def _validate_config(args):
    """`--dry-run`'s config validation: static path/shape checks only, no
    GPU/jax import. Returns a list of human-readable problem strings (empty
    = looks fine)."""
    problems = []
    if not os.path.isdir(args.run) and not os.path.isdir(_final_dir(args.run)):
        problems.append(f"--run {args.run!r}: no such directory (and no 'final/' under it)")
    final = _final_dir(args.run)
    meta_path = os.path.join(final, "mvq_run.json")
    if os.path.isdir(final) and not os.path.isfile(meta_path):
        problems.append(f"{final} has no mvq_run.json -- does not look like an mvq run dir")
    for key, path in (("session_dir", SESSION0_20_04["session_dir"]),
                      ("calib_dir", SESSION0_20_04["calib_dir"]),
                      ("predictions_dir", SESSION0_20_04["predictions_dir"]),
                      ("centerdetect", SESSION0_20_04["centerdetect"])):
        if not os.path.exists(path):
            problems.append(f"Session0/20_04 {key} not found: {path}")
    if not os.path.isdir(args.val_root):
        problems.append(f"--val-root not found: {args.val_root}")
    return problems


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="mvq run's 'final/' dir (or its parent)")
    ap.add_argument("--out", default=OUT_DIR_DEFAULT, help="dir for scorecard.json/.md")
    ap.add_argument("--fig-dir", default=FIG_DIR_DEFAULT)
    ap.add_argument("--work-dir", default="OutFiles/v2_accept", help="scratch dir for lift/coarse-pass outputs")
    ap.add_argument("--val-root", default=VAL_ROOT_DEFAULT)
    ap.add_argument("--val-batch", type=int, default=32)
    ap.add_argument("--val-workers", type=int, default=8)
    ap.add_argument("--attn-impl", default=None)
    ap.add_argument("--gpu", action="store_true",
                    help="actually execute GPU-heavy check groups (default: SKIP them)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands each selected group would run and validate config; write nothing")
    for g in GROUPS:
        ap.add_argument(f"--{g.replace('_', '-')}", action="store_true", help=f"run the '{g}' check group")
    ap.add_argument("--all", action="store_true", help="run every check group")
    return ap


def selected_groups(args):
    if args.all:
        return list(GROUPS)
    return [g for g in GROUPS if getattr(args, g)]


def main():
    args = build_arg_parser().parse_args()
    groups = selected_groups(args)
    if not groups:
        build_arg_parser().error("pass --all or at least one check-group flag (--val, --shift, ...)")

    if args.dry_run:
        problems = _validate_config(args)
        if problems:
            print("[dry-run] config problems found:")
            for p in problems:
                print(f"  - {p}")
        else:
            print("[dry-run] config looks valid (paths exist; run dir looks like an mvq run)")
        print(f"[dry-run] groups: {groups}")
        # Every check_* function, called with gpu=False, returns SKIP rows
        # whose `evidence` field IS the exact command it would otherwise run
        # (built before the gpu branch) -- so a plain gpu=False pass through
        # `run_groups` is exactly "print the commands", no separate code path.
        dry_args = argparse.Namespace(**vars(args)); dry_args.gpu = False
        for r in run_groups(args.run, groups, args=dry_args):
            print(f"  [{r['group']}/{r['check']}] {r['evidence']}")
            if r.get("note"):
                print(f"      note: {r['note']}")
        return

    rows = run_groups(args.run, groups, args=args)
    existing = load_existing_rows(args.out)
    merged = merge_rows(existing, rows, set(groups))
    scorecard = build_scorecard(merged, run=args.run,
                                generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    jpath, mpath = write_scorecard(args.out, scorecard)
    print(f"accepted={scorecard['accepted']}")
    print("wrote", jpath)
    print("wrote", mpath)


if __name__ == "__main__":
    main()
