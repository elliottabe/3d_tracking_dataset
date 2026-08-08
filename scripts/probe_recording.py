#!/usr/bin/env python3
"""Scale probe: robust body scale + pre-flight quality report for a BRAND-NEW
recording, WITHOUT any bout segmentation.

Key insight (verified): ``jarvis_jax.predict.sam3_driver.run_sam3_masks``
uses the bouts CSV for exactly ONE thing -- ``parse_bouts`` turns it into a
list of ``{bout_idx, start, end, n}`` frame windows, and everything
downstream (SAM3, ``scripts/run_bout.py``) is window-based. Nothing in the
pipeline knows what a "bout" (a courtship interaction) actually is. So a new
recording -- with no bout summary at all -- needs only a SYNTHETIC list of
evenly-spaced frame windows in the same CSV schema (``fly_id,bout_idx,
start_frame,end_frame``, ``fly_id`` == the session tag ``parse_bouts``
filters on -- see ``jarvis_jax.predict.sam3_driver.session_tag_for``) to
drive the existing SAM3 + ``run_bout.py`` machinery via
``recording.bouts_csv=<probe.csv>``. No changes to SAM3 are needed.

Workflow (three subcommands, mirroring the real GPU pipeline's stages):

  1. ``plan``     -- pure, no GPU/video-read beyond one ``cv2`` frame-count
                     query: plans ``n_windows`` evenly-spaced frame windows
                     and writes ``<out-root>/probe_bouts.csv``.
  2. ``commands`` -- prints the exact shell commands to run the SAM3 mask
                     stage and ``run_bout.py`` (limited to the 2D+
                     triangulation stages via ``pipeline.stop_after=
                     triangulate``, see ``scripts/run_bout.py``) over the
                     probe CSV. Does NOT run them (GPU work is the caller's
                     job, on a compute node).
  3. ``collect``  -- after those commands have been run, computes the
                     recording-level scale + a quality report over the probe
                     windows, dropping low-quality windows (out-of-view
                     flies, physically-inconsistent keypoints) instead of
                     letting them poison the estimate. Reuses (does not
                     reimplement) ``scripts/estimate_recording_scale.
                     estimate_run_root`` and ``scripts/mask_coverage.
                     coverage_report``.

Over-sampling is intentional: ``plan`` proposes ``n_windows`` (default 16)
but ``collect`` keeps only the best ``keep_windows`` (default 12) after
quality gating.

CLI:
    python scripts/probe_recording.py --session-dir <raw recording dir> \\
        --out-root <work dir> [--n-windows 16] [--window-len 60] [--keep 12] \\
        [--num-animals 2] [--recording-cfg session0] \\
        [--anatomy configs/anatomy/v1.yaml] {plan|commands|collect}
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scripts.mask_coverage import MIN_VIEWS_DEFAULT, coverage_report
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from mask_coverage import MIN_VIEWS_DEFAULT, coverage_report

try:
    from scripts.estimate_recording_scale import (
        WITHIN_BONE_CV_WARN_THRESH,
        _load_anatomy_cfg,
        estimate_run_root,
        segment_scale_diagnostics,
    )
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from estimate_recording_scale import (
        WITHIN_BONE_CV_WARN_THRESH,
        _load_anatomy_cfg,
        estimate_run_root,
        segment_scale_diagnostics,
    )

PROJECT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = PROJECT_DIR / "third_party" / "jarvis_jax"

CSV_FIELDS = ["fly_id", "bout_idx", "start_frame", "end_frame"]
COVERAGE_ORDER = {"ok": 0, "degraded": 1, "insufficient": 2}


# ---------------------------------------------------------------------------
# 1. Window planning -- pure, unit-testable.
# ---------------------------------------------------------------------------

def recording_length(session_dir) -> int:
    """Frame count of the recording, read from the first ``Cam*.mp4`` found
    under ``session_dir`` (no other metadata needed -- verified to work even
    on recordings with NO meta CSVs at all, e.g. Session0
    2025_10_20_13_20_04)."""
    session_dir = Path(session_dir)
    vids = sorted(glob.glob(str(session_dir / "Cam*.mp4")))
    if not vids:
        raise FileNotFoundError(
            f"recording_length: no Cam*.mp4 videos found under {session_dir}")
    import cv2
    cap = cv2.VideoCapture(vids[0])
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n <= 0:
        raise RuntimeError(
            f"recording_length: {vids[0]} reported a non-positive frame "
            f"count ({n}) -- corrupt/unreadable video?")
    return n


def plan_windows(n_frames: int, *, n_windows: int = 16, window_len: int = 60,
                 margin_frac: float = 0.02) -> List[Tuple[int, int]]:
    """``n_windows`` evenly-spaced, non-overlapping ``(start, length)``
    windows across ``[0, n_frames)``, skipping a ``margin_frac`` fraction of
    frames at each end (startup/teardown artifacts -- e.g. camera
    auto-exposure settling, sync jitter).

    Deterministic (pure function of the inputs). The margin is rounded UP
    (``math.ceil``) so every window strictly respects the requested
    ``margin_frac`` bound on both ends: ``start >= margin_frac * n_frames``
    and ``start + length <= n_frames - margin_frac * n_frames``.

    Windows are placed at ``lo + i * slot`` for ``slot = usable // n_windows``
    (integer floor division) -- since ``slot >= window_len`` is required to
    fit at all, consecutive windows can never overlap, and the last window is
    guaranteed (by construction, no rounding correction needed) to still fit
    inside the usable range.

    Raises ``ValueError`` if the request cannot fit (e.g. 100 windows of 60
    frames in a 1000-frame recording).
    """
    if n_frames <= 0:
        raise ValueError(f"plan_windows: n_frames must be positive, got {n_frames}")
    if n_windows <= 0:
        raise ValueError(f"plan_windows: n_windows must be positive, got {n_windows}")
    if window_len <= 0:
        raise ValueError(f"plan_windows: window_len must be positive, got {window_len}")
    if not (0 <= margin_frac < 0.5):
        raise ValueError(
            f"plan_windows: margin_frac must be in [0, 0.5), got {margin_frac}")

    margin = math.ceil(margin_frac * n_frames)
    lo, hi = margin, n_frames - margin
    usable = hi - lo
    slot = usable // n_windows if usable > 0 else 0

    if usable <= 0 or slot < window_len:
        raise ValueError(
            f"plan_windows: cannot fit n_windows={n_windows} windows of "
            f"window_len={window_len} frames inside the usable range "
            f"[{lo}, {hi}) ({usable} frames) of a {n_frames}-frame recording "
            f"with margin_frac={margin_frac} (slot size {slot} < window_len)")

    return [(lo + i * slot, window_len) for i in range(n_windows)]


def write_probe_csv(path, session_tag: str, windows) -> Path:
    """Write ``windows`` as a synthetic bouts CSV at ``path`` in the exact
    schema ``jarvis_jax.predict.sam3_driver.parse_bouts`` expects:
    ``fly_id,bout_idx,start_frame,end_frame`` with ``bout_idx`` 1..N and
    ``end_frame`` INCLUSIVE (``parse_bouts`` computes ``n = end - start + 1``
    -- this must match that exactly, since that ``n`` becomes the number of
    frames SAM3/run_bout actually reads for the window).

    ``fly_id`` is set to ``session_tag`` for every row so ``parse_bouts``'
    fly_id filter (active whenever the CSV has that column) keeps every
    window for this recording and rejects the CSV entirely for any other
    session tag -- this is the one piece of "identity" the synthetic CSV
    needs to carry.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_FIELDS)
        for i, (start, length) in enumerate(windows, start=1):
            end = int(start) + int(length) - 1  # inclusive
            writer.writerow([session_tag, i, int(start), end])
    os.replace(tmp, path)
    return path


def _print_window_table(windows: List[Tuple[int, int]]) -> None:
    header = f"{'idx':>4}  {'start':>10}  {'length':>7}  {'end':>10}"
    print(header)
    print("-" * len(header))
    for i, (start, length) in enumerate(windows, start=1):
        print(f"{i:>4}  {start:>10}  {length:>7}  {start + length - 1:>10}")


# ---------------------------------------------------------------------------
# 2. Commands -- the exact shell invocations for the two GPU stages.
# ---------------------------------------------------------------------------

def sam3_command(session_dir, probe_csv, out_root, *, num_animals: int = 2) -> str:
    """The SAM3 mask+identity front-end stage over every window in
    ``probe_csv`` (mirrors ``scripts/slurm_bout_array.py::
    build_sam3_array_script`` / ``docs/running_the_pipeline.md``, but a
    single non-array invocation covering all windows at once -- ``sam3.
    bout_ids`` left at its default ('' -> all bouts in the CSV) since the
    probe CSV contains nothing but this recording's own synthetic windows)."""
    masks_out = str(Path(out_root) / "sam3_masks")
    return (
        f"cd {PKG_DIR} && python -u scripts/sam3_masks.py "
        f"sam3.session_dir={session_dir} sam3.bouts_csv={probe_csv} "
        f"sam3.out={masks_out} sam3.num_animals={num_animals} "
        f"sam3.sam3_compile=false sam3.reuse_masks=true"
    )


def _bout_ids_override(bout_idxs: List[int]) -> str:
    """``bout_ids=<n>`` for a single window, ``bout_ids='1,2,...'`` (quoted,
    so Hydra parses it as a string, not a list) for several -- mirrors
    ``scripts/benchmark/run_variant.py::variant_commands``, the existing
    tested convention for driving ``run_bout.py`` over an explicit multi-bout
    set."""
    if len(bout_idxs) == 1:
        return str(bout_idxs[0])
    return "'" + ",".join(str(b) for b in bout_idxs) + "'"


def run_bout_command(recording_cfg: str, session_dir, probe_csv, out_root,
                     bout_idxs: List[int], *, num_animals: int = 2) -> str:
    """``run_bout.py`` limited to the 2D+triangulation stages (Stage A/B, plus
    optional B2 smoothing) via ``pipeline.stop_after=triangulate`` (see
    ``scripts/run_bout.py::should_stop_after_triangulate``) -- no STAC/
    silhouette/outputs/overlay, so this needs no ``mjcf``/silhouette
    resources beyond what the default config already composes.

    ``recording.session_dir``/``recording.predictions_dir``/``recording.
    bouts_csv``/``recording.num_animals`` are overridden onto the chosen
    ``recording_cfg`` group so calibration/cameras/rig config are reused from
    an existing recording config while session_dir + the (synthetic) bouts
    CSV + the SAM3 output dir point at the new recording's probe artifacts.
    """
    masks_out = str(Path(out_root) / "sam3_masks")
    bout_str = _bout_ids_override(bout_idxs)
    parts = [
        "cd", str(PROJECT_DIR), "&&", "python", "-u", "scripts/run_bout.py",
        "paths=hyak", f"recording={recording_cfg}",
        f"recording.session_dir={session_dir}",
        f"recording.predictions_dir={masks_out}",
        f"recording.bouts_csv={probe_csv}",
        f"recording.num_animals={num_animals}",
        f"outputs.out={out_root}",
        "pipeline.stop_after=triangulate",
        f"bout_ids={bout_str}",
    ]
    return " ".join(parts)


# ---------------------------------------------------------------------------
# 3. Collect -- recording-level scale + quality report over the probe windows.
# ---------------------------------------------------------------------------

def select_windows(window_quality: Dict[int, dict], *, keep: int,
                   cv_warn_thresh: float = WITHIN_BONE_CV_WARN_THRESH) -> dict:
    """Pure keep/drop selection over per-window quality summaries.

    ``window_quality``: ``{bout_idx: {"coverage_status": "ok"|"degraded"|
    "insufficient", "within_bone_cv": float}}``.

    A window is dropped outright if its coverage is ``"insufficient"`` (a fly
    is out of view in too many cameras -- see ``scripts.mask_coverage.
    coverage_report``) or its ``within_bone_cv`` exceeds ``cv_warn_thresh``
    (physically-inconsistent keypoints -- see ``scripts.
    estimate_recording_scale.segment_scale_diagnostics``); among the
    survivors, the best ``keep`` (ascending ``within_bone_cv``, ties broken
    by ascending ``bout_idx`` for a deterministic report) are kept and any
    excess is dropped as over-sampling.

    Returns ``{"kept": [bout_idx, ...] (sorted ascending),
    "dropped": {bout_idx: reason}}``.
    """
    dropped: Dict[int, str] = {}
    candidates: List[int] = []
    for idx, q in window_quality.items():
        if q["coverage_status"] == "insufficient":
            dropped[idx] = "insufficient_coverage"
        elif q["within_bone_cv"] > cv_warn_thresh:
            dropped[idx] = "high_within_bone_cv"
        else:
            candidates.append(idx)

    candidates.sort(key=lambda idx: (window_quality[idx]["within_bone_cv"], idx))
    kept = candidates[:keep]
    for idx in candidates[keep:]:
        dropped[idx] = "excess_oversample"

    return {"kept": sorted(kept), "dropped": dropped}


def _read_probe_csv(path) -> List[Tuple[int, int, int]]:
    """``[(bout_idx, start_frame, end_frame), ...]`` sorted by bout_idx, from
    a probe CSV written by ``write_probe_csv`` (any ``fly_id`` -- unlike
    ``parse_bouts`` this reads every row, since ``collect`` already knows it
    is looking at its own probe CSV)."""
    rows: List[Tuple[int, int, int]] = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((int(r["bout_idx"]), int(r["start_frame"]), int(r["end_frame"])))
    rows.sort(key=lambda t: t[0])
    return rows


def _kp3d_path(run_root, bout_idx: int, fly: int) -> Optional[Path]:
    """``<run_root>/bouts/bout_<idx:05d>/fly<fly>/{kp3d_filt.npz,kp3d.npz}``,
    preferring the filtered (smoothed) array -- mirrors ``scripts.
    estimate_recording_scale.bout_kp3d_paths``' own fallback, just for a
    single (bout, fly) instead of scanning the whole tree."""
    d = Path(run_root) / "bouts" / f"bout_{bout_idx:05d}" / f"fly{fly}"
    for name in ("kp3d_filt.npz", "kp3d.npz"):
        p = d / name
        if p.exists():
            return p
    return None


def _window_quality(run_root, predictions_dir, bout_idx: int, num_animals: int,
                    kp_names: List[str], mjcf_path: str, *,
                    min_views: int = MIN_VIEWS_DEFAULT) -> Tuple[Optional[dict], Optional[str]]:
    """One window's combined (worst-of-every-fly) coverage status +
    ``within_bone_cv``, or ``(None, reason)`` when the window is unusable
    (missing SAM3 masks or a fly's triangulated ``kp3d`` artifact -- e.g. the
    GPU commands haven't been run yet, or genuinely failed for that window).
    """
    bout_npz = Path(predictions_dir) / f"bout_{bout_idx:05d}" / "sam3_masks.npz"
    if not bout_npz.exists():
        return None, "missing_sam3_masks"

    coverage = coverage_report(str(bout_npz), min_views=min_views)
    per_fly: Dict[str, dict] = {}
    worst_status = "ok"
    worst_cv = 0.0
    for fly in range(num_animals):
        kp3d_path = _kp3d_path(run_root, bout_idx, fly)
        if kp3d_path is None:
            return None, f"missing_kp3d_fly{fly}"
        with np.load(kp3d_path) as z:
            kp3d = z["kp3d"]
        status = coverage["per_fly"].get(str(fly), {}).get("status", "insufficient")
        try:
            diag = segment_scale_diagnostics(kp3d, kp_names, mjcf_path)
            cv = float(diag["within_bone_cv"])
            if not math.isfinite(cv):
                cv = float("inf")
        except ValueError:
            cv = float("inf")
        per_fly[str(fly)] = {"coverage_status": status, "within_bone_cv": cv}
        if COVERAGE_ORDER[status] > COVERAGE_ORDER[worst_status]:
            worst_status = status
        worst_cv = max(worst_cv, cv)

    return {"coverage_status": worst_status, "within_bone_cv": worst_cv,
            "per_fly": per_fly, "n_cams": coverage["n_cams"]}, None


def _build_kept_view(run_root: Path, kept_idxs: List[int]) -> Path:
    """A throwaway temp dir mirroring ``run_root``'s structure but with
    ``bouts/`` containing only the KEPT windows (as symlinks to the real bout
    dirs) -- so ``estimate_run_root`` (which globs ``run_root/bouts/bout_*``
    with no bout-selection argument of its own) computes scale/identity over
    kept windows only, WITHOUT moving, copying, or otherwise mutating any of
    the real pose data under ``run_root``."""
    tmp_root = Path(tempfile.mkdtemp(prefix="probe_scale_"))
    bouts_dir = tmp_root / "bouts"
    bouts_dir.mkdir(parents=True, exist_ok=True)
    for idx in kept_idxs:
        src = Path(run_root) / "bouts" / f"bout_{idx:05d}"
        if src.exists():
            os.symlink(src.resolve(), bouts_dir / src.name)
    return tmp_root


def run_collect(*, out_root, num_animals: int, anatomy: str, keep: int,
                min_views: int = MIN_VIEWS_DEFAULT) -> dict:
    """The ``collect`` subcommand's implementation: reads the probe CSV +
    the pose artifacts the emitted commands should have already produced,
    quality-gates the windows, computes the final recording-level scale via
    the existing (unmodified) ``estimate_run_root``, and writes
    ``<out_root>/scale.json`` + ``<out_root>/probe_report.json``.
    """
    out_root = Path(out_root)
    probe_csv = out_root / "probe_bouts.csv"
    predictions_dir = out_root / "sam3_masks"
    windows = _read_probe_csv(probe_csv)
    if not windows:
        raise ValueError(f"run_collect: no windows found in {probe_csv} (run `plan` first)")

    cfg = _load_anatomy_cfg(anatomy)
    kp_names = list(cfg.model.KP_NAMES)
    mjcf_path = str(cfg.mjcf_path)

    per_window: Dict[int, dict] = {}
    usable_quality: Dict[int, dict] = {}
    unusable: Dict[int, str] = {}
    for idx, start, end in windows:
        quality, reason = _window_quality(
            out_root, predictions_dir, idx, num_animals, kp_names, mjcf_path,
            min_views=min_views)
        per_window[idx] = {"start_frame": start, "end_frame": end, "quality": quality}
        if quality is None:
            unusable[idx] = reason
        else:
            usable_quality[idx] = quality

    selection = select_windows(usable_quality, keep=keep)
    dropped = dict(selection["dropped"])
    dropped.update(unusable)
    kept = selection["kept"]

    if not kept:
        raise ValueError(
            f"run_collect: every window was dropped or unusable under {out_root} "
            f"-- dropped/unusable={dropped}")

    tmp_view = _build_kept_view(out_root, kept)
    try:
        scale_result = estimate_run_root(tmp_view, cfg, scale_keypoints="rigid_segment")
    finally:
        import shutil
        shutil.rmtree(tmp_view, ignore_errors=True)

    report = {
        "out_root": str(out_root),
        "n_windows": len(windows),
        "n_kept": len(kept),
        "kept_windows": kept,
        "dropped_windows": dropped,
        "per_window": per_window,
        "scale": scale_result,
        "identity": scale_result["identity"],
        "identity_reason": scale_result["identity_reason"],
    }

    scale_path = out_root / "scale.json"
    report_path = out_root / "probe_report.json"
    for path, payload in ((scale_path, scale_result), (report_path, report)):
        tmp = str(path) + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True,
                    help="raw recording dir (contains Cam*.mp4 + calibration/)")
    ap.add_argument("--out-root", required=True, help="probe working/output dir")
    ap.add_argument("--n-windows", type=int, default=16)
    ap.add_argument("--window-len", type=int, default=60)
    ap.add_argument("--margin-frac", type=float, default=0.02)
    ap.add_argument("--keep", type=int, default=12,
                    help="best-by-quality windows kept by `collect` (default 12)")
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--recording-cfg", default="session0",
                    help="Hydra `recording=` group reused for calibration/cameras/rig config")
    ap.add_argument("--anatomy", default="configs/anatomy/v1.yaml")
    ap.add_argument("--min-views", type=int, default=MIN_VIEWS_DEFAULT)
    ap.add_argument("mode", choices=["plan", "commands", "collect"])
    return ap


def main(argv=None) -> int:
    args = _build_argparser().parse_args(argv)
    out_root = Path(args.out_root)
    probe_csv = out_root / "probe_bouts.csv"

    if args.mode == "plan":
        try:
            from jarvis_jax.predict.sam3_driver import session_tag_for
        except ModuleNotFoundError:
            def session_tag_for(session_dir):  # noqa: ANN001 -- CLI fallback
                return "/".join(str(session_dir).rstrip("/").split("/")[-2:])

        n_frames = recording_length(args.session_dir)
        windows = plan_windows(n_frames, n_windows=args.n_windows,
                               window_len=args.window_len, margin_frac=args.margin_frac)
        session_tag = session_tag_for(args.session_dir)
        write_probe_csv(probe_csv, session_tag, windows)
        print(f"[probe_recording] session_dir={args.session_dir}")
        print(f"[probe_recording] recording_length={n_frames} frames")
        print(f"[probe_recording] session_tag={session_tag}")
        _print_window_table(windows)
        print(f"\nwrote {probe_csv}")
        return 0

    if args.mode == "commands":
        if not probe_csv.exists():
            print(f"error: {probe_csv} not found -- run `plan` first", file=sys.stderr)
            return 1
        windows = _read_probe_csv(probe_csv)
        bout_idxs = [idx for idx, _s, _e in windows]
        print("# Stage 0: SAM3 masks over every probe window")
        print(sam3_command(args.session_dir, probe_csv, out_root,
                           num_animals=args.num_animals))
        print()
        print("# Stage A/B: 2D keypoints + triangulation only (pipeline.stop_after=triangulate)")
        print(run_bout_command(args.recording_cfg, args.session_dir, probe_csv, out_root,
                               bout_idxs, num_animals=args.num_animals))
        return 0

    if args.mode == "collect":
        report = run_collect(out_root=out_root, num_animals=args.num_animals,
                             anatomy=args.anatomy, keep=args.keep,
                             min_views=args.min_views)
        print(f"[probe_recording] {report['n_kept']}/{report['n_windows']} windows kept")
        if report["dropped_windows"]:
            print(f"[probe_recording] dropped: {report['dropped_windows']}")
        print(f"[probe_recording] identity={report['identity']} ({report['identity_reason']})")
        print(f"[probe_recording] scale={report['scale']['scale']:.6f}")
        print(f"wrote {out_root / 'scale.json'}")
        print(f"wrote {out_root / 'probe_report.json'}")
        return 0

    return 1  # pragma: no cover -- argparse `choices` makes this unreachable


if __name__ == "__main__":
    raise SystemExit(main())
