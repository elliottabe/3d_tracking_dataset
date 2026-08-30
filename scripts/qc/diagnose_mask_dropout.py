#!/usr/bin/env python3
"""Task 19 (Phase 5) diagnostic: locate the bout-28 fly0 dropout in the stage
chain. DIAGNOSIS ONLY -- this script reads sam3_masks.npz + the archived
Task-1 kp2d/kp3d/qc_perframe arrays and reproduces the pipeline's OWN gates
in read-only numpy; it never touches pipeline source, configs, or re-runs
anything. See .superpowers/sdd/2026-08-29-coarse-to-fine-3d/task-19-brief.md.

Background (why this script exists): Task 1's fix round attributed the
bout-28 fly0 (female) NaN-3D stretch (frames 1502-2006) to "the SAM3 mask
drifts onto empty background". Direct measurement of sam3_masks.npz
contradicts that -- masks are smooth (centroid step 0.1-3.2 px) and never
merge with fly1 (separation ~250-300 px throughout); fly0 is instead marked
`valid=False` in up to 4 of 7 cameras from ~frame 1502, with `in_frame`
byte-identical to `valid` (SAM3 flagged her out-of-frame, not mis-segmented).
Three cameras (Cam2012853, Cam2012855, Cam2012630) stay 100% valid through
the whole dead zone -- more than DLT's >=2-view minimum -- yet the bout's
final 3D (qc_perframe.npz `reproj_px`) is NaN for a clean, contiguous 100% of
frames 1502-2006. This script finds the gate(s) responsible.

Camera-identity warning: sam3_masks.npz's own `cameras` field is NOT the
order kp2d.npz/kp3d.npz use. `scripts/run_bout.py` loads masks via
`load_bout_masks(..., expected_cameras=list(cfg.recording.cameras))`, which
name-reorders the mask arrays into `cfg.recording.cameras` order (see
configs/recording/session0.yaml) -- THAT order (not the npz's stored order)
is what kp2d/kp3d's camera axis is in. Every function below that compares
mask fields against kp2d/kp3d takes care of this reorder; never index the
raw npz's `valid`/`centroids`/`packed` directly against kp2d without it.

Subcommands (`python scripts/qc/diagnose_mask_dropout.py <cmd> ...`):
  report          Step 2/3 per-camera per-stage survival table (dead + live
                  frame) as JSON, plus the Step-3 repair-trigger determination.
  attrition       Step 5 per-frame attrition figure (also written by `all`).
  reproject-check Step 4 reprojection-in-bounds check at chosen frames.
  render-valid    Step 1 valid-honouring raw-frame mask overlay grid.
  all             Runs every step above and writes the full report + figures
                  under figures/2026-08-29-c2f-3d/phase5-dropout/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "third_party" / "jarvis_jax")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Gate constants -- kept as plain floats/ints (not imported from
# jarvis_jax.tracking.triangulate) so this script stays jax-free and cheap to
# run repeatedly on a shared GPU node; each is cited to its source of truth so
# it can be diffed against the live config by hand.
# ---------------------------------------------------------------------------
CONF_THRESH = 0.3                  # configs/detector/vitpose_v3.yaml: conf_thresh
VIEW_CONF_THRESH = 0.6             # configs/detector/vitpose_v3.yaml: view_conf_thresh
REPROJ_RESID_PX = 10.0             # configs/detector/vitpose_v3.yaml: reproj_resid_px
TRIGGER_FACTOR = 3.0               # jarvis_jax/tracking/triangulate.py: TRIGGER_FACTOR
MIN_CONSENSUS_VIEWS = 3            # jarvis_jax/tracking/triangulate.py: MIN_CONSENSUS_VIEWS
MIN_VIEWS = 4                      # configs/pipeline.yaml: masks.min_views
KP_MASK_AGREE_FLY_LENGTHS = 3.0    # configs/pipeline.yaml: masks.kp_mask_agree_fly_lengths
NAN_SOLVE_MAX_GAP = 10             # scripts/run_bout.py: NAN_SOLVE_MAX_GAP
NAN_SOLVE_MIN_SEG = 30             # scripts/run_bout.py: NAN_SOLVE_MIN_SEG
IMG_W, IMG_H = 1936, 448           # native camera resolution (== sam3_masks 'shape')

DEFAULT_BOUT = 28
DEFAULT_SPLIT = 1502
DEFAULT_TARGETS = (1550, 1700, 1900)
FIG_DIR = REPO_ROOT / "figures" / "2026-08-29-c2f-3d" / "phase5-dropout"


# ---------------------------------------------------------------------------
# Recording metadata (camera order, calib/session/predictions dirs) --
# reuses viz.config.courtship_recording so this script agrees with every
# other viz tool on what "camera order" and "which files" mean.
# ---------------------------------------------------------------------------
def recording_info(config_name: str = "pipeline") -> dict:
    from viz.config import courtship_recording
    return courtship_recording(config_name=config_name)


# ---------------------------------------------------------------------------
# Mask loading + camera reordering (native npz order -> recording.cameras
# order, matching what kp2d.npz/kp3d.npz are actually indexed by).
# ---------------------------------------------------------------------------
def load_fly_masks_reordered(mask_npz: str, fly: int, recording_cameras: list[str]) -> dict:
    """valid/centroids/in_frame/area for `fly`, all (T,C) or (T,C,2), reordered
    into `recording_cameras` order. Area is popcount over the bit-packed mask
    (exact: this rig's W=1936 is a multiple of 8, so packbits pads nothing),
    which avoids materialising the full (T,C,H,W) bool array."""
    with np.load(mask_npz, allow_pickle=True) as z:
        native_cams = [str(c) for c in np.asarray(z["cameras"]).tolist()]
        perm = np.array([native_cams.index(c) for c in recording_cameras], dtype=int)
        valid = np.asarray(z["valid"])[fly][perm].T                       # (T,C)
        in_frame = np.asarray(z["in_frame"])[fly][perm].T                 # (T,C)
        centroids = np.asarray(z["centroids"], np.float32)[fly][perm].transpose(1, 0, 2)  # (T,C,2)
        packed = np.asarray(z["packed"])[fly][perm]                       # (C,T,H,Wbytes)
        area = np.bitwise_count(packed).sum(axis=(-2, -1)).T.astype(np.int64)  # (T,C)
    return dict(valid=valid, in_frame=in_frame, centroids=centroids, area=area,
                cameras=list(recording_cameras), native_cameras=native_cams, perm=perm)


# ---------------------------------------------------------------------------
# Per-view / per-frame gates -- faithful, read-only copies of the logic in
# scripts/run_bout.py (view_mask_agreement, gate_low_coverage_frames) and
# scripts/mask_coverage.py (per_frame_views), reproduced here rather than
# imported so this script never imports run_bout.py's module (which drags in
# torch/jax/mujoco at import time via jarvis_jax.tracking.stac/polish/outputs
# -- a needless GPU-memory cost on a shared node for what is pure-numpy
# analysis of already-computed arrays). Cite scripts/run_bout.py if this ever
# needs re-checked against the live source.
# ---------------------------------------------------------------------------
def view_mask_agreement(kp2d, centroids, valid, mask_areas, *, max_fly_lengths):
    """(T,C) bool -- copy of scripts/run_bout.py:view_mask_agreement. Did this
    view's 2-D prediction (median over keypoints) land within
    `max_fly_lengths * sqrt(mask_area)` of the mask centroid SAM3 found here?"""
    valid = np.asarray(valid, bool)
    if max_fly_lengths is None:
        return valid.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(np.asarray(kp2d, float), axis=2)      # (T,C,2)
    d = np.linalg.norm(centre - np.asarray(centroids, float), axis=-1)   # (T,C)
    scale = np.sqrt(np.maximum(np.asarray(mask_areas, float), 1.0))
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(d) & (d <= float(max_fly_lengths) * scale)
    return ok & valid


def finite_frame_mask(kp3d) -> np.ndarray:
    """(T,) bool -- copy of scripts/run_bout.py:finite_frame_mask. ALL
    keypoints (all 3 coords) finite -- the all-or-nothing criterion STAC's
    NaN-robust solve uses to decide whether a frame is usable input."""
    a = np.asarray(kp3d)
    return np.isfinite(a).all(axis=tuple(range(1, a.ndim)))


def fill_short_gaps_mask(ok: np.ndarray, max_gap: int = NAN_SOLVE_MAX_GAP) -> np.ndarray:
    """(T,) bool -- which False entries of `ok` would be bridged by
    scripts/run_bout.py:fill_short_gaps's <= max_gap interpolation."""
    ok = np.asarray(ok, bool)
    filled = np.zeros(len(ok), bool)
    idx = np.flatnonzero(ok)
    if len(idx) < 2:
        return filled
    for lo, hi in zip(idx[:-1], idx[1:]):
        n = hi - lo - 1
        if 0 < n <= max_gap:
            filled[lo + 1:hi] = True
    return filled


def contiguous_segments(ok, min_len: int = NAN_SOLVE_MIN_SEG):
    """[(start, stop)) -- copy of scripts/run_bout.py:contiguous_segments."""
    ok = np.asarray(ok, bool)
    out, start = [], None
    for i, v in enumerate(ok):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i))
            start = None
    if start is not None and len(ok) - start >= min_len:
        out.append((start, len(ok)))
    return out


# ---------------------------------------------------------------------------
# Step 2 / Step 3: per-camera per-stage survival table + repair-trigger check
# ---------------------------------------------------------------------------
def per_camera_stage_table(mask_npz, kp2d, conf, masks, frame: int) -> list[dict]:
    """One row per camera for a single frame: which of the 6 Step-2 gates the
    female (fly0) survives at that (frame, camera)."""
    view_med = np.median(conf[frame], axis=-1)          # (C,)
    rows = []
    for c, cam in enumerate(masks["cameras"]):
        any_kp_finite = bool(np.isfinite(kp2d[frame, c]).all(axis=-1).any())
        any_conf_pass = bool((conf[frame, c] >= CONF_THRESH).any())
        vmed = float(view_med[c])
        rows.append(dict(
            camera=cam,
            mask_valid=bool(masks["valid"][frame, c]),
            in_frame=bool(masks["in_frame"][frame, c]),
            mask_area_px=int(masks["area"][frame, c]),
            keypoints_emitted=any_kp_finite,
            per_kp_conf_pass=any_conf_pass,          # any kp >= CONF_THRESH
            view_median_conf=round(vmed, 3),
            view_conf_pass=bool(vmed >= VIEW_CONF_THRESH),
            centroid_disagree_px=None,   # filled by caller when kp2d/centroids available
        ))
    return rows


def dropout_report(mask_npz: str, kp2d_path: str, kp3d_path: str, *, split_frame: int,
                    recording_cameras: list[str] | None = None, fly: int = 0,
                    example_dead: int = 1600, example_live: int = 1400) -> dict:
    """Per-camera per-stage survival counts across the whole bout for `fly`,
    plus the two worked frame examples the brief's Step 2 table asks for.

    Returns a dict with:
      cameras, T
      per_frame: views_raw, views_agree, frame_gated (min_views), any_kp3d_finite,
                 all_kp3d_finite, n_kp3d_finite (0..50)
      examples: {offset: [per-camera row, ...]} for example_dead/example_live
      split_summary: before/after `split_frame` aggregate stats
      stac_segments: contiguous_segments(...) over the whole bout (the ONLY
                     runs STAC would ever attempt to solve)
    """
    if recording_cameras is None:
        recording_cameras = recording_info()["cameras"]
    masks = load_fly_masks_reordered(mask_npz, fly, recording_cameras)
    with np.load(kp2d_path) as z:
        kp2d, conf = z["kp2d"], z["conf"]
    with np.load(kp3d_path) as z:
        kp3d = z["kp3d"]

    T = kp2d.shape[0]
    agree = view_mask_agreement(kp2d, masks["centroids"], masks["valid"], masks["area"],
                                 max_fly_lengths=KP_MASK_AGREE_FLY_LENGTHS)
    views_raw = masks["valid"].sum(axis=1)
    views_agree = agree.sum(axis=1)
    frame_gated = views_agree < MIN_VIEWS               # masks.min_views post-hoc NaN

    finite_per_kp = np.isfinite(kp3d).all(axis=-1)       # (T,K)
    n_kp3d_finite = finite_per_kp.sum(axis=1)
    all_finite = finite_frame_mask(kp3d)                 # STAC's own "ok" criterion
    any_finite = n_kp3d_finite > 0

    filled = fill_short_gaps_mask(all_finite, NAN_SOLVE_MAX_GAP)
    stac_segments = contiguous_segments(all_finite | filled, NAN_SOLVE_MIN_SEG)

    def _example(offset):
        rows = per_camera_stage_table(mask_npz, kp2d, conf, masks, offset)
        d = np.linalg.norm(
            np.nanmedian(kp2d[offset], axis=1) - masks["centroids"][offset], axis=-1)
        for row, dd in zip(rows, d):
            row["centroid_disagree_px"] = None if not np.isfinite(dd) else round(float(dd), 1)
            row["mask_agree"] = bool(agree[offset, recording_cameras.index(row["camera"])])
        return dict(
            offset=offset,
            views_raw=int(views_raw[offset]), views_agree=int(views_agree[offset]),
            frame_gated_min_views=bool(frame_gated[offset]),
            n_kp3d_finite=int(n_kp3d_finite[offset]),
            per_camera=rows,
        )

    split = int(split_frame)
    report = dict(
        bout_mask_npz=str(mask_npz), kp2d_path=str(kp2d_path), kp3d_path=str(kp3d_path),
        fly=fly, cameras=recording_cameras, T=int(T), split_frame=split,
        gates=dict(conf_thresh=CONF_THRESH, view_conf_thresh=VIEW_CONF_THRESH,
                   reproj_resid_px=REPROJ_RESID_PX, trigger_factor=TRIGGER_FACTOR,
                   min_consensus_views=MIN_CONSENSUS_VIEWS, min_views=MIN_VIEWS,
                   kp_mask_agree_fly_lengths=KP_MASK_AGREE_FLY_LENGTHS,
                   nan_solve_max_gap=NAN_SOLVE_MAX_GAP, nan_solve_min_seg=NAN_SOLVE_MIN_SEG),
        per_frame=dict(views_raw=views_raw.tolist(), views_agree=views_agree.tolist(),
                       frame_gated_min_views=frame_gated.tolist(),
                       n_kp3d_finite=n_kp3d_finite.tolist(),
                       any_kp3d_finite=any_finite.tolist(), all_kp3d_finite=all_finite.tolist()),
        split_summary=dict(
            before=dict(n=split, frame_gated_frac=float(frame_gated[:split].mean()),
                       all_finite_frac=float(all_finite[:split].mean()),
                       any_finite_frac=float(any_finite[:split].mean())),
            after=dict(n=T - split, frame_gated_frac=float(frame_gated[split:].mean()),
                      all_finite_frac=float(all_finite[split:].mean()),
                      any_finite_frac=float(any_finite[split:].mean())),
        ),
        stac_segments=[[int(a), int(b)] for a, b in stac_segments],
        stac_segments_after_split=[[int(a), int(b)] for a, b in stac_segments if a >= split],
        examples={str(example_dead): _example(example_dead), str(example_live): _example(example_live)},
    )
    return report


def repair_trigger_check(mask_npz: str, fly: int = 0) -> dict:
    """Step 3: did repair_outliers / repair_missing fire for this fly, and if
    not, which of (a)/(b)/(c) in the brief explains it? Reads only the npz's
    own bookkeeping fields (`suspect_cameras`, `gap_repair`, `in_frame` vs
    `valid`) -- no re-running of SAM3."""
    with np.load(mask_npz, allow_pickle=True) as z:
        has_suspect = "suspect_cameras" in z.files
        has_gap_repair = "gap_repair" in z.files
        valid = np.asarray(z["valid"])[fly].astype(bool)
        in_frame = np.asarray(z["in_frame"])[fly]
        suspect = [str(c) for c in np.asarray(z["suspect_cameras"]).tolist()] if has_suspect else []
        gap_repair = json.loads(str(np.asarray(z["gap_repair"]))) if has_gap_repair else []
    in_frame_bool = in_frame.astype(bool)
    identical = bool(np.array_equal(in_frame_bool, valid))
    n_invalid = int((~valid).sum())
    n_in_frame_yes_but_invalid = int(((~valid) & in_frame_bool).sum())  # would be a "gap"
    return dict(
        repair_outliers_fired=has_suspect, suspect_cameras=suspect,
        repair_missing_fired=has_gap_repair, gap_repair_report=gap_repair,
        in_frame_equals_valid=identical,
        n_invalid_views=n_invalid,
        n_in_frame_yes_but_invalid=n_in_frame_yes_but_invalid,
        conclusion=(
            "(b): repair_missing's own in_frame_codes geometric check "
            "(triangulate from >=3 valid cams, reproject, test image bounds) "
            "resolved EVERY invalid fly0 view as out-of-FOV (in_frame==valid "
            "bit-for-bit; 0 views coded IN_FRAME_YES-but-invalid), so "
            "find_gap_cameras() found zero (fly,camera) gaps to repair -- "
            "repair_missing never had a candidate, it did not fail an accept "
            "threshold. repair_outliers is similarly inapplicable ((b), not "
            "(a)): its per-camera residual is NaN for an absent (not merely "
            "wrong) mask, so these cameras never entered its candidate pool "
            "either (no suspect_cameras written)."
            if (not has_suspect and not has_gap_repair and identical
                and n_in_frame_yes_but_invalid == 0) else
            "See fields above -- does not match the (b) pattern; inspect directly."
        ),
    )


# ---------------------------------------------------------------------------
# Step 4: is fly0 genuinely out of view, or recoverably mis-tracked?
# ---------------------------------------------------------------------------
def reprojection_check(kp3d_fly_path: str, kp3d_other_path: str, calib_dir: str,
                       recording_cameras: list[str], mask_npz: str, fly: int,
                       targets, kp_index: int = 0, search_radius: int = 80) -> dict:
    """Reproject the nearest-finite trunk-keypoint 3D position (kp_index,
    default 0 == Scutellum) for `fly` and for the other fly into every camera
    at each frame in `targets`, and test image-bounds membership. The OTHER
    fly (always 7/7 valid) is included as a sanity check on the reprojection
    math itself: it must land in-bounds everywhere or the calibration/tool is
    suspect, not the female's out-of-view finding."""
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    rt = ReprojectionTool(calib_dir)
    if list(rt.cameras.keys()) != list(recording_cameras):
        raise RuntimeError(
            f"ReprojectionTool camera order {list(rt.cameras.keys())} != "
            f"recording.cameras {recording_cameras} -- do not reuse this "
            f"function's output until that's reconciled.")

    with np.load(kp3d_fly_path) as z:
        kp3d_fly = z["kp3d"]
    with np.load(kp3d_other_path) as z:
        kp3d_other = z["kp3d"]
    masks = load_fly_masks_reordered(mask_npz, fly, recording_cameras)

    def nearest_finite(finite, target):
        for r in range(search_radius):
            for t in (target - r, target + r):
                if 0 <= t < len(finite) and finite[t]:
                    return t
        return None

    finite_fly = np.isfinite(kp3d_fly[:, kp_index]).all(axis=-1)
    finite_other = np.isfinite(kp3d_other[:, kp_index]).all(axis=-1)

    out = {"kp_index": kp_index, "targets": {}}
    for tgt in targets:
        entry = {}
        for label, kp3d, finite in ((f"fly{fly}", kp3d_fly, finite_fly),
                                    ("other_fly", kp3d_other, finite_other)):
            src = nearest_finite(finite, tgt)
            if src is None:
                entry[label] = {"source_frame": None, "cameras": {}}
                continue
            X = kp3d[src, kp_index].astype(np.float64)
            uv = rt.reproject_point(X)
            cams = {}
            for c, cam in enumerate(recording_cameras):
                x, y = float(uv[c, 0]), float(uv[c, 1])
                inb = np.isfinite(x) and np.isfinite(y) and (0 <= x < IMG_W) and (0 <= y < IMG_H)
                row = {"u_px": round(x, 1), "v_px": round(y, 1), "in_bounds": bool(inb)}
                if label == f"fly{fly}":
                    row["mask_valid_at_target"] = bool(masks["valid"][tgt, c])
                    row["in_frame_at_target"] = bool(masks["in_frame"][tgt, c])
                cams[cam] = row
            entry[label] = {"source_frame": int(src), "source_delta": int(src - tgt),
                            "cameras": cams}
        out["targets"][str(tgt)] = entry
    return out


# ---------------------------------------------------------------------------
# Step 1: valid-honouring raw-frame mask overlay (settles the Task-1
# contradiction -- drifted mask vs. absent-and-flagged).
# ---------------------------------------------------------------------------
def render_valid_overlay(run_out: Path, bout: int, frames, recording_cameras,
                         session_dir: str, predictions_dir: str, out_path: Path) -> Path:
    """One PNG: rows = `frames`, cols = all 7 cameras. Draws fly0's mask (cyan,
    PALETTE["fly0"]) and fly1's mask (orange, PALETTE["fly1"]) ONLY where each
    fly's own valid[fly, cam, frame] is True, and stamps each panel with
    fly0's 'valid=T/F' -- so a panel with NO cyan overlay and 'valid=False' is
    "absent and flagged", not "drifted"; a cyan mask floating off the real
    animal (however unlikely given Step 2/3's numbers) would be the drift
    Task 1 described. Drawing fly1 too (matching viz.views.maskvid's
    convention) disambiguates identity: a fly visible in a valid=False panel
    is very often the OTHER fly wandering through that camera's view, not
    evidence fly0 herself was missed -- an orange mask on it settles that at
    a glance instead of requiring a numeric cross-check."""
    import cv2
    from hydra import initialize_config_dir, compose
    from viz.core import io as vio
    from viz.core import overlays
    from viz.core import colors as vcolors

    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "configs")):
        cfg = compose(config_name="pipeline")
    from scripts.run_bout import bout_start_frame   # lazy: pulls in heavy pipeline deps once
    start_abs = bout_start_frame(cfg, bout)

    masks0 = vio.load_masks(predictions_dir, bout, 0, recording_cameras)  # fly0 (cyan)
    masks1 = vio.load_masks(predictions_dir, bout, 1, recording_cameras)  # fly1 (orange)

    panel_h = 260
    rows = []
    for f in frames:
        imgs = list(next(iter(vio.read_frames_synced(session_dir, recording_cameras,
                                                       start_abs + f, 1))))
        cols = []
        for c, cam in enumerate(recording_cameras):
            rgb = imgs[c]
            bgr = (cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb is not None
                  else np.zeros((IMG_H, IMG_W, 3), np.uint8))
            v = bool(masks0["valid"][f, c])
            v1 = bool(masks1["valid"][f, c])
            if v1:
                overlays.draw_mask(bgr, masks1["masks"][f, c], vcolors.PALETTE["fly1"])
            if v:
                overlays.draw_mask(bgr, masks0["masks"][f, c], vcolors.PALETTE["fly0"])
            panel_w = int(round(panel_h * IMG_W / IMG_H))
            panel = cv2.resize(bgr, (panel_w, panel_h))
            label = f"{cam} f{f} fly0.valid={v} fly1.valid={v1}"
            cv2.rectangle(panel, (0, 0), (panel_w, 22), (0, 0, 0), -1)
            cv2.putText(panel, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                       (0, 255, 0) if v else (0, 0, 255), 1, cv2.LINE_AA)
            cols.append(panel)
        rows.append(np.hstack(cols))
    grid = np.vstack(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path


# ---------------------------------------------------------------------------
# Step 5: attrition figure
# ---------------------------------------------------------------------------
def attrition_figure(report: dict, out_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf = report["per_frame"]
    T = report["T"]
    x = np.arange(T)
    views_raw = np.array(pf["views_raw"])
    views_agree = np.array(pf["views_agree"])
    n_kp3d_finite = np.array(pf["n_kp3d_finite"])
    split = report["split_frame"]

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(x, views_raw, color="#4C72B0", lw=0.8, label="mask present (valid cams)")
    ax[0].plot(x, views_agree, color="#DD8452", lw=0.8,
              label="mask present AND kp-agrees (views_per_frame)")
    ax[0].axhline(MIN_VIEWS, color="grey", ls="--", lw=1, label=f"min_views={MIN_VIEWS}")
    ax[0].axvline(split, color="k", ls=":", lw=1)
    ax[0].set_ylabel("cameras surviving")
    ax[0].set_title("bout 28, fly0 (female): per-frame gate attrition")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].plot(x, n_kp3d_finite, color="#55A868", lw=0.8)
    ax[1].axvline(split, color="k", ls=":", lw=1)
    ax[1].axhline(50, color="grey", ls="--", lw=1)
    ax[1].set_ylabel("kp3d.npz\nkeypoints finite (/50)")

    all_finite = np.array(pf["all_kp3d_finite"], dtype=bool)
    ax[2].fill_between(x, 0, all_finite.astype(int), step="pre", color="#55A868",
                       label="all 50 finite (STAC-usable frame)")
    for a, b in report["stac_segments"]:
        ax[2].axvspan(a, b, color="#55A868", alpha=0.15)
    ax[2].axvline(split, color="k", ls=":", lw=1, label=f"split={split}")
    ax[2].set_ylim(-0.1, 1.3)
    ax[2].set_ylabel("STAC-usable\n(all-or-nothing)")
    ax[2].set_xlabel("frame (bout offset)")
    ax[2].legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _default_paths(bout: int):
    rec = recording_info()
    mask_npz = os.path.join(rec["predictions_dir"], f"bout_{bout:05d}", "sam3_masks.npz")
    arrays = (REPO_ROOT / "figures" / "2026-08-29-c2f-3d" / "phase0-baseline" / "arrays"
             / f"bout_{bout:05d}")
    return rec, mask_npz, arrays


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["report", "attrition", "reproject-check",
                                     "render-valid", "repair-check", "all"])
    ap.add_argument("--bout", type=int, default=DEFAULT_BOUT)
    ap.add_argument("--fly", type=int, default=0)
    ap.add_argument("--split", type=int, default=DEFAULT_SPLIT)
    ap.add_argument("--dead-frame", type=int, default=1600)
    ap.add_argument("--live-frame", type=int, default=1400)
    ap.add_argument("--targets", type=int, nargs="+", default=list(DEFAULT_TARGETS))
    ap.add_argument("--out-dir", type=str, default=str(FIG_DIR))
    args = ap.parse_args()

    rec, mask_npz, arrays = _default_paths(args.bout)
    fly_dir = arrays / f"fly{args.fly}"
    other_fly = 1 - args.fly
    other_dir = arrays / f"fly{other_fly}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    if args.cmd in ("report", "all"):
        rpt = dropout_report(mask_npz, str(fly_dir / "kp2d.npz"), str(fly_dir / "kp3d.npz"),
                             split_frame=args.split, recording_cameras=rec["cameras"],
                             fly=args.fly, example_dead=args.dead_frame,
                             example_live=args.live_frame)
        result["report"] = rpt
        (out_dir / "dropout_report.json").write_text(json.dumps(rpt, indent=2))
        print(f"[report] wrote {out_dir / 'dropout_report.json'}")

    if args.cmd in ("repair-check", "all"):
        rep = repair_trigger_check(mask_npz, fly=args.fly)
        result["repair_check"] = rep
        (out_dir / "repair_check.json").write_text(json.dumps(rep, indent=2))
        print(f"[repair-check] wrote {out_dir / 'repair_check.json'}")

    if args.cmd in ("reproject-check", "all"):
        rc = reprojection_check(str(fly_dir / "kp3d.npz"), str(other_dir / "kp3d.npz"),
                                rec["calib_dir"], rec["cameras"], mask_npz, args.fly,
                                args.targets)
        result["reproject_check"] = rc
        (out_dir / "reproject_check.json").write_text(json.dumps(rc, indent=2))
        print(f"[reproject-check] wrote {out_dir / 'reproject_check.json'}")

    if args.cmd in ("attrition", "all"):
        rpt = result.get("report") or dropout_report(
            mask_npz, str(fly_dir / "kp2d.npz"), str(fly_dir / "kp3d.npz"),
            split_frame=args.split, recording_cameras=rec["cameras"], fly=args.fly)
        p = attrition_figure(rpt, out_dir / "step5_attrition.png")
        print(f"[attrition] wrote {p}")

    if args.cmd in ("render-valid", "all"):
        p = render_valid_overlay(None, args.bout, args.targets, rec["cameras"],
                                 rec["session_dir"], rec["predictions_dir"],
                                 out_dir / "step1_valid_overlay.png")
        print(f"[render-valid] wrote {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
