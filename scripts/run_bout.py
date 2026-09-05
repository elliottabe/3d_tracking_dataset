#!/usr/bin/env python3
"""Resumable per-bout courtship inference driver (stages A-E).

For a bout index and each fly (``range(cfg.recording.num_animals)``), runs:

  A  ViTPose 2-D on SAM3-masked crops                       -> kp2d.npz
  B  DLT triangulation                                      -> kp3d.npz
  C  STAC ik_only (offsets fit once, shared across bouts)   -> stac_ik.h5
  D  model->mm bridge (per-frame s,R,t)                     -> qpos_refined.npz
  D2 wing-pitch vs the SAM masks (OPT-IN, off by default)   -> qpos_wingfit.npz
  E  FK outputs.h5 + qc.json + qc_perframe.npz + per-camera reprojection overlay videos

Every artifact is written atomically (tmp -> os.replace) and every stage is
skipped when its artifact already exists (see jarvis_jax.tracking.resume),
so a preempted/resumed run picks up where it left off. A ``DONE`` marker per
``<run_root>/bouts/bout_<idx:05d>/fly<f>/`` gates re-processing an already
completed bout/fly.

This driver is SHARED between courtship (``cfg.recording.num_animals == 2``)
and free-running (``num_animals == 1``) -- ``range(cfg.recording.num_animals)``
above is the only branch point. In particular ``configs/pipeline.yaml``'s
``scaling.scale_keypoints: rigid_segment`` default (Task 16) applies to BOTH
assays: free-running silently inherits it too. This is structurally safe --
same rig, same anatomy config, and the rigid-segment estimator has no
sex.json/identity dependency (unlike the per-fly canonicalization path) --
but it is a real, deliberate change of behaviour for free-running runs, not
just courtship ones.

Usage:
    python scripts/run_bout.py paths=hyak +bout_ids=3
    python scripts/run_bout.py paths=hyak            # bout_ids='' -> all bouts
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import sys
import glob
import json
import re
import time

from pathlib import Path

import warnings

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from jarvis_jax.tracking.resume import (
    atomic_save_npz, atomic_save_json, stage_done, mark_done, bout_complete)
from jarvis_jax.tracking.bout_masks import load_bout_masks, check_bout_camera_order
from jarvis_jax.tracking.predict_2d import (
    load_detector, predict_bout_2d, reorder_detector_to_model,
    verify_detector_kp_order)
from jarvis_jax.tracking.triangulate import (triangulate_keypoints,
                                             view_median_conf)
from jarvis_jax.tracking.filter import filter_bout_kp3d
from jarvis_jax.tracking.scale import compute_trunk_scale
from jarvis_jax.tracking.stac import fit_offsets_once, ik_only_bout
from jarvis_jax.tracking.bridge import compute_bridges
from jarvis_jax.tracking.outputs import build_fly_outputs
from jarvis_jax.tracking.qc import qc_report
from jarvis_jax.tracking.reproj_video import render_camera_overlays
from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for, masks_are_stale
from jarvis_jax.predict.synced_reader import load_plan, read_window
try:
    from scripts.scale_keypoints import resolve_scale_keypoints
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from scale_keypoints import resolve_scale_keypoints
try:
    from scripts.sync_policy import stale_invalidation_enabled
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from sync_policy import stale_invalidation_enabled
try:
    from scripts.mask_coverage import (
        MIN_VIEWS_DEFAULT, coverage_report, load_valid, per_frame_views, write_report)
except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
    from mask_coverage import (
        MIN_VIEWS_DEFAULT, coverage_report, load_valid, per_frame_views, write_report)

# Register the `basename` OmegaConf resolver used by configs/outputs/default.yaml
# (out = .../${recording.name}/${basename:${recording.session_dir}}/pose). Done as
# an import side effect here so @hydra.main composition resolves outputs.out
# whether it is the default pattern or an explicit override.
OmegaConf.register_new_resolver(
    "basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)


# ---------------------------------------------------------------------------
# Small, pure(-ish) helpers factored out for readability / testability.
# ---------------------------------------------------------------------------

def _bout_dirs(predictions_dir):
    """Sorted bout indices discovered as bout_<idx> dirs under predictions_dir."""
    idxs = []
    for d in sorted(glob.glob(os.path.join(predictions_dir, "bout_*"))):
        m = re.match(r"bout_(\d+)$", os.path.basename(d))
        if m:
            idxs.append(int(m.group(1)))
    return sorted(idxs)


def resolve_bout_ids(cfg):
    """cfg.bout_ids: '' -> all bouts under predictions_dir; else comma-separated ints."""
    spec = str(cfg.bout_ids).strip()
    if not spec:
        return _bout_dirs(cfg.recording.predictions_dir)
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def should_stop_after_triangulate(cfg) -> bool:
    """``cfg.pipeline.stop_after == 'triangulate'`` (Task 18:
    scripts/probe_recording.py). Absent ``pipeline`` block, or ``stop_after``
    null/unset/any other value, is a strict no-op returning False -- a config
    predating this feature (or simply not using it) runs every stage exactly
    as before. Used by ``process_bout_fly`` to return immediately after
    Stage A/B (2D keypoints + triangulation, + optional B2 smoothing)
    without spending GPU time on scale/offsets/STAC/polish/outputs/overlay --
    e.g. the scale probe, which only needs kp2d/kp3d to estimate a
    recording's body scale + quality from many short frame windows."""
    pipeline_cfg = cfg.get("pipeline") or {}
    return str(pipeline_cfg.get("stop_after", None)) == "triangulate"


def bout_start_frame(cfg, bout_idx):
    """Absolute start-frame for bout_idx from cfg.recording.bouts_csv.

    NOTE: this reads the session's bouts CSV (the same source D2/SAM3 itself
    parses via jarvis_jax.predict.sam3_driver.parse_bouts), not the D2
    manifest.json -- on real data the manifest can be stale/partial (e.g. it
    only reflects the most recent SAM3 invocation) while `bout_<idx>/
    sam3_masks.npz` dirs persist across runs, so a manifest-only lookup can
    KeyError on a bout whose masks are perfectly usable. The CSV is the
    authoritative, append-only bout definition and cfg.recording.bouts_csv is
    already wired up for exactly this."""
    tag = session_tag_for(str(cfg.recording.session_dir))
    bouts = parse_bouts(str(cfg.recording.bouts_csv), tag, bout_ids=[bout_idx])
    if not bouts:
        raise KeyError(
            f"bout_idx {bout_idx} not found in {cfg.recording.bouts_csv} (fly_id tag={tag})")
    return int(bouts[0]["start"])


def open_video_captures(session_dir, cameras):
    import cv2
    return [cv2.VideoCapture(os.path.join(session_dir, f"{c}.mp4")) for c in cameras]


def all_cams_frames(caps, start, T):
    """Yield T (C,H,W,3) uint8 RGB frames, sequential read from `start` across
    all cameras (one decoded frame per camera in memory at a time)."""
    import cv2
    for cap in caps:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for t in range(T):
        imgs = []
        for cap in caps:
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {start + t}")
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        yield np.stack(imgs)


def one_cam_frames(video_path, start, T):
    """Yield T (H,W,3) uint8 RGB frames from a single camera; never buffers
    more than one decoded frame (mirrors reproj_video's streaming contract)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    try:
        for t in range(T):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"failed to read frame {start + t} from {video_path}")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def project_points(cam_mat_4x3, pts_mm):
    """(4,3) camera matrix (the `p_h @ M` convention of ReprojectionTool) +
    (N,3) mm points -> (N,2) pixel coords, vectorized (no per-vertex loop --
    the FK'd mesh-subset overlay can have hundreds of vertices per frame)."""
    pts_mm = np.asarray(pts_mm, np.float64)
    if pts_mm.shape[0] == 0:
        return np.zeros((0, 2), np.float64)
    ph = np.concatenate([pts_mm, np.ones((pts_mm.shape[0], 1))], axis=1)  # (N,4)
    proj = ph @ np.asarray(cam_mat_4x3, np.float64)                      # (N,3)
    return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)


def high_confidence_sample(kp3d, max_frames=None):
    """Frame indices with every keypoint finite (fallback: the most-complete
    frames when none are fully finite) -- the STAC fit_offsets sample should
    not see NaN markers, and an all-NaN frame is strictly worse than a
    mostly-finite one if no frame is perfect."""
    K = kp3d.shape[1]
    finite_counts = np.isfinite(kp3d).all(axis=-1).sum(axis=-1)   # (T,)
    idx = np.where(finite_counts == K)[0]
    if idx.size == 0:
        idx = np.argsort(-finite_counts)
    if max_frames:
        idx = idx[:max_frames]
    return idx


def check_track_merge(bout_dir, *, min_separation_body_lengths=0.5,
                      max_merged_frac=0.02, kp_index=0):
    """Did the two flies' tracks collapse onto the same animal?

    Non-fatal QC, in the spirit of check_bout_camera_order: it reports, it does
    not mutate or fail the bout. Two flies cannot occupy the same point, so a
    separation far below one body length means one track has been captured by
    the other animal -- the failure that produced the Figure 4 exemplar's
    apparent male/female identity switching.

    Scale-free: separation is scored against each fly's OWN body length,
    measured as the median distance from `kp_index` (Scutellum) to its furthest
    keypoint, so no absolute units are assumed. Measured on
    Session0/2025_10_20_13_20_04 bout_00028 before the gray-fill fix: 310
    frames under 0.25 body lengths, bottoming at 0.03, against a 1.08 median --
    and afterwards, zero.

    Returns a dict; `status` is "ok", "merged", or "unknown" (a fly's kp3d is
    missing or has no finite frame).
    """
    out = {"status": "unknown", "n_merged": 0, "frac_merged": 0.0,
           "min_separation": None, "median_separation": None,
           "body_length": None, "threshold": None}
    try:
        a = np.load(os.path.join(bout_dir, "fly0", "kp3d.npz"))["kp3d"]
        b = np.load(os.path.join(bout_dir, "fly1", "kp3d.npz"))["kp3d"]
    except Exception:                                    # noqa: BLE001
        return out
    if a.shape != b.shape or a.ndim != 3 or a.shape[0] == 0:
        return out
    sep = np.linalg.norm(a[:, kp_index, :] - b[:, kp_index, :], axis=1)
    ok = np.isfinite(sep)
    if not ok.any():
        return out
    # body length = median over frames of each fly's own max keypoint spread
    spans = []
    for k in (a, b):
        d = np.linalg.norm(k - k[:, kp_index:kp_index + 1, :], axis=2)
        with warnings.catch_warnings():
            # All-NaN frames are ordinary here (gated/unmeasured); they must
            # drop out of the median, not warn.
            warnings.simplefilter("ignore", RuntimeWarning)
            spans.append(np.nanmedian(np.nanmax(d, axis=1)))
    body = float(np.nanmean(spans))
    if not np.isfinite(body) or body <= 0:
        return out
    thr = float(min_separation_body_lengths) * body
    merged = ok & (sep < thr)
    n = int(merged.sum())
    frac = n / float(ok.sum())
    out.update(status="merged" if frac > float(max_merged_frac) else "ok",
               n_merged=n, frac_merged=frac,
               min_separation=float(np.nanmin(sep[ok])),
               median_separation=float(np.nanmedian(sep[ok])),
               body_length=body, threshold=thr)
    return out


def mask_areas_per_view(masks):
    """(T,C) float: pixel area of each (frame, camera) mask.

    Looped over cameras rather than a single `masks.sum(axis=(2,3))` because
    the mask array is (T,C,H,W) and materialising the reduction over the whole
    thing at once is needlessly heavy on a full bout.
    """
    masks = np.asarray(masks)
    T, C = masks.shape[0], masks.shape[1]
    out = np.zeros((T, C), np.float64)
    for c in range(C):
        out[:, c] = masks[:, c].reshape(T, -1).sum(axis=1)
    return out


def view_mask_agreement(kp2d, centroids, valid, mask_areas, *, max_fly_lengths):
    """(T,C) bool: did this view's 2-D prediction land on THIS camera's own mask?

    The detector emits a full set of keypoints for whatever crop it is handed,
    including a crop whose target is tiny, edge-on, or absent, and reports high
    confidence while doing it. `view_conf_thresh` cannot see that: measured on
    Session0/2025_10_20_13_20_04 bout_00028, views where the female had NO mask
    at all still scored a median confidence of 0.856 -- above the 0.6 gate --
    and entered the DLT. Comparing the prediction against the mask that SAM3
    actually found is an independent signal that does see it.

    Distance is scored in units of the fly's own size, `sqrt(mask area)`, so a
    single threshold transfers across cameras, resolutions and body sizes
    rather than being a pixel count tuned to one rig. Measured on that bout:
    0.60 fly-lengths through the healthy stretch against 4.51 while the
    female's track was being dragged onto the male.

    A view with no valid mask can never agree: nothing says where the animal
    is, so its keypoints are not evidence.

    `max_fly_lengths=None` is a strict no-op (every valid view agrees), so a
    config without the key behaves exactly as before this feature.
    """
    valid = np.asarray(valid, bool)
    if max_fly_lengths is None:
        return valid.copy()
    with warnings.catch_warnings():
        # A view whose keypoints are entirely NaN is an ordinary outcome here
        # (no fly in the crop); it must resolve to "does not agree", not noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmedian(np.asarray(kp2d, float), axis=2)      # (T,C,2)
    d = np.linalg.norm(centre - np.asarray(centroids, float), axis=-1)   # (T,C)
    scale = np.sqrt(np.maximum(np.asarray(mask_areas, float), 1.0))
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(d) & (d <= float(max_fly_lengths) * scale)
    return ok & valid


def gate_low_coverage_frames(kp3d, views_per_frame, min_views):
    """NaN-out triangulated keypoints for frames with too few valid-camera
    views (Task 17: scripts/mask_coverage.py). Triangulating from very few
    views is ill-conditioned and produces garbage (coincident-looking flies,
    bones flexing 20-50%) that is worse than simply having no pose for that
    frame -- so below `min_views`, mark the whole frame NaN instead.

    Frames at/above `min_views` are returned byte-identical to the input.
    `min_views=None` (i.e. `cfg.masks.min_views` absent, the default) is a
    strict no-op -- gated_kp3d is a copy of `kp3d` with `n_gated == 0` -- so
    a config without a `masks` block behaves exactly as before this feature.

    Parameters
    ----------
    kp3d : (T, K, 3) array
        Triangulated keypoints for one fly.
    views_per_frame : (T,) int array
        Number of cameras with a valid mask for this fly, per frame (see
        scripts.mask_coverage.per_frame_views).
    min_views : int | None

    Returns
    -------
    (gated_kp3d, n_gated) : gated_kp3d is a (T, K, 3) array (always a copy,
        never the input object); n_gated is the number of frames NaN'd out.
    """
    gated = np.array(kp3d, copy=True)
    if min_views is None:
        return gated, 0
    below = np.asarray(views_per_frame) < int(min_views)
    gated[below] = np.nan
    return gated, int(np.count_nonzero(below))


def _import_segment_calibration():
    """Import the segment-calibration entry points, adding the repo root to
    sys.path if the pipeline's cwd/PYTHONPATH didn't already expose `utils`
    (mirrors jarvis_jax.tracking.filter._filter_keypoints)."""
    try:
        from jarvis_jax.tracking.segment_fit import optimize_segment_scales
        from utils.segment_calibration import build_segment_map
    except ImportError:
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        if repo not in sys.path:
            sys.path.insert(0, repo)
        from jarvis_jax.tracking.segment_fit import optimize_segment_scales
        from utils.segment_calibration import build_segment_map
    return optimize_segment_scales, build_segment_map


def import_keypoint_groups():
    """`viz.core.colors.keypoint_groups`, importable however this file was run.

    `scripts/viz/` is an unrelated package of one-off viz scripts with no
    `core` submodule. When run_bout.py is invoked directly, `sys.path[0]` is
    `scripts/`, so a plain `import viz` binds THAT package into
    `sys.modules['viz']` as an empty namespace package and `viz.core` does not
    exist. Recovering needs BOTH halves:

      * the repo root moved to the FRONT of sys.path, unconditionally. The
        earlier `if _repo not in sys.path: insert(0, ...)` looked equivalent
        and was not: the mvq lift/IK sbatch text runs run_bout.py under
        `PYTHONPATH=third_party/jarvis_jax:.`, so the repo root is already on
        sys.path but BEHIND `scripts/` -- the membership test skipped the
        insert, the retry re-resolved `viz` to `scripts/viz` again, and
        Session0 bout 28 died at the viz stage AFTER a completed 36-minute
        STAC solve (job 39606799, 2026-09-04).
      * the poisoned `viz*` entries dropped from sys.modules, since a cached
        binding is not revisited by a later import.

    Returns the function; raises ModuleNotFoundError if the repo really has no
    top-level `viz` package.
    """
    try:
        from viz.core.colors import keypoint_groups
        return keypoint_groups
    except ModuleNotFoundError:
        pass
    _repo = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    # Only drop entries that NAME the repo root. An empty string (or ".") is
    # the "current directory" entry, whose meaning is not "the repo root" even
    # when cwd happens to be it -- resolving those and dropping them would
    # quietly change how every other module in the process resolves.
    sys.path = [p for p in sys.path if not (p and os.path.abspath(p) == _repo)]
    sys.path.insert(0, _repo)
    for _m in list(sys.modules):
        if _m == "viz" or _m.startswith("viz."):
            del sys.modules[_m]
    from viz.core.colors import keypoint_groups
    return keypoint_groups


def stage_b_gate_signature(cfg):
    """A stable string naming every setting that changes what kp3d.npz contains.

    Stage B's gates (view-conf, mask-agreement, wing-collapse, rigid-repair)
    all run INSIDE `if not stage_done(kp3d_path)`, so on a resumed run an
    existing kp3d.npz silently bypasses them: turning a gate on in the config
    does nothing and the log shows no line for it. Measured on Session0 bout 28
    -- fly1 re-ran and got `[rigid-repair] ... frames [3, 10, 11]`, fly0 kept a
    kp3d.npz from a run predating the gate and was never reprocessed, with no
    warning. Recording the signature makes that detectable.

    ONLY Stage-B settings belong here, and the boundary is load-bearing rather
    than tidy. This string is stored INSIDE kp3d.npz and a mismatch REFUSES the
    bout, telling the operator to delete kp3d.npz, stac_ik.h5, qpos_refined.npz
    and the rest -- a re-triangulation plus a 12-minute STAC solve per bout-fly.
    Enrolling a setting that cannot change kp3d.npz therefore charges that price
    for nothing: `wing_mask_fit` (Stage D2) was tried here and removed, because
    it rewrites qpos AFTER the bridges and provably cannot alter kp3d.npz or
    stac_ik.h5, yet toggling it would have forced ~5.2 h of re-solving per sweep
    point on the 13-bout benchmark. Post-STAC stages carry their OWN provenance
    in their own artifact instead -- see `wing_mask_fit_signature` /
    `wing_fit_action`.
    """
    def _plain(v):
        """OmegaConf ListConfig is NOT a list/tuple instance, so an isinstance
        check lets it reach json.dumps unconverted and raises. Coerce anything
        iterable-but-not-a-string."""
        if v is not None and not isinstance(v, (str, bytes)) and hasattr(v, "__iter__"):
            return [_plain(x) for x in v]
        return v

    # With the mvq lifter (spec 2026-09-04-mvq-maskfree-frontend-design §4.5)
    # Stages A and B are REPLACED by `jarvis_jax.tracking.lift_mvq`, so none of
    # the DLT gates below ran on that kp3d.npz and none of them describes it.
    # What determines its contents is which checkpoint produced it, the
    # existence threshold that decided which frames are NaN, and which
    # IDENTITY rule named the two written flies (`mvq.identity`: the masks'
    # human review or the model's sex head -- they disagree on 40% of one
    # recording's frames) plus whether the per-keypoint mask-CONTAINMENT
    # filter deleted keypoints from it (`mvq.containment`), so that is what
    # the signature names -- same
    # contract (stored inside kp3d.npz, a mismatch refuses the bout), and an
    # mvq bout passes without `allow_stale_kp3d`.
    if str((cfg.get("pipeline") or {}).get("lifter", "dlt")) == "mvq":
        from jarvis_jax.tracking.lift_mvq import mvq_gate_string
        _mv = cfg.get("mvq") or {}
        return mvq_gate_string(_mv.get("checkpoint"), step=_mv.get("step"),
                               exist_thresh=_mv.get("exist_thresh"),
                               identity=_mv.get("identity"),
                               containment=_mv.get("containment"))

    _wc = cfg.get("wing_collapse") or {}
    _rr = cfg.get("rigid_repair") or {}
    _mk = cfg.get("masks") or {}
    return json.dumps({
        "conf_thresh": float(cfg.detector.conf_thresh),
        "view_conf_thresh": cfg.detector.get("view_conf_thresh", None),
        "reproj_resid_px": cfg.detector.get("reproj_resid_px", None),
        "kp_mask_agree_fly_lengths": _mk.get("kp_mask_agree_fly_lengths", None),
        "min_views": _mk.get("min_views", None),
        "wing_collapse": ({k: _plain(_wc.get(k)) for k in
                           ("enabled", "abs_floor", "rel_frac", "min_views_kept")}
                          if _wc.get("enabled", False) else {"enabled": False}),
        "rigid_repair": ({k: _plain(_rr.get(k))
                          for k in ("enabled", "edges_containing", "rel_tol",
                                    "max_gap", "max_cv", "min_conf", "min_frames")}
                         if _rr.get("enabled", False) else {"enabled": False}),
    }, sort_keys=True)


def sidebyside_pose_args(right_mode, action):
    """``--pose`` argv for the Stage-F sidebyside subprocess, or ``[]``.

    ``--right mujoco`` renders ``qpos`` straight out of ``stac_ik.h5`` -- the
    PRE-BRIDGE STAC pose -- and never reads ``qpos_refined.npz`` /
    ``qpos_wingfit.npz``, so ``viz.views.sidebyside.check_pose_mode`` REFUSES a
    non-'auto' ``--pose`` there. That refusal is right when a human asks for it;
    asking for it unconditionally from here was not.

    THE REGRESSION THIS EXISTS TO PREVENT. ``cfg.outputs.sidebyside_right`` is a
    live config knob (``mujoco`` is the older setting). Emitting an explicit,
    never-'auto' ``--pose`` alongside it made every Stage-F subprocess exit 2 --
    on every bout, whether or not ``wing_mask_fit`` was enabled. Stage F's
    failure is deliberately non-fatal, so nothing crashed and nothing was
    logged as broken: ``sidebyside.mp4`` simply stopped being produced. The
    combination is pinned by ``viz/tests/test_cli.py``, which drives this
    function's output through the real argparse and the real ``check_pose_mode``.

    For the panels that DO read the pose, name it explicitly rather than let the
    renderer's 'auto' choose: the render must show the pose that is actually in
    outputs.h5, and Task 7 renders both arms deliberately.
    """
    if str(right_mode) == "mujoco":
        return []
    return ["--pose", "wingfit" if action in ("fit", "reuse") else "refined"]


def wing_mask_fit_enabled(cfg) -> bool:
    """True only when the opt-in ``wing_mask_fit`` block exists AND is enabled.

    An absent block (a config predating the stage) or an explicit null is a
    strict no-op returning False -- same contract as
    ``should_stop_after_triangulate``. Nothing is imported and no artifact is
    written when this is False.
    """
    return bool((cfg.get("wing_mask_fit") or {}).get("enabled", False))


def wing_mask_fit_signature(cfg):
    """Provenance for ``qpos_wingfit.npz``: every ``wing_mask_fit`` setting that
    changes the FITTED POSE, and nothing that only changes how fast it computes.

    Stage D2 runs inside a ``stage_done`` guard, so without this a resumed run
    silently reuses a fit made under different weights. This is deliberately NOT
    folded into ``stage_b_gate_signature`` (see that function's docstring): the
    wing fit cannot change kp3d.npz or stac_ik.h5, so invalidating them would
    charge a 12-minute STAC re-solve per bout-fly for nothing. It is stored in,
    and invalidates, only its own artifact -- plus outputs.h5/qc.json, which DO
    consume the refined qpos.

    THE KEYS ARE ENUMERATED, not taken as ``sorted(wf)``. `frame_chunk` is a
    pure performance knob (frames per device call; the YAML says so) and
    enrolling it would invalidate a perfectly good fit merely for moving to a
    smaller GPU. A blanket ``sorted(wf)`` would also auto-enrol any key added
    later, including a future ``prefetch``. The exact-partition test in
    tests/test_run_bout_pipeline_structure.py fails if a new YAML key is neither
    listed here nor declared performance-only, so the choice cannot be skipped.

    Returns a JSON string whose KEYS are exactly the semantic set (the tests
    read them back from it rather than duplicating the list).
    """
    def _plain(v):
        """Same coercion as stage_b_gate_signature: an OmegaConf ListConfig is
        not a list instance and would reach json.dumps unconverted."""
        if v is not None and not isinstance(v, (str, bytes)) and hasattr(v, "__iter__"):
            return [_plain(x) for x in v]
        return v

    semantic = ("bbox_margin", "body_vertex_stride", "containment_weight",
                "coverage_normalize", "coverage_weight", "dilate_px",
                "exclude_cameras", "gate_area_ref_pct", "gate_enabled",
                "gate_min_area_frac", "gate_min_body_inside", "huber_delta",
                "knot_spacing", "limit_weight", "lr", "max_dpitch_deg",
                "min_present_cameras", "n_steps", "n_target_points",
                "out_hw", "param_mode", "smooth_weight")
    wf = cfg.get("wing_mask_fit") or {}
    return json.dumps({k: _plain(wf.get(k)) for k in semantic}, sort_keys=True)


def stored_wing_fit_signature(wingfit_path):
    """The `wing_mask_fit_sig` recorded in ``qpos_wingfit.npz``, or None.

    None means "no fit on disk, or one whose provenance is unknown" -- which
    includes a file written before this record existed. Unknown is treated as
    stale by `wing_fit_action`, which is the safe direction: the fit is ~75 s,
    not a 12-minute STAC solve.
    """
    if not stage_done(wingfit_path):
        return None
    with np.load(wingfit_path) as z:
        return str(z["wing_mask_fit_sig"]) if "wing_mask_fit_sig" in z.files else None


def wing_fit_action(cfg, stored_sig):
    """What Stage D2 must do. Returns ``(action, want_sig)``.

    ``stored_sig`` is the ``wing_mask_fit_sig`` entry of an existing
    ``qpos_wingfit.npz``, or None when there is no such file (or when it
    predates this provenance record -- those are unknown, hence refitted).

    ``'off'``    the stage is off: the STAC/bridge pose is used.
    ``'fit'``    (re)compute.
    ``'reuse'``  the stored fit was made under exactly this config.

    There is deliberately NO 'refuse'. An earlier round raised on "stage
    disabled but a fit and an outputs.h5 are both on disk". That was the wrong
    tool three ways: it could not fire on the common case (a completed bout has
    DONE and returns long before this runs, so the contaminated bout sailed
    through -- loud where the risk is small, silent where it is large); its
    blast radius was the whole SLURM array task, since nothing wraps
    ``process_bout_fly`` in try/except, so one leftover file aborted every
    remaining bout under the DEFAULT shipped config; and it was inconsistent
    with 'reuse', which assumed on exactly the same missing information. The
    per-artifact pose STAMP makes the state decidable instead: disable-after-
    enable simply rebuilds once from the STAC pose and stamps "none".

    Three routes reach a stale reuse with the Stage-B kp3d gate never firing --
    deleting kp3d.npz by hand (the ordinary "just re-triangulate this bout"),
    ``pipeline.allow_stale_kp3d``, and scripts/analysis/stage_b_restage.py --
    which is why the fit needs provenance of its own at all.
    """
    if not wing_mask_fit_enabled(cfg):
        return "off", None
    want = wing_mask_fit_signature(cfg)
    return ("reuse" if stored_sig == want else "fit"), want


def pose_artifact_stale(recorded, want):
    """True when a derived artifact must be rebuilt for pose provenance `want`.

    `recorded` is what the artifact says it was built from, or None for one that
    is missing OR predates the stamp. An unstamped artifact reads as the
    pre-wing-fit pose ("none"), which is exactly what keeps a default run a
    strict no-op on every bout already on disk: `want` is "none" too, they
    match, and nothing rebuilds. Callers still OR this with `not stage_done(p)`,
    which is what covers a missing file.

    THE DECISION LIVES ON DISK, deliberately. It used to be an in-memory
    "did we recompute this run" flag, and that was wrong twice: a preemption
    between build_fly_outputs and qc_report left QC PERMANENTLY stale (next run
    the wing-fit signature matched, so the action was 'reuse', the flag was
    False, and both QC guards skipped forever while outputs.h5 held the new
    pose -- and we run on preemptible ckpt nodes); and on the 'reuse' path an
    outputs.h5 that was never built from the fit was never rebuilt.
    """
    return (recorded if recorded is not None else "none") != want


def outputs_pose_source(path):
    """The `pose_source` stamped into outputs.h5, or None if absent/unreadable.

    Read with h5py rather than `ioh5.load`: the file carries (T, Kmesh, 3)
    mesh_mm and all we want is one small string.
    """
    if not stage_done(path):
        return None
    import h5py
    try:
        with h5py.File(path, "r") as f:
            if "pose_source" not in f:
                return None
            v = f["pose_source"][()]
    except OSError:
        return None                     # truncated/corrupt -> rebuild, the safe way
    return v.decode() if isinstance(v, bytes) else str(v)


def json_pose_source(path):
    """The `pose_source` recorded in a JSON artifact (qc.json, render sidecars)."""
    if not stage_done(path):
        return None
    try:
        with open(path) as f:
            return (json.load(f) or {}).get("pose_source")
    except (OSError, ValueError):
        return None


def npz_pose_source(path):
    """The `pose_source` recorded in an npz artifact (qc_perframe.npz)."""
    if not stage_done(path):
        return None
    try:
        with np.load(path) as z:
            return str(z["pose_source"]) if "pose_source" in z.files else None
    except (OSError, ValueError):
        return None


def overlay_pose_sources(path):
    """{camera: pose_source} recorded for the per-camera overlay videos.

    PER CAMERA, not per group. The first version stamped the group only when
    EVERY camera's mp4 existed -- but an overlay failure is caught and logged
    non-fatally, so one camera that keeps erroring left the group permanently
    stale and re-rendered the six good cameras on every resumed run (~60 ms per
    frame per camera), where before this feature each existing mp4 was simply
    skipped. Missing file, unreadable file, or the old flat group format all
    read as {} -- i.e. nothing is current, which re-renders once and then
    settles.
    """
    if not stage_done(path):
        return {}
    try:
        with open(path) as f:
            obj = json.load(f) or {}
    except (OSError, ValueError):
        return {}
    cams = obj.get("cameras")
    return {str(k): str(v) for k, v in cams.items()} if isinstance(cams, dict) else {}


def stamp_json_pose_source(path, pose_source):
    """Add `pose_source` to a JSON artifact another writer just produced.

    Used for qc.json, which `qc_report` writes itself. A preemption between that
    write and this one leaves the file unstamped, which reads as "none" and so
    rebuilds next run -- the safe direction.
    """
    with open(path) as f:
        obj = json.load(f)
    obj["pose_source"] = str(pose_source)
    atomic_save_json(path, obj)


def wing_mask_fit_refine_kwargs(wf):
    """``refine_wing_pitch`` kwargs from the ``wing_mask_fit`` config block.

    EVERY knob is passed explicitly and none is allowed to fall back to the
    module default, because the module defaults are NOT the measured-best
    configuration: `huber_delta` defaults to 0.0, at which the coverage term is
    an unrobustified L2 whose gradient grows with distance, so SAM halo and
    body-silhouette crescents dominate and the term loses on its own metric
    (Task 5 measured huber_delta=8 better at every coverage weight tried).
    A missing key raises rather than silently taking the default.

    Keys the CALLER consumes instead of the refiner -- `out_hw`/`bbox_margin`
    (the SDF stack), `body_vertex_stride` (the body basis), `exclude_cameras`
    and `min_present_cameras` (the per-frame camera gate) -- are handled in
    `wing_mask_fit_bout`; `tests/test_run_bout_pipeline_structure.py` pins that
    the two sets partition the YAML block exactly.
    """
    # `max_dpitch_deg: null` means "no bound" and must stay None, not become 0.0
    def _opt_float(v):
        return None if v is None else float(v)

    keys = (("containment_weight", float), ("coverage_weight", float),
            ("coverage_normalize", bool), ("huber_delta", float),
            ("smooth_weight", float), ("limit_weight", float),
            ("param_mode", str), ("knot_spacing", int),
            ("max_dpitch_deg", _opt_float),
            ("n_steps", int), ("lr", float), ("frame_chunk", int),
            ("n_target_points", int), ("dilate_px", int))
    missing = [k for k, _ in keys if k not in wf]
    if missing:
        raise KeyError(
            f"wing_mask_fit is enabled but the block is missing {missing}; "
            f"copy the full block from configs/pipeline.yaml -- these are not "
            f"allowed to fall back to refine_wing_pitch's defaults (huber_delta "
            f"defaults to 0.0, which measurably degrades the fit)")
    return {k: cast(wf[k]) for k, cast in keys}


def wing_mask_fit_bout(cfg, qpos, bridge_s, bridge_R, bridge_t, bridge_ok,
                       masks_dict, cameras):
    """Refine wing pitch against the SAM masks. Returns ``(qpos_refined, stats)``.

    Everything but the two wing-pitch DOFs comes back bit-identical, and a frame
    with no usable mask evidence (or no bridge) is handed back at its STAC pose.

    CAMERA ORDER -- the trap this whole pipeline has been bitten by twice. The
    mask camera axis and the DLT projection matrices must be in the SAME order,
    and `refine_wing_pitch` only checks the camera COUNT, so a PERMUTATION is
    invisible to it: it would project one camera's wing onto another camera's
    mask and the residual would still look plausible. Both axes are therefore
    built BY NAME off the single canonical list `cameras` (==
    `cfg.recording.cameras`): `load_bout_masks(expected_cameras=cameras)`
    reordered the mask axis before this was called, and
    `affine_cameras_by_name(calib_dir, cameras)` selects the matrices out of
    ReprojectionTool's name-keyed dict. The assertion below re-checks the mask
    side rather than trusting the convention, and a legacy sam3_masks.npz with
    no `cameras` name array (whose axis order cannot be checked by name at all)
    is refused outright.
    """
    from jarvis_jax.tracking.appendage_dof import appendage_vertex_indices
    from jarvis_jax.tracking.fk import load_anatomy, make_fk_repose
    from jarvis_jax.tracking.mask_quality import wing_fit_validity_gate
    from jarvis_jax.tracking.mask_sdf import sdf_stack_from_masks
    from jarvis_jax.tracking.wing_mask_refine import (
        affine_cameras_by_name, body_uv_track, body_vertex_indices,
        qpos_limits, refine_wing_pitch, wing_pitch_dof_mask)
    import mujoco

    wf = cfg.get("wing_mask_fit") or {}
    cameras = [str(c) for c in cameras]

    if "cameras" not in masks_dict:
        raise RuntimeError(
            "wing_mask_fit needs a name-labelled mask camera axis, but this "
            "sam3_masks.npz predates the `cameras` array, so its camera order "
            "cannot be verified by name -- and a permutation would silently "
            "project each wing onto the wrong camera's mask. Re-run SAM3 for "
            "this bout, or leave wing_mask_fit.enabled=false.")
    if list(masks_dict["cameras"]) != cameras:
        raise RuntimeError(
            f"mask camera axis {list(masks_dict['cameras'])} != canonical "
            f"{cameras}; load_bout_masks(expected_cameras=...) must be given "
            f"the SAME list that builds the projection matrices")
    cam_Ms, cam_ts = affine_cameras_by_name(cfg.recording.calib_dir, cameras)

    masks = np.asarray(masks_dict["masks"])
    valid = np.array(masks_dict["valid"], bool, copy=True)          # (T,C)
    # A frame with no bridge has no model->mm map, hence no usable geometry.
    # Kept as its OWN reason: these are the frames STAC could not solve, which
    # is a completely different diagnosis from "the fly was seen by too few
    # cameras", and reporting them as the latter sent the reader looking at SAM
    # coverage for a solver problem.
    no_bridge = ~np.asarray(bridge_ok, bool)                        # (T,)
    valid &= ~no_bridge[:, None]
    # Cameras excluded for the WINGS specifically: Cam2012631's masks are
    # truncated (the right wing is inside them on only 43-46% of frames), so its
    # silhouette would pull the blade in rather than out.
    _excl = [str(c) for c in (wf.get("exclude_cameras") or [])]
    _unknown = [c for c in _excl if c not in cameras]
    if _unknown:
        raise ValueError(f"wing_mask_fit.exclude_cameras names camera(s) "
                         f"{_unknown} that are not in cfg.recording.cameras "
                         f"{cameras}")
    for c in _excl:
        valid[:, cameras.index(c)] = False                          # BY NAME

    _min_cams = int(wf["min_present_cameras"])
    anat = load_anatomy(str(cfg.ik.xml), str(cfg.ik.mesh_npz))
    lb, ub = qpos_limits(anat["m"])
    opt_mask = wing_pitch_dof_mask(anat["m"])       # raises if the joints moved
    fk_repose = make_fk_repose(anat)
    body_idx = body_vertex_indices(str(cfg.ik.mesh_npz),
                                   stride=int(wf["body_vertex_stride"]))

    # ---- THE VALIDITY GATE: skipping a badly-tracked fly-frame is a FEATURE --
    # `min_present_cameras` counts masks that are PRESENT, and on Session0 bout
    # 28 fly0's frames 1500-2006 they are: 90.7% of them, at a third of her own
    # area, 40% of them slivers (the female pressed against a wall). The fit
    # then solves against a fragment and every band-limited parameterisation
    # renders the result as a SMOOTH 130 deg swing.
    #
    # TWO SIGNALS, and only the second is general. Area against that camera's
    # own healthy reference catches the truncated sliver; the fraction of the
    # projected BODY landing inside the mask catches a wrong-fly or merged-fly
    # mask, which has perfectly normal area and is invisible to any area test.
    #
    # AND THE RESULT IS USED TWICE. `camera_ok` goes into the SDF stack, so
    # sliver evidence never enters the objective; `frame_keep` goes into
    # refine_wing_pitch, so the skipped frames are HELD at the STAC pose. The
    # first alone is not enough: in spline/lowpass zeroing `present` does not
    # freeze a frame, the knots interpolate across it.
    gate = None
    if bool(wf["gate_enabled"]):
        # a coarse body basis is plenty for an inside FRACTION (the dense one
        # exists for the coverage raster, where the footprint has to cover the
        # silhouette), and it keeps this an O(seconds) precompute
        _gidx = body_idx[::max(1, len(body_idx) // 512)]
        _uv = body_uv_track(qpos, bridge_s, bridge_R, bridge_t,
                            fk_repose=fk_repose, body_idx=_gidx,
                            cam_Ms=cam_Ms, cam_ts=cam_ts)
        gate = wing_fit_validity_gate(
            masks, valid, body_uv=_uv,
            min_area_frac=float(wf["gate_min_area_frac"]),
            min_body_inside=float(wf["gate_min_body_inside"]),
            min_cameras=_min_cams,
            area_ref_pct=float(wf["gate_area_ref_pct"]))
        valid_fit = gate["camera_ok"]
    else:
        valid_fit = valid

    sdf, grid_scale, grid_offset, present = sdf_stack_from_masks(
        masks, valid_fit,
        out_hw=tuple(int(v) for v in wf["out_hw"]),
        bbox_margin=float(wf["bbox_margin"]))
    # Too few views is ill-conditioned for a silhouette fit exactly as it is for
    # triangulation. An all-False `present` row freezes that frame at its STAC
    # pose inside refine_wing_pitch (its "no evidence means no change" gate).
    _few = present.sum(axis=1) < _min_cams
    present[_few] = False
    # THREE disjoint camera-side diagnoses, peeled in order. "STAC could not
    # solve it", "too few cameras saw the fly AT ALL" and "enough cameras saw
    # it but their masks are slivers or disagree with the pose" send a reader to
    # three different places, so they are never merged into one count.
    _few_valid = valid.sum(axis=1) < _min_cams
    _thin = _few_valid & ~no_bridge
    _gated = _few & ~_few_valid & ~no_bridge

    q_ref = refine_wing_pitch(
        qpos,
        fk_repose=fk_repose,
        wing_vert_idx=appendage_vertex_indices(
            str(cfg.ik.mesh_npz), subset=str(cfg.ik.mesh_subset), include=("wing",)),
        body_vert_idx=body_idx,
        cam_Ms=cam_Ms, cam_ts=cam_ts,
        sdf=sdf, grid_scale=grid_scale, grid_offset=grid_offset,
        present=present, masks=masks,
        bridge_s=bridge_s, bridge_R=bridge_R, bridge_t=bridge_t,
        opt_mask=opt_mask, lb=lb, ub=ub,
        # HOLD the skipped frames at the STAC pose. `present` alone cannot do
        # it in a band-limited mode; see refine_wing_pitch's `frame_keep`.
        # None when the gate is off, so the arm the measured negative result was
        # taken on stays byte-reproducible.
        frame_keep=(present.any(axis=1) if gate is not None else None),
        **wing_mask_fit_refine_kwargs(wf))

    # Report the two wings by NAME, never by array position: which qpos address
    # is left and which is right is a property of the XML, and `opt_mask` is
    # just a sorted boolean over addresses.
    adr = {}
    for _name in ("wing_pitch_left", "wing_pitch_right"):
        _jid = mujoco.mj_name2id(anat["m"], mujoco.mjtObj.mjOBJ_JOINT, _name)
        adr[_name] = int(anat["m"].jnt_qposadr[_jid])
    q0 = np.asarray(qpos, np.float32)
    q1 = np.asarray(q_ref, np.float64)
    finite_pose = np.isfinite(q0).all(axis=1)
    _padr = np.array([adr["wing_pitch_left"], adr["wing_pitch_right"]])
    # `moved` is a MEASUREMENT of the pose delta, not the predicate
    # `present.any(1) & finite_pose`. That predicate is only equivalent in
    # `param_mode='free'`, where the frame gate lives on the parameter update.
    # In `spline`/`lowpass` one knot spans many frames and the gate lives on the
    # COST, so a frame with no mask evidence is INTERPOLATED by its neighbouring
    # knots and genuinely moves -- reporting it as "left at the STAC pose", and
    # excluding it from the dpitch medians, was simply false. Pinned by
    # tests/test_run_bout_pipeline_structure.py.
    moved = (finite_pose & np.isfinite(q1).all(axis=1)
             & (q1[:, _padr] != q0[:, _padr].astype(np.float64)).any(axis=1))
    d = np.rad2deg(q1[moved] - q0[moved].astype(np.float64))
    # THREE disjoint reasons a frame carries NO EVIDENCE, and in `free` mode they
    # sum to n_skipped exactly. Disjoint by construction: no_bridge and _thin both
    # have an all-False `present` row, and _thin excludes no_bridge, so the third
    # bucket is exactly "evidence present, pose not finite". In a band-limited
    # mode a no-evidence frame can still move, so `n_interpolated` reports how
    # many did -- without it the effect of the per-frame evidence gate, which is
    # the only frame-level safety this stage has, is invisible in the log.
    nonfinite_pose = ~finite_pose & present.any(axis=1)
    no_evidence = no_bridge | _thin | _gated | ~finite_pose
    # Frames sitting ON a wing-pitch joint stop that were not there before. The
    # hard clamp is a per-frame nonlinearity applied AFTER the parameterisation,
    # so a clamp hit makes the band-limited guarantee CONDITIONAL: measured on
    # bout 28 fly0, 17 clamped frames took a spline64 correction 11.45 deg (0.146
    # of its own amplitude) out of the knot span it is supposed to lie in.
    _lo, _hi = lb[_padr], ub[_padr]
    _on_stop = (np.isclose(q1[:, _padr], _lo[None, :], atol=1e-9)
                | np.isclose(q1[:, _padr], _hi[None, :], atol=1e-9))
    _was_on_stop = (np.isclose(q0[:, _padr], _lo[None, :], atol=1e-9)
                    | np.isclose(q0[:, _padr], _hi[None, :], atol=1e-9))
    clamp_hits = (np.isfinite(q1[:, _padr]) & _on_stop & ~_was_on_stop).any(axis=1)
    # Frames sitting AT the physiological |dpitch| bound. Like the joint clamp
    # this is a per-frame nonlinearity applied after the parameterisation, and
    # unlike it, its whole purpose is to fire on evidence that should not have
    # been believed -- so a non-zero count is a pointer at the MASKS, not a
    # defect in the fit.
    _dmax = wf.get("max_dpitch_deg")
    n_at_bound = 0
    if _dmax is not None:
        _dp = np.abs(q1[:, _padr] - q0[:, _padr].astype(np.float64))
        n_at_bound = int((np.isfinite(_dp)
                          & (_dp >= np.deg2rad(float(_dmax)) - 1e-9)).any(axis=1).sum())
    stats = {
        "n_frames": int(q0.shape[0]),
        "n_refined": int(moved.sum()),
        "n_skipped": int(q0.shape[0] - moved.sum()),
        "n_interpolated": int((moved & no_evidence).sum()),
        "n_no_evidence_frames": int(no_evidence.sum()),
        "n_clamp_hits": int(clamp_hits.sum()),
        "n_thin_frames": int(_thin.sum()),
        # The NEW gate's own bucket: enough cameras saw the fly, but their masks
        # were slivers or disagreed with the STAC pose. Kept separate from
        # `n_thin_frames` because the remedy is different -- SAM, not coverage.
        "n_gated_frames": int(_gated.sum()),
        "n_sliver_camera_frames": (int(gate["n_sliver_camera_frames"])
                                   if gate is not None else 0),
        "n_pose_reject_camera_frames": (int(gate["n_pose_reject_camera_frames"])
                                        if gate is not None else 0),
        "gate_enabled": bool(wf["gate_enabled"]),
        "n_at_dpitch_bound": n_at_bound,
        "n_no_bridge_frames": int(no_bridge.sum()),
        "n_nonfinite_pose_frames": int(nonfinite_pose.sum()),
        "min_present_cameras": _min_cams,
        # WHICH PARAMETERISATION produced this pose. `free` fits Delta-pitch per
        # frame and was measured to destroy the courtship song in wing pitch;
        # `spline`/`lowpass` are band-limited. It is in the provenance signature,
        # but a reader of the npz or the log should not have to reconstruct the
        # config to know which one they are holding.
        "param_mode": str(wf.get("param_mode", "free")),
        "knot_spacing": int(wf.get("knot_spacing", 32)),
        "dpitch_left_deg": float(np.median(d[:, adr["wing_pitch_left"]])) if moved.any() else 0.0,
        "dpitch_right_deg": float(np.median(d[:, adr["wing_pitch_right"]])) if moved.any() else 0.0,
    }
    return q_ref, stats


def compute_segment_scales(cfg, kp3d, kp_names, scale, run_root):
    """Per-segment (per-limb) SHAPE calibration M-step (the validated recipe).

    A single global trunk scale + STAC marker offsets leaves a per-limb
    proportion mismatch (model femur too long, tarsus too short) so the IK can't
    reach the keypoints (~0.9mm on the femur-tibia joint). This runs an
    UN-MORPHED offsets+ik on a high-confidence calibration sample in a TEMP dir
    (so the real run_root/offsets.h5 is untouched and, crucially, so the temp
    fit sees NO SEGMENT_SCALES), reads the resulting qpos + kp_data (model
    units) from that stac_ik.h5, runs the Adam-on-MJX differentiable M-step
    (jarvis_jax.tracking.segment_fit.optimize_segment_scales), and returns a
    JSON-serializable list of per-segment scale entries ready for
    cfg.model.SEGMENT_SCALES / stac_mjx.rescale.rescale_per_segment.
    """
    import tempfile
    import shutil
    import mujoco
    import stac_mjx.io_dict_to_hdf5 as ioh5

    optimize_segment_scales, build_segment_map = _import_segment_calibration()

    # High-confidence calibration sample (no NaN markers), capped for a fast fit.
    sample_idx = high_confidence_sample(kp3d, max_frames=300)

    tmp_dir = tempfile.mkdtemp(prefix="segcalib_", dir=run_root)
    try:
        # UN-MORPHED offsets + ik. cfg.model.SEGMENT_SCALES must be unset here
        # (it is: calibration runs BEFORE it is set), so run_stac fits/solves on
        # the base model and the resulting qpos is a base-model pose.
        fit_offsets_once(cfg, kp3d[sample_idx], kp_names,
                         offsets_path="offsets.h5", save_path=tmp_dir, scale=scale)
        ik_only_bout(cfg, kp3d[sample_idx], kp_names, offsets_path="offsets.h5",
                     out_h5="stac_ik.h5", save_path=tmp_dir, scale=scale)
        d = ioh5.load(os.path.join(tmp_dir, "stac_ik.h5"))
        qpos = np.asarray(d["qpos"])                                  # (T,nq)
        kp_model = np.asarray(d["kp_data"]).reshape(len(qpos), len(kp_names), 3)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    mj_model = mujoco.MjModel.from_xml_path(str(cfg.model.MJCF_PATH))
    segment_map = build_segment_map(dict(cfg.model.KEYPOINT_MODEL_PAIRS))
    scales, _info = optimize_segment_scales(
        mj_model, segment_map, kp_names, qpos, kp_model,
        lam_reg=0.001, lam_target=0.0, lr=0.01, iters=500, clamp=(0.6, 1.6),
        verbose=True)
    return [{"name": s["name"], "geom_body": s.get("geom_body", ""),
             "length_body": s.get("length_body", ""),
             "scale": float(scales[s["name"]]),
             "scale_sites_on_body": s.get("scale_sites_on_body", "")}
            for s in segment_map]


def apply_segment_scales(cfg, seg_entries):
    """Set cfg.model.SEGMENT_SCALES so stac_mjx morphs the model (Stac.__init__
    reads it and calls rescale_per_segment before compiling). Must run BEFORE
    the offsets fit so the real offsets.h5 is fit on the MORPHED model."""
    OmegaConf.set_struct(cfg.model, False)
    cfg.model.SEGMENT_SCALES = OmegaConf.create(seg_entries)


def bridges_to_arrays(bridges):
    """length-T list of (s,R,t)|None -> (bridge_s (T,), bridge_R (T,3,3),
    bridge_t (T,3), bridge_ok (T,) bool) for npz persistence (RESOLUTIONS #2:
    bridges must survive a resume without recomputing polish_bout)."""
    T = len(bridges)
    s = np.ones((T,), np.float32)
    R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    t = np.zeros((T, 3), np.float32)
    ok = np.zeros((T,), bool)
    for i, br in enumerate(bridges):
        if br is None:
            continue
        s[i], R[i], t[i] = float(br[0]), np.asarray(br[1], np.float32), np.asarray(br[2], np.float32)
        ok[i] = True
    return s, R, t, ok


def arrays_to_bridges(s, R, t, ok):
    """Inverse of bridges_to_arrays: rebuild the length-T list of (s,R,t)|None."""
    return [(float(s[i]), np.asarray(R[i]), np.asarray(t[i])) if ok[i] else None
            for i in range(len(ok))]


# ---------------------------------------------------------------------------
# Per (bout, fly) driver
# ---------------------------------------------------------------------------

def _backfill_qc_perframe(cfg, bout_idx: int, fly: int, bout_dir: str) -> None:
    """I2: for an already-DONE bout/fly that predates Stage-E's per-frame QC
    write, `qc_perframe.npz` is missing even though DONE + qc.json +
    outputs.h5 + kp2d.npz all exist -- and the pseudo-label finetune driver
    (`scripts/run_pseudolabel_finetune.py`) silently skips any fly-dir
    lacking `qc_perframe.npz`. Rebuild it from artifacts ALREADY on disk
    (outputs.h5's kp3d_mm/mesh_mm, kp2d.npz, SAM3 masks) -- this needs none
    of stac_ik.h5/qpos_refined.npz/bridges, so no earlier stage is
    recomputed. Idempotent + atomic (atomic_save_npz): no-ops if
    qc_perframe.npz already exists, or if outputs.h5/kp2d.npz are missing
    (nothing to backfill from -- leaves the bout as-is rather than raising,
    since this is a best-effort backfill on an already-DONE bout)."""
    outputs_h5_path = os.path.join(bout_dir, "outputs.h5")
    kp2d_path = os.path.join(bout_dir, "kp2d.npz")
    qc_perframe_path = os.path.join(bout_dir, "qc_perframe.npz")
    if stage_done(qc_perframe_path):
        return
    if not (stage_done(outputs_h5_path) and stage_done(kp2d_path)):
        return
    # Fly-sexing (jarvis_jax.tracking.sexing) may have physically swapped this
    # bout's fly0/fly1 dirs, decoupling the dir INDEX from its SAM3 mask SLOT.
    # This best-effort backfill reads masks by dir index (load_bout_masks(.., fly)),
    # so on a canonicalized bout it would pair this dir's pose with the OTHER
    # fly's masks and write a mismatched qc_perframe.npz. The per-run sex.json
    # flag can't reconstruct the cumulative mapping across reruns, so for any
    # sexing-managed bout (sex.json present in the parent bout dir) we skip the
    # backfill rather than risk corrupting per-frame QC (which feeds pseudo-label
    # training). New bouts always get qc_perframe from Stage E, so this only
    # skips rare legacy+canonicalized bouts -- the pre-feature status quo (the
    # pseudo-label driver already skips any fly-dir lacking qc_perframe.npz).
    if os.path.exists(os.path.join(os.path.dirname(bout_dir), "sex.json")):
        print(f"[courtship] bout {bout_idx} fly{fly}: skipping qc_perframe backfill "
              f"(sexing-managed bout; dir<->mask-slot mapping not guaranteed)")
        return

    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.tracking.qc_perframe import per_frame_qc

    predictions_dir = str(cfg.recording.predictions_dir)
    bout_npz = os.path.join(predictions_dir, f"bout_{bout_idx:05d}", "sam3_masks.npz")
    masks_dict = load_bout_masks(bout_npz, fly, expected_cameras=list(cfg.recording.cameras))
    T, C = masks_dict["T"], masks_dict["C"]

    with np.load(kp2d_path) as z:
        kp2d, conf = z["kp2d"], z["conf"]

    d = ioh5.load(outputs_h5_path)
    kp3d_mm = np.asarray(d["kp3d_mm"])                  # (T,K,3) FK'd world-mm sites
    mesh_mm = np.asarray(d["mesh_mm"])                  # (T,Kmesh,3) FK'd world-mm mesh subset
    valid2d = conf >= float(cfg.detector.conf_thresh)   # (T,C,K), same threshold as triangulation

    kp3d_by_frame = [kp3d_mm[t] for t in range(T)]
    mesh_by_frame = [mesh_mm[t] for t in range(T)]
    kp2d_by_frame = [{c: kp2d[t, c] for c in range(C)} for t in range(T)]
    vis_by_frame = [{c: valid2d[t, c] for c in range(C)} for t in range(T)]
    masks_by_frame = [
        {c: (masks_dict["masks"][t, c] if masks_dict["valid"][t, c] else None)
         for c in range(C)}
        for t in range(T)
    ]

    rt = ReprojectionTool(cfg.recording.calib_dir)
    pf = per_frame_qc(rt, mesh_by_frame=mesh_by_frame, kp3d_by_frame=kp3d_by_frame,
                      kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                      masks_by_frame=masks_by_frame)
    # Stamp what this was rebuilt FROM. Everything above comes out of
    # outputs.h5, so its own stamp is the answer -- inventing "none" here would
    # claim a plain STAC pose for a file rebuilt from a wing-refined one. This
    # writer lives outside process_bout_fly, so the AST guard over the stage
    # guards cannot see it; tests/test_run_bout_pipeline_structure.py covers it
    # separately.
    _bf_pose = outputs_pose_source(outputs_h5_path) or "none"
    atomic_save_npz(qc_perframe_path, pose_source=_bf_pose, **pf)
    print(f"[courtship] bout {bout_idx} fly{fly}: backfilled qc_perframe.npz "
          f"for already-DONE bout -> {qc_perframe_path}")


# ---------------------------------------------------------------------------
# NaN-robust STAC solve (frozen-joint bug).
#
# The pose optimisation is ONE batched jaxls problem over all frames, with
# ~T smoothness costs chaining consecutive frames. A NaN in any frame
# therefore propagates NaN gradients along that chain to EVERY frame, the
# optimiser applies no update at all, and all 86 joint DOFs stay at their
# initialisation while only the separately-solved root moves.
#
# Measured on Session0/2025_10_20_13_20_04 (60 bout-flies): the correlation is
# perfect and has no overlap --
#     frozen  (9): 40..1393 NaN kp3d frames (mean 452)
#     healthy(51):  0..12   NaN kp3d frames (mean 0.3)
# Nothing errors: STAC reports "Pose Optimization finished", a finite mean
# error, and the bout is marked DONE. It is invisible to reprojection (NaN),
# to LOO (unaffected -- triangulation is fine) and to IoU.
# ---------------------------------------------------------------------------
NAN_SOLVE_MAX_GAP = 10      # frames; gaps this short are interpolated
NAN_SOLVE_MIN_SEG = 30      # frames; shorter finite runs are not worth solving
NAN_SOLVE_MIN_KEYPOINTS = None  # None = all-or-nothing (unchanged); see cfg.nan_solve.min_keypoints


def finite_frame_mask(kp3d, *, min_keypoints: int | None = None,
                       required_indices=None) -> np.ndarray:
    """(T,) bool: frames usable for the IK solve.

    min_keypoints=None (default): every keypoint must be finite -- today's
    all-or-nothing behaviour, byte-identical to before this parameter existed.
    `required_indices` is IGNORED in this branch (see the safety-rail test
    `test_min_keypoints_none_ignores_required_indices`): when every keypoint
    is already required, an additional required-subset check can only ever
    agree with it.
    min_keypoints=N: a frame is usable if at least N of its K keypoints are
    themselves entirely finite (all coordinates present), even if others are
    NaN. This is safe to relax because the IK residual already ignores a NaN
    marker per-frame, per-keypoint (stac_core_jaxls.py's marker_cost: `finite
    = jnp.isfinite(kp)`, zeroing that marker's contribution and gradient) --
    see marker_validity_mask below for the matching per-marker mask.

    required_indices (fix round 1, Task 20): an extra AND condition -- these
    specific keypoints (if given) must ALSO be finite, regardless of the
    min_keypoints count being cleared. This exists because relaxing the frame
    gate alone is not enough: stac_mjx/compute_stac.py's per-frame IK
    warm-start reads the ROOT keypoint
    (`kp_root_xyz = kp_flat[:, root_kp_idx*3:root_kp_idx*3+3]`, line ~468) and
    the four trunk ORIENTATION keypoints
    (`_estimate_orientation_from_keypoints`) directly out of the keypoint
    array with NO finite check -- unlike the residual, which is NaN-safe. A
    partial frame missing exactly one of those hands the solver a NaN initial
    guess, which can reproduce the very frozen-joint failure mode this task
    is trying to avoid. Pass `required_keypoint_indices(cfg, kp_names)` here
    so that set always matches what the solver actually warm-starts from.
    """
    a = np.asarray(kp3d)
    # Per-keypoint finiteness (all coordinates present), i.e. marker_validity_mask.
    per_kp = np.isfinite(a).all(axis=tuple(range(2, a.ndim)))
    if min_keypoints is None:
        return per_kp.all(axis=1)
    ok = per_kp.sum(axis=1) >= int(min_keypoints)
    if required_indices:
        ok &= per_kp[:, list(required_indices)].all(axis=1)
    return ok


def required_keypoint_indices(cfg, kp_names) -> list[int]:
    """Keypoint indices (into `kp_names`) a partial frame must still have
    finite, so `finite_frame_mask`'s `required_indices` gate can never accept
    a frame the solver's own per-frame warm-start would silently mishandle.

    Mirrors -- deliberately, verbatim in structure -- how
    `stac_mjx.stac.Stac.__init__` resolves the SAME two indices for its
    per-frame warm-start (`stac-mjx/stac_mjx/stac.py:208-213` for the root,
    `:298-309` for orientation), reading them from the SAME two anatomy-config
    keys (`cfg.model.ROOT_OPTIMIZATION_KEYPOINT`,
    `cfg.model.JAXLS_ORIENTATION_KEYPOINTS`) and the SAME ordered `kp_names`
    list already threaded through this file -- rather than hardcoding
    keypoint names here. If the anatomy config is ever changed to pick a
    different root or orientation keypoint, this function picks it up too;
    a hardcoded name would silently diverge from the solver instead.

    Why not import/reuse stac.py's own resolution code directly: building a
    `Stac` instance is the only place that logic lives, and doing so loads
    the MJCF and constructs a full mujoco/mjx model -- expensive, and not
    available at the point in `run_bout.py`'s Stage C where the frame-gating
    decision must be made (before `ik_only_bout`/`fit_offsets_once` run).
    Re-deriving the same ~15 lines from the same two config keys is the least
    fragile alternative to that: a name (not an index) is the single source
    of truth in both places, so the only way to diverge is for stac.py's own
    resolution algorithm to change shape -- the same risk every other
    Stage-C caller of `cfg.model.KP_NAMES` already carries.

    Tolerant by construction, matching stac.py's own fallbacks: a missing
    config key, or a configured name absent from `kp_names` (e.g. after
    keypoint pruning for amputated/missing markers), contributes no
    requirement rather than raising -- exactly mirroring stac.py's own
    `_root_kp_idx = -1` / `_orientation_kp_indices = None` "warm-start simply
    skipped" behaviour, never a hard failure.
    """
    names = list(kp_names)
    required: set[int] = set()
    model_cfg = cfg.model

    # Root warm-start -- stac.py:208-213 gates on key PRESENCE, not truthiness.
    if "ROOT_OPTIMIZATION_KEYPOINT" in model_cfg:
        try:
            required.add(names.index(model_cfg.ROOT_OPTIMIZATION_KEYPOINT))
        except ValueError:
            pass  # stac.py would raise here too (name not in kp_names); not our call to fix

    # Orientation warm-start -- stac.py:298-309: rear/left/right required,
    # front only if given; if ANY name fails to resolve, stac.py disables the
    # WHOLE orientation warm-start (sets _orientation_kp_indices = None), so
    # none of the four become required in that case either -- mirrored here
    # as one all-or-nothing try, not four independent ones.
    orient_cfg = model_cfg.get("JAXLS_ORIENTATION_KEYPOINTS", {})
    if orient_cfg and len(orient_cfg) >= 3:
        keys = ["rear", "left", "right"] + (["front"] if "front" in orient_cfg else [])
        try:
            required.update(names.index(orient_cfg[k]) for k in keys)
        except (KeyError, ValueError):
            pass  # matches stac.py's except -> _orientation_kp_indices = None

    return sorted(required)


def marker_validity_mask(kp3d) -> np.ndarray:
    """(T, K) bool: which individual keypoints are entirely finite.

    The per-marker counterpart to finite_frame_mask -- exactly the markers the
    IK solver already treats as absent (NaN in, zero residual/gradient out;
    see stac_core_jaxls.py's marker_cost). Used to gate/report partial-frame
    acceptance; the mask itself does not need separate plumbing into the
    solver because a NaN marker already carries that meaning there.
    """
    a = np.asarray(kp3d)
    return np.isfinite(a).all(axis=tuple(range(2, a.ndim)))


def fill_short_gaps(kp3d, max_gap=NAN_SOLVE_MAX_GAP):
    """Linearly interpolate NaN runs of <= `max_gap` frames.

    Short dropouts are the common case and interpolating them keeps the
    smoothness chain intact, which is what stops one bad frame freezing the
    whole solve. Long runs are LEFT as NaN for the segmenter -- interpolating
    across (e.g.) bout_00022's 1393 missing frames would invent a trajectory.
    Returns (filled, filled_mask) where filled_mask marks interpolated frames.
    """
    a = np.array(kp3d, dtype=float, copy=True)
    ok = finite_frame_mask(a)
    filled = np.zeros(len(a), bool)
    if ok.all() or not ok.any():
        return a, filled
    idx = np.flatnonzero(ok)
    # every maximal run of bad frames strictly between two good ones
    for lo, hi in zip(idx[:-1], idx[1:]):
        n = hi - lo - 1
        if n <= 0 or n > max_gap:
            continue
        w = (np.arange(1, n + 1) / (n + 1.0)).reshape((-1,) + (1,) * (a.ndim - 1))
        a[lo + 1:hi] = a[lo] * (1.0 - w) + a[hi] * w
        filled[lo + 1:hi] = True
    return a, filled


def contiguous_segments(ok, min_len=NAN_SOLVE_MIN_SEG):
    """[(start, stop)) runs of True in `ok` with length >= min_len."""
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


def _h5_safe(v):
    """Restore stac_mjx's on-disk string encoding after an ioh5 round-trip.

    stac_mjx writes strings as fixed-width BYTES -- `|S12` for kp_names,
    `|S20`/`|S16` for names_qpos/names_xpos, and a single `|S24514` scalar for
    the YAML config -- and `stac_mjx.io.load_stac_data` calls `.decode("utf-8")`
    on each. `ioh5.load` hands them back as numpy UNICODE arrays and Python
    `str`, which h5py either refuses to write ("No conversion path for
    dtype('<U12')") or writes as a vlen-str dataset whose elements are `str`
    and have no `.decode`. Segment stitching is the only load->save path, so
    convert back here and keep the stitched file byte-compatible with the
    single-solve path.
    """
    if isinstance(v, np.ndarray) and v.dtype.kind == "U":
        return v.astype("S")
    if isinstance(v, str):
        return v.encode("utf-8")
    return v


def _solve_segments_into(cfg, kp_solve, kp_names, segs, *, offsets_path,
                         out_h5, bout_dir, scale, n_frames, bout_idx, fly):
    """Solve each finite segment independently and stitch into ONE h5.

    Per-frame arrays (first axis == segment length) are written back into a
    full-length array at the segment's own offset; frames in no segment stay
    NaN. Non-per-frame entries (config, kp_names, offsets, names_*) are taken
    from the first segment, which owns them identically.
    """
    import stac_mjx.io_dict_to_hdf5 as ioh5
    total = sum(b - a for a, b in segs)
    print(f"[stac] bout {bout_idx} fly{fly}: {len(segs)} finite segment(s) "
          f"covering {total}/{n_frames} frames "
          f"({[f'{a}:{b}' for a, b in segs]}) -- solving separately so a NaN "
          f"gap cannot freeze the whole batch", flush=True)
    merged, meta = None, None
    for si, (a, b) in enumerate(segs):
        tmp = f"stac_seg{si}.tmp.h5"
        ik_only_bout(cfg, kp_solve[a:b], kp_names, offsets_path=offsets_path,
                     out_h5=tmp, save_path=bout_dir, scale=scale)
        d = ioh5.load(os.path.join(bout_dir, tmp))
        if merged is None:
            merged, meta = {}, {}
            for k, v in d.items():
                arr = np.asarray(v)
                if arr.ndim >= 1 and arr.shape[0] == (b - a):
                    # Keep the solver's own dtype (float32): upcasting to
                    # float64 doubles the file and diverges from the
                    # single-solve path's output.
                    full = np.full((n_frames,) + arr.shape[1:],
                                   np.nan if arr.dtype.kind == "f" else 0,
                                   dtype=arr.dtype)
                    merged[k] = full
                else:
                    meta[k] = v
        for k in merged:
            merged[k][a:b] = np.asarray(d[k])
        os.remove(os.path.join(bout_dir, tmp))
    out = {k: _h5_safe(v) for k, v in meta.items()}
    out.update(merged)
    ioh5.save(os.path.join(bout_dir, out_h5), out)


def _restore_unsolved_nan(h5_path, ok_original, segs):
    """Re-NaN frames whose INPUT keypoints were NaN.

    Short gaps are interpolated purely so the solve does not break; the poses
    that come back for them are not measurements, so they must not be written
    out as if they were.
    """
    import stac_mjx.io_dict_to_hdf5 as ioh5
    d = ioh5.load(h5_path)
    q = np.asarray(d.get("qpos"))
    if q is None or q.ndim != 2 or len(ok_original) != q.shape[0]:
        return
    bad = ~np.asarray(ok_original, bool)
    if not bad.any():
        return
    n = 0
    for k, v in list(d.items()):
        arr = np.asarray(v)
        if arr.ndim >= 1 and arr.shape[0] == q.shape[0] and arr.dtype.kind == "f":
            arr = arr.copy()          # keep the solver's float32; see _h5_safe
            arr[bad] = np.nan
            d[k] = arr
            n += 1
    # This is a load->save round-trip like the stitch above, so it needs the
    # same string re-encoding -- otherwise the '<U12' kp_names read back from
    # the file we just wrote cannot be written out again.
    ioh5.save(h5_path, {k: _h5_safe(v) for k, v in d.items()})
    print(f"[stac] re-NaN'd {int(bad.sum())} unmeasured frame(s) across "
          f"{n} per-frame array(s)", flush=True)


def joints_frozen(qpos, tol=1e-6) -> bool:
    """True when EVERY non-root DOF is constant across time -- the signature of
    a solve that never updated (see the module note above). qpos[:, :7] is the
    free-joint root, solved in a separate earlier stage that succeeds anyway."""
    q = np.asarray(qpos, dtype=float)
    if q.ndim != 2 or q.shape[0] < 2 or q.shape[1] <= 7:
        return False
    joint = q[:, 7:]
    if not np.isfinite(joint).any():
        return False
    return bool(np.all(np.nanstd(joint, axis=0) < tol))


def process_bout_fly(cfg, bout_idx: int, fly: int):
    run_root = str(cfg.outputs.out)
    bout_dir = os.path.join(run_root, "bouts", f"bout_{bout_idx:05d}", f"fly{fly}")
    if bout_complete(bout_dir):
        # I2: bouts completed before qc_perframe.npz existed have DONE +
        # qc.json + outputs.h5 + kp2d.npz but no qc_perframe.npz, which the
        # pseudo-label finetune driver treats as "incomplete" and silently
        # skips. Backfill it (cheap: no stage recomputation) before the
        # early-return short-circuit below.
        _backfill_qc_perframe(cfg, bout_idx, fly, bout_dir)
        print(f"[courtship] bout {bout_idx} fly{fly}: already DONE, skipping")
        return

    predictions_dir = str(cfg.recording.predictions_dir)
    bout_npz = os.path.join(predictions_dir, f"bout_{bout_idx:05d}", "sam3_masks.npz")
    cameras = list(cfg.recording.cameras)

    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    rt = ReprojectionTool(cfg.recording.calib_dir)
    cam_mats = np.asarray(rt.camera_matrices, np.float32)  # (C,4,3); order matches recording.cameras

    # Canonical-slot sync plan for this session (frame_sync.py / Task 2-4): None or
    # status "clean" makes every synced_reader read below byte-identical/positional
    # to the pre-sync-gate behavior; a trim/reindex plan realigns per-camera reads
    # to the shared canonical slot axis instead of raw mp4 frame index.
    sync_plan = load_plan(cfg.recording.session_dir)

    # Non-mutating camera-order QC (cheap, once per bout). Camera identity is
    # trusted from the self-labelling `cameras` array sam3_driver writes (the
    # mask write path is order-preserving: masks are packed by cam_idx and the
    # `cameras` array records that same order), and name-based reordering in
    # load_bout_masks(expected_cameras=) below applies it. This check only
    # SURFACES bad mask-centroid geometry (almost always a per-camera SAM3
    # tracking error in the bout) and NEVER mutates the file or fails the bout
    # -- auto-remapping on coarse mask centroids was found to corrupt
    # correctly-ordered masks when a single camera is mis-tracked.
    _cam = check_bout_camera_order(bout_npz, fly, cameras, rt)
    print(f"[camera-order] bout {bout_idx} fly{fly}: {_cam['status']}"
          f" (identity={_cam.get('identity_resid')}, worst_cam={_cam.get('worst_cam')})")

    masks_dict = load_bout_masks(bout_npz, fly, expected_cameras=cameras)
    T, C = masks_dict["T"], masks_dict["C"]

    # -- Per-camera mask coverage (Task 17, scripts/mask_coverage.py): a
    #    first-class QC signal computed BEFORE triangulation, straight from
    #    the sam3_masks.npz BOTH flies share. SAM3 correctly reports "not
    #    found" per (fly,cam,frame) rather than mis-assigning the same
    #    animal to both slots -- the measured failure mode is the female
    #    (fly0) dropping to as few as 3/7 cameras in several Session0 bouts
    #    while the male (fly1) stays 7/7 in every bout examined -- and
    #    triangulating from too few views is ill-conditioned (produces
    #    garbage that downstream looks like coincident flies or bones
    #    flexing 20-50%). coverage.json covers BOTH flies from one npz read,
    #    so it is written ONCE per bout to the bout dir (parent of
    #    fly0/fly1), mirroring scale.json's "whoever gets there first"
    #    pattern, not duplicated per fly-dir.
    masks_cfg = cfg.get("masks") or {}
    _min_views_cfg = masks_cfg.get("min_views", None)
    _min_views_for_report = int(_min_views_cfg) if _min_views_cfg is not None else MIN_VIEWS_DEFAULT
    views_per_frame = per_frame_views(load_valid(bout_npz))[fly]  # (T,) for this fly
    coverage_path = os.path.join(os.path.dirname(bout_dir), "coverage.json")
    if not stage_done(coverage_path):
        _coverage = coverage_report(bout_npz, min_views=_min_views_for_report)
        write_report(os.path.dirname(bout_dir), _coverage)

    kp2d_path = os.path.join(bout_dir, "kp2d.npz")
    kp3d_path = os.path.join(bout_dir, "kp3d.npz")
    kp3d_filt_path = os.path.join(bout_dir, "kp3d_filt.npz")
    stac_h5_path = os.path.join(bout_dir, "stac_ik.h5")
    qpos_path = os.path.join(bout_dir, "qpos_refined.npz")
    wingfit_path = os.path.join(bout_dir, "qpos_wingfit.npz")
    outputs_h5_path = os.path.join(bout_dir, "outputs.h5")
    qc_json_path = os.path.join(bout_dir, "qc.json")
    offsets_path = os.path.join(run_root, "offsets.h5")

    os.makedirs(run_root, exist_ok=True)
    os.makedirs(bout_dir, exist_ok=True)

    # -- staleness cascade: if this bout's masks were (re)written under a sync plan
    #    the previously-computed pose artifacts don't reflect (e.g. the plan's
    #    status/delta changed since kp2d/kp3d/... were computed from the old
    #    masks), those downstream artifacts are stale and must recompute. The
    #    masks themselves are the SAM3 stage's (Task 4) responsibility -- this
    #    only clears the POSE artifacts derived from them. Session-shared
    #    offsets.h5/scale.json/segment_scales.json are deliberately left alone
    #    (they are per-fly body constants, not per-bout).
    # cfg.sync.invalidate_stale (scripts.sync_policy) gates this cascade so
    # benchmark/A-B runs can keep pre-seeded (frozen) artifacts instead of
    # having them wiped and recomputed per-variant.
    if (stale_invalidation_enabled(cfg)
            and os.path.exists(bout_npz) and masks_are_stale(bout_npz, sync_plan)):
        print(f"[sync] bout {bout_idx} fly{fly}: masks stale for plan "
              f"status={getattr(sync_plan, 'status', None)} -- invalidating downstream artifacts")
        for _p in (kp2d_path, kp3d_path, kp3d_filt_path, stac_h5_path, qpos_path,
                   wingfit_path, outputs_h5_path, qc_json_path,
                   os.path.join(bout_dir, "qc_perframe.npz")):
            try:
                if os.path.exists(_p):
                    os.remove(_p)
            except OSError:
                pass

    # Centroids come from masks_dict (already reordered/autofixed above), NOT a
    # fresh raw npz read -- they must stay in lockstep with masks_dict["masks"]'s
    # camera axis (see autofix_bout_camera_order above).
    # Bound HERE, outside Stage A, because Stage B's mask-agreement gate needs it
    # too: when kp2d.npz already exists but kp3d.npz does not (the re-run path
    # after invalidating stale triangulation), Stage A is skipped and a
    # Stage-A-local binding raised UnboundLocalError.
    centroids = masks_dict["centroids"]

    # `pipeline.lifter: mvq` (spec 2026-09-04-mvq-maskfree-frontend-design
    # §4.5) means Stages A (ViTPose) and B (DLT) are REPLACED entirely by
    # `jarvis_jax.tracking.lift_mvq` / `scripts/mvq_lift_bout.py`, which write
    # kp2d.npz/kp3d.npz BEFORE run_bout.py ever runs. Stage A/B below are
    # gated ONLY by `stage_done(kp2d_path)`/`stage_done(kp3d_path)` -- file
    # existence, not who produced the file -- so a bout whose mvq lift never
    # ran (or was interrupted before writing these files) would silently fall
    # through to ViTPose+DLT here and get its DLT-triangulated kp3d.npz
    # stamped with the mvq `gates` string (`stage_b_gate_signature` returns
    # the mvq checkpoint's signature under this lifter, unconditionally) --
    # indistinguishable on disk from a real mvq lift, and never re-triangulated
    # correctly by a later run (the gate signature would MATCH). Refuse
    # instead of computing anything.
    if str((cfg.get("pipeline") or {}).get("lifter", "dlt")) == "mvq":
        _missing_mvq = []
        if not stage_done(kp2d_path):
            _missing_mvq.append(kp2d_path)
        if not stage_done(kp3d_path):
            _missing_mvq.append(kp3d_path)
        if _missing_mvq:
            raise RuntimeError(
                f"bout {bout_idx} fly{fly}: pipeline.lifter=mvq requires kp2d.npz and "
                f"kp3d.npz to already exist (written by the mvq lift), but missing: "
                f"{_missing_mvq}. Stage A (ViTPose) / Stage B (DLT) must NEVER run under "
                f"pipeline.lifter=mvq -- doing so would stamp DLT-triangulated keypoints "
                f"with the mvq gate signature, indistinguishable on disk from a real mvq "
                f"lift. Run the lift first, e.g.:\n"
                f"    PYTHONPATH=third_party/jarvis_jax:. python scripts/mvq_lift_bout.py "
                f"--session-dir <session_dir> --predictions-dir <sam3_masks_dir> "
                f"--out {run_root} --run <mvq_checkpoint_dir> --bout {bout_idx}")

    # -- Stage A: ViTPose 2-D ---------------------------------------------------
    if not stage_done(kp2d_path):
        start = bout_start_frame(cfg, bout_idx)

        def _frames_iter():
            # read_window yields (frames (C,H,W,3), present (C,) bool) per canonical
            # slot; predict_bout_2d only wants the frames -- dropped/absent cameras
            # already come back as zero frames from read_window and are masked out
            # downstream via masks_dict["valid"].
            for _frames, _present in read_window(
                    cfg.recording.session_dir, cameras, sync_plan, start, T):
                yield _frames

        # Distractor gray-fill: the OTHER fly's pixels are replaced with the
        # crop mean so the detector sees one clean animal + the target-mask
        # channel (JARVIS's dataset2D training crop). Without it, a crop
        # containing both flies gives the detector no signal about which is the
        # target -- measured on this dataset, keypoints landed on the WRONG fly
        # in >90% of unambiguous frames in 5 bouts and 10-90% in 33 more.
        # Single-animal recordings pass None and are unaffected.
        _distractor = None
        if int(cfg.recording.num_animals) == 2:
            _other = load_bout_masks(bout_npz, 1 - fly, expected_cameras=cameras)
            _distractor = _other["masks"]
            print(f"[gray-fill] bout {bout_idx} fly{fly}: using fly{1 - fly} masks "
                  f"as distractor ({int(_other['valid'].sum())} valid views)")
        # cfg.detector.kp_names is an ASSERTION about this checkpoint's channel
        # order; a wrong one permutes every keypoint silently (residuals stay
        # plausible). Check it against the checkpoint's own training order first.
        verify_detector_kp_order(cfg.detector.ckpt, list(cfg.detector.kp_names))
        vit = load_detector(cfg.detector.ckpt, num_keypoints=int(cfg.detector.num_keypoints))
        kp2d, conf = predict_bout_2d(
            vit, _frames_iter(), masks_dict["masks"], centroids, masks_dict["valid"], cam_mats,
            crop=int(cfg.detector.crop), batch=int(cfg.detector.get("batch", 64)),
            decode_sharpen=float(cfg.detector.get("decode_sharpen", 1.0)),
            distractor_masks=_distractor,
            distractor_dilate=int(cfg.detector.get("distractor_dilate", 0)),
            target_protect=cfg.detector.get("target_protect", None),
            # MUST travel with `ckpt`: a checkpoint trained with
            # train.mask_ablation=true never saw a populated 4th channel, so
            # feeding it one raises NO error and silently degrades accuracy.
            # The flag was added in 821bcd6 and, until now, was passed by no
            # caller at all -- inert.
            zero_mask_channel=bool(cfg.detector.get("zero_mask_channel", False)))
        _distractor = None          # free the (T,C,H,W) mask array promptly
        # The detector emits channels in its training (tracking/COCO) order, which
        # is NOT the XML/model order the rest of the pipeline (triangulation, STAC,
        # silhouette IK, QC) assumes. Reorder O -> model order here so kp2d.npz and
        # every downstream stage are consistently in cfg.model.KP_NAMES order.
        kp2d, conf = reorder_detector_to_model(
            kp2d, conf, list(cfg.detector.kp_names), list(cfg.model.KP_NAMES))
        # Optional per-(camera,keypoint) displacement gate: drops (zeroes the
        # confidence of) a frame whose 2D position hops further than that
        # keypoint's own threshold from the immediately preceding frame.
        # Targets a correlated majority-flip artifact (WingL_V13, Session0
        # bout 28: a landmark alternating between two real-but-wrong visible
        # structures at high confidence, camera majority flipping frame to
        # frame) that confidence gating, multi-view consensus, and 1-Euro
        # smoothing all measurably fail to catch -- see
        # jarvis_jax.tracking.kp2d_displacement_gate's module docstring and
        # .superpowers/sdd/2026-08-29-coarse-to-fine-3d/displacement-gate.md.
        # Applied BEFORE kp2d_filter (raw detector signal, post-reorder):
        # thresholds were derived from RAW frame-to-frame displacement, and a
        # smoothed signal would systematically under-trigger it. Default-off,
        # magnitude-only -- read the module docstring's LIMITATION before
        # enabling on a new bout/camera rig.
        _dg = cfg.detector.get("kp2d_displacement_gate", None)
        if _dg is not None and bool(_dg.get("enabled", False)):
            from jarvis_jax.tracking.kp2d_displacement_gate import displacement_gate_kp2d
            _conf_before = conf
            conf = displacement_gate_kp2d(
                kp2d, conf, list(cfg.model.KP_NAMES),
                default_px=float(_dg.get("default_px", 40.0)),
                per_keypoint_px=dict(_dg.get("per_keypoint_px", {}) or {}),
                conf_thresh=float(cfg.detector.conf_thresh))
            _n_dropped = int(((_conf_before >= float(cfg.detector.conf_thresh))
                              & (conf < float(cfg.detector.conf_thresh))).sum())
            print(f"[displacement-gate] bout {bout_idx} fly{fly}: dropped "
                  f"{_n_dropped} (frame,camera,keypoint) views exceeding their "
                  f"displacement threshold")
        # Optional per-camera 1-Euro 2D smoothing (kills ViTPose soft-argmax
        # high-freq wobble at the source, before triangulation). Applied in
        # MODEL kp order (post-reorder) so preserve_raw_patterns match KP_NAMES.
        # Default-off; validated via the 2D-wobble diagnostic. Wings are kept raw.
        _kf = cfg.detector.get("kp2d_filter", None)
        if _kf is not None and bool(_kf.get("enabled", False)):
            from jarvis_jax.tracking.kp2d_oneeuro import filter_kp2d_oneeuro
            kp2d = filter_kp2d_oneeuro(
                kp2d, conf, list(cfg.model.KP_NAMES),
                conf_thresh=float(cfg.detector.conf_thresh),
                min_cutoff=float(_kf.get("min_cutoff", 1.0)),
                beta=float(_kf.get("beta", 0.0)),
                d_cutoff=float(_kf.get("d_cutoff", 1.0)),
                preserve_raw_patterns=tuple(_kf.get("preserve_raw_patterns", ["Wing"])))
        atomic_save_npz(kp2d_path, kp2d=kp2d, conf=conf)
    with np.load(kp2d_path) as z:
        kp2d, conf = z["kp2d"], z["conf"]

    # -- Stage B: DLT triangulation ----------------------------------------------
    _gate_sig = stage_b_gate_signature(cfg)
    if stage_done(kp3d_path) and not bool(
            (cfg.get("pipeline") or {}).get("allow_stale_kp3d", False)):
        with np.load(kp3d_path) as _z:
            _prev = str(_z["gates"]) if "gates" in _z.files else None
        if _prev != _gate_sig:
            # REFUSE rather than silently reuse, and rather than auto-deleting a
            # 12-minute STAC solve. Same contract as the stac_ik.h5 T-mismatch
            # check below: name the artifacts and let the operator remove them.
            _what = ("carries no gate signature (written before this check "
                     "existed)" if _prev is None else
                     f"was produced under DIFFERENT Stage-B gates\n  stored:  {_prev}")
            raise RuntimeError(
                f"bout {bout_idx} fly{fly}: {kp3d_path} {_what}\n"
                f"  current: {_gate_sig}\n"
                f"Stage B's gates (view-conf, mask-agreement, wing-collapse, "
                f"rigid-repair) only run when kp3d.npz is (re)computed, so "
                f"reusing this file would silently ignore the current config. "
                f"Delete these and rerun this bout/fly:\n"
                f"    rm -f {os.path.join(bout_dir, 'kp3d.npz')} "
                f"{os.path.join(bout_dir, 'kp3d_filt.npz')} "
                f"{os.path.join(bout_dir, 'stac_ik.h5')} "
                f"{os.path.join(bout_dir, 'qpos_refined.npz')} "
                f"{os.path.join(bout_dir, 'qpos_wingfit.npz')} "
                f"{os.path.join(bout_dir, 'outputs.h5')} "
                f"{os.path.join(bout_dir, 'qc.json')} "
                f"{os.path.join(bout_dir, 'DONE')}\n"
                f"(or set pipeline.allow_stale_kp3d=true to accept the stored "
                f"file as-is, which is only right if you know the gates match.)")
    if not stage_done(kp3d_path):
        # Per-view confidence gate: drop a whole (frame, camera) whose median
        # keypoint confidence says the crop probably does not contain the fly.
        # Absent from the config (None) is a strict no-op, so a config without
        # the key triangulates exactly as before. See triangulate_keypoints.
        _view_thresh = cfg.detector.get("view_conf_thresh", None)
        _view_thresh = None if _view_thresh is None else float(_view_thresh)
        if _view_thresh is not None:
            _vmed = view_median_conf(conf)                       # (T,C)
            _dropped = int((_vmed < _view_thresh).sum())
            if _dropped:
                print(f"[view-gate] bout {bout_idx} fly{fly}: dropped "
                      f"{_dropped}/{_vmed.size} (frame,camera) views below "
                      f"median conf {_view_thresh}")
        # Reprojection-residual outlier rejection: the gate confidence cannot
        # provide (a single camera's 2D swapped to the wrong leg at conf
        # 0.4-0.8 passes both thresholds above and drags the DLT). Absent from
        # the config (None) is a strict no-op, same contract as
        # view_conf_thresh. See triangulate_keypoints.
        _resid_px = cfg.detector.get("reproj_resid_px", None)
        _resid_px = None if _resid_px is None else float(_resid_px)
        # Per-view MASK-AGREEMENT gate: a view only counts if its prediction
        # landed on the mask SAM3 found for this fly on this camera. This is
        # independent of confidence -- see view_mask_agreement -- and it also
        # feeds the frame-level min_views gate below, which until now counted
        # views by mask VALIDITY and so never fired on the failure it was meant
        # to catch (bout_00028: a median of 5 valid views throughout, 0 frames
        # gated, while the keypoints were on the other fly).
        _agree_len = masks_cfg.get("kp_mask_agree_fly_lengths", None)
        _agree_len = None if _agree_len is None else float(_agree_len)
        if _agree_len is not None:
            _areas = mask_areas_per_view(masks_dict["masks"])
            _agree = view_mask_agreement(
                kp2d, centroids, masks_dict["valid"], _areas,
                max_fly_lengths=_agree_len)
            _n_drop = int((~_agree & np.asarray(masks_dict["valid"], bool)).sum())
            conf = np.where(_agree[..., None], conf, 0.0)
            views_per_frame = _agree.sum(axis=1)
            print(f"[kp-mask-agree] bout {bout_idx} fly{fly}: dropped {_n_drop} "
                  f"(frame,camera) views whose keypoints missed their own mask by "
                  f">{_agree_len} fly-lengths; agreeing views/frame median "
                  f"{int(np.median(views_per_frame))}")
        # Per-view WING L/R COLLAPSE gate. A view that puts BOTH wing labels on
        # the SAME wing carries no left/right information, and consensus cannot
        # help when it is the MAJORITY doing it: measured on Session0 bout 28
        # fly1, on the 11 frames whose 3-D WingL_V12-V13 length flips from ~4.8u
        # to ~21.6u, FOUR of seven views collapse |WingL_V12 - WingR_V12| to
        # 0.30-0.54 body lengths while 630/853/862 hold 2.0-2.6 -- so the
        # reproj_resid_px gate above discards the three CORRECT views. Dropping
        # the collapsed views for the WING keypoints only: WingL_V12 jumps
        # 13 -> 6, WingL_V13 7 -> 2, frames 300+ unchanged, no new NaNs.
        # min_views_kept is the guard that makes this safe on a folded-wing fly
        # (the female's wings genuinely superpose in most views); absent config
        # is a strict no-op. See jarvis_jax.tracking.wing_lr_assign.
        _wc = cfg.get("wing_collapse") or {}
        if bool(_wc.get("enabled", False)):
            from jarvis_jax.tracking.wing_lr_assign import (
                detect_wing_lr_collapse, mask_collapsed_wing_views)
            _collapsed, _cinfo = detect_wing_lr_collapse(
                kp2d, conf, list(cfg.model.KP_NAMES),
                abs_floor=float(_wc.get("abs_floor", 0.6)),
                rel_frac=float(_wc.get("rel_frac", 0.35)),
                conf_thresh=float(cfg.detector.conf_thresh))
            conf, _minfo = mask_collapsed_wing_views(
                conf, _collapsed, list(cfg.model.KP_NAMES),
                min_views_kept=int(_wc.get("min_views_kept", 4)))
            print(f"[wing-collapse] bout {bout_idx} fly{fly}: masked wings in "
                  f"{_minfo['views_masked']} (frame,camera) views over "
                  f"{_minfo['frames_masked']}/{T} frames; left "
                  f"{_minfo['frames_left_alone_too_few_views']} frames alone "
                  f"(<{_minfo['min_views_kept']} views would survive); "
                  f"per-camera collapse rate "
                  + ", ".join(f"{cameras[i][-3:]}={100*r:.0f}%"
                              for i, r in _cinfo['per_camera_collapse_rate'].items()),
                  flush=True)
        kp3d, conf3d = triangulate_keypoints(
            kp2d, conf, cam_mats, conf_thresh=float(cfg.detector.conf_thresh),
            view_conf_thresh=_view_thresh, reproj_resid_px=_resid_px)
        # RIGID-INVARIANT repair. The collapse gate above cuts the wing L/R
        # flip from 11 frames to 3 on Session0 bout 28 fly1, but the survivors
        # reach the IK: qpos[7] swings 0.7 rad (~40 deg) and the fitted wing
        # site jumps 13-17 mm against a 0.33 mm baseline. The FIT cannot show
        # it (the model's wing is rigid, so outputs.h5 reports a perfect vein
        # length while the POSE is wrong), and no length projection can fix it
        # because the error is angular. So delete the frames the invariant
        # proves impossible and interpolate the short gap; a run longer than
        # max_gap, or one touching a bout edge, is left NaN rather than
        # invented. Absent config is a strict no-op.
        _rr = cfg.get("rigid_repair") or {}
        if bool(_rr.get("enabled", False)):
            from jarvis_jax.tracking.filter import courtship_skeleton_edges
            from jarvis_jax.tracking.rigid_lengths import (
                estimate_bone_lengths, repair_flipped_segments)
            _names = list(cfg.model.KP_NAMES)
            _edges = courtship_skeleton_edges(_names)
            _only = [n for n in (_rr.get("edges_containing") or [])]
            if _only:
                _edges = np.array([e for e in _edges
                                   if any(p in _names[e[0]] or p in _names[e[1]]
                                          for p in _only)])
            _tg, _trep = estimate_bone_lengths(
                kp3d, conf3d, _names, _edges,
                min_conf=float(_rr.get("min_conf", 0.5)),
                min_frames=int(_rr.get("min_frames", 50)),
                max_cv=float(_rr.get("max_cv", 0.20)))
            kp3d, _rrep = repair_flipped_segments(
                kp3d, conf3d, _names, _edges, _tg,
                rel_tol=float(_rr.get("rel_tol", 0.5)),
                max_gap=int(_rr.get("max_gap", 5)),
                also_nan=tuple(_rr.get("also_nan") or ()))
            for _e, _r in _rrep.items():
                print(f"[rigid-repair] bout {bout_idx} fly{fly}: {_e} target "
                      f"{_r['target']:.2f}u -- flagged {_r['n_flagged']} frames "
                      f"in {_r['n_runs']} run(s), interpolated "
                      f"{_r['n_interpolated']}, left NaN {_r['n_left_nan']}; "
                      f"frames {_r['frames'][:12]}", flush=True)
            if not _rrep:
                print(f"[rigid-repair] bout {bout_idx} fly{fly}: no rigid "
                      f"violations over {len(_tg)} edges "
                      f"({sum(v is not None for v in _tg.values())} measurable)",
                      flush=True)
        # Gate frames with too few valid-camera masks to NaN instead of
        # triangulating from too few views (see mask-coverage comment above
        # masks_dict). cfg.masks.min_views absent (_min_views_cfg is None)
        # is a strict no-op -- gate_low_coverage_frames returns kp3d
        # unchanged, so a config without a `masks` block behaves exactly as
        # before this feature.
        kp3d, _n_gated = gate_low_coverage_frames(kp3d, views_per_frame, _min_views_cfg)
        if _min_views_cfg is not None:
            print(f"[mask-coverage] bout {bout_idx} fly{fly}: gated {_n_gated}/{T} frames "
                  f"below min_views={_min_views_cfg} (too few valid camera masks)")
        atomic_save_npz(kp3d_path, kp3d=kp3d, conf3d=conf3d,
                        gates=np.asarray(_gate_sig))
    with np.load(kp3d_path) as z:
        kp3d, conf3d = z["kp3d"], z["conf3d"]
        _kp3d_kp_names = [str(n) for n in z["kp_names"]] if "kp_names" in z.files else None

    kp_names = list(cfg.model.KP_NAMES)

    # `jarvis_jax.tracking.lift_mvq.lift_masked_bout` stamps kp3d.npz with its
    # own `kp_names` array (permuted BY NAME into cfg.model.KP_NAMES already
    # -- see that function's docstring), and `scripts/viz/mvq_bout_video.py`
    # writes mvq-order files under `pose_mvq/` with the SAME `gates` string
    # this stage's staleness check accepts. Neither of those write paths is
    # re-checked by anything else in run_bout.py, so this assertion is the
    # ONLY defence against a kp3d.npz whose keypoint axis silently disagrees
    # with cfg.model.KP_NAMES -- exactly the CLAUDE.md trap (a wrong index
    # space reads a real body part, just the WRONG one, with every metric
    # still confident). A DLT-written kp3d.npz has no `kp_names` array at all
    # (it is implicitly cfg.model.KP_NAMES by construction), so this is a
    # strict no-op there.
    if _kp3d_kp_names is not None and _kp3d_kp_names != kp_names:
        _i = next((i for i, (a, b) in enumerate(zip(_kp3d_kp_names, kp_names)) if a != b),
                  min(len(_kp3d_kp_names), len(kp_names)))
        _a = _kp3d_kp_names[_i] if _i < len(_kp3d_kp_names) else "<missing>"
        _b = kp_names[_i] if _i < len(kp_names) else "<missing>"
        raise RuntimeError(
            f"bout {bout_idx} fly{fly}: {kp3d_path} carries kp_names that do not match "
            f"cfg.model.KP_NAMES -- first mismatch at index {_i}: stored={_a!r} vs "
            f"cfg={_b!r}.\n"
            f"  stored ({len(_kp3d_kp_names)}): {_kp3d_kp_names}\n"
            f"  cfg    ({len(kp_names)}): {kp_names}\n"
            f"Reading this file with the wrong keypoint order measures a real body part, "
            f"just the WRONG one, with every downstream metric still confident -- refusing "
            f"rather than silently mis-indexing every keypoint from here on.")

    # -- Stage B2: temporal smoothing / outlier rejection of the triangulated
    #    kp3d BEFORE scale/offsets/STAC. Distal leg tips occasionally
    #    mistriangulate and jump many mm; STAC then bends the leg to chase the
    #    outlier (jittery/curling IK legs). filter_bout_kp3d reuses the
    #    free-walking preprocessing filter (conf mask -> MAD bone-length reject
    #    -> spike removal -> spline fill -> savgol); wings are excluded so fast
    #    wing motion survives. Raw kp3d.npz is kept as the DLT reference; the
    #    cleaned array (saved to kp3d_filt.npz) feeds scale, offsets, STAC and
    #    the keypoint bridge. No-op when cfg.filtering.enabled is false.
    if bool(cfg.get("filtering") or {}) and bool(cfg.filtering.get("enabled", False)):
        if not stage_done(kp3d_filt_path):
            kp3d_f = filter_bout_kp3d(kp3d, conf3d, kp_names, cfg.filtering)
            atomic_save_npz(kp3d_filt_path, kp3d=kp3d_f, conf3d=conf3d)
        with np.load(kp3d_filt_path) as z:
            kp3d, conf3d = z["kp3d"], z["conf3d"]

    # -- Task 18 stage limit: cfg.pipeline.stop_after='triangulate' (see
    #    should_stop_after_triangulate) stops right here, after 2D keypoints +
    #    triangulation (+ optional B2 smoothing) -- no scale/offsets/STAC/
    #    polish/outputs/overlay. Absent/null (the default) is a strict no-op,
    #    so this is byte-identical to pre-Task-18 behaviour otherwise. Used by
    #    scripts/probe_recording.py's scale probe, which only needs kp3d to
    #    estimate a brand-new recording's body scale + quality from many short
    #    frame windows, without spending GPU time on IK. Deliberately no
    #    mark_done/DONE marker: kp2d.npz/kp3d.npz/kp3d_filt.npz are already
    #    individually stage-checkpointed above, so a later full (non-limited)
    #    run resumes from here instead of recomputing them.
    if should_stop_after_triangulate(cfg):
        print(f"[courtship] bout {bout_idx} fly{fly}: pipeline.stop_after=triangulate "
              f"-- stopping after kp2d/kp3d (no STAC)")
        return

    # -- scale.json: trunk Procrustes body-size scale, computed ONCE (shared
    #    across all bouts/flies, since body size is constant per fly) from
    #    whichever bout/fly gets there first -- see jarvis_jax.tracking.scale.
    #    Without this, raw triangulated keypoints are ~78x the MuJoCo model's
    #    rest-pose scale, which stalls the STAC jaxls LM-batch solve.
    scale_path = os.path.join(run_root, "scale.json")
    if not stage_done(scale_path):
        scale_names = resolve_scale_keypoints(cfg, kp_names)
        _scale_keypoints_mode = str(cfg.scaling.get("scale_keypoints", "trunk"))
        if _scale_keypoints_mode == "rigid_segment":
            # Pose-invariant direct rigid-leg-segment-length measurement (see
            # scripts/estimate_recording_scale.py) instead of a trunk-marker
            # Procrustes/norm-ratio fit -- `cfg.scaling.estimator` is IGNORED
            # here (a rigid segment's length is measured, not fit). Warn (not
            # silently no-op) if an operator has a stale non-default
            # estimator override, and record that it was ignored in
            # scale.json itself so the artifact is self-describing -- do NOT
            # echo the (unused) configured value, which would be
            # indistinguishable from a run where it actually mattered.
            try:
                from scripts.estimate_recording_scale import (
                    per_bout_segment_scale, warn_if_estimator_ignored)
            except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/
                from estimate_recording_scale import (
                    per_bout_segment_scale, warn_if_estimator_ignored)
            warn_if_estimator_ignored(str(cfg.scaling.estimator), caller="run_bout.py")
            # Pool across EVERY bout of this recording that is already
            # triangulated, rather than fitting from whichever bout reached
            # this line first. scale.json is a RECORDING-level artifact but
            # was being written by the first process to arrive, so one
            # arbitrary bout-fly defined the whole recording: measured on
            # Session0/2025_10_20_13_20_04 the shipped 0.009914 is
            # bout_00004/fly0's value EXACTLY -- the 8th percentile of that
            # recording's 59 bout-flies, and 15.5% below the pooled estimate
            # (0.011738). Body size is constant per individual across a
            # recording, so that spread is estimator noise.
            #
            # `estimate_run_root` also returns `scale_by_fly` when the
            # recording's fly0/fly1 identity is stable (sex-canonicalized);
            # run_bout.py already PREFERS that below, but nothing ever wrote
            # it. Same call fixes both.
            try:
                from scripts.estimate_recording_scale import (
                    bout_kp3d_paths, estimate_run_root)
            except ModuleNotFoundError:
                from estimate_recording_scale import (
                    bout_kp3d_paths, estimate_run_root)
            _rec = OmegaConf.create({
                "model": {"KP_NAMES": list(kp_names)},
                "mjcf_path": str(cfg.ik.xml)})
            _est = None
            try:
                _est = estimate_run_root(
                    Path(run_root), _rec, scale_keypoints="rigid_segment")
            except Exception as _e:              # noqa: BLE001 - fall back below
                print(f"[scale] recording-level estimate failed ({type(_e).__name__}: "
                      f"{_e}); falling back to this bout only", flush=True)
            # `estimate_run_root` does not report how many bout-flies it
            # pooled, so count the inputs it would have seen. Without this the
            # count is always 0 and the pooled branch can never be taken.
            _n_pooled = sum(len(bout_kp3d_paths(Path(run_root), _f))
                            for _f in (0, 1))
            if _est and _est.get("scale") and _n_pooled >= 2:
                _payload = {
                    "scale": float(_est["scale"]),
                    "scale_by_fly": _est.get("scale_by_fly"),
                    "n_bout_flies": _n_pooled,
                    "identity": _est.get("identity"),
                    "identity_reason": _est.get("identity_reason"),
                    "source": "estimate_run_root (pooled over the recording)",
                }
                print(f"[scale] pooled over {_n_pooled} bout-flies -> "
                      f"{_est['scale']:.6f}  identity={_est.get('identity')} "
                      f"scale_by_fly={_est.get('scale_by_fly')}", flush=True)
            else:
                # Too early in the run to pool (this may be the first bout
                # triangulated). Record HOW MANY bouts backed the number so a
                # later run can tell this apart from a real pooled estimate.
                _pair_scales = per_bout_segment_scale(
                    kp3d, kp_names, cfg.ik.xml)
                _payload = {
                    "scale": float(np.median(_pair_scales)),
                    "scale_by_fly": None,
                    "n_bout_flies": max(_n_pooled, 1),
                    "identity": (_est or {}).get("identity"),
                    "identity_reason": (_est or {}).get("identity_reason"),
                    "source": "this bout only (too few bouts triangulated to pool)",
                }
                print(f"[scale] WARNING: only {_payload['n_bout_flies']} bout-fly "
                      f"available; scale.json is a SINGLE-BOUT estimate "
                      f"({_payload['scale']:.6f}). Re-run "
                      f"scripts/estimate_recording_scale.py once the recording "
                      f"is fully triangulated.", flush=True)
            _payload.update({
                "scale_keypoints": _scale_keypoints_mode,
                "trunk_keypoints": list(scale_names),
                "estimator": "ignored (rigid_segment)",
                "method": "rigid_segment"})
            atomic_save_json(scale_path, _payload)
        else:
            _scale = compute_trunk_scale(
                kp3d, kp_names, cfg.ik.xml,
                trunk_names=scale_names,
                estimator=cfg.scaling.estimator,
                robust_stat=cfg.scaling.robust_stat,
                robust=cfg.scaling.robust)
            atomic_save_json(scale_path, {
                "scale": float(_scale),
                "scale_keypoints": _scale_keypoints_mode,
                "trunk_keypoints": list(scale_names),
                "estimator": str(cfg.scaling.estimator)})
    with open(scale_path) as _f:
        _scale_data = json.load(_f)
    # Prefer a per-fly scale when present (scripts/estimate_recording_scale.py
    # writes scale_by_fly for recordings whose fly0/fly1 identity is stable
    # across bouts); otherwise fall back to the single shared scale above.
    _scale_by_fly = _scale_data.get("scale_by_fly")
    if _scale_by_fly and str(fly) in _scale_by_fly:
        scale = float(_scale_by_fly[str(fly)])
    else:
        # No per-fly scale: BOTH flies get one number. Body size is constant
        # per individual, and the male and female differ measurably (0.011617
        # vs 0.011744 on Session0/2025_10_20_13_20_04), so this is a real
        # approximation, not a formality -- and until now it happened SILENTLY.
        # It has two causes, and they need different fixes, so name which:
        #   identity != canonical -> some bout lacks sex.json; fly0/fly1 is not
        #     a stable individual label and pooling per slot would blend the
        #     two animals. Fix: scripts/apply_id_review.py.
        #   n_bout_flies < 2      -> one arbitrary bout-fly defined the whole
        #     recording. That is the scale-from-first-bout defect (measured
        #     15.5% low, the 8th percentile of 59 bout-flies). Fix: triangulate
        #     more bouts before the scale stage runs.
        _ident = _scale_data.get("identity")
        _n_pooled = _scale_data.get("n_bout_flies")
        if not bool(cfg.scaling.get("allow_shared_scale", False)):
            raise RuntimeError(
                f"bout {bout_idx} fly{fly}: refusing a SHARED body scale. "
                f"{scale_path} has scale_by_fly={_scale_by_fly!r}, "
                f"identity={_ident!r} ({_scale_data.get('identity_reason')}), "
                f"n_bout_flies={_n_pooled!r}. Both flies would be fitted at one "
                f"size. Fix the cause (apply_id_review.py for identity; more "
                f"triangulated bouts for pooling), or set "
                f"scaling.allow_shared_scale=true to accept it deliberately.")
        print(f"[scale] WARNING bout {bout_idx} fly{fly}: SHARED scale "
              f"(identity={_ident!r}, n_bout_flies={_n_pooled!r}) -- both flies "
              f"fitted at one body size, allowed by scaling.allow_shared_scale",
              flush=True)
        scale = float(_scale_data["scale"])

    # Physical sanity gate: whatever branch/mode produced `scale` above (fresh
    # rigid_segment pool, fresh single-bout fallback, fresh trunk/all fit, OR a
    # stale scale.json read back on resume), `scale` does double duty as BOTH
    # the world->model unit conversion and the per-animal body-size fit, and
    # nothing else checks either factor independently -- a badly wrong scale
    # still looks dimensionally plausible and gets absorbed by marker offsets,
    # which is exactly how the historical 38x scale-from-first-bout defect
    # survived both residual and NaN checks. Checking HERE, after the
    # per-fly-vs-shared resolution above, covers every path that can produce
    # the `scale` this bout-fly actually uses. Error, not warning -- see
    # scripts.estimate_recording_scale.assert_plausible_body_scale.
    try:
        from scripts.estimate_recording_scale import assert_plausible_body_scale
    except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/, not repo root
        from estimate_recording_scale import assert_plausible_body_scale
    assert_plausible_body_scale(
        scale, cfg.ik.xml, context=f"bout {bout_idx} fly{fly} ({scale_path})")

    # -- segment_scales.json: per-segment (per-limb) SHAPE calibration. Like
    #    scale.json, this is a per-fly-constant morph computed ONCE per session
    #    (shared across all bouts/flies, whoever gets there first) and persisted
    #    to run_root/segment_scales.json. The single trunk `scale` above fixes
    #    overall body SIZE but leaves a per-limb PROPORTION mismatch (model femur
    #    too long, tarsus too short) that STAC marker offsets can't absorb, so the
    #    IK can't reach the keypoints. cfg.model.SEGMENT_SCALES is then set in
    #    EVERY process (each bout is a separate process) so stac_mjx morphs the
    #    model (Stac.__init__ -> rescale_per_segment, prints "[calibration]
    #    morphed N body segments"). CRITICAL: this must run BEFORE the offsets fit
    #    so the real offsets.h5 is fit on the MORPHED model. Gated by
    #    cfg.model.segment_calibration (anatomy config).
    #
    #    OPT-IN ONLY, and the absent-key default is False (user decision,
    #    2026-08-31). It used to default TRUE when the key was missing, so any
    #    anatomy config that simply did not declare `segment_calibration`
    #    silently morphed the body -- the opposite of opt-in. The objection is
    #    structural: this gives every calibratable segment its own free scale,
    #    an unbounded change to the body model with nothing holding the result
    #    near a real fly, so a better reprojection number is not evidence the
    #    anatomy improved. Measured 2026-08-31 it buys ~10% site error and
    #    makes the trunk-vs-leg disagreement slightly WORSE (12.26 -> 12.65%).
    if bool(cfg.model.get("segment_calibration", False)):
        print("=" * 78, flush=True)
        print("[segment-calibration] ENABLED -- the body model will be MORPHED "
              "per segment.\n  This is opt-in and off by default: each segment "
              "gets its own free scale, an\n  unbounded change to the anatomy. "
              "Reprojection/site-error gains from this are\n  NOT evidence the "
              "anatomy is more correct. See configs/anatomy/v1.yaml.",
              flush=True)
        print("=" * 78, flush=True)
        seg_scales_path = os.path.join(run_root, "segment_scales.json")
        if not stage_done(seg_scales_path):
            seg_entries = compute_segment_scales(cfg, kp3d, kp_names, scale, run_root)
            atomic_save_json(seg_scales_path, seg_entries)
        with open(seg_scales_path) as _f:
            seg_entries = json.load(_f)
        apply_segment_scales(cfg, seg_entries)

    # -- offsets_fly<f>.h5: marker offsets are a per-INDIVIDUAL constant, like
    #    scale. They used to be fit ONCE per run root on whichever bout-fly got
    #    here first and shared: on Session0/2025_10_20_13_20_04 bout 28 that
    #    sample was fly0 (the female) frames 0..499, so the male ran IK with her
    #    offsets -- abdomen fitted 1.10x too long, wings 1.01-1.06x (2026-09-04).
    #    Now: per fly, pooled over EVERY triangulated bout of this fly in the
    #    recording, high-confidence + physically-plausible frames only,
    #    stratified across bouts, and fit with temporal smoothing OFF because
    #    the sample is not a time series. Per-fly needs canonical identity
    #    (sex.json everywhere); otherwise refuse unless stac.allow_shared_offsets.
    #    See scripts/offsets_sample.py.
    try:
        from scripts.offsets_sample import (
            aligned_per_frame_scales, load_fly_bouts, offsets_fit_cfg,
            resolve_offsets_path, select_offsets_sample)
        from scripts.estimate_recording_scale import _determine_identity, per_frame_scales
    except ModuleNotFoundError:  # direct invocation: sys.path[0] is scripts/
        from offsets_sample import (
            aligned_per_frame_scales, load_fly_bouts, offsets_fit_cfg,
            resolve_offsets_path, select_offsets_sample)
        from estimate_recording_scale import _determine_identity, per_frame_scales
    _off_ident, _off_reason = _determine_identity(Path(run_root))
    offsets_path, _off_mode = resolve_offsets_path(
        run_root, fly, _off_ident,
        allow_shared=bool(cfg.stac.get("allow_shared_offsets", False)),
        reason=_off_reason)
    if _off_mode == "shared":
        print(f"[offsets] WARNING bout {bout_idx} fly{fly}: SHARED marker offsets "
              f"({_off_reason}) -- both flies fitted with one anatomy, allowed by "
              f"stac.allow_shared_offsets", flush=True)
    if not stage_done(offsets_path):
        _min_conf = float(cfg.stac.get("offsets_min_conf", 0.7))
        _bouts = load_fly_bouts(run_root, fly) if _off_mode == "per_fly" else {}
        if not _bouts:            # shared mode, or nothing on disk yet: this bout
            _conf3d = None
            if os.path.exists(kp3d_path):
                with np.load(kp3d_path) as _z:
                    _conf3d = np.asarray(_z["conf3d"]) if "conf3d" in _z.files else None
            _bouts = {int(bout_idx): (np.asarray(kp3d), _conf3d)}
        _scales = {b: aligned_per_frame_scales(
                       k, lambda kk: per_frame_scales(kk, kp_names, cfg.ik.xml))
                   for b, (k, _) in _bouts.items()}
        _sample = select_offsets_sample(
            {b: k for b, (k, _) in _bouts.items()},
            {b: c for b, (_, c) in _bouts.items()},
            n_frames=int(cfg.stac.get("n_fit_frames", 500)), min_conf=_min_conf,
            scale_by_bout=_scales)
        print(f"[offsets] bout {bout_idx} fly{fly}: {_off_mode} fit on "
              f"{_sample.provenance['n_selected']} frames pooled from "
              f"{_sample.provenance['n_bouts']} bout(s) (min_conf {_min_conf}, "
              f"scale gate {_sample.provenance['scale_gate']}); per bout: "
              f"{_sample.provenance['bouts']}", flush=True)
        _tmp_name = os.path.basename(offsets_path) + ".tmp"
        fit_offsets_once(offsets_fit_cfg(cfg), _sample.kp3d, kp_names,
                         offsets_path=_tmp_name, save_path=run_root, scale=scale)
        _sample.provenance.update({"fly": int(fly), "mode": _off_mode,
                                   "identity": _off_ident, "identity_reason": _off_reason,
                                   "scale": float(scale), "smoothing": "off"})
        atomic_save_json(os.path.splitext(offsets_path)[0] + ".json", _sample.provenance)
        os.replace(os.path.join(run_root, _tmp_name), offsets_path)

# -- Stage C: STAC ik_only ----------------------------------------------------
    if not stage_done(stac_h5_path):
        # NaN-robust solve (see the frozen-joint note above). Short dropouts
        # are interpolated so the smoothness chain survives; long ones split
        # the bout into independently-solved segments so a gap cannot
        # propagate NaN gradients across it. A fully-finite bout takes the
        # original single-solve path unchanged.
        #
        # nan_solve.min_keypoints (absent/null by default -> None) relaxes the
        # frame-acceptance gate from "every keypoint finite" to "at least N of
        # K finite": a frame missing a few markers is still a real partial
        # measurement, and the IK residual already ignores a NaN marker
        # per-frame/per-keypoint (see finite_frame_mask's docstring), so it
        # costs nothing to let the solver see it. Absent/null reproduces
        # today's all-or-nothing behaviour exactly.
        #
        # required_indices (fix round 1): min_keypoints alone protects only
        # the solver's residual, which is already NaN-safe per marker. It does
        # NOT protect stac_mjx/compute_stac.py's per-frame warm-start, which
        # reads the root keypoint and the four trunk-orientation keypoints
        # straight out of the array with no finite check -- so a partial
        # frame missing exactly one of those would warm-start the solver from
        # NaN. required_keypoint_indices derives which keypoints those are
        # from the SAME anatomy-config keys the solver itself resolves them
        # from (see its docstring), not a hardcoded name, so this can't
        # silently diverge from the solver if the anatomy config changes.
        _nan_cfg = cfg.get("nan_solve") or {}
        _min_kp_cfg = _nan_cfg.get("min_keypoints", None)
        _required_idx = required_keypoint_indices(cfg, kp_names)
        _kp_solve, _filled = fill_short_gaps(kp3d)
        _ok = finite_frame_mask(_kp_solve, min_keypoints=_min_kp_cfg,
                                required_indices=_required_idx)
        _segs = contiguous_segments(_ok)
        if _min_kp_cfg is not None:
            _strict_ok = finite_frame_mask(_kp_solve)
            _partial_ok = _ok & ~_strict_ok
            if _partial_ok.any():
                _mm = marker_validity_mask(_kp_solve)[_partial_ok]
                print(f"[stac] bout {bout_idx} fly{fly}: nan_solve.min_keypoints="
                      f"{_min_kp_cfg} additionally accepted {int(_partial_ok.sum())} "
                      f"partial frame(s) (mean {float(_mm.sum(axis=1).mean()):.1f}/"
                      f"{_mm.shape[1]} markers present) -- these are real but "
                      f"incomplete observations; treat the fitted pose there with "
                      f"suspicion, not as a full measurement.", flush=True)
        _solver = str(cfg.stac.get("solver", "per_frame"))
        if _solver == "per_frame" and _ok.any():
            # Per-frame multi-start IK (jarvis_jax.tracking.stac_perframe): no
            # temporal coupling, so NaN gaps need no segmenting -- unsolved
            # frames simply stay NaN. Frames only "ok" through fill_short_gaps
            # are re-NaN'd below like the batch path's.
            from jarvis_jax.tracking.stac_perframe import solve_per_frame_ik
            solve_per_frame_ik(cfg, np.asarray(_kp_solve) * float(scale), kp_names,
                               xml_path=str(cfg.ik.xml), offsets_h5=offsets_path,
                               out_h5=os.path.join(bout_dir, "stac_ik.tmp.h5"), solve_mask=_ok,
                               log_prefix=f"[ik-perframe] bout {bout_idx} fly{fly}:")
        elif _solver not in ("per_frame", "batch"):
            raise ValueError(f"stac.solver must be 'per_frame' or 'batch', got {_solver!r}")
        elif _ok.all():
            if _filled.any():
                print(f"[stac] interpolated {_filled.sum()} short-gap frame(s) "
                      f"before the solve", flush=True)
            ik_only_bout(cfg, _kp_solve, kp_names, offsets_path=offsets_path,
                        out_h5="stac_ik.tmp.h5", save_path=bout_dir, scale=scale)
        elif not _segs:
            # Nothing to solve: after gating, this fly has no run of measured
            # frames long enough to fit. That is a real, recorded outcome --
            # the female is edge-on or out of frame for essentially the whole
            # bout -- not a crash. Raising here killed the whole SLURM array
            # task, taking the OTHER fly and the dependent aggregate job with
            # it (measured: Session0 bouts 8/19/26). Record why, write NO
            # DONE and NO stac_ik.h5 so nothing downstream mistakes this for a
            # processed fly, and let the run continue.
            _reason = {
                "status": "unsolvable",
                "reason": "no_finite_segment",
                "min_segment_frames": int(NAN_SOLVE_MIN_SEG),
                "min_keypoints": None if _min_kp_cfg is None else int(_min_kp_cfg),
                "n_frames": int(len(_ok)),
                "n_nan_frames": int((~_ok).sum()),
                "note": ("no run of measured frames long enough to fit; the "
                         "keypoints for this fly were gated away (see "
                         "kp-mask-agree / mask-coverage above)"),
            }
            atomic_save_json(os.path.join(bout_dir, "unsolvable.json"), _reason)
            print(f"[stac] bout {bout_idx} fly{fly}: UNSOLVABLE -- no finite run "
                  f"of >= {NAN_SOLVE_MIN_SEG} frames "
                  f"({_reason['n_nan_frames']}/{_reason['n_frames']} frames have "
                  f"NaN keypoints). Wrote unsolvable.json; leaving this fly "
                  f"without a pose and continuing.", flush=True)
            return
        else:
            _solve_segments_into(
                cfg, _kp_solve, kp_names, _segs, offsets_path=offsets_path,
                out_h5="stac_ik.tmp.h5", bout_dir=bout_dir, scale=scale,
                n_frames=len(_kp_solve), bout_idx=bout_idx, fly=fly)
        # Frames we never measured must not masquerade as solved. Same gate as
        # above -- including required_indices -- as above: a partial-but-accepted
        # frame (raw kp3d, not gap-filled) is a real measurement and must
        # survive; a frame only "ok" in _kp_solve because fill_short_gaps
        # interpolated it is not, and stays re-NaN'd. Using a DIFFERENT
        # (weaker) gate here than for `_ok`/`_segs` above would let a frame
        # the segment-builder correctly excluded (missing root/orientation)
        # survive as "ok" here anyway, since it was never part of any solved
        # segment to begin with.
        _restore_unsolved_nan(os.path.join(bout_dir, "stac_ik.tmp.h5"),
                              finite_frame_mask(kp3d, min_keypoints=_min_kp_cfg,
                                                required_indices=_required_idx),
                              _segs if not _ok.all() else None)
        os.replace(os.path.join(bout_dir, "stac_ik.tmp.h5"),
                  os.path.join(bout_dir, "stac_ik.h5"))

    # RESOLUTIONS #3: the STAC ik and the SAM masks must cover the exact same
    # bout frame range. On resume, a stale stac_ik.h5 left from a different
    # (e.g. differently-trimmed) run would silently desync outputs/QC from
    # masks_dict -- fail loudly instead of polishing garbage.
    import stac_mjx.io_dict_to_hdf5 as ioh5
    q = np.asarray(ioh5.load(stac_h5_path)["qpos"])
    if joints_frozen(q):
        raise RuntimeError(
            f"bout {bout_idx} fly{fly}: STAC returned a FROZEN pose -- every "
            f"non-root DOF is constant across {q.shape[0]} frames, i.e. the "
            f"pose optimisation never updated (only the separately-solved "
            f"root moves). This writes a plausible-looking stac_ik.h5 with a "
            f"finite mean error and is invisible to reprojection/LOO/IoU, so "
            f"it is failed here rather than marked DONE. "
            f"({int((~finite_frame_mask(kp3d)).sum())}/{len(kp3d)} input "
            f"frames have NaN keypoints; see {stac_h5_path})")
    if q.shape[0] != T:
        raise RuntimeError(
            f"bout {bout_idx} fly{fly}: stac_ik.h5 qpos T={q.shape[0]} != "
            f"masks T={T} ({stac_h5_path} vs {bout_npz}); stale/mismatched "
            f"resumed artifact -- delete it and rerun this bout/fly.")

    # -- Stage C2: per-frame weak-DOF polish (wings, abdomen) ---------------------
    #    The batch solve stalls the folded wing's pitch against the yaw stop
    #    (see jarvis_jax.tracking.stac_polish). Runs in place on stac_ik.h5,
    #    idempotent on its signature; only when no pose-derived artifact exists
    #    yet, so a resumed run never silently changes a pose that outputs/qc/viz
    #    were already built from.
    _pcfg = cfg.stac.get("polish") or {}
    if bool(_pcfg.get("enabled", False)) and str(cfg.stac.get("solver", "per_frame")) == "per_frame":
        pass                    # the per-frame solver already multi-starts and selects per frame
    elif bool(_pcfg.get("enabled", False)):
        if os.path.exists(qpos_path):
            import h5py as _h5
            with _h5.File(stac_h5_path, "r") as _f:
                _done_sig = _f.attrs.get("polish_sig")
            if _done_sig is None:
                print(f"[polish] bout {bout_idx} fly{fly}: stac_ik.h5 is UNPOLISHED but "
                      f"downstream pose artifacts already exist ({qpos_path}); leaving it. "
                      f"Delete qpos_refined.npz/outputs.h5/qc*.{{json,npz}}/DONE to polish "
                      f"on the next run.", flush=True)
        else:
            from jarvis_jax.tracking.stac_polish import polish_stac_h5
            polish_stac_h5(cfg, stac_h5_path, kp_names, xml_path=str(cfg.ik.xml),
                           log_prefix=f"[polish] bout {bout_idx} fly{fly}:")

    # -- Stage D: model->mm bridge --------------------------------------------------
    #    Was "silhouette-containment polish". The polish was dead -- with
    #    silhouette_weight == containment_weight == 0 (the shipped default) it
    #    early-returned q = q_init, measured equal to the STAC qpos at
    #    maxabsdiff 0.0 on both flies of Session0 bout 28 -- but the same call
    #    computed the BRIDGE, which is not optional (Stage E maps model units
    #    to mm through it, and a None bridge is the "this frame is NaN"
    #    contract). The bridge was extracted to tracking.bridge and the
    #    silhouette deleted around it. Under the default bridge_mode
    #    'keypoint' no masks are read at all.
    if not stage_done(qpos_path):
        qpos_refined, bs, bR, bt, bok = compute_bridges(
            stac_h5_path, cfg.ik.xml, kp3d, conf3d, cfg.recording.calib_dir,
            kp_scale=scale, bridge_mode=str(cfg.ik.get("bridge_mode", "keypoint")),
            masks_dict=masks_dict)
        atomic_save_npz(qpos_path, qpos=qpos_refined,
                        bridge_s=bs, bridge_R=bR, bridge_t=bt, bridge_ok=bok)
    with np.load(qpos_path) as zq:
        qpos_refined = zq["qpos"]
        bridge_s, bridge_R = zq["bridge_s"], zq["bridge_R"]
        bridge_t, bridge_ok = zq["bridge_t"], zq["bridge_ok"]
    bridges = arrays_to_bridges(bridge_s, bridge_R, bridge_t, bridge_ok)

    # -- Stage D2: wing-pitch refinement against the SAM masks (OPT-IN) -------------
    #    Wing pitch is 99.6% of the marker Jacobian's null direction (per-column
    #    sensitivity yaw 0.271 / roll 0.344 / pitch 0.042): each wing body
    #    carries only two markers and WingX_base maps to the THORAX, so nothing
    #    in the keypoint solve constrains the blade and STAC rotates it into the
    #    abdomen. The masks DO contain wing pixels, so pitch -- and only pitch --
    #    is recovered here. Yaw carries the courtship song and roll is the
    #    best-observed wing DOF; both stay with the marker solve.
    #
    #    ORDER. This sits AFTER compute_bridges and before Stage E, not between
    #    Stages C and D as the design first said: the refinement projects the
    #    wing mesh onto the masks through the per-frame bridge
    #    (br_s * (v @ br_R.T) + br_t), so the bridge must already exist.
    #    Those SAME bridges are REUSED rather than refitted afterwards -- the
    #    bridge is a similarity fitted by Umeyama to the KEYPOINTS, and
    #    refining wing pitch moves only the two wing markers of the full marker
    #    set (and only the two markers whose Jacobian columns are the null
    #    direction the bridge is least sensitive to), so a refit is a no-op to
    #    within noise. If that ever stops being true, recompute here and report
    #    the delta rather than assuming.
    #
    #    Off by default; an absent wing_mask_fit block is a strict no-op that
    #    imports nothing and writes nothing.
    #
    #    PROVENANCE. The fit records the config it was made under INSIDE
    #    qpos_wingfit.npz and `wing_fit_action` compares it, rather than relying
    #    on the Stage-B gate signature: the wing fit cannot change kp3d.npz or
    #    stac_ik.h5, so putting it there would charge a re-triangulation plus a
    #    12-minute STAC solve per bout-fly for a change that provably cannot
    #    alter either -- and the escape hatch (allow_stale_kp3d) removes the
    #    check altogether, which is worse. Three routes reach a stale reuse with
    #    the kp3d gate never firing: deleting kp3d.npz by hand,
    #    allow_stale_kp3d=true, and scripts/analysis/stage_b_restage.py.
    _action, _wf_sig = wing_fit_action(cfg, stored_wing_fit_signature(wingfit_path))
    # WHICH POSE Stage E and everything downstream will hold. Derived from the
    # ACTION, not from "did we recompute this run": on the `reuse` path the fit
    # is loaded and qpos_refined replaced, so an outputs.h5 that was never built
    # from the fit still has to be rebuilt. "none" is the plain STAC/bridge pose.
    _pose_sig = _wf_sig if _action in ("fit", "reuse") else "none"
    if _action == "fit":
        _q_wf, _wf_stats = wing_mask_fit_bout(
            cfg, qpos_refined, bridge_s, bridge_R, bridge_t, bridge_ok,
            masks_dict, cameras)
        atomic_save_npz(wingfit_path, qpos=_q_wf, wing_mask_fit_sig=_wf_sig,
                        **_wf_stats)
    if _action in ("fit", "reuse"):
        with np.load(wingfit_path) as zw:
            qpos_refined = zw["qpos"]
            _st = {k: zw[k] for k in zw.files
                   if k not in ("qpos", "wing_mask_fit_sig")}
        # Same contract as the stac_ik.h5 T-mismatch check above: a resumed
        # artifact from a differently-trimmed run would silently desync Stage E
        # from masks_dict, and build_fly_outputs would fail later with a bare
        # `assert scale.shape[0] == T`. Refuse here, where it can be explained.
        if qpos_refined.shape[0] != T:
            raise RuntimeError(
                f"bout {bout_idx} fly{fly}: qpos_wingfit.npz qpos T="
                f"{qpos_refined.shape[0]} != masks T={T} ({wingfit_path}); "
                f"stale/mismatched resumed artifact -- delete it and rerun "
                f"this bout/fly.")
        # Printed on a resumed run too (the stats live in the npz), so the log
        # of a re-run says what the reused fit did rather than going silent.
        print(f"[wing-mask-fit] bout {bout_idx} fly{fly}: {_action}; wing pitch "
              f"CHANGED on {int(_st['n_refined'])}/{int(_st['n_frames'])} frames; "
              f"{int(_st['n_skipped'])} unchanged; "
              f"{int(_st['n_no_evidence_frames'])} frames carry no mask evidence "
              f"({int(_st['n_no_bridge_frames'])} unsolved by STAC, "
              f"{int(_st['n_thin_frames'])} seen by fewer than "
              f"{int(_st['min_present_cameras'])} mask cameras, "
              f"{int(_st.get('n_gated_frames', 0))} GATED as badly tracked "
              f"({int(_st.get('n_sliver_camera_frames', 0))} sliver + "
              f"{int(_st.get('n_pose_reject_camera_frames', 0))} pose-disagreeing "
              f"camera-frames), "
              f"{int(_st['n_nonfinite_pose_frames'])} non-finite pose), of which "
              f"{int(_st['n_interpolated'])} moved anyway; "
              f"{int(_st['n_clamp_hits'])} hit a joint stop; "
              f"{int(_st.get('n_at_dpitch_bound', 0))} at the |dpitch| bound; "
              f"param_mode {str(_st['param_mode'])}"
              + (f" (knots every {int(_st['knot_spacing'])} frames)"
                 if str(_st['param_mode']) != 'free' else '')
              + f"; median "
              f"wing_pitch change L {float(_st['dpitch_left_deg']):+.2f} deg / "
              f"R {float(_st['dpitch_right_deg']):+.2f} deg", flush=True)

    # -- Stage E: outputs.h5 + qc.json ---------------------------------------------
    #    Every artifact built from the fitted pose -- outputs.h5, qc.json,
    #    qc_perframe.npz, the per-camera overlays and sidebyside.mp4 -- is
    #    guarded by `stage_done`, so without a provenance check enabling Stage D2
    #    on a bout that already has them computes the fit, writes it, logs a
    #    plausible line, and changes nothing anyone reads. Each one therefore
    #    carries the pose it was built from and is compared against `_pose_sig`.
    #    Every write overwrites atomically (stac_mjx.io_dict_to_hdf5.save /
    #    atomic_save_npz), so nothing is deleted.
    _outputs_stale = pose_artifact_stale(outputs_pose_source(outputs_h5_path), _pose_sig)
    if not stage_done(outputs_h5_path) or _outputs_stale:
        build_fly_outputs(
            cfg.recording, ik_h5=stac_h5_path, model_xml=cfg.ik.xml,
            mesh_npz=cfg.ik.mesh_npz, qpos=qpos_refined, bridges=bridges,
            out_path=outputs_h5_path, mesh_subset=str(cfg.outputs.mesh_subset),
            pose_source=_pose_sig)

    # NOTE: bout_dir already ends in f"fly{fly}" (see its construction above),
    # matching qc_json_path -- qc_perframe_path is a sibling, NOT another
    # nested fly{fly} segment.
    qc_perframe_path = os.path.join(bout_dir, "qc_perframe.npz")
    # Separate staleness per artifact, not one shared flag: a preemption between
    # build_fly_outputs and qc_report used to leave QC permanently stale, because
    # the next run saw a matching wing-fit signature ('reuse'), no in-memory
    # flag, and both QC files present -- so both guards skipped forever while
    # outputs.h5 held the new pose. We run on preemptible ckpt nodes.
    _qc_stale = pose_artifact_stale(json_pose_source(qc_json_path), _pose_sig)
    _qcpf_stale = pose_artifact_stale(npz_pose_source(qc_perframe_path), _pose_sig)
    if (not stage_done(qc_json_path) or _qc_stale
            or not stage_done(qc_perframe_path) or _qcpf_stale):
        d = ioh5.load(outputs_h5_path)
        kp3d_mm = np.asarray(d["kp3d_mm"])                  # (T,K,3) FK'd world-mm sites
        mesh_mm = np.asarray(d["mesh_mm"])                  # (T,Kmesh,3) FK'd world-mm mesh subset
        valid2d = conf >= float(cfg.detector.conf_thresh)   # (T,C,K), same threshold as triangulation
        kp3d_by_frame = [kp3d_mm[t] for t in range(T)]
        mesh_by_frame = [mesh_mm[t] for t in range(T)]
        kp2d_by_frame = [{c: kp2d[t, c] for c in range(C)} for t in range(T)]
        vis_by_frame = [{c: valid2d[t, c] for c in range(C)} for t in range(T)]
        masks_by_frame = [
            {c: (masks_dict["masks"][t, c] if masks_dict["valid"][t, c] else None)
             for c in range(C)}
            for t in range(T)
        ]
        # qc.json and qc_perframe.npz need the SAME per-(frame, camera)
        # mesh-vs-mask IoU + keypoint reprojection error, and computing them
        # twice was the single largest cost in each stage (3m39s + 2m52s of a
        # ~35 min bout, both dominated by these two metrics). Compute once here
        # and hand the result to both; each function still computes its own
        # when the kwarg is omitted, so nothing else changes.
        from jarvis_jax.tracking.qc import frame_metrics as _qc_frame_metrics
        _fm = _qc_frame_metrics(rt, kp3d_by_frame=kp3d_by_frame,
                                mesh_by_frame=mesh_by_frame,
                                kp2d_by_frame=kp2d_by_frame,
                                vis_by_frame=vis_by_frame,
                                masks_by_frame=masks_by_frame)
        if not stage_done(qc_json_path) or _qc_stale:
            # ik_reproj metric (additive qc.json key): FITTED (kp3d_by_frame,
            # the FK'd outputs.h5 sites above) vs MEASURED -- the RAW
            # triangulated kp3d.npz, reloaded explicitly here rather than
            # reused from the `kp3d` local, which Stage B2 (cfg.filtering)
            # may have already overwritten with the filtered kp3d_filt.npz
            # array by this point in the function. Same kp2d_by_frame/
            # vis_by_frame gate as every other metric above -- the point is
            # to compare fitted vs. measured against the SAME 2-D, not to
            # introduce a second gate.
            with np.load(kp3d_path) as _zm:
                kp3d_measured_raw = np.asarray(_zm["kp3d"])
            kp3d_measured_by_frame = [kp3d_measured_raw[t] for t in range(T)]

            # Per-group breakdown from viz.core.colors.keypoint_groups (the
            # repo's shared keypoint-semantics module) -- trunk/legs/wings/
            # abdomen, the split that matters for IK work: "wings" is
            # keypoint_groups' "thorax" bucket split by the "Wing" name
            # prefix, "trunk" is what's left of head+thorax. jarvis_jax
            # (qc.py) deliberately has no dependency on the top-level `viz`
            # package, so that split is built HERE and passed in as
            # `group_defs`, not inside qc.py.
            _kg = import_keypoint_groups()(kp_names)
            _wings_idx = [i for i in _kg["thorax"] if kp_names[i].startswith("Wing")]
            group_defs = {
                "trunk": _kg["head"] + [i for i in _kg["thorax"] if i not in _wings_idx],
                "legs": _kg["legs"],
                "wings": _wings_idx,
                "abdomen": _kg["abdomen"],
            }

            qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                     kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                     masks_by_frame=masks_by_frame, out_json=qc_json_path,
                     kp3d_measured_by_frame=kp3d_measured_by_frame,
                     kp_names=kp_names, group_defs=group_defs,
                     frame_metrics=_fm)
            # qc_report writes the file itself, so the provenance is added
            # after. A preemption in between leaves it unstamped, which reads as
            # "none" and rebuilds next run -- the safe direction.
            stamp_json_pose_source(qc_json_path, _pose_sig)

        # -- per-frame QC (Gate A inputs): soft/hard silhouette IoU, marker
        #    reproj, n_cams, one row per frame -- next to qc.json. Reuses the
        #    same *_by_frame locals built for qc_report above.
        if not stage_done(qc_perframe_path) or _qcpf_stale:
            from jarvis_jax.tracking.qc_perframe import per_frame_qc
            pf = per_frame_qc(rt, mesh_by_frame=mesh_by_frame, kp3d_by_frame=kp3d_by_frame,
                              kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                              masks_by_frame=masks_by_frame, frame_metrics=_fm)
            assert "pose_source" not in pf, "per_frame_qc now collides with the stamp"
            atomic_save_npz(qc_perframe_path, pose_source=_pose_sig, **pf)

    # -- overlays: project the per-frame ARTICULATED mesh into each camera,
    #    plus the triangulated kp3d; skip cameras whose video already exists.
    #    mesh_mm comes from outputs.h5 (Stage E's build_fly_outputs): it is
    #    the FK'd `mesh_subset` in world mm, per frame -- i.e. the actual
    #    qpos posed through the skeleton, NOT a rigidly-bridged rest-pose
    #    (T-pose) mesh (the old approach). A rest-pose mesh has no skinning, so
    #    rigidly bridging it could only ever show a canonical T-pose fly, never
    #    the articulated fit.
    if cfg.outputs.overlay:
        d_out = ioh5.load(outputs_h5_path)
        mesh_mm_all = np.asarray(d_out["mesh_mm"])   # (T,Kmesh,3) FK'd world-mm mesh subset
        overlay_dir = os.path.join(bout_dir, "overlays")
        start = bout_start_frame(cfg, bout_idx)
        # An .mp4 cannot carry a stamp, so one sidecar records the pose PER
        # CAMERA. These are FK'd from outputs.h5's mesh_mm, so after a refit
        # they would otherwise keep showing the previous wings -- the
        # `stage_done` skip below is exactly the "plausible artifact, stale
        # content" failure the stamps exist for. Per camera rather than per
        # group because an overlay failure is non-fatal: a group stamp written
        # only when all 7 exist meant one permanently-failing camera re-rendered
        # the other six on every resume, forever.
        overlay_stamp_path = os.path.join(overlay_dir, "pose_source.json")
        _overlay_srcs = overlay_pose_sources(overlay_stamp_path)
        # Build one render JOB per missing camera, then render them
        # CONCURRENTLY -- jarvis_jax.tracking.reproj_video.render_camera_overlays
        # runs one ISOLATED SUBPROCESS per camera (same pattern as Stage F's
        # `python -m viz sidebyside` below, and for the same reason: fork is
        # unsafe while this process holds jax's threads, and multiprocessing's
        # spawn would re-import THIS module -- jax, mujoco, hydra, stac_mjx --
        # in every worker). The cameras share nothing, and the per-frame cost is
        # ~60 ms of matplotlib-rasterise + h264 that no vectorisation can
        # remove, so concurrency is the only large win here that leaves every
        # output byte untouched. cfg.outputs.overlay_workers (default 4) caps
        # it: raise it to 7 on an idle node, set it to 1 for strictly
        # sequential, in-process rendering.
        jobs = []
        for ci, cam in enumerate(cameras):
            overlay_path = os.path.join(overlay_dir, f"{cam}_reproj.mp4")
            _overlay_stale = _overlay_srcs.get(cam) != _pose_sig
            if stage_done(overlay_path) and not _overlay_stale:
                continue
            # Per-camera overlays are cosmetic QC and outputs.h5 is already
            # written above -- a render failure on one camera must never fail
            # the bout (mirrors Stage F's "logged but never fails the bout").
            try:
                mesh2d_by_frame, kp2d_by_frame_overlay = [], []
                for t in range(T):
                    mesh_t = mesh_mm_all[t]
                    mesh_ok = np.isfinite(mesh_t).all(-1)
                    if not mesh_ok.any():
                        mesh2d_by_frame.append(np.zeros((0, 2)))
                    else:
                        mesh2d_by_frame.append(project_points(cam_mats[ci], mesh_t[mesh_ok]))
                    kp_t = kp3d[t]
                    kp_ok = np.isfinite(kp_t).all(-1)
                    kp2d_by_frame_overlay.append(project_points(cam_mats[ci], kp_t[kp_ok]))
                import cv2
                _cap = cv2.VideoCapture(
                    os.path.join(str(cfg.recording.session_dir), f"{cam}.mp4"))
                H = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                W = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _cap.release()
                if H <= 0 or W <= 0:
                    H, W = 1080, 1920  # last-resort fallback; normally valid
                # The worker re-derives the sync plan with the same
                # load_plan(session_dir) this function used (SyncPlan objects
                # are not worth pickling), so its reads are the same slots.
                jobs.append(dict(cam=cam, session_dir=str(cfg.recording.session_dir),
                                 start=int(start), T=int(T), H=H, W=W,
                                 mesh2d=mesh2d_by_frame, kp2d=kp2d_by_frame_overlay,
                                 out_path=overlay_path,
                                 fps=int(cfg.outputs.overlay_fps)))
            except Exception as e:  # noqa: BLE001 -- QC artifact, never fatal
                print(f"[overlay] WARNING: bout {bout_idx} fly{fly} cam {cam} "
                      f"reproj overlay setup failed ({type(e).__name__}: {e}) -- skipping.")
        if jobs:
            _t_ov = time.time()
            _errs = render_camera_overlays(
                jobs, max_workers=int(cfg.outputs.get("overlay_workers", 4)))
            for cam, err in _errs.items():
                if err:
                    print(f"[overlay] WARNING: bout {bout_idx} fly{fly} cam {cam} "
                          f"reproj overlay failed ({err}) -- skipping.")
            print(f"[overlay] bout {bout_idx} fly{fly}: {len(jobs)} camera(s) in "
                  f"{time.time() - _t_ov:.0f}s "
                  f"({sum(1 for e in _errs.values() if e is None)} written)", flush=True)
            # Stamp exactly the cameras that rendered. One that failed keeps no
            # entry, so it retries next run while the others stay skipped.
            for _j in jobs:
                if _errs.get(_j["cam"]) is None and stage_done(_j["out_path"]):
                    _overlay_srcs[_j["cam"]] = _pose_sig
            atomic_save_json(overlay_stamp_path, {"cameras": _overlay_srcs})

    # -- Stage F: standard QC side-by-side video (raw video + SAM mask + ViTPose
    #    2-D skeleton  |  MuJoCo IK render + 3-D sites). Auto-generated per
    #    (bout, fly) so every run ships a visual QC artifact next to outputs.h5.
    #    Run as an ISOLATED SUBPROCESS (python -m viz sidebyside): the viz view
    #    re-composes its own Hydra config, which would raise "GlobalHydra is
    #    already initialized" if called in-process under this @hydra.main app.
    #    Capped to outputs.sidebyside_frames to bound the MuJoCo render on long
    #    bouts. A viz failure is logged but never fails the bout (QC, not core).
    if bool(cfg.outputs.get("sidebyside", True)):
        sbs_path = os.path.join(bout_dir, "sidebyside.mp4")
        # Same sidecar treatment as the overlays, and for a sharper reason: this
        # is the pipeline's DEFAULT visual QC artifact, the one a reader judges
        # the fit by, and `viz sidebyside` loads the pose itself -- so it is also
        # told WHICH pose to draw (--pose below) instead of defaulting to
        # qpos_refined.npz, which is the PRE-FIT pose.
        sbs_stamp_path = os.path.join(bout_dir, "sidebyside.pose_source.json")
        _sbs_stale = pose_artifact_stale(json_pose_source(sbs_stamp_path), _pose_sig)
        if not stage_done(sbs_path) or _sbs_stale:
            import subprocess
            n_sbs = int(cfg.outputs.get("sidebyside_frames", 300))
            _sbs_right = str(cfg.outputs.get("sidebyside_right", "rigcam"))
            # Pass THIS recording's session/predictions dir + the bout's absolute
            # start frame, else the viz resolves the default (Session0) recording
            # and renders the wrong video/masks/frames.
            cmd = [sys.executable, "-m", "viz", "sidebyside",
                   "--run", run_root, "--bout", str(bout_idx), "--fly", str(fly),
                   "--n", str(n_sbs),
                   # View-matched right panel, configurable via
                   # cfg.outputs.sidebyside_right (default "rigcam"): a MuJoCo
                   # render from a camera built off the LEFT panel's own
                   # calibration. The rig is orthographic (verified:
                   # P[2,:3]==0 for all 7 cameras), so an orthographic MuJoCo
                   # camera matches it exactly -- an actual view-matched IK
                   # render, not just fitted points. "reproj" (points only, no
                   # MuJoCo) is cheaper but isn't "the IK rendering with the
                   # frames" a reader asked for. The OLD default, `mujoco`,
                   # rendered MuJoCo's model-space `track1` camera, whose
                   # extrinsics are unrelated to the left camera -- so the fly
                   # appeared rotated between the panes and a correct fit read
                   # as "facing the wrong way", and per its own --help it
                   # "cannot be used to judge orientation"; `rigcam` (added
                   # after this comment was first written) is the fix for
                   # exactly that, not `reproj`.
                   "--right", _sbs_right,
                   # --pose only where the right panel actually reads a pose;
                   # see sidebyside_pose_args (the mujoco panel refuses one).
                   *sidebyside_pose_args(_sbs_right, _action),
                   "--conf", str(float(cfg.detector.conf_thresh)),
                   "--session-dir", str(cfg.recording.session_dir),
                   "--predictions-dir", str(cfg.recording.predictions_dir),
                   # The right panel is rendered from THESE camera poses, so it
                   # must be this recording's calibration, not the default one.
                   "--calib-dir", str(cfg.recording.calib_dir),
                   "--start-frame", str(int(bout_start_frame(cfg, bout_idx))),
                   "--fps", str(int(cfg.outputs.overlay_fps)), "--out", sbs_path]
            # The parent process still holds ~90% of the GPU (jax preallocated),
            # so the viz subprocess must NOT try to grab GPU memory. Its render is
            # MuJoCo/EGL (a small GL context, fine alongside the parent) and needs
            # no jax-GPU, so force jax onto CPU for the subprocess.
            sbs_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
            # One row per view: three real camera views beside three
            # view-matched MuJoCo renders. Roles resolve from THIS recording's
            # calibration (viz.core.rigviews), so the same setting works for
            # Session0 and Session1. Empty string -> the old single-view render.
            _views = str(cfg.outputs.get("sidebyside_views", "") or "").strip()
            if _views:
                cmd += ["--views", _views]
            r = subprocess.run(cmd, capture_output=True, text=True, env=sbs_env)
            if r.returncode != 0:
                print(f"[courtship] bout {bout_idx} fly{fly}: sidebyside viz FAILED "
                      f"(non-fatal):\n{r.stderr[-1500:]}")
            else:
                print(f"[courtship] bout {bout_idx} fly{fly}: sidebyside -> {sbs_path}")
                if stage_done(sbs_path):
                    atomic_save_json(sbs_stamp_path, {"pose_source": _pose_sig})

    mark_done(bout_dir)
    print(f"[courtship] bout {bout_idx} fly{fly}: DONE -> {bout_dir}")


def _log_gpu_env():
    """Log the physical GPUs (nvidia-smi -L) and how many JAX actually recognizes,
    at job start. If JAX silently falls back to CPU (transient CUDA-init failure),
    every stage runs on CPU and the job appears to hang -- this line makes that
    obvious ('jax sees 0 GPU(s)') instead of requiring a live process autopsy."""
    import subprocess
    if os.environ.get("JAX_PLATFORMS", "") == "cpu":
        print("[gpu-check] WARNING: JAX_PLATFORMS=cpu is set -> the pipeline will run on CPU "
              "(extremely slow). If unintended, it was likely inherited via `sbatch "
              "--export=ALL` from the submit shell; `unset JAX_PLATFORMS` in the job.", flush=True)
    try:
        r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30)
        print("[gpu-check] nvidia-smi -L:\n" + (r.stdout.strip() or r.stderr.strip() or "(no output)"),
              flush=True)
    except Exception as e:
        print(f"[gpu-check] nvidia-smi -L failed: {e}", flush=True)
    try:
        import jax
        devs = jax.devices()
        ngpu = sum(1 for d in devs if getattr(d, "platform", "") == "gpu")
        print(f"[gpu-check] jax sees {ngpu} GPU(s) of {len(devs)} device(s): {devs}", flush=True)
        if ngpu == 0 and os.environ.get("JAX_PLATFORMS", "") != "cpu":
            # Fail fast instead of grinding on CPU for hours (which also blocks the
            # dependent chain). On a preemptible/--requeue array this reschedules
            # onto another node. Set JAX_PLATFORMS=cpu to intentionally allow CPU.
            raise RuntimeError(
                "JAX has NO GPU (fell back to CpuDevice) but a GPU is present per "
                "nvidia-smi -L above. Batch nodes need `module load cuda/12.9.1` to "
                "expose libcuda. Failing fast so this task requeues on another node. "
                "Set JAX_PLATFORMS=cpu to force CPU intentionally.")
    except RuntimeError:
        raise
    except Exception as e:
        print(f"[gpu-check] jax.devices() failed: {e}", flush=True)


def _canonicalize_bout_sex(cfg, bout_idx: int):
    """After both flies of a courtship bout are written, canonicalize male -> fly1.

    The human ID review is the authority when one exists. It normally arrives
    baked into the mask npz's `sex_meta` (written by
    scripts/canonicalize_sam_masks.py), which is the form that survives a
    re-run because it is expressed in the mask slot space the fly dirs inherit.
    `cfg.sexing.review` may additionally point at an id_review manifest, used
    only when the mask carries no human decision.
    """
    from jarvis_jax.tracking.sexing import (
        canonicalize_bout, load_review, read_sex_meta, review_key_for)
    run_root = str(cfg.outputs.out)
    bout_dir = os.path.join(run_root, "bouts", f"bout_{bout_idx:05d}")
    mask_npz = os.path.join(str(cfg.recording.predictions_dir),
                            f"bout_{bout_idx:05d}", "sam3_masks.npz")
    sx = cfg.get("sexing", {}) or {}
    review_entry = None
    review_path = sx.get("review", None)
    if review_path:
        key = review_key_for(bout_dir, pose_dir=os.path.basename(run_root))
        review_entry = load_review(str(review_path)).get(key)
        if review_entry is None:
            print(f"[sexing] bout {bout_idx}: no entry for {key!r} in "
                  f"{review_path} -- falling back to the mask/pose signal")
    res = canonicalize_bout(
        bout_dir, list(cfg.model.KP_NAMES),
        mask_sex_meta=read_sex_meta(mask_npz),
        review_entry=review_entry,
        ratio_thr=float(sx.get("ratio_thr", 1.5)),
        high_ratio=float(sx.get("high_ratio", 2.5)),
        conf_min=float(sx.get("conf_min", 0.2)),
        min_frames=int(sx.get("min_frames", 20)))
    male_label = "fly?" if res["male_fly"] is None else f"fly{res['male_fly']}"
    print(f"[sexing] bout {bout_idx}: male={male_label} conf={res['confidence']} "
          f"authority={res['authority']} method={res['method']} "
          f"swap={res['applied_swap']} heuristic_agrees={res['heuristic_agrees']}")
    return res


def main_from_cfg(cfg: DictConfig):
    _log_gpu_env()
    bout_ids = resolve_bout_ids(cfg)
    n_anim = int(cfg.recording.num_animals)
    print(f"[courtship] processing {len(bout_ids)} bout(s): {bout_ids}")
    for bout_idx in bout_ids:
        for fly in range(n_anim):
            process_bout_fly(cfg, bout_idx, fly)
        # Two flies cannot be in the same place: a collapsed separation means
        # one track was captured by the other animal. Reported, never fatal --
        # it is a QC signal about the INPUT, and the bout's artifacts are still
        # what they are. Written beside coverage.json so it is inspectable
        # without re-deriving it. Skipped for single-animal assays.
        if n_anim == 2:
            _bd = os.path.join(str(cfg.outputs.out), "bouts", f"bout_{bout_idx:05d}")
            _tm = check_track_merge(
                _bd, min_separation_body_lengths=float(
                    (cfg.get("masks") or {}).get("merge_body_lengths", 0.5)))
            if _tm["status"] == "merged":
                print(f"[track-merge] bout {bout_idx}: WARNING the two flies' tracks "
                      f"collapse on {_tm['n_merged']} frame(s) "
                      f"({100 * _tm['frac_merged']:.1f}%): min separation "
                      f"{_tm['min_separation']:.2f} vs body length "
                      f"{_tm['body_length']:.2f}. One track is probably following "
                      f"the other fly.")
            else:
                print(f"[track-merge] bout {bout_idx}: {_tm['status']} "
                      f"(min separation {_tm['min_separation']}, "
                      f"body length {_tm['body_length']})")
            try:
                atomic_save_json(os.path.join(_bd, "track_qc.json"), _tm)
            except Exception as _e:                       # noqa: BLE001
                print(f"[track-merge] bout {bout_idx}: could not write track_qc.json ({_e})")
        # Fly sexing is now done authoritatively at SAM3 step-0 (mask-area vote in
        # jarvis_jax.predict.sam3_driver.canonicalize_male_fly), which packs masks
        # in canonical order so pose fly0/fly1 already has male=fly1. The old
        # pose-level wing-CV sexing (_canonicalize_bout_sex) was unreliable and is
        # no longer auto-invoked. Manual identity overrides go through
        # scripts/canonicalize_session_sex.py --labels (apply_manual_labels ->
        # jarvis_jax.tracking.sexing._swap_fly_dirs), NOT _canonicalize_bout_sex,
        # which is retained only for ad-hoc/debug use.


@hydra.main(version_base=None, config_path="../configs", config_name="pipeline")
def main(cfg: DictConfig):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
