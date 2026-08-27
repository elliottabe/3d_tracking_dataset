#!/usr/bin/env python3
"""Resumable per-bout courtship inference driver (stages A-E).

For a bout index and each fly (``range(cfg.recording.num_animals)``), runs:

  A  ViTPose 2-D on SAM3-masked crops                       -> kp2d.npz
  B  DLT triangulation                                      -> kp3d.npz
  C  STAC ik_only (offsets fit once, shared across bouts)   -> stac_ik.h5
  D  silhouette-containment polish                          -> qpos_refined.npz
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

from pathlib import Path

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf

from jarvis_jax.tracking.resume import (
    atomic_save_npz, atomic_save_json, stage_done, mark_done, bout_complete)
from jarvis_jax.tracking.bout_masks import load_bout_masks, check_bout_camera_order
from jarvis_jax.tracking.predict_2d import (
    load_detector, predict_bout_2d, reorder_detector_to_model)
from jarvis_jax.tracking.triangulate import (triangulate_keypoints,
                                             view_median_conf)
from jarvis_jax.tracking.filter import filter_bout_kp3d
from jarvis_jax.tracking.scale import compute_trunk_scale
from jarvis_jax.tracking.stac import fit_offsets_once, ik_only_bout
from jarvis_jax.tracking.polish import polish_bout
from jarvis_jax.tracking.outputs import build_fly_outputs
from jarvis_jax.tracking.qc import qc_report
from jarvis_jax.tracking.reproj_video import write_camera_video
from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for, masks_are_stale
from jarvis_jax.predict.synced_reader import load_plan, read_window, read_one_cam
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


def atomic_write_mp4(path, write_fn):
    """Like resume.atomic_write, but keeps the '.mp4' suffix on the
    temp file: imageio's writer picks its backend by sniffing the URI's
    extension, so a bare '<path>.tmp' (no '.mp4') makes it fall back to the
    wrong plugin (e.g. TIFF) and crash on the 'fps' kwarg."""
    tmp = path[:-4] + ".tmp.mp4" if path.endswith(".mp4") else path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_fn(tmp)
    if not os.path.exists(tmp):
        # imageio never creates the file when zero frames are appended (e.g. a
        # bout whose frame window runs past the end of the source video), so
        # os.replace would crash with FileNotFoundError. This artifact is
        # cosmetic QC -- warn and skip rather than fail the bout.
        print(f"[atomic_write_mp4] WARNING: writer produced no file for {path} "
              f"(0 frames?) -- skipping this video.")
        return None
    os.replace(tmp, path)
    return path


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
    atomic_save_npz(qc_perframe_path, **pf)
    print(f"[courtship] bout {bout_idx} fly{fly}: backfilled qc_perframe.npz "
          f"for already-DONE bout -> {qc_perframe_path}")


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
                   outputs_h5_path, qc_json_path, os.path.join(bout_dir, "qc_perframe.npz")):
            try:
                if os.path.exists(_p):
                    os.remove(_p)
            except OSError:
                pass

    # -- Stage A: ViTPose 2-D ---------------------------------------------------
    if not stage_done(kp2d_path):
        start = bout_start_frame(cfg, bout_idx)
        # Centroids come from masks_dict (already reordered/autofixed above),
        # NOT a fresh raw npz read -- they must stay in lockstep with
        # masks_dict["masks"]'s camera axis (see autofix_bout_camera_order
        # above).
        centroids = masks_dict["centroids"]

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
        vit = load_detector(cfg.detector.ckpt, num_keypoints=int(cfg.detector.num_keypoints))
        kp2d, conf = predict_bout_2d(
            vit, _frames_iter(), masks_dict["masks"], centroids, masks_dict["valid"], cam_mats,
            crop=int(cfg.detector.crop), batch=int(cfg.detector.get("batch", 64)),
            decode_sharpen=float(cfg.detector.get("decode_sharpen", 1.0)),
            distractor_masks=_distractor)
        _distractor = None          # free the (T,C,H,W) mask array promptly
        # The detector emits channels in its training (tracking/COCO) order, which
        # is NOT the XML/model order the rest of the pipeline (triangulation, STAC,
        # silhouette IK, QC) assumes. Reorder O -> model order here so kp2d.npz and
        # every downstream stage are consistently in cfg.model.KP_NAMES order.
        kp2d, conf = reorder_detector_to_model(
            kp2d, conf, list(cfg.detector.kp_names), list(cfg.model.KP_NAMES))
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
        kp3d, conf3d = triangulate_keypoints(
            kp2d, conf, cam_mats, conf_thresh=float(cfg.detector.conf_thresh),
            view_conf_thresh=_view_thresh, reproj_resid_px=_resid_px)
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
        atomic_save_npz(kp3d_path, kp3d=kp3d, conf3d=conf3d)
    with np.load(kp3d_path) as z:
        kp3d, conf3d = z["kp3d"], z["conf3d"]

    kp_names = list(cfg.model.KP_NAMES)

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
                "mjcf_path": str(cfg.silhouette.xml)})
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
                    kp3d, kp_names, cfg.silhouette.xml)
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
                kp3d, kp_names, cfg.silhouette.xml,
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
        scale = float(_scale_data["scale"])

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
    if bool(cfg.model.get("segment_calibration", True)):
        seg_scales_path = os.path.join(run_root, "segment_scales.json")
        if not stage_done(seg_scales_path):
            seg_entries = compute_segment_scales(cfg, kp3d, kp_names, scale, run_root)
            atomic_save_json(seg_scales_path, seg_entries)
        with open(seg_scales_path) as _f:
            seg_entries = json.load(_f)
        apply_segment_scales(cfg, seg_entries)

    # -- offsets.h5: fit ONCE (shared across all bouts/flies) on a
    #    high-confidence kp3d sample from whichever bout gets there first --
    if not stage_done(offsets_path):
        sample_idx = high_confidence_sample(kp3d)
        fit_offsets_once(cfg, kp3d[sample_idx], kp_names,
                         offsets_path="offsets.h5.tmp", save_path=run_root, scale=scale)
        os.replace(os.path.join(run_root, "offsets.h5.tmp"),
                  os.path.join(run_root, "offsets.h5"))

    
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


def finite_frame_mask(kp3d) -> np.ndarray:
    """(T,) bool: frames whose keypoints are entirely finite."""
    a = np.asarray(kp3d)
    return np.isfinite(a).all(axis=tuple(range(1, a.ndim)))


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
                    full = np.full((n_frames,) + arr.shape[1:], np.nan,
                                   dtype=float if arr.dtype.kind == "f" else arr.dtype)
                    merged[k] = full
                else:
                    meta[k] = v
        for k in merged:
            merged[k][a:b] = np.asarray(d[k])
        os.remove(os.path.join(bout_dir, tmp))
    out = dict(meta)
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
            arr = arr.astype(float, copy=True)
            arr[bad] = np.nan
            d[k] = arr
            n += 1
    ioh5.save(h5_path, d)
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


# -- Stage C: STAC ik_only ----------------------------------------------------
    if not stage_done(stac_h5_path):
        # NaN-robust solve (see the frozen-joint note above). Short dropouts
        # are interpolated so the smoothness chain survives; long ones split
        # the bout into independently-solved segments so a gap cannot
        # propagate NaN gradients across it. A fully-finite bout takes the
        # original single-solve path unchanged.
        _kp_solve, _filled = fill_short_gaps(kp3d)
        _ok = finite_frame_mask(_kp_solve)
        _segs = contiguous_segments(_ok)
        if _ok.all():
            if _filled.any():
                print(f"[stac] interpolated {_filled.sum()} short-gap frame(s) "
                      f"before the solve", flush=True)
            ik_only_bout(cfg, _kp_solve, kp_names, offsets_path=offsets_path,
                        out_h5="stac_ik.tmp.h5", save_path=bout_dir, scale=scale)
        elif not _segs:
            raise RuntimeError(
                f"bout {bout_idx} fly{fly}: no finite run of >= "
                f"{NAN_SOLVE_MIN_SEG} frames to solve "
                f"({int((~_ok).sum())}/{len(_ok)} frames have NaN keypoints)")
        else:
            _solve_segments_into(
                cfg, _kp_solve, kp_names, _segs, offsets_path=offsets_path,
                out_h5="stac_ik.tmp.h5", bout_dir=bout_dir, scale=scale,
                n_frames=len(_kp_solve), bout_idx=bout_idx, fly=fly)
        # Frames we never measured must not masquerade as solved.
        _restore_unsolved_nan(os.path.join(bout_dir, "stac_ik.tmp.h5"),
                              finite_frame_mask(kp3d), _segs if not _ok.all() else None)
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

    # -- Stage D: silhouette-containment polish -----------------------------------
    if not stage_done(qpos_path):
        qpos_refined, bridges = polish_bout(
            stac_h5_path, cfg, kp3d, conf3d, masks_dict, cfg.recording.calib_dir,
            kp_scale=scale, bridge_mode=str(cfg.silhouette.get("bridge_mode", "mask")))
        bs, bR, bt, bok = bridges_to_arrays(bridges)
        atomic_save_npz(qpos_path, qpos=qpos_refined,
                        bridge_s=bs, bridge_R=bR, bridge_t=bt, bridge_ok=bok)
    with np.load(qpos_path) as zq:
        qpos_refined = zq["qpos"]
        bridges = arrays_to_bridges(zq["bridge_s"], zq["bridge_R"], zq["bridge_t"], zq["bridge_ok"])

    # -- Stage E: outputs.h5 + qc.json ---------------------------------------------
    if not stage_done(outputs_h5_path):
        build_fly_outputs(
            cfg.recording, ik_h5=stac_h5_path, model_xml=cfg.silhouette.xml,
            mesh_npz=cfg.silhouette.mesh_npz, qpos=qpos_refined, bridges=bridges,
            out_path=outputs_h5_path, mesh_subset=str(cfg.outputs.mesh_subset))

    # NOTE: bout_dir already ends in f"fly{fly}" (see its construction above),
    # matching qc_json_path -- qc_perframe_path is a sibling, NOT another
    # nested fly{fly} segment.
    qc_perframe_path = os.path.join(bout_dir, "qc_perframe.npz")
    if not stage_done(qc_json_path) or not stage_done(qc_perframe_path):
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
        if not stage_done(qc_json_path):
            qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                     kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                     masks_by_frame=masks_by_frame, out_json=qc_json_path)

        # -- per-frame QC (Gate A inputs): soft/hard silhouette IoU, marker
        #    reproj, n_cams, one row per frame -- next to qc.json. Reuses the
        #    same *_by_frame locals built for qc_report above.
        if not stage_done(qc_perframe_path):
            from jarvis_jax.tracking.qc_perframe import per_frame_qc
            pf = per_frame_qc(rt, mesh_by_frame=mesh_by_frame, kp3d_by_frame=kp3d_by_frame,
                              kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                              masks_by_frame=masks_by_frame)
            atomic_save_npz(qc_perframe_path, **pf)

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
        for ci, cam in enumerate(cameras):
            overlay_path = os.path.join(overlay_dir, f"{cam}_reproj.mp4")
            if stage_done(overlay_path):
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

                def _frames_rgb(cam=cam, H=H, W=W):
                    # read_one_cam yields (frame_rgb (H,W,3)|None, present); a dropped
                    # slot (frame None) still needs a placeholder frame so
                    # write_camera_video's frame-count bookkeeping stays in lockstep
                    # with mesh2d_by_frame/kp2d_by_frame_overlay (one entry per T).
                    # The placeholder must match the real (H,W) -- imageio's ffmpeg
                    # writer raises "All images in a movie should have same size"
                    # if any yielded frame's shape differs from the first.
                    for _fr, _present in read_one_cam(
                            cfg.recording.session_dir, cam, sync_plan, start, T):
                        yield _fr if _fr is not None else np.zeros((H, W, 3), np.uint8)

                def _write(tmp, mesh2d_by_frame=mesh2d_by_frame,
                          kp2d_by_frame_overlay=kp2d_by_frame_overlay):
                    write_camera_video(
                        tmp, frames_rgb_iter=_frames_rgb(),
                        mesh2d_by_frame=mesh2d_by_frame, kp2d_by_frame=kp2d_by_frame_overlay,
                        fps=int(cfg.outputs.overlay_fps))

                atomic_write_mp4(overlay_path, _write)
            except Exception as e:  # noqa: BLE001 -- QC artifact, never fatal
                print(f"[overlay] WARNING: bout {bout_idx} fly{fly} cam {cam} "
                      f"reproj overlay failed ({type(e).__name__}: {e}) -- skipping.")

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
        if not stage_done(sbs_path):
            import subprocess
            n_sbs = int(cfg.outputs.get("sidebyside_frames", 300))
            # Pass THIS recording's session/predictions dir + the bout's absolute
            # start frame, else the viz resolves the default (Session0) recording
            # and renders the wrong video/masks/frames.
            cmd = [sys.executable, "-m", "viz", "sidebyside",
                   "--run", run_root, "--bout", str(bout_idx), "--fly", str(fly),
                   "--n", str(n_sbs), "--camera", "track1",
                   "--conf", str(float(cfg.detector.conf_thresh)),
                   "--session-dir", str(cfg.recording.session_dir),
                   "--predictions-dir", str(cfg.recording.predictions_dir),
                   "--start-frame", str(int(bout_start_frame(cfg, bout_idx))),
                   "--fps", str(int(cfg.outputs.overlay_fps)), "--out", sbs_path]
            # The parent process still holds ~90% of the GPU (jax preallocated),
            # so the viz subprocess must NOT try to grab GPU memory. Its render is
            # MuJoCo/EGL (a small GL context, fine alongside the parent) and needs
            # no jax-GPU, so force jax onto CPU for the subprocess.
            sbs_env = {**os.environ, "JAX_PLATFORMS": "cpu"}
            r = subprocess.run(cmd, capture_output=True, text=True, env=sbs_env)
            if r.returncode != 0:
                print(f"[courtship] bout {bout_idx} fly{fly}: sidebyside viz FAILED "
                      f"(non-fatal):\n{r.stderr[-1500:]}")
            else:
                print(f"[courtship] bout {bout_idx} fly{fly}: sidebyside -> {sbs_path}")

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
    """After both flies of a courtship bout are written, canonicalize male -> fly1."""
    from jarvis_jax.tracking.sexing import canonicalize_bout, read_sex_meta
    run_root = str(cfg.outputs.out)
    bout_dir = os.path.join(run_root, "bouts", f"bout_{bout_idx:05d}")
    mask_npz = os.path.join(str(cfg.recording.predictions_dir),
                            f"bout_{bout_idx:05d}", "sam3_masks.npz")
    sx = cfg.get("sexing", {}) or {}
    res = canonicalize_bout(
        bout_dir, list(cfg.model.KP_NAMES),
        mask_sex_meta=read_sex_meta(mask_npz),
        ratio_thr=float(sx.get("ratio_thr", 1.5)),
        high_ratio=float(sx.get("high_ratio", 2.5)),
        conf_min=float(sx.get("conf_min", 0.2)),
        min_frames=int(sx.get("min_frames", 20)))
    male_label = "fly?" if res["male_fly"] is None else f"fly{res['male_fly']}"
    print(f"[sexing] bout {bout_idx}: male={male_label} conf={res['confidence']} "
          f"method={res['method']} swap={res['applied_swap']}")
    return res


def main_from_cfg(cfg: DictConfig):
    _log_gpu_env()
    bout_ids = resolve_bout_ids(cfg)
    n_anim = int(cfg.recording.num_animals)
    print(f"[courtship] processing {len(bout_ids)} bout(s): {bout_ids}")
    for bout_idx in bout_ids:
        for fly in range(n_anim):
            process_bout_fly(cfg, bout_idx, fly)
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
