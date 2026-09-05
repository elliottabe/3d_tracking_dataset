#!/usr/bin/env python3
"""Phase 1 coarse pass: SAM3 @ reduced frame rate over a WHOLE recording.

Design: docs/specs/2026-08-31-pipeline-schematic-and-inventory-design.md
sections 1-2 (binding -- thresholds/rates there are measured, not re-derived
here). This script produces artifact 1 of that design ("Approach C"): the
recording-level per-(coarse_frame, camera, fly) TRACKS file. Artifact 2 (the
bout-summary CSV, today's exact schema) is produced by the companion
`scripts/coarse_pass_gates.py`, which reads this file WITHOUT touching the GPU
-- gate thresholds can be iterated locally without re-running SAM3.

Reuses the existing SAM3 front-end (JARVIS `SAM3VideoTracker`,
`assign_identities`, `canonicalize_male_fly`, `in_frame_codes` from
`jarvis_jax.predict.sam3_driver`) completely unchanged -- this is NOT a new
detector. The only new mechanism is striding: `positions_per_cam` (already
supported by `SAM3VideoTracker.process_bout` to realign frames against a sync
plan) is repurposed here to select every `stride`-th real frame per camera.

Memory note (measured this session, not in the design doc -- the design's
"chunking is already solved" section describes the tracker's OWN internal
`_track_camera_chunked` box-prompt+IoU boundary stitching, which fires when a
single `process_bout` call exceeds `chunk_len` frames). A literal single call
spanning the WHOLE 1/16-rate recording (~31k coarse frames for Session0) would
keep `BoutMasks`' full-resolution boolean masks (7 cams x 2 flies x 448x1936 px
~= 867 KB/mask) resident for EVERY coarse frame at once: ~378 GB host RAM for
Session0 -- infeasible on a shared node. So this script instead processes the
recording in `chunk_len`-sized (1000 coarse-frame) WINDOWS, each an
independent `process_bout()` call -- exactly the per-bout call
`run_sam3_masks` already makes for a real bout, just aimed at a synthetic
window instead of a human-reviewed one. Because each window's length equals
`chunk_len` (never exceeds it), the tracker's internal chunked path never
fires within a window (fresh text-prompt detection + `assign_identities` +
`canonicalize_male_fly` per window instead). `chunk_overlap` (120 coarse
frames) still staggers window starts as configured (same `step = chunk_len -
chunk_overlap` grid the internal mechanism uses) so windows overlap the same
amount; the overlap is deduplicated on merge (earliest window wins) rather
than reconciled by box-prompt IoU matching. Per the task brief, this is
deliberately NOT new cross-chunk identity stitching: `canonicalize_male_fly`
already gives an ABSOLUTE per-window identity signal (male is always fly1),
so windows agree on identity without any boundary matching at all. Peak host
RAM is bounded to one window (~12 GB), freed before the next.

Usage (SAM3 needs the queue -- see scripts/slurm/submit_task.sh; this must NOT
be run directly on a shared interactive GPU node):

    export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
    python scripts/coarse_pass.py \\
        --session-dir /gscratch/.../Session0/2025_10_20_13_20_04 \\
        --out /gscratch/.../processed/courtship/Session0/2025_10_20_13_20_04/coarse_pass/coarse_tracks.npz \\
        --stride 16 --sam3-compile false
"""
import argparse
import glob
import json
import os
import sys
import time

# Pre-load huggingface_hub.file_download BEFORE the SAM3 import chain (jarvis ->
# timm -> torch) runs -- see scripts/sam3_masks.py (third_party/jarvis_jax) for
# the full rationale. Verified fix; without it SAM3 array jobs die at import.
import huggingface_hub.file_download  # noqa: E402,F401

import numpy as np

from jarvis_jax.predict.sam3_driver import (  # noqa: E402
    IN_FRAME_UNKNOWN, _camera_matrices, _triangulate_batch, as_numpy_repro,
    canonicalize_male_fly, ensure_sync_plan, in_frame_codes, session_tag_for,
    video_paths_for)

# in_frame_codes' MIN_CAMS hard floor -- same value as jarvis_jax/data/build_v5.py
# MIN_CAMS (cited there as "unchanged"); hardcoded here rather than importing a
# data-prep module for an unrelated concern.
MIN_CAMS = 3


def build_jarvis_paths(jarvis_root=None):
    """(code_root, jarvis_root) exactly as jarvis_jax.predict.sam3_driver.run_sam3_masks
    resolves them, so `import jarvis.*` / project loading behave identically."""
    here = os.path.dirname(os.path.abspath(
        __import__("jarvis_jax.predict.sam3_driver", fromlist=["x"]).__file__))
    code_root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(here))), "JARVIS-HybridNet")
    assert os.path.isdir(code_root), f"JARVIS-HybridNet not found at {code_root}"
    if jarvis_root is None:
        jarvis_root = os.environ.get("JARVIS_ROOT", code_root)
    assert os.path.isdir(jarvis_root), f"jarvis_root not found: {jarvis_root}"
    return code_root, jarvis_root


def load_project(project, session_dir, jarvis_root=None):
    code_root, jarvis_root = build_jarvis_paths(jarvis_root)
    for p in (code_root, os.path.join(code_root, "tools")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from jarvis.config.project_manager import ProjectManager
    from jarvis.utils.reprojection import get_repro_tool

    pm = ProjectManager()
    if os.path.realpath(pm.parent_dir) != os.path.realpath(jarvis_root):
        pm.parent_dir = jarvis_root
    assert pm.load(project), f"Could not load JARVIS project '{project}'"
    cfg = pm.get_cfg()
    session_calib = os.path.join(session_dir, "calibration")
    repro_tool = get_repro_tool(cfg, session_calib if os.path.isdir(session_calib) else None)
    assert repro_tool is not None, "ReprojectionTool not available"
    return repro_tool


def video_frame_count_and_size(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return n, w, h


def window_starts(n_coarse, chunk_len, chunk_overlap):
    """Same grid SAM3VideoTracker._track_camera_chunked uses internally
    (step = chunk_len - chunk_overlap), just driven from the outside."""
    step = max(1, chunk_len - chunk_overlap)
    return list(range(0, max(n_coarse, 1), step))


def build_positions_per_cam(cameras, coarse_indices, stride, plan):
    """positions_per_cam for SAM3VideoTracker.process_bout: real mp4 frame
    index for each coarse index, per camera. `plan` (a sync_plan.SyncPlan or
    None) is Session0-inapplicable (no Cam*_meta.csv there -> positional
    fallback) but honored generically."""
    if plan is None:
        row = [int(c) * stride for c in coarse_indices]
        return [list(row) for _ in cameras]
    out = []
    for cam in cameras:
        cp = plan.cams[cam]
        row = []
        for c in coarse_indices:
            s = int(c) * stride
            row.append(int(cp.pos(s)) if cp.has(s) else None)
        out.append(row)
    return out


def extract_window_stats(bm, num_animals, W, H):
    """Per-(fly, camera, local-frame) area/centroid/valid/border_dist for one
    already-tracked+identified window. Reads `bm.get_frame` one frame at a
    time (transient per-frame mask arrays only -- NOT the persistent
    BoutMasks.masks accumulator, which is what the module docstring's memory
    note is about)."""
    T = bm.num_frames
    C = bm.num_cameras
    area = np.full((num_animals, C, T), np.nan, dtype=np.float32)
    cent = np.zeros((num_animals, C, T, 2), dtype=np.float32)
    valid = np.zeros((num_animals, C, T), dtype=bool)
    for fi in range(T):
        slots = bm.get_frame(fi, num_animals=num_animals)
        if slots is None:
            continue
        for fly_idx, slot in enumerate(slots):
            if slot is None:
                continue
            v = slot["valid"].numpy()
            m = slot["masks"].numpy()
            c = slot["centroids"].numpy()
            valid[fly_idx, :, fi] = v
            cent[fly_idx, :, fi] = c
            a = m.reshape(m.shape[0], -1).sum(axis=1).astype(np.float32)
            area[fly_idx, :, fi] = np.where(v, a, np.nan)
    bx = np.minimum(cent[..., 0], (W - 1) - cent[..., 0])
    by = np.minimum(cent[..., 1], (H - 1) - cent[..., 1])
    border = np.where(valid, np.minimum(bx, by), np.nan)
    return area, cent, valid, border


def triangulate_window(cent, valid, repro_tool):
    """(2,T,3) triangulated centroid per fly; NaN where <2 cams valid."""
    A, C, T, _ = cent.shape
    cam_mats = _camera_matrices(repro_tool)
    pts = np.transpose(cent, (0, 2, 1, 3)).reshape(A * T, C, 2)
    vld = np.transpose(valid, (0, 2, 1)).reshape(A * T, C)
    X = _triangulate_batch(pts, cam_mats, vld)
    return X.reshape(A, T, 3)


def setup_recording(session_dir, *, stride, chunk_len, chunk_overlap, num_animals,
                     start_frame, end_frame, project, jarvis_root):
    """Everything every mode ('full'/'window'/'collect') needs before touching
    SAM3: project/camera geometry + the (SAM3-independent, purely arithmetic)
    window grid. None of this needs the GPU -- 'collect' never builds a
    tracker at all, so it can run CPU-only."""
    tag = session_tag_for(session_dir)
    repro_tool = load_project(project, session_dir, jarvis_root)
    cameras = list(repro_tool.cameras)
    video_paths = video_paths_for(session_dir, cameras)
    n_full, W, H = video_frame_count_and_size(video_paths[0])
    if end_frame is None:
        end_frame = n_full
    plan = ensure_sync_plan(session_dir)

    n_coarse = (end_frame - start_frame) // stride
    coarse0 = start_frame // stride
    starts = window_starts(n_coarse, chunk_len, chunk_overlap)
    return dict(tag=tag, repro_tool=repro_tool, cameras=cameras, video_paths=video_paths,
                n_full=n_full, W=W, H=H, plan=plan, n_coarse=n_coarse, coarse0=coarse0,
                starts=starts)


def build_tracker(*, gpu_id, sam3_text, sam3_version, sam3_compile, sam3_checkpoint,
                   chunk_len, chunk_overlap, lowmem):
    from jarvis.prediction.sam3_video_tracker import SAM3VideoTracker  # noqa
    tracker = SAM3VideoTracker(
        gpu_id=gpu_id, text_prompt=sam3_text, sam3_version=sam3_version,
        compile=sam3_compile, checkpoint_path=sam3_checkpoint,
        chunk_len=chunk_len, chunk_overlap=chunk_overlap)
    if lowmem:
        from jarvis_jax.predict.sam3_driver import _enable_sam3_lowmem
        _enable_sam3_lowmem(tracker.predictor)
    return tracker


def check_parts_complete(parts_dir, n_windows):
    """Refuse to collect unless EVERY window_NNNN.npz in [0, n_windows) exists.

    A window that found no flies still writes a part file (empty=True), so
    this only catches a window that never ran/finished -- exactly the case
    that would otherwise silently produce a plausible-looking but WRONG bout
    summary (task brief: "the worst outcome here")."""
    missing = [i for i in range(n_windows)
               if not os.path.isfile(os.path.join(parts_dir, f"window_{i:04d}.npz"))]
    if missing:
        raise SystemExit(
            f"[coarse_pass] COLLECT REFUSED: {len(missing)}/{n_windows} window part "
            f"file(s) missing under {parts_dir}: {missing}. Submit/resume the array "
            f"task(s) for these window index/indices, then re-run --mode collect.")


def process_one_window(tracker, video_paths, repro_tool, cameras, coarse0, stride, plan,
                        num_animals, W, H, parts_dir, wi, gs, ge, n_coarse):
    """Compute window `wi` (grid offsets [gs, ge) in coarse-index space,
    clamped to n_coarse) and write ONLY its part file. Shared by 'full'
    (looped over every window) and 'window' (one SLURM array task, one
    index) modes -- this is the one piece of real work either mode does."""
    ge = min(ge, n_coarse)
    part_path = os.path.join(parts_dir, f"window_{wi:04d}.npz")
    t0 = time.time()
    coarse_indices = [coarse0 + c for c in range(gs, ge)]
    res = process_window(tracker, video_paths, repro_tool, cameras,
                          coarse_indices, stride, plan, num_animals, W, H)
    if res is None:
        print(f"[coarse_pass] window {wi} [{gs}:{ge}): no flies found -- skipping")
        np.savez_compressed(part_path, empty=np.array(True), gs=gs, ge=ge)
        return
    np.savez_compressed(
        part_path, gs=gs, ge=ge,
        area=res["area"], cent=res["cent"], valid=res["valid"],
        border=res["border"], in_frame=res["in_frame"], X3d=res["X3d"],
        sex_status=np.array(res["sex_status"]),
        sex_info=np.array(json.dumps(res["sex_info"])))
    print(f"[coarse_pass] window {wi} [{gs}:{ge}) sex={res['sex_status']} "
          f"({res['sex_info']}) {time.time()-t0:.1f}s -> {part_path}")


def process_window(tracker, video_paths, repro_tool, cameras, coarse_indices,
                    stride, plan, num_animals, W, H):
    """Run one chunk_len-sized window through the existing per-bout pipeline
    (process_bout -> assign_identities -> canonicalize_male_fly), returning
    per-window stats. Returns None if SAM3 found nothing at all in the window
    (identity_map never populated -- e.g. a window with zero flies visible)."""
    import torch

    positions_per_cam = build_positions_per_cam(cameras, coarse_indices, stride, plan)
    bm = tracker.process_bout(video_paths, 0, len(coarse_indices),
                              num_animals=num_animals,
                              positions_per_cam=positions_per_cam)
    # See run_sam3_masks: SAM3's predictors leave a process-wide bf16 autocast
    # active, which silently corrupts the sub-pixel triangulation geometry in
    # assign_identities unless forced back to fp32.
    with torch.autocast(device_type="cuda", enabled=False):
        bm.assign_identities(repro_tool, num_animals=num_animals)
    sex_status, sex_info = "n/a", {}
    if num_animals == 2:
        sex_status, sex_info = canonicalize_male_fly(bm, num_animals, male_slot=1)
    if bm.identity_map[0] is None:
        return None
    area, cent, valid, border = extract_window_stats(bm, num_animals, W, H)
    rt_np = as_numpy_repro(repro_tool)
    codes = in_frame_codes(cent, valid, rt_np, W, H, min_support=MIN_CAMS)
    X3d = triangulate_window(cent, valid, rt_np)
    del bm  # release this window's full-resolution masks before the next one
    return dict(area=area, cent=cent, valid=valid, border=border, in_frame=codes,
                X3d=X3d, sex_status=sex_status, sex_info=sex_info)


def run(session_dir, out_path, *, stride=16, chunk_len=1000, chunk_overlap=120,
        num_animals=2, start_frame=0, end_frame=None, project="unified_V3_masked",
        jarvis_root=None, gpu_id=0, sam3_version="sam3.1", sam3_text="insect",
        sam3_compile=False, sam3_checkpoint=None, lowmem=True, resume=True,
        mode="full", window_index=None):
    """mode='full' (default): today's exact behaviour, unchanged -- every
    window computed in-process, sequentially, then merged. mode='window':
    compute ONE window (`window_index`) and write only its part file --
    what one SLURM array task runs (see scripts/slurm_coarse_pass_array.py);
    never touches the GPU/SAM3 if that window's part already exists and
    `resume` is set. mode='collect': CPU-only -- no tracker is built at all
    -- verify every window's part file exists (loudly refusing otherwise,
    see `check_parts_complete`) and merge them into the single tracks
    artifact + meta.json, exactly as 'full' already did at the end of its
    loop."""
    ctx = setup_recording(session_dir, stride=stride, chunk_len=chunk_len,
                           chunk_overlap=chunk_overlap, num_animals=num_animals,
                           start_frame=start_frame, end_frame=end_frame,
                           project=project, jarvis_root=jarvis_root)
    tag, repro_tool = ctx["tag"], ctx["repro_tool"]
    cameras, video_paths = ctx["cameras"], ctx["video_paths"]
    n_full, W, H, plan = ctx["n_full"], ctx["W"], ctx["H"], ctx["plan"]
    n_coarse, coarse0, starts = ctx["n_coarse"], ctx["coarse0"], ctx["starts"]
    print(f"[coarse_pass] {tag}: {n_full} full-rate frames, stride={stride} "
          f"-> {n_coarse} coarse frames, {len(starts)} windows "
          f"(chunk_len={chunk_len}, chunk_overlap={chunk_overlap}) mode={mode}")

    parts_dir = out_path + ".parts"
    os.makedirs(parts_dir, exist_ok=True)

    def _merge():
        merge_parts(parts_dir, out_path, cameras=cameras, coarse0=coarse0, stride=stride,
                    n_coarse=n_coarse, W=W, H=H, num_animals=num_animals,
                    session_dir=session_dir, chunk_len=chunk_len,
                    chunk_overlap=chunk_overlap, project=project)

    if mode == "collect":
        check_parts_complete(parts_dir, len(starts))
        _merge()
        print(f"[coarse_pass] collected {len(starts)} window part(s) -> {out_path}")
        return

    if mode == "window":
        if window_index is None:
            raise SystemExit("[coarse_pass] --mode window requires --window-index")
        if not (0 <= window_index < len(starts)):
            raise SystemExit(f"[coarse_pass] --window-index {window_index} out of range "
                             f"[0, {len(starts)})")
        gs = starts[window_index]
        part_path = os.path.join(parts_dir, f"window_{window_index:04d}.npz")
        if resume and os.path.isfile(part_path):
            print(f"[coarse_pass] window {window_index} [{gs}:...) -- reusing {part_path}")
            return
        tracker = build_tracker(gpu_id=gpu_id, sam3_text=sam3_text, sam3_version=sam3_version,
                                 sam3_compile=sam3_compile, sam3_checkpoint=sam3_checkpoint,
                                 chunk_len=chunk_len, chunk_overlap=chunk_overlap, lowmem=lowmem)
        process_one_window(tracker, video_paths, repro_tool, cameras, coarse0, stride, plan,
                            num_animals, W, H, parts_dir, window_index, gs, gs + chunk_len,
                            n_coarse)
        return

    assert mode == "full", f"unknown mode {mode!r}"
    tracker = build_tracker(gpu_id=gpu_id, sam3_text=sam3_text, sam3_version=sam3_version,
                             sam3_compile=sam3_compile, sam3_checkpoint=sam3_checkpoint,
                             chunk_len=chunk_len, chunk_overlap=chunk_overlap, lowmem=lowmem)
    for wi, gs in enumerate(starts):
        ge = min(gs + chunk_len, n_coarse)
        part_path = os.path.join(parts_dir, f"window_{wi:04d}.npz")
        if resume and os.path.isfile(part_path):
            print(f"[coarse_pass] window {wi} [{gs}:{ge}) -- reusing {part_path}")
            continue
        process_one_window(tracker, video_paths, repro_tool, cameras, coarse0, stride, plan,
                            num_animals, W, H, parts_dir, wi, gs, gs + chunk_len, n_coarse)

    check_parts_complete(parts_dir, len(starts))
    _merge()
    print(f"[coarse_pass] wrote {out_path}")


def merge_parts(parts_dir, out_path, *, cameras, coarse0, stride, n_coarse, W, H,
                num_animals, session_dir, chunk_len, chunk_overlap, project):
    C = len(cameras)
    area = np.full((num_animals, C, n_coarse), np.nan, dtype=np.float32)
    cent = np.zeros((num_animals, C, n_coarse, 2), dtype=np.float32)
    valid = np.zeros((num_animals, C, n_coarse), dtype=bool)
    border = np.full((num_animals, C, n_coarse), np.nan, dtype=np.float32)
    in_frame = np.full((num_animals, C, n_coarse), IN_FRAME_UNKNOWN, dtype=np.int8)
    X3d = np.full((num_animals, n_coarse, 3), np.nan, dtype=np.float32)
    filled = np.zeros(n_coarse, dtype=bool)
    windows_meta = []

    for part_path in sorted(glob.glob(os.path.join(parts_dir, "window_*.npz"))):
        with np.load(part_path, allow_pickle=True) as z:
            gs, ge = int(z["gs"]), int(z["ge"])
            if "empty" in z.files:
                windows_meta.append(dict(gs=gs, ge=ge, sex_status="empty"))
                continue
            # First-window-wins dedup over the chunk_overlap region shared
            # with the previous window (see module docstring). Fancy-index
            # by explicit global/local index arrays (not slice+bool) so this
            # works uniformly for the (fly,cam,T) arrays AND the (fly,cam,T,2)
            # `cent` array -- a chained `arr[:, :, sl][..., new]` silently
            # applies the boolean mask to the WRONG axis (the trailing xy
            # axis) for the latter.
            new_local = np.nonzero(~filled[gs:ge])[0]
            new_global = new_local + gs
            for arr, part in ((area, z["area"]), (cent, z["cent"]), (valid, z["valid"]),
                             (border, z["border"]), (in_frame, z["in_frame"]),
                             ):
                arr[:, :, new_global] = part[:, :, new_local]
            X3d[:, new_global] = z["X3d"][:, new_local]
            filled[new_global] = True
            sex_info = json.loads(str(z["sex_info"]))
            windows_meta.append(dict(gs=gs, ge=ge, sex_status=str(z["sex_status"]),
                                     **sex_info))

    # Derived per-fly, per-frame gate inputs (auditable alongside the raw
    # per-camera arrays -- see design doc s2 "recording-level TRACKS").
    with np.errstate(invalid="ignore"):
        area_med = np.nanmedian(np.where(valid, area, np.nan), axis=1)
        border_med = np.nanmedian(np.where(valid, border, np.nan), axis=1)
    n_valid_cams = valid.sum(axis=1).astype(np.int16)
    sep3d = np.linalg.norm(X3d[0] - X3d[1], axis=-1).astype(np.float32) if num_animals >= 2 else None
    # Per-camera 2D centroid separation, median over cams where BOTH flies
    # are valid there (a "flies merged/overlapping" signal independent of
    # the 3D triangulation, which needs >=2 valid cams to exist at all).
    sep2d_med = np.full(n_coarse, np.nan, dtype=np.float32)
    if num_animals >= 2:
        both = valid[0] & valid[1]                      # (C,T)
        d2 = np.linalg.norm(cent[0] - cent[1], axis=-1)  # (C,T)
        with np.errstate(invalid="ignore"):
            d2m = np.where(both, d2, np.nan)
            sep2d_med = np.nanmedian(d2m, axis=0).astype(np.float32)

    coarse_frame = coarse0 * stride + np.arange(n_coarse) * stride

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(
        tmp, coarse_frame=coarse_frame, cameras=np.array(cameras),
        area=area, centroid=cent, valid=valid, border_dist=border, in_frame=in_frame,
        area_med=area_med, border_med=border_med, n_valid_cams=n_valid_cams,
        X3d=X3d, sep3d=sep3d if sep3d is not None else np.array([]),
        sep2d_med=sep2d_med, filled=filled)
    os.replace(tmp, out_path)
    meta = dict(session_dir=session_dir, stride=stride, chunk_len=chunk_len,
                chunk_overlap=chunk_overlap, project=project, cameras=cameras,
                W=W, H=H, num_animals=num_animals, n_coarse=n_coarse,
                coarse0=coarse0, min_cams=MIN_CAMS, windows=windows_meta,
                frac_filled=float(filled.mean()))
    with open(out_path.rsplit(".npz", 1)[0] + ".meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-dir", required=True)
    ap.add_argument("--out", required=True, help="output coarse_tracks.npz path")
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--chunk-len", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=120)
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=None,
                    help="exclusive; default = whole recording (full frame count)")
    ap.add_argument("--project", default="unified_V3_masked")
    ap.add_argument("--jarvis-root", default=None)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--sam3-version", default="sam3.1")
    ap.add_argument("--sam3-text", default="insect")
    ap.add_argument("--sam3-compile", type=lambda s: s.lower() not in ("0", "false", "no"),
                    default=False)
    ap.add_argument("--sam3-checkpoint", default=None)
    ap.add_argument("--no-lowmem", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--mode", choices=("full", "window", "collect"), default="full",
                    help="full (default) = today's sequential sweep, unchanged: every "
                         "window computed in-process then merged. window = compute ONE "
                         "window (--window-index) and write only its parts/window_NNNN.npz "
                         "-- what a SLURM array task runs, see "
                         "scripts/slurm_coarse_pass_array.py. collect = CPU-only, no "
                         "SAM3/GPU: merge existing parts/window_*.npz into the tracks "
                         "artifact, refusing loudly if any window's part is missing.")
    ap.add_argument("--window-index", type=int, default=None,
                    help="0-based window index to compute; required for --mode window")
    args = ap.parse_args()

    run(args.session_dir, args.out, stride=args.stride, chunk_len=args.chunk_len,
        chunk_overlap=args.chunk_overlap, num_animals=args.num_animals,
        start_frame=args.start_frame, end_frame=args.end_frame, project=args.project,
        jarvis_root=args.jarvis_root, gpu_id=args.gpu, sam3_version=args.sam3_version,
        sam3_text=args.sam3_text, sam3_compile=args.sam3_compile,
        sam3_checkpoint=args.sam3_checkpoint, lowmem=not args.no_lowmem,
        resume=not args.no_resume, mode=args.mode, window_index=args.window_index)


if __name__ == "__main__":
    main()
