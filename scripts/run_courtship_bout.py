#!/usr/bin/env python3
"""Resumable per-bout courtship inference driver (stages A-E).

For a bout index and each fly (``range(cfg.recording.num_animals)``), runs:

  A  ViTPose 2-D on SAM3-masked crops                       -> kp2d.npz
  B  DLT triangulation                                      -> kp3d.npz
  C  STAC ik_only (offsets fit once, shared across bouts)   -> stac_ik.h5
  D  silhouette-containment polish                          -> qpos_refined.npz
  E  FK outputs.h5 + qc.json + per-camera reprojection overlay videos

Every artifact is written atomically (tmp -> os.replace) and every stage is
skipped when its artifact already exists (see jarvis_jax.cse.courtship_resume),
so a preempted/resumed run picks up where it left off. A ``DONE`` marker per
``<run_root>/bouts/bout_<idx:05d>/fly<f>/`` gates re-processing an already
completed bout/fly.

Usage:
    python scripts/run_courtship_bout.py paths=hyak +bout_ids=3
    python scripts/run_courtship_bout.py paths=hyak            # bout_ids='' -> all bouts
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import glob
import re

import numpy as np
import hydra
from omegaconf import DictConfig

from jarvis_jax.cse.courtship_resume import (
    atomic_save_npz, stage_done, mark_done, bout_complete)
from jarvis_jax.cse.courtship_bout_masks import load_bout_masks
from jarvis_jax.cse.courtship_predict_2d import load_detector, predict_bout_2d
from jarvis_jax.cse.courtship_triangulate import triangulate_keypoints
from jarvis_jax.cse.courtship_stac import fit_offsets_once, ik_only_bout
from jarvis_jax.cse.courtship_polish import polish_bout
from jarvis_jax.cse.outputs import build_fly_outputs
from jarvis_jax.cse.qc import qc_report
from jarvis_jax.cse.reproj_video import write_camera_video
from jarvis_jax.cse.mesh_decimate import decimate_mesh_npz
from jarvis_jax.predict.sam3_driver import parse_bouts, session_tag_for


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


def load_centroids(bout_npz, fly):
    """npz 'centroids' (A,C,T,2) -> this fly's (T,C,2) (load_bout_masks itself
    does not surface centroids, so this is read directly from the npz)."""
    with np.load(bout_npz) as z:
        return np.asarray(z["centroids"], np.float32)[fly].transpose(1, 0, 2)


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
    the decimated-mesh overlay can have thousands of vertices per frame)."""
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
        idx = np.sort(np.argsort(-finite_counts))
    if max_frames:
        idx = idx[:max_frames]
    return idx


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
    """Like courtship_resume.atomic_write, but keeps the '.mp4' suffix on the
    temp file: imageio's writer picks its backend by sniffing the URI's
    extension, so a bare '<path>.tmp' (no '.mp4') makes it fall back to the
    wrong plugin (e.g. TIFF) and crash on the 'fps' kwarg."""
    tmp = path[:-4] + ".tmp.mp4" if path.endswith(".mp4") else path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_fn(tmp)
    os.replace(tmp, path)
    return path


# ---------------------------------------------------------------------------
# Per (bout, fly) driver
# ---------------------------------------------------------------------------

def process_bout_fly(cfg, bout_idx: int, fly: int):
    run_root = str(cfg.outputs.out)
    bout_dir = os.path.join(run_root, "bouts", f"bout_{bout_idx:05d}", f"fly{fly}")
    if bout_complete(bout_dir):
        print(f"[courtship] bout {bout_idx} fly{fly}: already DONE, skipping")
        return

    predictions_dir = str(cfg.recording.predictions_dir)
    bout_npz = os.path.join(predictions_dir, f"bout_{bout_idx:05d}", "sam3_masks.npz")
    masks_dict = load_bout_masks(bout_npz, fly)
    T, C = masks_dict["T"], masks_dict["C"]

    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    rt = ReprojectionTool(cfg.recording.calib_dir)
    cam_mats = np.asarray(rt.camera_matrices, np.float32)  # (C,4,3); order matches recording.cameras
    cameras = list(cfg.recording.cameras)

    kp2d_path = os.path.join(bout_dir, "kp2d.npz")
    kp3d_path = os.path.join(bout_dir, "kp3d.npz")
    stac_h5_path = os.path.join(bout_dir, "stac_ik.h5")
    qpos_path = os.path.join(bout_dir, "qpos_refined.npz")
    outputs_h5_path = os.path.join(bout_dir, "outputs.h5")
    qc_json_path = os.path.join(bout_dir, "qc.json")
    offsets_path = os.path.join(run_root, "offsets.h5")
    decimated_mesh_path = os.path.join(run_root, "decimated_mesh.npz")

    # -- one-time, run-root-level precompute: whichever bout/fly gets here
    #    first does the work; every later bout/fly reuses the artifact.
    #    decimate_mesh_npz writes with a plain np.savez (not atomic_save_npz),
    #    so wrap it in a tmp-then-replace ourselves to keep the same
    #    crash-safety guarantee as every other artifact in this driver. --
    if not stage_done(decimated_mesh_path):
        _tmp_mesh = decimated_mesh_path + ".tmp.npz"
        decimate_mesh_npz(cfg.silhouette.mesh_npz, _tmp_mesh,
                          target_faces=int(cfg.outputs.decimated_faces))
        os.replace(_tmp_mesh, decimated_mesh_path)

    # -- Stage A: ViTPose 2-D ---------------------------------------------------
    if not stage_done(kp2d_path):
        start = bout_start_frame(cfg, bout_idx)
        centroids = load_centroids(bout_npz, fly)
        caps = open_video_captures(cfg.recording.session_dir, cameras)
        try:
            vit = load_detector(cfg.detector.ckpt, num_keypoints=int(cfg.detector.num_keypoints))
            frames_iter = all_cams_frames(caps, start, T)
            kp2d, conf = predict_bout_2d(
                vit, frames_iter, masks_dict["masks"], centroids, masks_dict["valid"], cam_mats,
                crop=int(cfg.detector.crop), batch=int(cfg.detector.get("batch", 64)))
        finally:
            for cap in caps:
                cap.release()
        atomic_save_npz(kp2d_path, kp2d=kp2d, conf=conf)
    with np.load(kp2d_path) as z:
        kp2d, conf = z["kp2d"], z["conf"]

    # -- Stage B: DLT triangulation ----------------------------------------------
    if not stage_done(kp3d_path):
        kp3d, conf3d = triangulate_keypoints(
            kp2d, conf, cam_mats, conf_thresh=float(cfg.detector.conf_thresh))
        atomic_save_npz(kp3d_path, kp3d=kp3d, conf3d=conf3d)
    with np.load(kp3d_path) as z:
        kp3d, conf3d = z["kp3d"], z["conf3d"]

    kp_names = list(cfg.model.KP_NAMES)

    # -- offsets.h5: fit ONCE (shared across all bouts/flies) on a
    #    high-confidence kp3d sample from whichever bout gets there first --
    if not stage_done(offsets_path):
        sample_idx = high_confidence_sample(kp3d)
        fit_offsets_once(cfg, kp3d[sample_idx], kp_names,
                         offsets_path=offsets_path, save_path=run_root)

    # -- Stage C: STAC ik_only ----------------------------------------------------
    if not stage_done(stac_h5_path):
        ik_only_bout(cfg, kp3d, kp_names, offsets_path=offsets_path,
                    out_h5="stac_ik.h5", save_path=bout_dir)

    # RESOLUTIONS #3: the STAC ik and the SAM masks must cover the exact same
    # bout frame range. On resume, a stale stac_ik.h5 left from a different
    # (e.g. differently-trimmed) run would silently desync outputs/QC from
    # masks_dict -- fail loudly instead of polishing garbage.
    import stac_mjx.io_dict_to_hdf5 as ioh5
    q = np.asarray(ioh5.load(stac_h5_path)["qpos"])
    if q.shape[0] != T:
        raise RuntimeError(
            f"bout {bout_idx} fly{fly}: stac_ik.h5 qpos T={q.shape[0]} != "
            f"masks T={T} ({stac_h5_path} vs {bout_npz}); stale/mismatched "
            f"resumed artifact -- delete it and rerun this bout/fly.")

    # -- Stage D: silhouette-containment polish -----------------------------------
    if not stage_done(qpos_path):
        qpos_refined, bridges = polish_bout(
            stac_h5_path, cfg, kp3d, conf3d, masks_dict, cfg.recording.calib_dir)
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

    if not stage_done(qc_json_path):
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
        qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                 kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                 masks_by_frame=masks_by_frame, out_json=qc_json_path)

    # -- overlays: rigidly transform the DECIMATED mesh via each frame's
    #    STAC->mm bridge (no re-articulation -- mesh_decimate.py is explicit
    #    that the decimated copy is raster-only, never kinematics/outputs) +
    #    the triangulated kp3d, project into each camera; skip cameras whose
    #    video already exists --
    if cfg.outputs.overlay:
        with np.load(decimated_mesh_path) as dmesh:
            dverts = np.asarray(dmesh["vertices"], np.float32)   # (Nd,3) rest-pose, model units
        overlay_dir = os.path.join(bout_dir, "overlays")
        start = bout_start_frame(cfg, bout_idx)
        for ci, cam in enumerate(cameras):
            overlay_path = os.path.join(overlay_dir, f"{cam}_reproj.mp4")
            if stage_done(overlay_path):
                continue
            mesh2d_by_frame, kp2d_by_frame_overlay = [], []
            for t in range(T):
                br = bridges[t]
                if br is None:
                    mesh2d_by_frame.append(np.zeros((0, 2)))
                    kp2d_by_frame_overlay.append(np.zeros((0, 2)))
                    continue
                s, R, tr = br
                mesh_mm_t = s * (np.asarray(R) @ dverts.T).T + np.asarray(tr)
                mesh2d_by_frame.append(project_points(cam_mats[ci], mesh_mm_t))
                kp_t = kp3d[t]
                kp_ok = np.isfinite(kp_t).all(-1)
                kp2d_by_frame_overlay.append(project_points(cam_mats[ci], kp_t[kp_ok]))
            video_path = os.path.join(cfg.recording.session_dir, f"{cam}.mp4")

            def _write(tmp, video_path=video_path, mesh2d_by_frame=mesh2d_by_frame,
                      kp2d_by_frame_overlay=kp2d_by_frame_overlay):
                write_camera_video(
                    tmp, frames_rgb_iter=one_cam_frames(video_path, start, T),
                    mesh2d_by_frame=mesh2d_by_frame, kp2d_by_frame=kp2d_by_frame_overlay,
                    fps=int(cfg.outputs.overlay_fps))

            atomic_write_mp4(overlay_path, _write)

    mark_done(bout_dir)
    print(f"[courtship] bout {bout_idx} fly{fly}: DONE -> {bout_dir}")


def main_from_cfg(cfg: DictConfig):
    bout_ids = resolve_bout_ids(cfg)
    print(f"[courtship] processing {len(bout_ids)} bout(s): {bout_ids}")
    for bout_idx in bout_ids:
        for fly in range(int(cfg.recording.num_animals)):
            process_bout_fly(cfg, bout_idx, fly)


@hydra.main(version_base=None, config_path="../configs", config_name="courtship_pipeline")
def main(cfg: DictConfig):
    main_from_cfg(cfg)


if __name__ == "__main__":
    main()
