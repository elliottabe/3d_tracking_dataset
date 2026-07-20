"""Render driver: overlay predicted 3D keypoints (courtship-bout 3D pipeline)
onto the source camera videos -> a VS-Code-playable (H.264/yuv420p) tracking
video, one mp4 per camera.

No such driver existed before this file: ``jarvis_jax/tracking/reproj_video.py``
is a streaming-mp4-writer *library*; its only prior caller
(``jarvis_jax/tracking/run_outputs_qc.py``) draws a single-fly IK/COCO crop
overlay, not full-frame two-fly bout predictions.

Camera order / geometry: uses
``jarvis_jax.predict.session_predict.session_geometry`` -- the AUTHORITATIVE
camera order that matches the sam3 mask npz axis and ``Cam<name>.mp4`` reads.
Do NOT build a bare ``ReprojectionTool`` directly; its lexicographic camera
order can disagree with this (see module docstring there / D3 camera-order
gotcha).

Frame timeline: the bout's absolute [start, end] frame range is read from the
per-fly ``fly{0,1}.csv`` ``frame`` column (min/max), NOT trusted blindly from
``manifest.json``. Empirically (Session0, 2026-07-18) ``Predictions_3D_sam3_all30/
manifest.json`` on this cluster holds only the SINGLE most-recently-written
bout entry (looks like a sharded job overwriting instead of appending) --
using it as the sole source of ``start``/``num_frames`` would silently render
the wrong bout or crash. The CSV frame column is always self-consistent with
the ``pred_dir`` being rendered, so it is authoritative here; the manifest
entry (if present and if it matches ``bout_id``) is only used for an
informational cross-check.

Reprojection: full-image px = homogeneous ``(x,y,z,1) @ cameraMatrices[c]``
(shape (4,3)) then perspective-divide by the 3rd component -- identical
convention to ``scripts/viz_compare_3d_runs.py:project_full`` and
``jarvis_jax.geometry.reprojection_tool.ReprojectionTool.reproject_point``.
Overlay is drawn on the FULL camera frame (not a crop).

Example (Session0, red_data_unified project, bout 4)::

    cd third_party/jarvis_jax
    python -m jarvis_jax.scripts.render_bout_reproj \\
        --session-dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04 \\
        --masks-dir   /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/Predictions_3D_sam3_all30 \\
        --pred-dir    /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/Predictions_3D_36233268 \\
        --bout-id 4 \\
        --project red_data_unified \\
        --out /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/reproj_videos
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess

import numpy as np


# --------------------------------------------------------------------------
# Reprojection (pure numpy, unit-testable)
# --------------------------------------------------------------------------
def reproject_points(p3d, M):
    """Project 3D point(s) to full-image pixel coords.

    Parameters
    ----------
    p3d : array_like, shape (..., 3)
        World-space 3D point(s); NaN rows propagate to NaN output rows.
    M : array_like, shape (4, 3)
        Camera projection matrix in the ``ph @ M`` convention (matches
        ``ReprojectionTool.camera_matrices[c]`` /
        ``viz_compare_3d_runs.py:project_full``).

    Returns
    -------
    ndarray, shape (..., 2)
        Perspective-divided full-image pixel coordinates.
    """
    p3d = np.asarray(p3d, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)
    ones = np.ones(p3d.shape[:-1] + (1,), dtype=np.float64)
    ph = np.concatenate([p3d, ones], axis=-1)          # (...,4)
    proj = ph @ M                                       # (...,3)
    return proj[..., :2] / proj[..., 2:3]


def project_and_filter(kp3d_frame, conf_frame, M, min_conf=0.0):
    """One frame's (J,3) keypoints -> filtered (K,2) full-image px.

    Drops joints with NaN 3D coords or confidence below ``min_conf``.
    """
    kp3d_frame = np.asarray(kp3d_frame, dtype=np.float64)
    conf_frame = np.asarray(conf_frame, dtype=np.float64)
    pts2d = reproject_points(kp3d_frame, M)            # (J,2), NaN-preserving
    valid = (~np.isnan(kp3d_frame).any(axis=-1)) & (conf_frame >= min_conf)
    return pts2d[valid]


# --------------------------------------------------------------------------
# fly{0,1}.csv reader (session_io.py has writers only; no reader exists yet)
# --------------------------------------------------------------------------
def read_fly_csv(path):
    """Read a Predictions_3D fly CSV (schema: session_io.write_fly_csv).

    Robust to the header-2 joint-property label actually written in
    production output ("conf") vs. the label session_io.write_fly_csv itself
    emits ("confidence") -- columns are read POSITIONALLY (frame, then
    (x,y,z,confidence) groups of 4 per joint), never by label text.

    Returns
    -------
    joint_names : list[str], length J
    frames : ndarray (T,) int64 -- absolute (session-video) frame indices
    kp : ndarray (T, J, 3) float64 -- NaN where invalid
    conf : ndarray (T, J) float64 -- NaN where invalid
    """
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if len(rows) < 2 or rows[0][0] != "frame" or rows[1][0] != "frame":
        raise ValueError(f"{path}: does not look like a Predictions_3D fly CSV")
    header1 = rows[0]
    n_joints = (len(header1) - 1) // 4
    joint_names = [header1[1 + 4 * j] for j in range(n_joints)]
    data = rows[2:]
    T = len(data)
    if T == 0:
        return joint_names, np.zeros(0, np.int64), np.zeros((0, n_joints, 3)), \
            np.zeros((0, n_joints))
    frames = np.array([int(r[0]) for r in data], dtype=np.int64)
    vals = np.array([r[1:] for r in data], dtype=np.float64)   # (T, J*4), "nan"-safe
    vals = vals.reshape(T, n_joints, 4)
    kp = vals[..., :3]
    conf = vals[..., 3]
    return joint_names, frames, kp, conf


def dense_by_frame(frames, kp, conf, start, num_frames):
    """Scatter sparse (frame -> kp/conf) rows onto a dense [start, start+num_frames)
    timeline; missing frames (gaps, or frames outside range) -> NaN rows.
    """
    frames = np.asarray(frames, dtype=np.int64)
    J = kp.shape[1] if kp.ndim == 3 else 0
    out_kp = np.full((num_frames, J, 3), np.nan, dtype=np.float64)
    out_conf = np.full((num_frames, J), np.nan, dtype=np.float64)
    idx = frames - int(start)
    keep = (idx >= 0) & (idx < num_frames)
    out_kp[idx[keep]] = kp[keep]
    out_conf[idx[keep]] = conf[keep]
    return out_kp, out_conf


# --------------------------------------------------------------------------
# manifest.json (best-effort informational cross-check only; see module docstring)
# --------------------------------------------------------------------------
def find_bout_in_manifest(manifest_path, bout_id):
    """Return the manifest['bouts'] entry with bout_idx == bout_id, or None
    (missing file / unparsable JSON / bout absent -- never raises)."""
    try:
        with open(manifest_path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    for b in d.get("bouts", []):
        if int(b.get("bout_idx", -1)) == int(bout_id):
            return b
    return None


# --------------------------------------------------------------------------
# Video I/O helpers (cv2; not exercised by the fast unit tests)
# --------------------------------------------------------------------------
def _norm_cam_name(name):
    """Strip a leading 'Cam' so 'Cam2012853' and '2012853' compare equal."""
    name = str(name)
    return name[3:] if name.startswith("Cam") else name


def camera_video_path(session_dir, cam_name):
    """session_geometry's camera_names already include the 'Cam' prefix
    (e.g. 'Cam2012853'); tolerate a bare numeric-id name too."""
    fname = cam_name if str(cam_name).startswith("Cam") else f"Cam{cam_name}"
    return os.path.join(str(session_dir), f"{fname}.mp4")


def probe_video_fps(video_path, default=30.0):
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        return float(fps) if fps and fps > 0 else float(default)
    finally:
        cap.release()


def iter_video_frames(video_path, start, num_frames):
    """Seek once to `start`, then read `num_frames` frames sequentially,
    yielding RGB uint8 arrays (one frame in memory at a time)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start))
        for t in range(num_frames):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(
                    f"failed to read frame {start + t} from {video_path} "
                    f"(requested {num_frames} frames from start={start})")
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def render_bout(*, session_dir, masks_dir, pred_dir, bout_id, project,
                jarvis_root, out, cameras=None, max_frames=None,
                min_conf=0.0, fps=None):
    """Render one bout's fly0+fly1 3D-keypoint reprojection overlay, one mp4
    per camera, into `out`. Returns the list of written mp4 paths.
    """
    from jarvis_jax.predict.session_predict import session_geometry

    camera_names, cam_mats = session_geometry(project, session_dir, jarvis_root)
    if cameras:
        wanted_norm = {_norm_cam_name(c) for c in cameras}
        keep = [_norm_cam_name(n) in wanted_norm for n in camera_names]
        kept_norm = {_norm_cam_name(n) for n, k in zip(camera_names, keep) if k}
        missing = wanted_norm - kept_norm
        if missing:
            raise KeyError(f"requested camera(s) not found: {sorted(missing)} "
                            f"(available: {camera_names})")
        camera_names = [n for n, k in zip(camera_names, keep) if k]
        cam_mats = cam_mats[np.asarray(keep, dtype=bool)]

    bout_dir = os.path.join(str(pred_dir), f"bout_{int(bout_id):05d}")
    names0, frames0, kp0, conf0 = read_fly_csv(os.path.join(bout_dir, "fly0.csv"))
    names1, frames1, kp1, conf1 = read_fly_csv(os.path.join(bout_dir, "fly1.csv"))
    if names0 != names1:
        raise ValueError(f"fly0/fly1 joint-name mismatch in {bout_dir}")

    start = int(min(frames0.min(), frames1.min()))
    end = int(max(frames0.max(), frames1.max()))
    num_frames = end - start + 1

    manifest_bout = find_bout_in_manifest(
        os.path.join(str(masks_dir), "manifest.json"), bout_id)
    if manifest_bout is not None:
        m_start, m_n = manifest_bout.get("start"), manifest_bout.get("num_frames")
        if m_start != start or m_n != num_frames:
            print(f"[render_bout_reproj] WARNING: manifest.json bout {bout_id} "
                  f"says start={m_start} num_frames={m_n}, but fly CSVs say "
                  f"start={start} num_frames={num_frames}. Trusting the CSVs "
                  f"(manifest.json on this session appears stale/partial -- "
                  f"see module docstring).")

    if max_frames:
        num_frames = min(num_frames, int(max_frames))

    kp0d, conf0d = dense_by_frame(frames0, kp0, conf0, start, num_frames)
    kp1d, conf1d = dense_by_frame(frames1, kp1, conf1, start, num_frames)

    os.makedirs(str(out), exist_ok=True)
    out_paths = []
    for c, cam_name in enumerate(camera_names):
        M = cam_mats[c]
        video_path = camera_video_path(session_dir, cam_name)
        cam_fps = fps if fps else probe_video_fps(video_path)

        proj0 = reproject_points(kp0d, M)      # (num_frames, J, 2)
        proj1 = reproject_points(kp1d, M)
        kp2d_by_frame = []
        mesh2d_by_frame = []
        for t in range(num_frames):
            valid0 = (~np.isnan(kp0d[t]).any(axis=-1)) & (conf0d[t] >= min_conf)
            valid1 = (~np.isnan(kp1d[t]).any(axis=-1)) & (conf1d[t] >= min_conf)
            kp2d_by_frame.append(proj0[t][valid0])      # fly0 -> magenta
            mesh2d_by_frame.append(proj1[t][valid1])    # fly1 -> cyan

        from jarvis_jax.tracking.reproj_video import write_camera_video
        out_path = os.path.join(str(out), f"bout{int(bout_id)}_{cam_name}.mp4")
        write_camera_video(
            out_path,
            frames_rgb_iter=iter_video_frames(video_path, start, num_frames),
            mesh2d_by_frame=mesh2d_by_frame,
            kp2d_by_frame=kp2d_by_frame,
            fps=cam_fps, codec="libx264", pixelformat="yuv420p")
        out_paths.append(out_path)
        print(f"[render_bout_reproj] wrote {out_path} "
              f"({num_frames} frames @ {cam_fps:.3f} fps)")

    return out_paths


def ffprobe_codec(path):
    """Return {'codec_name':..., 'pix_fmt':...} via ffprobe, or None if the
    ffprobe binary is unavailable (non-fatal -- purely a print/verify step)."""
    if shutil.which("ffprobe") is None:
        return None
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
          "-show_entries", "stream=codec_name,pix_fmt",
          "-of", "default=noprint_wrappers=1", path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        return None
    info = {}
    for line in result.stdout.strip().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k] = v
    return info


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(
        description="Overlay predicted 3D keypoints onto camera videos for "
                    "one courtship bout (H.264/yuv420p mp4 per camera).")
    ap.add_argument("--session-dir", required=True,
                    help="Recording dir with Cam<name>.mp4 + calibration/")
    ap.add_argument("--masks-dir", required=True,
                    help="<session_dir>/Predictions_3D_sam3_all30 (has manifest.json)")
    ap.add_argument("--pred-dir", required=True,
                    help="session_predict output dir with bout_<idx>/fly{0,1}.csv")
    ap.add_argument("--bout-id", required=True, type=int)
    ap.add_argument("--project", default="red_data_unified")
    ap.add_argument("--jarvis-root", default=os.environ.get("JARVIS_ROOT") or
                    "/gscratch/portia/eabe/Research/Github/JARVIS-HybridNet")
    ap.add_argument("--out", required=True, help="output dir for per-camera mp4s")
    ap.add_argument("--cameras", default=None,
                    help="comma-separated camera-name subset (e.g. "
                         "Cam2012630,Cam2012631); default = all cameras")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="cap the number of frames rendered (short clip)")
    ap.add_argument("--min-conf", type=float, default=0.0,
                    help="skip joints with confidence below this")
    ap.add_argument("--fps", type=float, default=None,
                    help="override output fps; default = probed from each "
                         "camera's source video")
    return ap


def main(argv=None):
    args = build_argparser().parse_args(argv)
    cameras = [c.strip() for c in args.cameras.split(",")] if args.cameras else None
    out_paths = render_bout(
        session_dir=args.session_dir, masks_dir=args.masks_dir,
        pred_dir=args.pred_dir, bout_id=args.bout_id, project=args.project,
        jarvis_root=args.jarvis_root, out=args.out, cameras=cameras,
        max_frames=args.max_frames, min_conf=args.min_conf, fps=args.fps)
    if out_paths:
        info = ffprobe_codec(out_paths[0])
        if info:
            print(f"[render_bout_reproj] ffprobe({out_paths[0]}): {info}")
        else:
            print(f"[render_bout_reproj] ffprobe unavailable or failed for "
                  f"{out_paths[0]}")
    return out_paths


if __name__ == "__main__":
    main()
