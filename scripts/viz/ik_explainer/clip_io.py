"""Clip I/O for the IK explainer: DLT calibration, video, shipped baseline.

This clip is NOT in the pipeline's usual format. It ships 11-coefficient DLT
`Cam*_dlt.csv` where the pipeline expects `Cam*.yaml`, so
`viz.core.reproject.ReprojectionTool` raises on it and we load the DLTs here
instead.

UNITS: the shipped `data3D_*.csv` is in 0.1 mm while the DLTs expect mm.
`load_shipped_kp3d_mm` applies the ONLY x0.1 conversion in this package.
Everything else in the explainer is already mm.
"""
import csv
import glob
import os
from pathlib import Path

import cv2
import numpy as np

CLIP_DEFAULT = ("/data2/users/eabe/datasets/3d_tracking/clips/Session6/"
                "2025_10_12_15_06_46")

# The shipped CSV is in 0.1 mm. See module docstring; do not use elsewhere.
_SHIPPED_CSV_TO_MM = 0.1


def shipped_csv_path(clip: str = CLIP_DEFAULT) -> str:
    hits = sorted(glob.glob(os.path.join(clip, "data3D_*.csv")))
    if not hits:
        raise FileNotFoundError(f"no data3D_*.csv under {clip}")
    return hits[0]


def load_dlt(calib_dir: str):
    """`Cam*_dlt.csv` (11 coefficients each) -> (cam_mats (C,4,3), names).

    Returns matrices in the `ph @ M` convention used by
    `viz.core.reproject.project` and `center3d.triangulate_dlt_batched`,
    i.e. M = P.T for the 3x4 DLT matrix P.
    """
    files = sorted(glob.glob(os.path.join(calib_dir, "Cam*_dlt.csv")))
    if not files:
        raise FileNotFoundError(f"no Cam*_dlt.csv under {calib_dir}")
    mats, names = [], []
    for f in files:
        L = np.loadtxt(f, dtype=np.float64)
        if L.shape != (11,):
            raise ValueError(f"{f}: expected 11 DLT coefficients, got {L.shape}")
        P = np.zeros((3, 4), np.float64)
        P[0, :3], P[0, 3] = L[0:3], L[3]
        P[1, :3], P[1, 3] = L[4:7], L[7]
        P[2, :3], P[2, 3] = L[8:11], 1.0
        mats.append(P.T)                                   # (4,3)
        names.append(os.path.basename(f).replace("_dlt.csv", ""))
    return np.stack(mats), names


def project(cam_mats: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    """pts_mm (K,3) -> (C,K,2) pixel coordinates."""
    pts = np.asarray(pts_mm, np.float64)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)     # (K,4)
    proj = np.einsum("kj,cjm->ckm", ph, np.asarray(cam_mats, np.float64))
    return proj[..., :2] / proj[..., 2:3]


def camera_view_dirs(cam_mats: np.ndarray) -> np.ndarray:
    """(C,4,3) -> (C,3) unit optical axes in world coords (affine cameras)."""
    from jarvis_jax.tracking.affine_camera import factor_affine
    dirs = []
    for M in np.asarray(cam_mats, np.float64):
        _K2, R, _t = factor_affine(M.T)                    # factor wants (3,4)
        dirs.append(R[2])
    return np.stack(dirs)


def load_shipped_kp3d_mm(csv_path: str):
    """Shipped data3D csv -> (xyz_mm (T,50,3), conf (T,50), names) in DETECTOR order.

    Applies the only x0.1 (0.1 mm -> mm) conversion in this package.
    """
    rows = list(csv.reader(open(csv_path)))
    names = [rows[0][i] for i in range(0, len(rows[0]), 4)]
    arr = np.array(
        [[np.nan if v in ("", "nan", "NaN") else float(v) for v in r]
         for r in rows[2:]], np.float64).reshape(-1, len(names), 4)
    return arr[:, :, :3] * _SHIPPED_CSV_TO_MM, arr[:, :, 3], names


def video_path(clip: str, cam_name: str) -> str:
    hits = sorted(glob.glob(os.path.join(clip, f"{cam_name}_*.mp4")))
    if not hits:
        raise FileNotFoundError(f"no video for {cam_name} under {clip}")
    return hits[0]


def n_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frames(path: str, indices) -> np.ndarray:
    """Read the given frame indices (BGR uint8). Sequential-friendly."""
    cap = cv2.VideoCapture(path)
    out, want = [], sorted(set(int(i) for i in indices))
    pos = {}
    for i in want:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, img = cap.read()
        if not ok:
            raise IOError(f"{path}: failed to read frame {i}")
        pos[i] = img
    cap.release()
    for i in indices:
        out.append(pos[int(i)])
    return np.stack(out)


def out_dirs(clip: str = CLIP_DEFAULT) -> dict:
    root = Path(clip) / "ik_explainer"
    d = {"root": root,
         "predictions": root / "predictions",
         "qc": root / "qc",
         "frames": root / "frames"}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d
