"""Clip I/O for the IK explainer: DLT calibration, video, shipped baseline.

This clip is NOT in the pipeline's usual format. It ships 11-coefficient DLT
`Cam*_dlt.csv` where the pipeline expects `Cam*.yaml`, so
`viz.core.reproject.ReprojectionTool` raises on it and we load the DLTs here
instead.

UNITS: the shipped `data3D_*.csv` is in 0.1 mm while the DLTs expect mm.
`load_shipped_kp3d_mm` applies the ONLY x0.1 conversion in this package.
Everything else in the explainer is already mm.

DISPLAY SOURCE (`enhanced/`): `video_path` defaults to the brightness/contrast
-lifted copies in `<clip>/enhanced/` instead of the raw `<clip>/Cam*.mp4`.
Verified before switching (not re-derived here): all 7 files, same
filenames, same dimensions 1936x448, same rate 800/1, same 921 frames as the
raw copies; normalised cross-correlation against the raw frames is
0.9984-0.9986 at frames 0/400/900 with lag 0 giving the highest correlation
(frame-aligned, no offset); grey mean 22.1 -> 40.5, std 32.3 -> 47.1 (that's
the whole difference -- lifted brightness/contrast, not different content).
This is a PRESENTATION-ONLY change: SAM3 masks and the ViTPose detector were
run against the RAW videos, and nothing downstream of the detector (2D, 3D,
STAC/IK) is re-derived from the enhanced copies. Call sites that feed the
actual detection/masking pipeline pin `enhanced=False` explicitly
(`detect2d.run_detect`'s `caps`, `stage_session_dir`'s SAM3 symlinks,
`prepare_clip.n_frames`) so a future re-run of those stages keeps reading
the exact frames the shipped predictions were derived from; every other
caller (the rendered acts, QC panels, diagnostics) picks up `enhanced/`
automatically through this one function.
"""
import csv
import glob
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# jarvis_jax is not pip-installed; the repo convention is an explicit path
# insert (scripts/slurm_bout_array.py:55,390). camera_view_dirs needs it.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT / "third_party" / "jarvis_jax") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "third_party" / "jarvis_jax"))

CLIP_DEFAULT = ("/data2/users/eabe/datasets/3d_tracking/clips/Session6/"
                "2025_10_12_15_06_46")

# The shipped CSV is in 0.1 mm. See module docstring; do not use elsewhere.
_SHIPPED_CSV_TO_MM = 0.1

# What every enhanced/ file is expected to match (see module docstring's
# DISPLAY SOURCE section for how these were verified). A mismatch here means
# a different/incomplete enhanced copy has been dropped in since verification
# -- fail loudly rather than let it silently desync from the overlays, which
# were computed against the raw video's frame indices.
_ENHANCED_EXPECTED_WH = (1936, 448)
_ENHANCED_EXPECTED_N_FRAMES = 921


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
    with open(csv_path) as fh:
        rows = list(csv.reader(fh))
    names = [rows[0][i] for i in range(0, len(rows[0]), 4)]
    arr = np.array(
        [[np.nan if v in ("", "nan", "NaN") else float(v) for v in r]
         for r in rows[2:]], np.float64).reshape(-1, len(names), 4)
    return arr[:, :, :3] * _SHIPPED_CSV_TO_MM, arr[:, :, 3], names


def _assert_enhanced_video_shape(path: str) -> None:
    """Fail loudly if an `enhanced/` file doesn't match the raw copy it
    stands in for -- see module docstring's DISPLAY SOURCE section for the
    verified numbers this checks against. A silently mismatched enhanced copy
    (wrong dimensions, or a truncated/longer re-encode) would desync every
    overlay computed against the raw video's frame indices without any other
    signal, so this runs on every resolution of an enhanced path, not once by
    hand.
    """
    cap = cv2.VideoCapture(path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    want_w, want_h = _ENHANCED_EXPECTED_WH
    if (w, h) != (want_w, want_h):
        raise ValueError(
            f"{path}: enhanced video is {w}x{h}, expected {want_w}x{want_h} "
            "-- refusing to use it as a drop-in for the raw video (overlays "
            "would desync)")
    if n != _ENHANCED_EXPECTED_N_FRAMES:
        raise ValueError(
            f"{path}: enhanced video has {n} frames, expected "
            f"{_ENHANCED_EXPECTED_N_FRAMES} -- refusing to use it as a "
            "drop-in for the raw video (overlays would desync)")


def video_path(clip: str, cam_name: str, *, enhanced: bool = True) -> str:
    """Locate this camera's video file.

    Prefers the brightness/contrast-lifted copy in `<clip>/enhanced/` (see
    module docstring's DISPLAY SOURCE section) -- checked at load time
    against the exact frame count/dimensions the raw copy has, so a
    mismatched enhanced file fails loudly instead of silently desyncing
    overlays. Falls back to the raw `<clip>/Cam*.mp4` when no enhanced copy
    exists for this camera, or when the caller explicitly passes
    `enhanced=False` (pipeline stages that must keep reading the exact frames
    SAM3/the detector were run against, e.g. `detect2d.run_detect`,
    `stage_session_dir`, `prepare_clip.n_frames` -- see module docstring).
    Raises `FileNotFoundError` if neither an enhanced nor a raw file exists.
    """
    if enhanced:
        hits = sorted(glob.glob(os.path.join(clip, "enhanced", f"{cam_name}_*.mp4")))
        if hits:
            path = hits[0]
            _assert_enhanced_video_shape(path)
            return path
    hits = sorted(glob.glob(os.path.join(clip, f"{cam_name}_*.mp4")))
    if not hits:
        where = "enhanced/ or raw" if enhanced else "raw"
        raise FileNotFoundError(f"no {where} video for {cam_name} under {clip}")
    return hits[0]


def n_video_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def read_frames(path: str, indices) -> np.ndarray:
    """Read the given frame indices (BGR uint8)."""
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


# --- Keypoint orders -------------------------------------------------------
# TWO orders exist and they differ. Mixing them scrambles anatomy while every
# numeric QC stays green (CLAUDE.md records exactly this bug).
#   DETECTOR order: configs/detector/vitpose_v3.yaml:kp_names == data/fly50.json
#                   == the shipped data3D csv header.
#   MODEL order:    configs/anatomy/v1.yaml:KP_NAMES == MuJoCo XML site order.
# Everything downstream of the detector is MODEL order.


def _yaml(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def detector_kp_names() -> list:
    cfg = _yaml(_REPO_ROOT / "configs" / "detector" / "vitpose_v3.yaml")
    return list(cfg["kp_names"])


def model_kp_names() -> list:
    cfg = _yaml(_REPO_ROOT / "configs" / "anatomy" / "v1.yaml")
    return list(cfg["model"]["KP_NAMES"])


def detector_to_model_index() -> np.ndarray:
    """(50,) int such that `arr[detector_to_model_index()]` is in MODEL order."""
    det, mod = detector_kp_names(), model_kp_names()
    missing = set(mod) - set(det)
    if missing:
        raise ValueError(f"model keypoints absent from detector order: {sorted(missing)}")
    pos = {n: i for i, n in enumerate(det)}
    return np.array([pos[n] for n in mod], np.int64)


# --- SAM3 staging (Task 4) --------------------------------------------------
# SAM3 needs a JARVIS-format calibration dir (Cam*.yaml); this clip ships
# Cam*_dlt.csv. See masks.py module docstring for the discovered prerequisite.


def write_jarvis_calibration(clip: str, out_dir) -> list:
    """DLT csvs -> OpenCV FileStorage Cam*.yaml with a `projectionMatrix` node.

    JARVIS's get_repro_tool (jarvis/utils/reprojection.py:152) requires
    Cam<id>.yaml; this clip ships Cam<id>_dlt.csv. The 3x4 matrix is the same
    object either way, so this is a format change, not a recalibration.
    """
    import cv2
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_mats, names = load_dlt(str(Path(clip) / "calibration"))
    written = []
    for M, name in zip(cam_mats, names):
        P = np.asarray(M, np.float64).T                      # (4,3) -> (3,4)
        path = out_dir / f"{name}.yaml"
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        fs.write("projectionMatrix", P)
        fs.release()
        written.append(path)
    return written


def assert_calibration_roundtrip(clip: str, calib_dir, tol: float = 1e-4) -> float:
    """Re-read the staged Cam*.yaml and confirm it reproduces load_dlt's matrices.

    A silent format error in write_jarvis_calibration would place every SAM3
    crop on the wrong pixels while every downstream shape stayed correct, so
    this runs on every staging, not just once by hand. Returns the max abs diff.
    """
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    expected, names = load_dlt(str(Path(clip) / "calibration"))
    rt = ReprojectionTool(str(calib_dir))
    got_names = [c.name for c in rt._camera_list]
    if got_names != list(names):
        raise ValueError(
            f"staged calibration camera order {got_names} != source order {list(names)}")
    diff = float(np.max(np.abs(np.asarray(rt.camera_matrices, np.float64)
                               - np.asarray(expected, np.float64))))
    if not diff < tol:
        raise ValueError(
            f"staged Cam*.yaml does not round-trip: max abs diff {diff:.3e} >= {tol:.0e}")
    return diff


def stage_session_dir(clip: str) -> Path:
    """Build the JARVIS-shaped session dir SAM3 needs, without touching raw input.

    Named <...>/Session6/<timestamp> so session_tag_for() yields the same tag
    the bouts CSV already uses.

    Symlinks are named `<cam>.mp4` (not the raw `Cam<id>_frames_*.mp4`
    filename): `sam3_driver.video_paths_for` looks up
    `<session_dir>/<cam>.mp4` for each camera name in the calibration, and
    verified by hand that keeping the raw suffix makes that lookup raise
    FileNotFoundError. The camera name is the same one `write_jarvis_calibration`
    uses, so both stay keyed off `load_dlt`'s names.

    After writing the calibration, this asserts it round-trips (see
    `assert_calibration_roundtrip`) -- a silent format error here would place
    every SAM3 crop on the wrong pixels while every downstream shape stayed
    correct, so it is checked on every staging, not just once by hand.
    """
    tag = str(clip).rstrip("/").split("/")[-2:]
    root = out_dirs(clip)["root"] / "session" / tag[0] / tag[1]
    root.mkdir(parents=True, exist_ok=True)
    _mats, names = load_dlt(str(Path(clip) / "calibration"))
    for name in names:
        # enhanced=False: SAM3 must see the exact raw frames the shipped
        # masks/detector predictions were derived from (see module
        # docstring's DISPLAY SOURCE section) -- this is provenance-critical
        # staging, not a display panel, and must not silently switch sources.
        src = Path(video_path(clip, name, enhanced=False))
        link = root / f"{name}.mp4"
        # Path.exists() follows symlinks, so a DANGLING symlink (target
        # missing/moved) reports False here and symlink_to would then raise
        # FileExistsError on the leftover link inode. Check is_symlink() too.
        if not link.is_symlink() and not link.exists():
            link.symlink_to(src)
    calib_dir = root / "calibration"
    write_jarvis_calibration(clip, calib_dir)
    diff = assert_calibration_roundtrip(clip, calib_dir)
    print(f"[stage_session_dir] calibration round-trip max abs diff: {diff:.3e} "
          f"(tol 1e-4) at {calib_dir}")
    return root
