# IK Explainer Animation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a ~75 s silent 1080p talk video that explains inverse kinematics by re-running the full pipeline (SAM3 → ViTPose → triangulation → STAC IK) on one clip and rendering the solver's own intermediate stages.

**Architecture:** Two phases. A *prediction phase* re-runs each pipeline stage on the clip, writing a numbered artifact per stage to `<CLIP>/ik_explainer/predictions/`. A *render phase* turns those artifacts into four independent PNG sequences, which an assembler concatenates into the mp4. Every stage writes to disk before the next reads it, so a failed render never re-runs SAM3.

**Tech Stack:** Python 3.12, NumPy, OpenCV, JAX (ViTPose, STAC), PyTorch (SAM3), MuJoCo (v1 fly model), h5py, imageio/ffmpeg.

**Spec:** `docs/specs/2026-08-13-ik-explainer-animation-design.md` — read it before starting. It records why each choice was made.

## Global Constraints

Every task's requirements implicitly include this section.

- **`CLIP` = `/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46`.** All outputs go under `<CLIP>/ik_explainer/`. Raw inputs (`calibration/`, `Cam*.mp4`, `enhanced/`, `preview/`, `data3D_*.csv`) are **read-only — never modify or delete them.**
- **`jarvis_jax` is NOT pip-installed.** Follow the repo convention from `scripts/slurm_bout_array.py:55,390`:
  ```python
  import sys
  from pathlib import Path
  _REPO = Path(__file__).resolve().parents[3]      # scripts/viz/ik_explainer/x.py -> repo root
  sys.path.insert(0, str(_REPO))
  sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
  ```
- **MuJoCo env vars must be set BEFORE `import mujoco`**, at module top:
  ```python
  import os
  os.environ.setdefault("MUJOCO_GL", "egl")
  os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
  ```
- **Run GPU work with `env -u JAX_PLATFORMS`** so a stray `JAX_PLATFORMS=cpu` from the shell does not silently force CPU.
- **Run directly on `glados`** (2× RTX A6000). Never `sbatch`, never a login node.
- **All 3D is in millimetres.** The only `× 0.1` conversion in the codebase is for the shipped baseline CSV, and it lives in exactly one function (`clip_io.load_shipped_kp3d_mm`). Never scale anywhere else.
- **Two keypoint orders exist and they differ.** Getting this wrong scrambles anatomy while every numeric QC stays green — CLAUDE.md records exactly this bug.
  - **DETECTOR order** — `configs/detector/vitpose_v3.yaml:kp_names`, identical to `data/fly50.json:node_names` and to the shipped CSV header. Starts `Antenna_Base, EyeL, EyeR, Scutellum, Abd_A4, Abd_tip, WingL_base, …`
  - **MODEL order** — `configs/anatomy/v1.yaml:KP_NAMES` (MuJoCo XML site order). Starts `Scutellum, WingL_base, WingR_base, Antenna_Base, EyeL, EyeR, WingL_V12, …`
  - Everything downstream of the detector (triangulation, filter, STAC, all rendering) is **MODEL order**. Convert once, immediately after the detector, using `jarvis_jax.tracking.predict_2d.reorder_detector_to_model`.
- **Camera axis order is a second silent trap.** The SAM3 mask npz's camera axis
  need not match the DLT order. Always call
  `load_bout_masks(npz, fly, expected_cameras=cam_names)` so the C axis is
  reordered **by name** into calibration order. A positional mismatch pairs each
  crop with the wrong camera matrix and collapses triangulation while every
  array shape still looks right (`scripts/fix_mask_camera_order.py`,
  `tests/test_mask_camera_order.py` exist because this happened).
- **Colours come from `viz/core/colors.py`** (`PALETTE`, `keypoint_groups`, `leg_chains`). White/cyan = observed/detector, green = fit, grey = mask/mesh. Never invent a palette.
- **Labels use real names and units** — `T1L_FeTi` not index 11, `Cam2012857` not "camera 4", px / mm / deg / frames always stated. Never bare array indices.
- **Anatomy is v1**: `models/fruitfly_v1/fruitfly_v1_free.xml` (`nq=93`, 86 meshes, named cameras `hero`/`side`/`bottom`/`track1`).
- **Frame count `N = 921`** — driven by video. The shipped CSV has 922 rows; truncate to `min()` and assert the difference is ≤ 1.
- **Tests** live flat in `tests/test_ik_explainer_*.py`, run with `python -m pytest`. Tests needing the real clip must skip cleanly when it is absent (see Task 1 Step 1).
- **Every figure is read back before any claim is made about it.** "Generated a figure" is not evidence; opening it and reporting what it shows is.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/viz/ik_explainer/__init__.py` | package marker |
| `scripts/viz/ik_explainer/clip_io.py` | DLT loading, camera matrices, video reading, shipped-CSV baseline, keypoint-order constants |
| `scripts/viz/ik_explainer/prepare_clip.py` | synthesise the one-bout summary CSV SAM3 needs; create output dirs |
| `scripts/viz/ik_explainer/masks.py` | SAM3 → `01_sam3_masks.npz` + mask QC |
| `scripts/viz/ik_explainer/detect2d.py` | 4-ch ViTPose → `02_kp2d.npz` (model order) + 2D QC |
| `scripts/viz/ik_explainer/triangulate3d.py` | triangulate → `03_kp3d.npz` (mm); filter → `04_kp3d_filt.npz`; baseline QC |
| `scripts/viz/ik_explainer/stage_ik.py` | staged STAC → `05_stac_ik.h5`, `06_stages.npz` + stage QC |
| `scripts/viz/ik_explainer/draw.py` | shared 2D drawing helpers (keypoints, chains, labels, fades) |
| `scripts/viz/ik_explainer/acts/act1_views.py` | Act 1 PNG sequence |
| `scripts/viz/ik_explainer/acts/act2_triangulate.py` | Act 2 PNG sequence |
| `scripts/viz/ik_explainer/acts/act3_align.py` | Act 3 PNG sequence |
| `scripts/viz/ik_explainer/acts/act4_solve.py` | Act 4 PNG sequence |
| `scripts/viz/ik_explainer/assemble.py` | concat + crossfade + titles → mp4; write README/manifest |
| `tests/test_ik_explainer_clip_io.py` | Task 1 tests |
| `tests/test_ik_explainer_orders.py` | Task 2 tests (keypoint order) |
| `tests/test_ik_explainer_triangulate.py` | Task 6 tests |
| `tests/test_ik_explainer_draw.py` | Task 8 tests |

---

### Task 1: `clip_io.py` — calibration, video, baseline

**Files:**
- Create: `scripts/viz/ik_explainer/__init__.py`
- Create: `scripts/viz/ik_explainer/clip_io.py`
- Test: `tests/test_ik_explainer_clip_io.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces:
  - `CLIP_DEFAULT: str`
  - `shipped_csv_path(clip: str = CLIP_DEFAULT) -> str`
  - `load_dlt(calib_dir: str) -> tuple[np.ndarray, list[str]]` → `cam_mats (C,4,3) float64`, `cam_names` sorted (`["Cam2012630", ...]`)
  - `project(cam_mats: np.ndarray, pts_mm: np.ndarray) -> np.ndarray` → `(C,K,2)` for `pts_mm (K,3)`
  - `camera_view_dirs(cam_mats) -> np.ndarray` → `(C,3)` unit optical axes in world coords
  - `load_shipped_kp3d_mm(csv_path) -> tuple[np.ndarray, np.ndarray, list[str]]` → `xyz_mm (T,50,3)`, `conf (T,50)`, `names` in **DETECTOR** order
  - `video_path(clip: str, cam_name: str) -> str`
  - `read_frames(path: str, indices: list[int]) -> np.ndarray` → `(n,H,W,3)` uint8 BGR
  - `n_video_frames(path: str) -> int`
  - `out_dirs(clip: str) -> dict[str,Path]` with keys `root, predictions, qc, frames`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ik_explainer_clip_io.py
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import clip_io

CLIP = clip_io.CLIP_DEFAULT
pytestmark = pytest.mark.skipif(
    not Path(CLIP).exists(), reason=f"source clip not present: {CLIP}")


def test_load_dlt_returns_seven_affine_cameras():
    cam_mats, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    assert cam_mats.shape == (7, 4, 3)
    assert names[0] == "Cam2012630" and len(names) == 7
    # affine/telecentric: P[2,:3] == 0, P[2,3] == 1  (P = cam_mat.T)
    P = np.swapaxes(cam_mats, -1, -2)                      # (7,3,4)
    assert np.allclose(P[:, 2, :3], 0.0)
    assert np.allclose(P[:, 2, 3], 1.0)


def test_shipped_kp3d_is_millimetres_and_detector_order():
    xyz, conf, names = clip_io.load_shipped_kp3d_mm(
        clip_io.shipped_csv_path(CLIP))
    assert xyz.shape[1] == 50 and conf.shape[1] == 50
    assert names[:4] == ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]
    # body length Antenna_Base -> Abd_tip is ~2.4 mm for Drosophila.
    i0, i1 = names.index("Antenna_Base"), names.index("Abd_tip")
    body_mm = np.nanmedian(np.linalg.norm(xyz[:, i0] - xyz[:, i1], axis=-1))
    assert 1.5 < body_mm < 4.0, f"body length {body_mm:.2f} mm — unit error"


def test_projection_lands_in_frame_and_never_wildly_outside():
    """Guards the 0.1mm/mm unit trap.

    MEASURED ground truth for this clip: 99.958% of all 322,700 keypoint
    projections land inside 1936x448. The 134 that do not are ALL on
    Cam2012630 (the vertical/top camera) and are all distal right-leg tips
    (T2R_TaTip, T1R_TaTip, T1R_TaT3, T2R_TaT3) leaving the bottom edge by at
    most 15.7 px -- genuine field-of-view clipping on a 448-px-tall strip, not
    a calibration error.

    The excursion bound is what makes this a unit-trap test: feeding the raw
    0.1mm CSV puts u in [7927, 10705], thousands of px outside, so the 32 px
    margin fails instantly.
    """
    cam_mats, _ = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    xyz, _, _ = clip_io.load_shipped_kp3d_mm(clip_io.shipped_csv_path(CLIP))
    uv = np.stack([clip_io.project(cam_mats, xyz[t]) for t in range(xyz.shape[0])])
    u, v = uv[..., 0], uv[..., 1]
    inside = (u >= 0) & (u < 1936) & (v >= 0) & (v < 448)
    assert inside.mean() > 0.999, f"only {100 * inside.mean():.3f}% in frame"
    margin = 32.0
    assert u.min() > -margin and u.max() < 1936 + margin
    assert v.min() > -margin and v.max() < 448 + margin


def test_camera_view_dirs_form_a_30_degree_arc():
    cam_mats, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    d = clip_io.camera_view_dirs(cam_mats)
    assert d.shape == (7, 3)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.0)
    ang = np.degrees(np.arccos(np.clip(d @ d.T, -1, 1)))
    off = ang[~np.eye(7, dtype=bool)]
    # every pair is a multiple of 30 deg within 1.5 deg
    assert np.all(np.abs(off - np.round(off / 30.0) * 30.0) < 1.5)


def test_frame_counts_agree_within_one():
    cam_mats, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    n_vid = clip_io.n_video_frames(clip_io.video_path(CLIP, names[0]))
    xyz, _, _ = clip_io.load_shipped_kp3d_mm(clip_io.shipped_csv_path(CLIP))
    assert abs(n_vid - xyz.shape[0]) <= 1
    assert n_vid == 921


def test_read_frames_returns_requested_frames():
    _, names = clip_io.load_dlt(os.path.join(CLIP, "calibration"))
    imgs = clip_io.read_frames(clip_io.video_path(CLIP, names[0]), [0, 5, 900])
    assert imgs.shape == (3, 448, 1936, 3)
    assert imgs.dtype == np.uint8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/eabe/Research/MyRepos/3d_tracking_dataset && python -m pytest tests/test_ik_explainer_clip_io.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.viz.ik_explainer'`

- [ ] **Step 3: Write the implementation**

```python
# scripts/viz/ik_explainer/__init__.py
"""Generator for the IK explainer animation. See
docs/specs/2026-08-13-ik-explainer-animation-design.md."""
```

```python
# scripts/viz/ik_explainer/clip_io.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/eabe/Research/MyRepos/3d_tracking_dataset && PYTHONPATH=third_party/jarvis_jax JAX_PLATFORMS=cpu python -m pytest tests/test_ik_explainer_clip_io.py -v`
Expected: 6 passed.

If `test_camera_view_dirs_form_a_30_degree_arc` fails, the DLT row packing in `load_dlt` is wrong — do not relax the test; the spec records the arc as exactly 30°.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/__init__.py scripts/viz/ik_explainer/clip_io.py tests/test_ik_explainer_clip_io.py
git commit -m "feat(ik-explainer): clip I/O for DLT calibration, video, baseline

The clip ships 11-coef Cam*_dlt.csv, which ReprojectionTool cannot read.
load_shipped_kp3d_mm holds the only 0.1mm->mm conversion in the package;
tests assert a 2.39mm body length and 100% in-frame projection so a unit
regression fails loudly rather than being absorbed downstream."
```

---

### Task 2: Keypoint-order constants and conversion

**Files:**
- Modify: `scripts/viz/ik_explainer/clip_io.py` (append)
- Test: `tests/test_ik_explainer_orders.py`

**Interfaces:**
- Consumes: Task 1's `clip_io` module.
- Produces:
  - `detector_kp_names() -> list[str]` (50, DETECTOR order, from `configs/detector/vitpose_v3.yaml`)
  - `model_kp_names() -> list[str]` (50, MODEL order, from `configs/anatomy/v1.yaml`)
  - `detector_to_model_index() -> np.ndarray` `(50,)` int, such that `arr_model = arr_detector[idx]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ik_explainer_orders.py
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import clip_io


def test_the_two_orders_are_different_but_same_set():
    det, mod = clip_io.detector_kp_names(), clip_io.model_kp_names()
    assert len(det) == len(mod) == 50
    assert set(det) == set(mod)
    assert det != mod, "orders must differ — see configs comments"
    assert det[:4] == ["Antenna_Base", "EyeL", "EyeR", "Scutellum"]
    assert mod[:3] == ["Scutellum", "WingL_base", "WingR_base"]


def test_index_maps_detector_array_to_model_order():
    det, mod = clip_io.detector_kp_names(), clip_io.model_kp_names()
    idx = clip_io.detector_to_model_index()
    assert idx.shape == (50,)
    assert [det[i] for i in idx] == mod


def test_reordering_is_a_permutation_not_a_reshape():
    idx = clip_io.detector_to_model_index()
    assert sorted(idx.tolist()) == list(range(50))


def test_shipped_csv_is_in_detector_order():
    xyz, conf, names = clip_io.load_shipped_kp3d_mm(
        clip_io.shipped_csv_path(clip_io.CLIP_DEFAULT))
    assert names == clip_io.detector_kp_names()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ik_explainer_orders.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'detector_kp_names'`

- [ ] **Step 3: Write the implementation**

Append to `scripts/viz/ik_explainer/clip_io.py`:

```python
# --- Keypoint orders -------------------------------------------------------
# TWO orders exist and they differ. Mixing them scrambles anatomy while every
# numeric QC stays green (CLAUDE.md records exactly this bug).
#   DETECTOR order: configs/detector/vitpose_v3.yaml:kp_names == data/fly50.json
#                   == the shipped data3D csv header.
#   MODEL order:    configs/anatomy/v1.yaml:KP_NAMES == MuJoCo XML site order.
# Everything downstream of the detector is MODEL order.
_REPO_ROOT = Path(__file__).resolve().parents[3]


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
```

If `configs/anatomy/v1.yaml` nests `KP_NAMES` under a different top-level key than `model`, adjust `model_kp_names` to match the file — verify with:
`python -c "import yaml;d=yaml.safe_load(open('configs/anatomy/v1.yaml'));print([k for k in d])"`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ik_explainer_orders.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/clip_io.py tests/test_ik_explainer_orders.py
git commit -m "feat(ik-explainer): pin the two keypoint orders and the permutation

Detector order (vitpose_v3.yaml == fly50.json == shipped csv) differs from
model order (anatomy/v1.yaml KP_NAMES == XML site order). Tests assert they
differ, cover the same set, and that the mapping is a true permutation, so a
silent reordering regression fails here rather than as scrambled anatomy."
```

---

### Task 3: `prepare_clip.py` — bout summary + output scaffold

**Files:**
- Create: `scripts/viz/ik_explainer/prepare_clip.py`

**Interfaces:**
- Consumes: `clip_io.load_dlt`, `clip_io.n_video_frames`, `clip_io.video_path`, `clip_io.out_dirs`.
- Produces: `write_bouts_csv(clip: str, *, n_override: int = 0, name: str = "bouts.csv") -> Path` writing `<CLIP>/ik_explainer/<name>` with header `fly_id,bout_idx,start_frame,end_frame,n_frames` and exactly one row covering `[0, N-1]`. `n_override > 0` truncates the bout — the only way to run a short SAM3 pilot, since `run_sam3_masks`'s `limit` counts bouts, not frames.

**Why:** `sam3_driver.run_sam3_masks` requires a bouts CSV; this clip has none. `scripts/data_prep/convert_underview_to_jarvis.py` sets the precedent of synthesising "one bout spanning every frame".

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""Prepare the clip for the explainer: output dirs + a one-bout summary CSV.

SAM3's driver (jarvis_jax.predict.sam3_driver.run_sam3_masks) discovers work
from a bouts CSV; this clip ships none. We synthesise a single bout spanning
every frame, matching the precedent in
scripts/data_prep/convert_underview_to_jarvis.py.
"""
import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io   # noqa: E402


def n_frames(clip: str) -> int:
    """Frames common to every camera; the clip is only as long as its shortest view."""
    _mats, names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    counts = [clip_io.n_video_frames(clip_io.video_path(clip, n)) for n in names]
    if max(counts) - min(counts) > 1:
        raise ValueError(f"cameras disagree on frame count by >1: {dict(zip(names, counts))}")
    return min(counts)


def write_bouts_csv(clip: str = clip_io.CLIP_DEFAULT, *, n_override: int = 0,
                    name: str = "bouts.csv") -> Path:
    """Write a one-bout summary. n_override>0 truncates the bout (pilot runs).

    run_sam3_masks has NO frame limit -- its `limit` counts BOUTS -- so the only
    way to time a short slice is to hand it a bout that covers fewer frames.
    """
    d = clip_io.out_dirs(clip)
    n = int(n_override) if n_override else n_frames(clip)
    # fly_id must equal the session tag '<Session>/<timestamp>' that
    # sam3_driver.parse_bouts filters on (see its docstring).
    session_tag = "/".join(str(clip).rstrip("/").split("/")[-2:])
    out = d["root"] / name
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fly_id", "bout_idx", "start_frame", "end_frame", "n_frames"])
        w.writerow([session_tag, 0, 0, n - 1, n])
    print(f"wrote {out}  (1 bout, {n} frames, fly_id={session_tag})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    a = ap.parse_args()
    clip_io.out_dirs(a.clip)
    write_bouts_csv(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it and verify the output**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
python scripts/viz/ik_explainer/prepare_clip.py
cat /data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46/ik_explainer/bouts.csv
```
Expected: `1 bout, 921 frames`, and a CSV with header plus one row
`Session6/2025_10_12_15_06_46,0,0,920,921`.

- [ ] **Step 3: Verify SAM3 will discover the videos**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
python -c "
import glob, os
CLIP='/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46'
v=sorted(glob.glob(os.path.join(CLIP,'Cam*.mp4')))
print(len(v),'videos match Cam*.mp4')
for p in v: print(' ',os.path.basename(p))
"
```
Expected: exactly 7 — `sam3_driver` globs `Cam*.mp4` (`sam3_driver.py:190`), and this clip's `Cam2012630_frames_1161383_1162303.mp4` matches. If it prints 0, stop: the naming assumption is wrong and Task 4 cannot run.

- [ ] **Step 4: Commit**

```bash
git add scripts/viz/ik_explainer/prepare_clip.py
git commit -m "feat(ik-explainer): synthesise the one-bout summary SAM3 needs

The clip ships no bouts CSV. Writes a single bout covering all frames with
fly_id set to the '<Session>/<timestamp>' session tag parse_bouts filters on,
and refuses to proceed if the 7 cameras disagree on frame count by >1."
```

---

### Task 4: `masks.py` — SAM3 masks (measure runtime first)

**Files:**
- Create: `scripts/viz/ik_explainer/masks.py`

**Interfaces:**
- Consumes: `prepare_clip.write_bouts_csv`, `clip_io`.
- Produces: `<CLIP>/ik_explainer/predictions/01_sam3_masks.npz` and `<CLIP>/ik_explainer/qc/01_masks.png`.

**This is the plan's main schedule risk** (spec Risk 1): SAM3 over 7 × 921 = 6,447 frames, cost unmeasured. Step 1 measures a 40-frame slice *before* committing to the full run.

- [ ] **Step 1: Time a 40-frame pilot**

Write `scripts/viz/ik_explainer/masks.py`:

```python
#!/usr/bin/env python3
"""SAM3 masks for the explainer clip.

The retrained ViTPose is 4-channel: session_frameset.build_frameset writes the
SAM3 target mask into channel 3 of every 448x448 crop, so masks are a hard
prerequisite for the 2D stage, not an optional QC aid.

num_animals=1 and distractor_masks stay None — predict_bout_2d's docstring
records that distractor gray-fill is "None for single-animal assays".
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, prepare_clip   # noqa: E402


def run_masks(clip: str, *, limit_frames: int = 0, jarvis_root: str | None = None,
              project: str = "fly50"):
    """Run SAM3 over the clip's single bout. limit_frames>0 => pilot slice."""
    from jarvis_jax.predict.sam3_driver import run_sam3_masks

    d = clip_io.out_dirs(clip)
    # A pilot MUST use a truncated bout and its OWN output dir: run_sam3_masks
    # has no frame limit (`limit` counts bouts), and reuse_masks=True would
    # otherwise let a 40-frame pilot npz stand in for the full run.
    if limit_frames:
        bouts = prepare_clip.write_bouts_csv(clip, n_override=limit_frames,
                                             name="bouts_pilot.csv")
        out_dir = d["predictions"] / "sam3_pilot"
    else:
        bouts = prepare_clip.write_bouts_csv(clip)
        out_dir = d["predictions"] / "sam3"
    t0 = time.time()
    manifest = run_sam3_masks(
        project=project,
        session_dir=str(clip),
        bouts_csv=str(bouts),
        out=str(out_dir),
        num_animals=1,
        reuse_masks=True,
        jarvis_root=jarvis_root or os.environ.get("JARVIS_ROOT"),
        sam3={"text_prompt": "insect", "sam3_version": "sam3.1", "gpu_id": 0},
        overlay=True, overlay_cams=3, overlay_frames=min(300, limit_frames or 300),
    )
    dt = time.time() - t0
    n = limit_frames or n_frames_full(clip)
    print(f"[timing] SAM3 {n} frames x 7 cams in {dt:.1f}s "
          f"({dt / max(n, 1):.3f} s/frame)")
    if limit_frames:
        full = prepare_clip.n_frames(clip)
        print(f"[timing] extrapolated full run ({full} frames): "
              f"{dt / limit_frames * full / 60:.1f} min")
    return manifest, dt


def n_frames_full(clip: str) -> int:
    return prepare_clip.n_frames(clip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--pilot", type=int, default=0,
                    help="if >0, run only this many frames and report timing")
    ap.add_argument("--jarvis-root", default=None)
    a = ap.parse_args()
    run_masks(a.clip, limit_frames=a.pilot, jarvis_root=a.jarvis_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Run the pilot:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/masks.py --pilot 40
```

The pilot prints its own extrapolation to the full 921 frames. **Record it before continuing.** If the extrapolation exceeds ~2 hours, stop and report to the user with the measured number rather than proceeding — the spec names this the main schedule risk, and the user should decide whether to sub-range the clip.

`run_sam3_masks` needs a JARVIS *project* directory (`projects/`). If it raises about a missing project, locate one with
`ls /home/eabe/Research/MyRepos/JARVIS-HybridNet/projects` and pass `--jarvis-root`. Do not invent a project name.

- [ ] **Step 2: Run the full mask pass**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/masks.py
```
Expected: `01_sam3_masks.npz` (or a per-bout `sam3_masks.npz` under `predictions/sam3/bout_00000/`) plus a timing line. Note the on-disk size; the pipeline convention is 50–500 MB compressed.

- [ ] **Step 3: Write the mask QC figure**

Append to `masks.py`:

```python
def qc_masks(clip: str, frames=(120, 450, 780)):
    """Mask QC montage.

    EXPECTATION if masks are good: exactly ONE connected fly-sized blob per
    camera per frame, covering the fly and not the arena wall or its
    reflection, present in all 7 views at every sampled frame.
    FALSIFICATION: blobs on arena features, multiple competing blobs, or an
    empty view -> the 2D stage will read garbage from crop channel 3.
    """
    import cv2
    from jarvis_jax.tracking.bout_masks import load_bout_masks

    d = clip_io.out_dirs(clip)
    _mats, names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    # load_bout_masks(npz_path, fly, *, expected_cameras=None) -> DICT with keys
    # masks (T,C,H,W) bool, valid (T,C), centroids (T,C,2). Passing
    # expected_cameras reorders the C axis BY NAME into our DLT order -- without
    # it the mask camera axis can silently disagree with the calibration (see
    # scripts/fix_mask_camera_order.py and tests/test_mask_camera_order.py).
    bm = load_bout_masks(str(mask_npz_path(clip)), 0, expected_cameras=names)
    panels = []
    for f in frames:
        row = []
        for ci, cam in enumerate(names):
            img = clip_io.read_frames(clip_io.video_path(clip, cam), [f])[0]
            m = np.asarray(bm["masks"][f, ci], bool)
            n_blobs, _lab = cv2.connectedComponents(m.astype(np.uint8))
            over = img.copy()
            over[m] = (0.5 * over[m] + 0.5 * np.array([200, 200, 200])).astype(np.uint8)
            cv2.putText(over, f"{cam} f{f} blobs={n_blobs - 1} px={int(m.sum())}",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                        cv2.LINE_AA)
            row.append(cv2.resize(over, (484, 112)))
        panels.append(np.hstack(row))
    out = d["qc"] / "01_masks.png"
    cv2.imwrite(str(out), np.vstack(panels))
    print(f"wrote {out}")
    return out
```

Wire it into `main()` behind `--qc`, then run:
```bash
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/masks.py --qc
```

Add this helper to `masks.py` so both this task and Task 5 locate the npz the
same way (SAM3 writes a per-bout `sam3_masks.npz`):

```python
def mask_npz_path(clip: str) -> Path:
    d = clip_io.out_dirs(clip)
    hits = sorted((d["predictions"] / "sam3").rglob("sam3_masks.npz"))
    if not hits:
        raise FileNotFoundError(f"no sam3_masks.npz under {d['predictions'] / 'sam3'}")
    return hits[0]
```

- [ ] **Step 4: READ the QC figure and report**

Open `<CLIP>/ik_explainer/qc/01_masks.png` with the Read tool. Report, against the stated expectation: how many blobs per view, whether any view is empty, whether any mask covers arena rather than fly. **Do not proceed to Task 5 if any sampled view has 0 blobs or a mask on the wall** — the detector reads this as crop channel 3 and everything downstream inherits the error.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/masks.py
git commit -m "feat(ik-explainer): SAM3 mask stage with pilot timing and QC gate

The detector is 4-channel and reads the mask as crop channel 3, so masks gate
every downstream stage. --pilot times a 40-frame slice before committing to
6447 frames (the plan's main schedule risk), and the QC montage counts
connected components per view so an empty or wall-latched mask is caught
before it silently corrupts 2D."
```

---

### Task 5: `detect2d.py` — 4-channel ViTPose in model order

**Files:**
- Create: `scripts/viz/ik_explainer/detect2d.py`

**Interfaces:**
- Consumes: masks from Task 4; `clip_io.load_dlt`, `clip_io.detector_to_model_index`, `clip_io.model_kp_names`.
- Produces: `02_kp2d.npz` with arrays `kp2d (N,C,50,2) float32` full-frame px and `conf (N,C,50) float32`, **both in MODEL order**, plus `cam_names` and `kp_names`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""4-channel ViTPose 2D for the explainer clip.

Output is written in MODEL order (configs/anatomy/v1.yaml KP_NAMES). The
detector emits DETECTOR order (configs/detector/vitpose_v3.yaml kp_names);
reorder_detector_to_model bridges the two. Getting this wrong scrambles
anatomy while every numeric QC stays green -- CLAUDE.md records that bug.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io            # noqa: E402
from scripts.viz.ik_explainer import masks as masks_mod  # noqa: E402

CKPT = ("/data2/users/eabe/datasets/3d_tracking/jax_vitpose_runs/"
        "v4_8gpu_20260808/final")


def run_detect(clip: str, *, ckpt: str = CKPT, batch: int = 64,
               decode_sharpen: float = 3.0):
    from jarvis_jax.tracking.bout_masks import load_bout_masks
    from jarvis_jax.tracking.predict_2d import (load_detector, predict_bout_2d,
                                                reorder_detector_to_model)

    d = clip_io.out_dirs(clip)
    cam_mats, cam_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    # fly=0 (single-animal clip). expected_cameras reorders the mask C axis BY
    # NAME into the DLT order -- a positional mismatch here silently pairs each
    # crop with the wrong camera matrix and collapses triangulation.
    bm = load_bout_masks(str(masks_mod.mask_npz_path(clip)), 0,
                         expected_cameras=cam_names)
    masks = np.asarray(bm["masks"], bool)              # (N,C,H,W)
    centroids = np.asarray(bm["centroids"], np.float32)  # (N,C,2)
    valid = np.asarray(bm["valid"], bool)              # (N,C)
    N = masks.shape[0]

    caps = [clip_io.video_path(clip, c) for c in cam_names]

    def frames_iter():
        import cv2
        readers = [cv2.VideoCapture(p) for p in caps]
        try:
            for _t in range(N):
                imgs = []
                for r in readers:
                    ok, img = r.read()
                    if not ok:
                        raise IOError("video ended early")
                    imgs.append(img[:, :, ::-1])          # BGR -> RGB
                yield np.stack(imgs)
        finally:
            for r in readers:
                r.release()

    vit = load_detector(ckpt, num_keypoints=50)
    kp2d, conf = predict_bout_2d(
        vit, frames_iter(), masks, centroids, valid, cam_mats,
        crop=448, batch=batch, decode_sharpen=decode_sharpen,
        distractor_masks=None)          # single-animal assay

    # DETECTOR order -> MODEL order. Single conversion point.
    det_names = clip_io.detector_kp_names()
    mod_names = clip_io.model_kp_names()
    kp2d, conf = reorder_detector_to_model(kp2d, conf, det_names, mod_names)

    out = d["predictions"] / "02_kp2d.npz"
    np.savez_compressed(out, kp2d=kp2d.astype(np.float32),
                        conf=conf.astype(np.float32),
                        cam_names=np.array(cam_names),
                        kp_names=np.array(mod_names))
    print(f"wrote {out}  kp2d={kp2d.shape} conf={conf.shape} (MODEL order)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()
    run_detect(a.clip, ckpt=a.ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Check `reorder_detector_to_model`'s real signature first
(`sed -n 52,67p third_party/jarvis_jax/jarvis_jax/tracking/predict_2d.py`) and match it; it returns `(kp2d, conf)` reordered along the keypoint axis.

- [ ] **Step 2: Run it**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/detect2d.py
```
Expected: `kp2d=(921, 7, 50, 2) conf=(921, 7, 50) (MODEL order)`

- [ ] **Step 3: Write the 2D QC overlay**

Append to `detect2d.py`:

```python
def qc_kp2d(clip: str, frames=(120, 450, 780)):
    """2D QC montage, one row per frame, all 7 cameras.

    EXPECTATION if the detector and the ordering are right: keypoints sit on
    the fly in every view, with Antenna_Base/EyeL/EyeR on the HEAD, Abd_tip at
    the posterior tip, and the six leg chains running proximal->distal without
    crossing the body. Confidence should DROP on views where a limb is
    occluded -- uniform high confidence everywhere means the ordering or the
    crop is wrong, not that tracking is perfect.
    FALSIFICATION: head-group markers on the abdomen, or leg chains zig-zagging
    across the body => detector->model reorder is broken.
    """
    import cv2
    from viz.core.colors import PALETTE, keypoint_groups, leg_chains

    d = clip_io.out_dirs(clip)
    z = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp2d, conf = z["kp2d"], z["conf"]
    cam_names = [str(c) for c in z["cam_names"]]
    kp_names = [str(n) for n in z["kp_names"]]
    groups, chains = keypoint_groups(kp_names), leg_chains(kp_names)
    gcol = {"head": PALETTE["head"], "abdomen": PALETTE["tail"],
            "thorax": (0, 255, 255), "legs": PALETTE["fly0"]}

    rows = []
    for f in frames:
        row = []
        for ci, cam in enumerate(cam_names):
            img = clip_io.read_frames(clip_io.video_path(clip, cam), [f])[0]
            p, c = kp2d[f, ci], conf[f, ci]
            cx, cy = float(np.nanmean(p[:, 0])), float(np.nanmean(p[:, 1]))
            x0 = int(np.clip(cx - 210, 0, img.shape[1] - 420))
            y0 = int(np.clip(cy - 150, 0, img.shape[0] - 300))
            crop = img[y0:y0 + 300, x0:x0 + 420].copy()
            q = p - np.array([x0, y0])
            for _leg, ch in chains.items():
                for a, b in zip(q[ch][:-1], q[ch][1:]):
                    if np.all(np.isfinite([a, b])):
                        cv2.line(crop, tuple(a.astype(int)), tuple(b.astype(int)),
                                 PALETTE["fly0"], 1, cv2.LINE_AA)
            for g, idxs in groups.items():
                for i in idxs:
                    if np.all(np.isfinite(q[i])):
                        r = 2 if c[i] >= 0.3 else 1
                        cv2.circle(crop, tuple(q[i].astype(int)), r, gcol[g], -1,
                                   cv2.LINE_AA)
            for lab in ("Antenna_Base", "Abd_tip"):
                i = kp_names.index(lab)
                if np.all(np.isfinite(q[i])):
                    cv2.putText(crop, lab, tuple((q[i] + [4, -4]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1,
                                cv2.LINE_AA)
            cv2.putText(crop, f"{cam} f{f} medconf={np.median(c):.2f}", (6, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            row.append(crop)
        rows.append(np.hstack(row))
    out = d["qc"] / "02_kp2d_overlay.png"
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"wrote {out}")
    return out
```

Wire behind `--qc` and run it.

- [ ] **Step 4: READ the QC figure and report**

Open `<CLIP>/ik_explainer/qc/02_kp2d_overlay.png` with the Read tool. Report against the expectation: are head markers on the head in all 7 views? Do leg chains run cleanly proximal→distal? Does median confidence vary across views? **A figure that was generated but not viewed is not evidence.**

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/detect2d.py
git commit -m "feat(ik-explainer): 4ch ViTPose 2D, written in model order

Converts detector order -> model order exactly once, immediately after
inference. QC overlay labels Antenna_Base/Abd_tip by name and dims
low-confidence markers, so a reorder regression shows up as head markers on
the abdomen rather than as a silently plausible number."
```

---

### Task 6: `triangulate3d.py` — 3D in mm, filtered, vs baseline

**Files:**
- Create: `scripts/viz/ik_explainer/triangulate3d.py`
- Test: `tests/test_ik_explainer_triangulate.py`

**Interfaces:**
- Consumes: `02_kp2d.npz`; `clip_io.load_dlt`, `clip_io.load_shipped_kp3d_mm`, `clip_io.detector_to_model_index`.
- Produces: `03_kp3d.npz` (`kp3d (N,50,3)` mm, `conf3d (N,50)`, MODEL order), `04_kp3d_filt.npz` (same shape).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ik_explainer_triangulate.py
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io, triangulate3d

CLIP = clip_io.CLIP_DEFAULT
pytestmark = pytest.mark.skipif(
    not Path(CLIP).exists(), reason="source clip not present")


def test_round_trip_recovers_known_3d_in_mm():
    """Project known mm 3D into 7 views, triangulate back, expect ~0 error.

    This is the guarantee that lets the explainer drop the 0.1 unit factor:
    triangulating our own 2D yields mm directly.
    """
    cam_mats, _ = clip_io.load_dlt(str(Path(CLIP) / "calibration"))
    xyz, _, _ = clip_io.load_shipped_kp3d_mm(clip_io.shipped_csv_path(CLIP))
    X = xyz[:50]                                        # (T,K,3) mm
    ph = np.concatenate([X, np.ones((*X.shape[:2], 1))], -1)
    proj = np.einsum("tkj,cjm->tckm", ph, cam_mats)
    uv = proj[..., :2] / proj[..., 2:3]
    conf = np.ones(uv.shape[:3], np.float32)
    kp3d, _ = triangulate3d.triangulate(uv, conf, cam_mats)
    err = np.linalg.norm(kp3d - X, axis=-1)
    assert np.nanmax(err) < 1e-3, f"max round-trip error {np.nanmax(err):.2e} mm"


def test_affine_projection_has_unit_depth():
    """The rig is telecentric, so the perspective divide must be a no-op."""
    cam_mats, _ = clip_io.load_dlt(str(Path(CLIP) / "calibration"))
    X = np.array([[10.0, 3.0, 1.5], [15.0, 2.0, 1.0]])
    ph = np.concatenate([X, np.ones((2, 1))], -1)
    proj = np.einsum("kj,cjm->ckm", ph, cam_mats)
    assert np.allclose(proj[..., 2], 1.0)


def test_filter_never_reduces_coverage():
    rng = np.random.default_rng(0)
    names = clip_io.model_kp_names()
    kp3d = rng.normal(size=(60, 50, 3)) * 0.2 + 10.0
    conf = np.ones((60, 50), np.float32)
    out = triangulate3d.filter_kp3d(kp3d, conf, names)
    assert out.shape == kp3d.shape
    finite_in = np.isfinite(kp3d).all(-1)
    finite_out = np.isfinite(out).all(-1)
    assert np.all(finite_out[finite_in]), "filter deleted a keypoint it was given"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=third_party/jarvis_jax JAX_PLATFORMS=cpu python -m pytest tests/test_ik_explainer_triangulate.py -v`
Expected: FAIL — `ImportError: cannot import name 'triangulate3d'`

- [ ] **Step 3: Write the implementation**

```python
#!/usr/bin/env python3
"""Triangulate the explainer clip's 2D into 3D (mm) and temporally filter it.

UNITS: the DLTs map mm -> px, so triangulating our own 2D yields mm directly.
No 0.1 factor appears here; that conversion exists only in
clip_io.load_shipped_kp3d_mm for the baseline comparison.

Filtering is a PREREQUISITE, not polish: jarvis_jax/tracking/filter.py records
that raw triangulated distal keypoints (esp. *_TaTip) jump many mm frame to
frame and STAC then bends the leg to chase the outlier. Act 4's whole claim is
that tarsal tips track the keypoints.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))

from scripts.viz.ik_explainer import clip_io   # noqa: E402

# Matches configs/detector/vitpose_v3.yaml.
CONF_THRESH = 0.3
VIEW_CONF_THRESH = 0.6

# Matches configs/pipeline.yaml `filtering:`.
FILTER_CFG = {"enabled": True, "preserve_raw_patterns": ["Wing"]}


def triangulate(kp2d, conf, cam_mats, *, conf_thresh=CONF_THRESH,
                view_conf_thresh=VIEW_CONF_THRESH):
    """kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3) -> (kp3d (T,K,3) mm, conf3d)."""
    from jarvis_jax.tracking.triangulate import triangulate_keypoints
    kp2d = np.nan_to_num(np.asarray(kp2d, np.float32), nan=0.0,
                         posinf=0.0, neginf=0.0)   # NaN px would poison the SVD
    return triangulate_keypoints(kp2d, np.asarray(conf, np.float32),
                                 np.asarray(cam_mats, np.float32),
                                 conf_thresh=conf_thresh,
                                 view_conf_thresh=view_conf_thresh)


def filter_kp3d(kp3d, conf3d, kp_names, filter_cfg=None):
    from jarvis_jax.tracking.filter import filter_bout_kp3d
    return filter_bout_kp3d(kp3d, conf3d, list(kp_names),
                            dict(filter_cfg or FILTER_CFG))


def run(clip: str = clip_io.CLIP_DEFAULT):
    d = clip_io.out_dirs(clip)
    cam_mats, _cams = clip_io.load_dlt(str(Path(clip) / "calibration"))
    z = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    kp_names = [str(n) for n in z["kp_names"]]

    kp3d, conf3d = triangulate(z["kp2d"], z["conf"], cam_mats)
    np.savez_compressed(d["predictions"] / "03_kp3d.npz", kp3d=kp3d,
                        conf3d=conf3d, kp_names=np.array(kp_names))
    print(f"wrote 03_kp3d.npz {kp3d.shape} mm  finite={np.isfinite(kp3d).all(-1).mean():.3f}")

    filt = filter_kp3d(kp3d, conf3d, kp_names)
    np.savez_compressed(d["predictions"] / "04_kp3d_filt.npz", kp3d=filt,
                        conf3d=conf3d, kp_names=np.array(kp_names))
    print(f"wrote 04_kp3d_filt.npz {filt.shape} mm")
    return kp3d, filt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default=clip_io.CLIP_DEFAULT)
    a = ap.parse_args()
    run(a.clip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=third_party/jarvis_jax JAX_PLATFORMS=cpu python -m pytest tests/test_ik_explainer_triangulate.py -v`
Expected: 3 passed. Then run the stage:
`env -u JAX_PLATFORMS python scripts/viz/ik_explainer/triangulate3d.py`

- [ ] **Step 5: Write the baseline + filter QC**

Append to `triangulate3d.py`:

```python
def qc_vs_baseline(clip: str = clip_io.CLIP_DEFAULT):
    """Compare our 3D against the shipped csv, and measure the filter's effect.

    EXPECTATION if ordering and units are right: per-keypoint median
    disagreement is SMALL and roughly uniform across keypoints. Given 1 px of
    2D error costs 9.3 um, a well-behaved run should sit in the tens of um to
    ~0.1 mm. A few hundred um on distal tarsi is plausible (different detector
    run); a UNIFORM offset of ~mm, or one keypoint wildly off, is a unit or
    ordering error.
    FALSIFICATION: disagreement clustered by keypoint GROUP (all head markers
    off by the same large amount) => the detector->model permutation is wrong.

    Filter expectation: *_TaTip max acceleration drops substantially
    (filter.py's own precedent: 12.5 -> 0.9 mm). No drop => filter not applied.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = clip_io.out_dirs(clip)
    ours = np.load(d["predictions"] / "03_kp3d.npz", allow_pickle=True)
    filt = np.load(d["predictions"] / "04_kp3d_filt.npz", allow_pickle=True)
    kp_names = [str(n) for n in ours["kp_names"]]

    base_xyz, _bc, base_names = clip_io.load_shipped_kp3d_mm(
        clip_io.shipped_csv_path(clip))
    # baseline is DETECTOR order; ours is MODEL order.
    idx = clip_io.detector_to_model_index()
    base = base_xyz[:, idx]                       # -> MODEL order
    n = min(len(base), len(ours["kp3d"]))
    diff = np.linalg.norm(ours["kp3d"][:n] - base[:n], axis=-1)      # (n,K)
    med = np.nanmedian(diff, axis=0)

    def max_accel(a, name_sub="TaTip"):
        cols = [i for i, nm in enumerate(kp_names) if name_sub in nm]
        v = np.diff(a[:, cols], axis=0)
        acc = np.linalg.norm(np.diff(v, axis=0), axis=-1)
        return float(np.nanmax(acc))

    raw_acc = max_accel(ours["kp3d"])
    filt_acc = max_accel(filt["kp3d"])

    fig, ax = plt.subplots(2, 1, figsize=(14, 8), constrained_layout=True)
    ax[0].bar(range(len(med)), med)
    ax[0].set_xticks(range(len(med)))
    ax[0].set_xticklabels(kp_names, rotation=90, fontsize=6)
    ax[0].set_ylabel("median |ours - shipped| (mm)")
    ax[0].set_title(f"re-predicted vs shipped baseline, n={n} frames "
                    f"(1 px of 2D error = 9.3 um in 3D)")
    ax[1].bar(["raw", "filtered"], [raw_acc, filt_acc], color=["grey", "green"])
    ax[1].set_ylabel("max *_TaTip acceleration (mm/frame^2)")
    ax[1].set_title(f"filter effect: {raw_acc:.2f} -> {filt_acc:.2f} mm")
    out = d["qc"] / "03_kp3d_vs_shipped.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    import json
    (d["qc"] / "qc.json").write_text(json.dumps(
        {"median_diff_mm": dict(zip(kp_names, med.round(5).tolist())),
         "tatip_max_accel_raw_mm": raw_acc,
         "tatip_max_accel_filtered_mm": filt_acc,
         "n_frames_compared": int(n)}, indent=2))
    print(f"wrote {out}")
    return out
```

Wire behind `--qc` and run.

- [ ] **Step 6: READ the QC figure and report**

Open `<CLIP>/ik_explainer/qc/03_kp3d_vs_shipped.png`. Report the per-keypoint median disagreement and whether it is uniform or group-clustered, plus the `*_TaTip` acceleration before/after. **Group-clustered disagreement means the permutation is wrong — stop and fix Task 2 before continuing.**

- [ ] **Step 7: Commit**

```bash
git add scripts/viz/ik_explainer/triangulate3d.py tests/test_ik_explainer_triangulate.py
git commit -m "feat(ik-explainer): triangulate to mm, filter, and diff vs baseline

Tests assert the 3D->2D->3D round trip closes below 1e-3 mm and that the
affine depth term is identically 1, which is what licenses dropping the 0.1
unit factor from the chain. The baseline diff reorders the shipped csv from
detector to model order, so a bad permutation shows up as group-clustered
disagreement rather than as a plausible small number."
```

---

### Task 7: `stage_ik.py` — staged STAC with per-stage snapshots

**Files:**
- Create: `scripts/viz/ik_explainer/stage_ik.py`

**Interfaces:**
- Consumes: `04_kp3d_filt.npz`; `configs/anatomy/v1.yaml`.
- Produces: `05_stac_ik.h5`, and `06_stages.npz` with arrays
  `qpos_default (nq,)`, `qpos_scaled (nq,)`, `qpos_root (nq,)`, `qpos_pose (nq,)`,
  `qpos_seq (N,nq)`, `residual_mm (4,)` (one per stage, in the order default,
  scaled, root, pose), `stage_names (4,)`, `kp_names (50,)`.

**Why staged:** `stac-mjx/stac_mjx/stac.py:296` (`fit_offsets`) already runs `root_optimization` then `pose_optimization`. We snapshot `mjx_data.qpos` after each rather than keeping only the final result — the video's "misaligned → aligned" beat is the optimiser's own frame zero.

- [ ] **Step 1: Read the solver before writing the driver**

Run and read:
```bash
sed -n 296,404p stac-mjx/stac_mjx/stac.py
sed -n 1,60p stac-mjx/stac_mjx/compute_stac.py
grep -n "def root_optimization\|def pose_optimization\|def offset_optimization" stac-mjx/stac_mjx/compute_stac.py
sed -n 1,50p stac-mjx/stac_mjx/rescale.py
sed -n 1,60p scripts/run_stac.py
```
Write down the exact signatures of `root_optimization` and `pose_optimization` and what each returns. **Do not guess them** — the snapshot driver mirrors `fit_offsets` and must call them the same way.

- [ ] **Step 2: Write the staged driver**

Create `scripts/viz/ik_explainer/stage_ik.py` mirroring `Stac.fit_offsets`' control flow, but capturing `mjx_data.qpos` after each stage. Structure:

```python
#!/usr/bin/env python3
"""Staged STAC IK: snapshot qpos after each solver stage.

stac.py:296 fit_offsets already runs root_optimization -> pose_optimization,
alternating with offset_optimization. The explainer needs the INTERMEDIATE
states, so this mirrors that control flow and records qpos after each stage
instead of keeping only the final result. Same solver, nothing hand-authored:
the "misaligned" state Act 3 opens on is the optimiser's frame zero.
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "third_party" / "jarvis_jax"))
sys.path.insert(0, str(_REPO / "stac-mjx"))

from scripts.viz.ik_explainer import clip_io   # noqa: E402


def marker_residual_mm(mjx_model, mjx_data, kp_frame, body_site_idxs):
    """Mean Euclidean distance (mm) between model marker sites and keypoints."""
    import numpy as np
    sites = np.asarray(mjx_data.site_xpos)[np.asarray(body_site_idxs)]
    tgt = np.asarray(kp_frame).reshape(-1, 3)
    return float(np.nanmean(np.linalg.norm(sites - tgt, axis=-1)))
```

Then implement `run(clip, frame_for_stills=450)` which:
1. loads `04_kp3d_filt.npz` (MODEL order, mm),
2. builds the `Stac` object from `configs/anatomy/v1.yaml` the same way `scripts/run_stac.py` composes it,
3. records `qpos_default` + residual from the rest pose,
4. applies `rescale` body scaling → `qpos_scaled` + residual,
5. calls `compute_stac.root_optimization(...)` → `qpos_root` + residual,
6. calls `compute_stac.pose_optimization(...)` → `qpos_pose` + residual and the full `qpos_seq`,
7. saves `06_stages.npz` and lets `stac_mjx` write `05_stac_ik.h5`.

**Units check to include inline:** assert the keypoints handed to STAC have a
median `Antenna_Base`→`Abd_tip` distance in `[1.5, 4.0]` mm before solving, and
raise if not. This is the 38× body-scale failure mode CLAUDE.md records.

- [ ] **Step 3: Run it**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/stage_ik.py
```
Expected: four residual values printed, **monotonically decreasing**, and `06_stages.npz` written.

- [ ] **Step 4: Write the stage QC figure**

Add `qc_stages(clip)` rendering one MuJoCo panel per stage (camera `hero`), mesh in grey with the keypoint cloud overlaid in cyan and the residual in mm burned into each panel.

```
EXPECTATION: residual decreases monotonically default -> scaled -> root ->
pose, and by the `root` panel the mesh is anatomically oriented (head end at
the head keypoints). FALSIFICATION: residual flat or rising, or the mesh
oriented 180 deg off with its head sitting on the abdomen keypoints -- which
a residual number alone will not reveal, because a flipped fit can score
similarly to a correct one on a roughly symmetric marker set.
```

Save to `<CLIP>/ik_explainer/qc/05_ik_stages.png`.

- [ ] **Step 5: READ the QC figure and report**

Open it. Report the four residuals and whether the mesh is correctly oriented by the `root` stage. **A 180° flip here invalidates Acts 3–4 — stop and fix before rendering.**

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/ik_explainer/stage_ik.py
git commit -m "feat(ik-explainer): staged STAC driver snapshotting qpos per stage

Mirrors fit_offsets' control flow but records qpos after rescale,
root_optimization and pose_optimization so the video can show the solver's own
intermediate states. Asserts a 1.5-4mm body length before solving, since the
38x body-scale failure mode CLAUDE.md records is invisible to residual checks."
```

---

### Task 8: `draw.py` — shared drawing helpers

**Files:**
- Create: `scripts/viz/ik_explainer/draw.py`
- Test: `tests/test_ik_explainer_draw.py`

**Interfaces:**
- Produces:
  - `draw_keypoints(img, uv, kp_names, conf=None, alpha=1.0) -> np.ndarray`
  - `draw_leg_chains(img, uv, kp_names, alpha=1.0) -> np.ndarray`
  - `label(img, text, xy, scale=0.5) -> np.ndarray`
  - `stage_title(img, title, subtitle="") -> np.ndarray`
  - `fade(img_a, img_b, t: float) -> np.ndarray` (`t=0` → a, `t=1` → b)
  - `scale_bar_mm(img, px_per_mm, mm=1.0) -> np.ndarray`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ik_explainer_draw.py
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import draw


def test_fade_endpoints_and_midpoint():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.full((10, 10, 3), 200, np.uint8)
    assert np.array_equal(draw.fade(a, b, 0.0), a)
    assert np.array_equal(draw.fade(a, b, 1.0), b)
    assert abs(int(draw.fade(a, b, 0.5)[0, 0, 0]) - 100) <= 1


def test_alpha_zero_leaves_image_untouched():
    img = np.full((60, 60, 3), 40, np.uint8)
    uv = np.full((50, 2), 30.0)
    names = ["Antenna_Base"] * 50
    assert np.array_equal(draw.draw_keypoints(img, uv, names, alpha=0.0), img)


def test_drawing_does_not_mutate_the_input():
    img = np.full((60, 60, 3), 40, np.uint8)
    before = img.copy()
    uv = np.full((50, 2), 30.0)
    draw.draw_keypoints(img, uv, ["EyeL"] * 50, alpha=1.0)
    assert np.array_equal(img, before), "helper must return a copy"


def test_nonfinite_keypoints_are_skipped_not_crashed():
    img = np.zeros((60, 60, 3), np.uint8)
    uv = np.full((50, 2), np.nan)
    out = draw.draw_keypoints(img, uv, ["EyeL"] * 50, alpha=1.0)
    assert np.array_equal(out, img)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ik_explainer_draw.py -v`
Expected: FAIL — `ImportError: cannot import name 'draw'`

- [ ] **Step 3: Write the implementation**

```python
"""Shared 2D drawing for the explainer acts.

Colours come from viz/core/colors.py so this video shares the repo's visual
language. Every helper returns a COPY -- acts composite the same base frame at
several alphas, so in-place drawing would smear across a fade.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from viz.core.colors import PALETTE, keypoint_groups, leg_chains   # noqa: E402

_GROUP_COLOR = {"head": PALETTE["head"], "abdomen": PALETTE["tail"],
                "thorax": (0, 255, 255), "legs": PALETTE["fly0"]}


def fade(img_a, img_b, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return np.asarray(img_a).copy()
    if t >= 1.0:
        return np.asarray(img_b).copy()
    return cv2.addWeighted(np.asarray(img_a), 1.0 - t, np.asarray(img_b), t, 0.0)


def draw_keypoints(img, uv, kp_names, conf=None, alpha=1.0, radius=3):
    out = np.asarray(img).copy()
    if alpha <= 0.0:
        return out
    layer = out.copy()
    groups = keypoint_groups(list(kp_names))
    for g, idxs in groups.items():
        for i in idxs:
            p = np.asarray(uv)[i]
            if not np.all(np.isfinite(p)):
                continue
            r = radius if (conf is None or conf[i] >= 0.3) else max(1, radius - 2)
            cv2.circle(layer, tuple(np.round(p).astype(int)), r,
                       _GROUP_COLOR[g], -1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0.0)


def draw_leg_chains(img, uv, kp_names, alpha=1.0, thickness=1):
    out = np.asarray(img).copy()
    if alpha <= 0.0:
        return out
    layer = out.copy()
    for _leg, chain in leg_chains(list(kp_names)).items():
        pts = np.asarray(uv)[chain]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.all(np.isfinite([a, b])):
                cv2.line(layer, tuple(np.round(a).astype(int)),
                         tuple(np.round(b).astype(int)), PALETTE["fly0"],
                         thickness, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0.0)


def label(img, text, xy, scale=0.5, color=(255, 255, 255)):
    out = np.asarray(img).copy()
    cv2.putText(out, text, (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, 1, cv2.LINE_AA)
    return out


def stage_title(img, title, subtitle=""):
    out = np.asarray(img).copy()
    cv2.putText(out, title, (48, 72), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (48, 116), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (190, 190, 190), 1, cv2.LINE_AA)
    return out


def scale_bar_mm(img, px_per_mm, mm=1.0, origin=None):
    out = np.asarray(img).copy()
    h, w = out.shape[:2]
    x0, y0 = origin if origin is not None else (w - int(mm * px_per_mm) - 40, h - 30)
    x1 = x0 + int(mm * px_per_mm)
    cv2.line(out, (x0, y0), (x1, y0), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{mm:g} mm", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_ik_explainer_draw.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/draw.py tests/test_ik_explainer_draw.py
git commit -m "feat(ik-explainer): shared drawing helpers on the repo palette

Helpers return copies because acts composite one base frame at several alphas
and in-place drawing would smear across fades; a test pins that. Non-finite
keypoints are skipped rather than crashing the render mid-sequence."
```

---

### Task 9: Act 1 — seven views

**Files:**
- Create: `scripts/viz/ik_explainer/acts/__init__.py`
- Create: `scripts/viz/ik_explainer/acts/act1_views.py`

**Interfaces:**
- Consumes: `02_kp2d.npz`, `clip_io`, `draw`.
- Produces: `<CLIP>/ik_explainer/frames/act1_views/f%05d.png`, 1920×1080, 450 frames (15 s @ 30 fps).

- [ ] **Step 1: Write the act**

Structure: a 4×2 grid of fly-centred crops (7 panels + 1 title cell) at 1920×1080. Over 450 output frames:
- frames 0–89: video only, panels labelled `Cam2012857  elev -0.6 deg` etc. (elevations from `clip_io.camera_view_dirs`).
- frames 90–209: keypoints fade in, `alpha = (f - 90) / 120`, low-confidence markers drawn dimmer via `draw_keypoints(..., conf=...)`.
- frames 210–449: full overlay, video advancing.

Source video frame for output frame `f` is `int(f * N / 450)` so the whole 921-frame clip spans the act.

Burn in: `800 fps -> 1/27 speed` and a 1 mm scale bar via `draw.scale_bar_mm(img, px_per_mm=80.7)`.

- [ ] **Step 2: Render and count frames**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/acts/act1_views.py
ls /data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46/ik_explainer/frames/act1_views | wc -l
```
Expected: 450.

- [ ] **Step 3: READ three sampled frames and report**

```
EXPECTATION: at f=0 video only, no markers. At f=150 markers ~50% faded in on
all 7 panels. At f=400 full overlay with markers ON the fly in every panel,
camera names and elevations legible, 1 mm scale bar present.
FALSIFICATION: markers present at f=0 (fade math inverted), or markers off the
fly in some panel (crop centre wrong for that camera).
```
Open `f00000.png`, `f00150.png`, `f00400.png` with the Read tool and report against this.

- [ ] **Step 4: Commit**

```bash
git add scripts/viz/ik_explainer/acts/__init__.py scripts/viz/ik_explainer/acts/act1_views.py
git commit -m "feat(ik-explainer): Act 1, seven views with keypoints fading in

Panels carry camera name and true elevation, and low-confidence markers are
drawn dimmer so per-view occlusion failure is visible -- that is the setup for
why Act 2 needs seven cameras."
```

---

### Task 10: Act 2 — triangulation

**Files:**
- Create: `scripts/viz/ik_explainer/acts/act2_triangulate.py`

**Interfaces:**
- Consumes: `02_kp2d.npz`, `03_kp3d.npz`, `clip_io.camera_view_dirs`, `draw`.
- Produces: `frames/act2_triangulate/f%05d.png`, 300 frames (10 s).

**Geometry (from the spec — do not invent it):** the 7 optical axes form a 180° arc at 30° spacing. Panels fly from the Act 1 grid to their true orientations. Rays are **parallel**, not converging — the rig is telecentric and the cameras sit at infinity. Panel *distance* is arbitrary staging; panel *direction* is real. Burn that caveat on screen.

- [ ] **Step 1: Write the act**

Use a simple hand-rolled 3D→2D camera (perspective, for the *presentation* view only — this is the viewer's camera, not a rig camera) to place:
- each panel as a textured quad at radius `R` along `-view_dir`, oriented to face the origin,
- the 3D keypoint cloud at the origin from `03_kp3d.npz`, centred on the fly's centroid,
- rays from each panel's keypoints along `view_dir`, drawn as parallel line segments.

Timeline over 300 frames: 0–119 panels fly out (ease-in-out); 120–199 rays extend; 200–259 cloud materialises (alpha ramp); 260–299 panels fade, cloud remains.

- [ ] **Step 2: Render and READ**

```
EXPECTATION: panels end on a visible 180 deg arc at even 30 deg spacing; the
rays are PARALLEL within each panel's bundle (never fanning to a point); the
cloud appears exactly where the bundles intersect and is fly-shaped (a
recognisable head-thorax-abdomen with six legs), not a blob.
FALSIFICATION: rays converging to a point => a perspective camera model leaked
into the rig geometry; cloud not at the intersection => wrong 3D or wrong
transform.
```
Open frames 0, 150, 299 and report.

- [ ] **Step 3: Commit**

```bash
git add scripts/viz/ik_explainer/acts/act2_triangulate.py
git commit -m "feat(ik-explainer): Act 2, panels fly to true rig geometry

Panel orientations are the optical axes recovered by factoring the DLTs (a 180
deg arc at 30 deg spacing), and rays are drawn parallel because the rig is
telecentric. The on-screen caveat states that panel distance is staging while
direction is real."
```

---

### Task 11: Act 3 — scale and root alignment

**Files:**
- Create: `scripts/viz/ik_explainer/acts/act3_align.py`

**Interfaces:**
- Consumes: `06_stages.npz` (`qpos_default`, `qpos_scaled`, `qpos_root`, `residual_mm`), `04_kp3d_filt.npz`.
- Produces: `frames/act3_align/f%05d.png`, 600 frames (20 s).

- [ ] **Step 1: Write the act**

Render with MuJoCo (`models/fruitfly_v1/fruitfly_v1_free.xml`, camera `hero`), time frozen on one keypoint frame, viewer camera slowly orbiting (azimuth sweep ~40° across the act).

Timeline: 0–119 mesh fades in at `qpos_default` beside the cloud (visibly wrong size and place); 120–299 interpolate `qpos_default → qpos_scaled` with the scale factor on screen in mm; 300–539 interpolate `qpos_scaled → qpos_root`; 540–599 hold.

Burn in the running residual in mm, interpolated between the recorded stage values, plus the stage name (`rescale`, `root_optimization`).

**Interpolating qpos:** the root free joint's quaternion (`qpos[3:7]`) must be SLERPed, not linearly interpolated — linear interpolation of quaternions produces a non-unit quaternion and a visibly wrong tumble. Interpolate `qpos[:3]` linearly, `qpos[3:7]` by SLERP, `qpos[7:]` linearly.

- [ ] **Step 2: Render and READ**

```
EXPECTATION: at f=0 mesh is clearly the wrong size and offset from the cloud.
By f=299 it is the right size but still misplaced. By f=539 it sits on the
cloud, ANATOMICALLY ORIENTED -- head end at the head keypoints. Residual
decreases monotonically across the act.
FALSIFICATION: mesh settles 180 deg flipped (head on the abdomen keypoints)
while the residual still falls -- a flipped fit can score similarly on a
roughly symmetric marker set, which is exactly why this is checked by eye.
```
Open frames 0, 299, 539 and report.

- [ ] **Step 3: Commit**

```bash
git add scripts/viz/ik_explainer/acts/act3_align.py
git commit -m "feat(ik-explainer): Act 3, body scale then root alignment

Shows the solver's real rescale and root_optimization stages with the residual
in mm on screen. Root quaternion is SLERPed rather than lerped, since a lerped
quaternion denormalises and tumbles visibly."
```

---

### Task 12: Act 4 — joint solve, then motion

**Files:**
- Create: `scripts/viz/ik_explainer/acts/act4_solve.py`

**Interfaces:**
- Consumes: `06_stages.npz` (`qpos_root`, `qpos_pose`, `qpos_seq`, `residual_mm`), `04_kp3d_filt.npz`, `02_kp2d.npz`, `clip_io`.
- Produces: `frames/act4_solve/f%05d.png`, 900 frames (30 s).

- [ ] **Step 1: Write the act**

Timeline: 0–239 interpolate `qpos_root → qpos_pose` (joints bend onto the keypoints), residual falling; 240–779 play `qpos_seq` with the keypoint cloud advancing in step; 780–899 the closing 2-up — MuJoCo fit on the left, one real camera view with the fit reprojected on the right.

For the 2-up, reproject the FK'd marker sites through that camera's DLT with `clip_io.project` and draw them in green (`PALETTE["fit"]`) over the video, with observed 2D in cyan for contrast.

- [ ] **Step 2: Render and READ**

```
EXPECTATION: by f=239 the mesh limbs lie along the keypoint chains. Through
240-779 tarsal tips TRACK the observed keypoints during leg swing -- the mesh
foot stays on the cyan marker as the leg moves, not merely near it. In the
closing 2-up the green reprojected fit overlies the fly in the real video.
FALSIFICATION: tips detaching during swing => pose stage not converged, or the
filter stage was skipped (raw *_TaTip jumps mm frame to frame and STAC chases
the outlier).
```
Open frames 239, 500, 850 and report.

- [ ] **Step 3: Commit**

```bash
git add scripts/viz/ik_explainer/acts/act4_solve.py
git commit -m "feat(ik-explainer): Act 4, joint solve then playback

Closes on a 2-up with the FK'd markers reprojected through the clip's DLT onto
real video, returning to where Act 1 began."
```

---

### Task 13: `assemble.py` — mp4, README, manifest

**Files:**
- Create: `scripts/viz/ik_explainer/assemble.py`

**Interfaces:**
- Consumes: all four `frames/act*/` sequences.
- Produces: `<CLIP>/ik_explainer/ik_explainer.mp4`, `README.md`, `manifest.json`.

- [ ] **Step 1: Write the assembler**

Concatenate the four sequences in order with 15-frame crossfades between acts (`draw.fade`), encode H.264 at 30 fps, 1920×1080, via `imageio` with `ffmpeg`. Assert the total frame count equals `450 + 300 + 600 + 900 - 3*15 = 2205` (~73.5 s) and fail loudly if any act directory is short — a truncated act must not silently ship.

Write `README.md` describing what each directory holds and the exact commands to regenerate, and `manifest.json` recording: repo commit (`git rev-parse HEAD`), detector checkpoint path, anatomy config, SAM3 prompt/version, per-stage wall times, `N`, and the four stage residuals in mm.

- [ ] **Step 2: Assemble and verify**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/assemble.py
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate,codec_name -of csv=p=0 \
  /data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46/ik_explainer/ik_explainer.mp4
```
Expected: `1920,1080,2205,30/1,h264`.

- [ ] **Step 3: READ sampled frames from the final mp4**

Extract one frame per act and open each with the Read tool:
```bash
CLIP=/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46
for f in 200 550 1000 1900; do
  ffmpeg -y -v error -i $CLIP/ik_explainer/ik_explainer.mp4 \
    -vf "select=eq(n\,$f)" -vframes 1 /tmp/ikx_$f.png
done
```
Report what each shows and confirm the four acts appear in order with no black or duplicated frames at the crossfades.

- [ ] **Step 4: Verify the full tree**

Run:
```bash
find /data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46/ik_explainer \
  -maxdepth 2 -not -path '*/frames/act*/*' | sort
du -sh /data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46/ik_explainer
```
Expected: matches the spec's on-disk layout — `README.md`, `manifest.json`, `bouts.csv`, `predictions/01..06`, `qc/01..05 + qc.json`, `frames/act1..4`, `ik_explainer.mp4`. Confirm the raw inputs (`calibration/`, `Cam*.mp4`, `enhanced/`, `preview/`, `data3D_*.csv`) are unchanged.

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=third_party/jarvis_jax JAX_PLATFORMS=cpu python -m pytest tests/test_ik_explainer_*.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/ik_explainer/assemble.py
git commit -m "feat(ik-explainer): assemble the four acts into the deliverable

Asserts the exact expected frame count so a truncated act fails loudly rather
than silently shipping a short video. manifest.json records the repo commit,
checkpoint, configs and per-stage residuals so a rendered video can be traced
back to what produced it."
```

---

## Verification Summary

Gates that must be passed by eye, not by number alone (each corresponds to a spec row):

| Gate | Task | Stop condition |
|---|---|---|
| Masks: one fly-sized blob per view | 4 | any view empty, or mask on the wall |
| 2D: head markers on the head, chains clean | 5 | group-scrambled anatomy ⇒ reorder broken |
| 3D vs baseline: uniform small disagreement | 6 | group-clustered offsets ⇒ permutation wrong |
| Filter: `*_TaTip` max accel drops | 6 | no drop ⇒ filter not applied |
| IK stages: residual monotone, mesh not flipped | 7 | 180° flip invalidates Acts 3–4 |
| Act 2: rays parallel, cloud at intersection | 10 | converging rays ⇒ perspective leaked in |
| Act 4: tarsal tips track through swing | 12 | tips detach ⇒ pose stage not converged |
