# Visualization Centralization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the scattered viz scripts with one importable top-level `viz/` package — a shared reprojection/overlay core plus a unified `python -m viz <subcommand>` CLI — covering both the original STAC pipeline and the new courtship mesh/silhouette pipeline.

**Architecture:** A pipeline-agnostic `viz/core/` (reproject, overlays, io, layout, colors) carries the reusable engine; `viz/views/` holds one module per CLI subcommand built on the core; `viz/cli.py` + `viz/__main__.py` dispatch. The package sits at the repo top level (peer to `scripts/`, `configs/`) so it can import both `jarvis_jax` and `stac_mjx` without coupling them.

**Tech Stack:** Python, numpy, OpenCV (`cv2`), matplotlib, `jarvis_jax` (`ReprojectionTool`, `cse.courtship_bout_masks`), `stac_mjx.io_dict_to_hdf5` (h5 load), `stac_mjx.viz` (fit-check only). Tests: pytest.

## Global Constraints

- Package is top-level `viz/`, importable as `viz.core.*` / `viz.views.*`, runnable as `python -m viz <subcommand> …`.
- The core is pipeline-agnostic: pure geometry + cv2/matplotlib + artifact loaders. NO pipeline-specific logic in `viz/core/`; pipeline specifics live in `viz/views/`.
- Reprojection uses the ReprojectionTool convention verbatim: `ph = [pts,1] (N,4); proj = ph @ M (N,3); uv = proj[:,:2]/proj[:,2:3]` where `M` is the `(4,3)` camera matrix. Do NOT reinvent it — this matches `scripts/run_bout.py::project_points`.
- Reuse, don't reimplement: camera matrices via `jarvis_jax.geometry.reprojection_tool.ReprojectionTool(calib_dir).camera_matrices` `(num_cam,4,3)`; SAM masks via `jarvis_jax.tracking.bout_masks.load_bout_masks(npz_path, fly, expected_cameras=...)`.
- Every core module is unit-tested; views are smoke-tested only (produce an output file without error), matching how the pipeline treats heavy renders.
- Run pytest from repo root: `cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset && OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest viz/tests/ -q`. (JAX_PLATFORMS=cpu keeps core/io tests off the GPU; the `fit-check` smoke needs a GPU and is skipped when none is present.)
- Reference implementations (source for the promoted views) live in `docs/plans/viz-reference/*.py` (preserved from the debugging session; see its README). Treat them as behavior references to rewrite on the core, not files to copy verbatim.
- Commit after each task. Do NOT `git add -A`; add explicit paths. End commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

```
viz/
├── __init__.py          # empty (marks package)
├── __main__.py          # delegates to viz.cli.main()
├── cli.py               # argparse subcommands -> view entrypoints
├── core/
│   ├── __init__.py
│   ├── colors.py        # BGR palette + keypoint semantic groups + per-leg chains derived from a KP_NAMES list
│   ├── reproject.py     # camera_matrices(calib_dir); project(cam_mat, pts3d); reproject_all(cam_mats, pts3d)
│   ├── overlays.py      # cv2 draw primitives on a BGR image
│   ├── io.py            # artifact loaders + run-path resolution + frame reading
│   └── layout.py        # montage grid, crop-to-points, banner
├── views/
│   ├── __init__.py
│   ├── overlay.py       # `viz overlay` (+ --bodyalign)
│   ├── legskel.py       # `viz legskel`
│   ├── kp_qc.py         # `viz kp-qc`
│   ├── fit_check.py     # `viz fit-check`
│   ├── reproj_video.py  # `viz reproj-video`
│   └── clip.py          # `viz clip cut|stack|render`
└── tests/
    ├── __init__.py
    ├── test_colors.py
    ├── test_reproject.py
    ├── test_overlays.py
    ├── test_io.py
    ├── test_layout.py
    └── test_cli.py
```

---

### Task 1: Package scaffold + `core.colors` + `core.reproject`

**Files:**
- Create: `viz/__init__.py` (empty), `viz/core/__init__.py` (empty), `viz/tests/__init__.py` (empty)
- Create: `viz/core/colors.py`, `viz/core/reproject.py`
- Test: `viz/tests/test_colors.py`, `viz/tests/test_reproject.py`

**Interfaces:**
- Produces:
  - `viz.core.colors.PALETTE: dict[str,tuple]` BGR colors with keys `fly0, fly1, head, tail, detector, fit, mask, mesh`.
  - `viz.core.colors.keypoint_groups(kp_names: list[str]) -> dict[str, list[int]]` with keys `head` (Antenna/Eye), `thorax` (Scutellum/Wing*), `abdomen` (Abd*), `legs` (all `T[1-3][LR]_*`).
  - `viz.core.colors.leg_chains(kp_names: list[str]) -> dict[str, list[int]]` — one proximal→distal index chain per leg `T1L,T2L,T3L,T1R,T2R,T3R` using segment order `[ThxCx,Tro,FeTi,TiTa,TaT1,TaT3,TaTip]` (only segments present for that leg).
  - `viz.core.reproject.camera_matrices(calib_dir: str) -> tuple[np.ndarray, list[str]]` returns `(cam_mats (C,4,3) float32, camera_names)`.
  - `viz.core.reproject.project(cam_mat_4x3, pts3d) -> np.ndarray` `(N,2)` (empty-safe).
  - `viz.core.reproject.reproject_all(cam_mats, pts3d) -> np.ndarray` `(C,N,2)`.

- [ ] **Step 1: Write failing tests**

```python
# viz/tests/test_colors.py
from viz.core import colors
KP = ["Scutellum","WingL_base","WingR_base","Antenna_Base","EyeL","EyeR",
      "WingL_V12","WingL_V13","WingR_V12","WingR_V13","Abd_A4","Abd_tip",
      "T1L_ThxCx","T1L_Tro","T1L_FeTi","T1L_TiTa","T1L_TaT1","T1L_TaT3","T1L_TaTip",
      "T2L_Tro","T2L_FeTi","T2L_TiTa","T2L_TaT1","T2L_TaT3","T2L_TaTip"]

def test_palette_has_core_keys():
    for k in ("fly0","fly1","head","tail","detector","fit","mask","mesh"):
        assert k in colors.PALETTE and len(colors.PALETTE[k]) == 3

def test_keypoint_groups_partition():
    g = colors.keypoint_groups(KP)
    assert KP.index("EyeL") in g["head"] and KP.index("Antenna_Base") in g["head"]
    assert KP.index("Abd_tip") in g["abdomen"]
    assert KP.index("Scutellum") in g["thorax"] and KP.index("WingL_base") in g["thorax"]
    assert KP.index("T1L_TaTip") in g["legs"] and KP.index("T2L_Tro") in g["legs"]

def test_leg_chains_proximal_to_distal():
    chains = colors.leg_chains(KP)
    t1l = chains["T1L"]
    assert [KP[i] for i in t1l] == ["T1L_ThxCx","T1L_Tro","T1L_FeTi","T1L_TiTa","T1L_TaT1","T1L_TaT3","T1L_TaTip"]
    # T2L has no ThxCx segment present -> starts at Tro
    assert [KP[i] for i in chains["T2L"]] == ["T2L_Tro","T2L_FeTi","T2L_TiTa","T2L_TaT1","T2L_TaT3","T2L_TaTip"]
```

```python
# viz/tests/test_reproject.py
import numpy as np
from viz.core import reproject

def test_project_matches_ph_at_M_convention():
    # M (4,3): pick a simple orthographic-ish matrix; verify uv = (ph@M)[:2]/[2]
    M = np.array([[2.,0.,0.],[0.,3.,0.],[0.,0.,0.],[10.,20.,1.]], np.float32)  # (4,3)
    pts = np.array([[1.,1.,5.],[0.,0.,0.]], np.float64)
    ph = np.concatenate([pts, np.ones((2,1))],1); proj = ph @ M
    expect = (proj[:,:2]/proj[:,2:3])
    got = reproject.project(M, pts)
    assert np.allclose(got, expect, atol=1e-4) and got.shape == (2,2)

def test_project_empty_safe():
    M = np.eye(4,3, dtype=np.float32)
    assert reproject.project(M, np.zeros((0,3))).shape == (0,2)

def test_reproject_all_stacks_per_camera():
    M = np.array([[2.,0.,0.],[0.,3.,0.],[0.,0.,0.],[10.,20.,1.]], np.float32)
    cam_mats = np.stack([M, M*1.0])
    out = reproject.reproject_all(cam_mats, np.array([[1.,1.,5.]]))
    assert out.shape == (2,1,2)
```

- [ ] **Step 2: Run tests, verify they fail** — `python -m pytest viz/tests/test_colors.py viz/tests/test_reproject.py -q` → FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# viz/core/colors.py
"""Single visual language for all viz: BGR palette + keypoint semantics.
Colours are BGR (cv2). Groups/chains are derived from a KP_NAMES list so they
work for any keypoint ordering."""
import re

PALETTE = {
    "fly0": (255, 255, 0),   # cyan
    "fly1": (0, 165, 255),   # orange
    "head": (0, 0, 255),     # red
    "tail": (255, 0, 0),     # blue
    "detector": (255, 255, 0),
    "fit": (0, 255, 0),      # green
    "mask": (200, 200, 200), # grey fill
    "mesh": (200, 200, 200),
}

def keypoint_groups(kp_names):
    g = {"head": [], "thorax": [], "abdomen": [], "legs": []}
    for i, n in enumerate(kp_names):
        if re.match(r"T[1-3][LR]_", n):        g["legs"].append(i)
        elif n.startswith(("Antenna", "Eye")): g["head"].append(i)
        elif n.startswith("Abd"):              g["abdomen"].append(i)
        elif n.startswith(("Scutellum", "Wing")): g["thorax"].append(i)
    return g

_SEGS = ["ThxCx", "Tro", "FeTi", "TiTa", "TaT1", "TaT3", "TaTip"]

def leg_chains(kp_names):
    idx = {n: i for i, n in enumerate(kp_names)}
    out = {}
    for leg in ("T1L", "T2L", "T3L", "T1R", "T2R", "T3R"):
        chain = [idx[f"{leg}_{s}"] for s in _SEGS if f"{leg}_{s}" in idx]
        if chain:
            out[leg] = chain
    return out
```

```python
# viz/core/reproject.py
"""3D(mm)->2D(px) reprojection, ReprojectionTool `ph @ M` convention.
M is a (4,3) camera matrix; ph=[pts,1]. Verbatim match to
scripts/run_bout.py::project_points."""
import numpy as np
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

def camera_matrices(calib_dir):
    rt = ReprojectionTool(calib_dir)
    names = [c.name for c in rt._camera_list]
    return np.asarray(rt.camera_matrices, np.float32), names

def project(cam_mat_4x3, pts3d):
    pts = np.asarray(pts3d, np.float64)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), np.float32)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)   # (N,4)
    proj = ph @ np.asarray(cam_mat_4x3, np.float64)                  # (N,3)
    return (proj[:, :2] / proj[:, 2:3]).astype(np.float32)

def reproject_all(cam_mats, pts3d):
    return np.stack([project(m, pts3d) for m in np.asarray(cam_mats)])
```

- [ ] **Step 4: Run tests, verify pass** — `python -m pytest viz/tests/test_colors.py viz/tests/test_reproject.py -q` → PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add viz/__init__.py viz/core/__init__.py viz/tests/__init__.py viz/core/colors.py viz/core/reproject.py viz/tests/test_colors.py viz/tests/test_reproject.py
git commit -m "feat(viz): package scaffold + colors + reproject core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `core.overlays` (cv2 draw primitives)

**Files:**
- Create: `viz/core/overlays.py`
- Test: `viz/tests/test_overlays.py`

**Interfaces:**
- Consumes: `viz.core.colors.PALETTE`.
- Produces (all take a BGR `img` `(H,W,3) uint8`, mutate + return it):
  - `draw_points(img, uv, color, radius=3) -> img` (skips non-finite / out-of-bounds)
  - `draw_chain(img, uv_list, color, thickness=1) -> img` (polyline through finite points + dots)
  - `draw_cloud(img, uv, color, radius=1) -> img` (mesh point splat)
  - `draw_mask(img, mask_bool, color, alpha=0.35, outline=True) -> img` (translucent fill + contour)
  - `draw_axis(img, tail_uv, head_uv, color) -> img` (arrow tail→head)
  - `legend(img, items) -> img` where `items: list[(text, color)]`

- [ ] **Step 1: Write failing tests**

```python
# viz/tests/test_overlays.py
import numpy as np
from viz.core import overlays

def _img(): return np.zeros((32, 32, 3), np.uint8)

def test_draw_points_marks_pixels_finite_only():
    img = _img()
    out = overlays.draw_points(img.copy(), np.array([[10.,10.],[np.nan,5.]]), (0,0,255), radius=2)
    assert out[10,10].tolist() == [0,0,255]           # drawn
    assert int((out[:,:,2] > 0).sum()) > 0            # some red
    assert out.shape == (32,32,3)

def test_draw_mask_fill_keeps_shape_and_tints():
    img = _img(); m = np.zeros((32,32), bool); m[4:8,4:8] = True
    out = overlays.draw_mask(img, m, (200,200,200), alpha=0.5, outline=True)
    assert out.shape == (32,32,3) and int(out[6,6].sum()) > 0

def test_draw_chain_connects_points():
    img = _img()
    out = overlays.draw_chain(img, [(2,2),(20,20)], (0,255,0), thickness=1)
    assert int((out[:,:,1] > 0).sum()) > 2            # a line of green
```

- [ ] **Step 2: Run, verify fail** — `python -m pytest viz/tests/test_overlays.py -q` → FAIL.

- [ ] **Step 3: Implement**

```python
# viz/core/overlays.py
"""cv2 overlay primitives on a BGR uint8 image. Each draws in place + returns
the image. Non-finite / out-of-bounds points are skipped."""
import cv2
import numpy as np

def _pt(u, v):
    return (int(round(u)), int(round(v)))

def draw_points(img, uv, color, radius=3):
    for p in np.asarray(uv, float).reshape(-1, 2):
        if np.isfinite(p).all():
            cv2.circle(img, _pt(*p), radius, color, -1)
    return img

def draw_chain(img, uv_list, color, thickness=1):
    pts = [_pt(u, v) for (u, v) in uv_list if np.isfinite([u, v]).all()]
    for a, b in zip(pts[:-1], pts[1:]):
        cv2.line(img, a, b, color, thickness)
    for p in pts:
        cv2.circle(img, p, max(2, thickness + 1), color, -1)
    return img

def draw_cloud(img, uv, color, radius=1):
    return draw_points(img, uv, color, radius=radius)

def draw_mask(img, mask_bool, color, alpha=0.35, outline=True):
    m = np.asarray(mask_bool, bool)
    if m.any():
        ov = img.copy(); ov[m] = color
        img = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
        if outline:
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, color, 1)
    return img

def draw_axis(img, tail_uv, head_uv, color):
    if np.isfinite(tail_uv).all() and np.isfinite(head_uv).all():
        cv2.arrowedLine(img, _pt(*tail_uv), _pt(*head_uv), color, 2, tipLength=0.3)
    return img

def legend(img, items):
    for i, (text, color) in enumerate(items):
        cv2.putText(img, text, (5, 18 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return img
```

- [ ] **Step 4: Run, verify pass** — `python -m pytest viz/tests/test_overlays.py -q` → PASS.

- [ ] **Step 5: Commit** — `git add viz/core/overlays.py viz/tests/test_overlays.py && git commit -m "feat(viz): cv2 overlay primitives core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 3: `core.io` (artifact loaders + run paths + frames)

**Files:**
- Create: `viz/core/io.py`
- Test: `viz/tests/test_io.py`

**Interfaces:**
- Produces:
  - `fly_dir(run_root, bout, fly) -> str` = `{run_root}/bouts/bout_{bout:05d}/fly{fly}`
  - `load_outputs(run_root, bout, fly) -> dict` with `kp3d_mm (T,K,3)`, `mesh_mm (T,M,3)`, `kp_names (list)` (via `stac_mjx.io_dict_to_hdf5.load`)
  - `load_kp2d(run_root, bout, fly) -> tuple(kp2d (T,C,K,2), conf (T,C,K))`
  - `load_kp3d(run_root, bout, fly) -> tuple(kp3d (T,K,3), conf3d (T,K))`
  - `load_masks(predictions_dir, bout, fly, cameras) -> dict` (wraps `jarvis_jax.tracking.bout_masks.load_bout_masks`, `expected_cameras=cameras`)
  - `read_frame(video_path, frame_idx) -> np.ndarray (H,W,3) BGR` and `read_frames(session_dir, cameras, start, count) -> generator of (C,H,W,3) RGB` (thin wrappers over cv2.VideoCapture; mirror `scripts/run_bout.py::open_video_captures/all_cams_frames`)

- [ ] **Step 1: Write failing tests** (use a tiny synthetic outputs.h5 + kp2d.npz written in the test):

```python
# viz/tests/test_io.py
import numpy as np, os
from viz.core import io
import stac_mjx.io_dict_to_hdf5 as ioh5

def test_fly_dir_format():
    assert io.fly_dir("/run", 3, 1).endswith("/bouts/bout_00003/fly1")

def test_load_outputs_and_kp2d(tmp_path):
    d = tmp_path/"bouts"/"bout_00001"/"fly0"; d.mkdir(parents=True)
    ioh5.save(str(d/"outputs.h5"), {"kp3d_mm": np.zeros((2,50,3),np.float32),
                                    "mesh_mm": np.zeros((2,300,3),np.float32),
                                    "kp_names": np.array(["a","b"])})
    np.savez(d/"kp2d.npz", kp2d=np.zeros((2,7,50,2),np.float32), conf=np.ones((2,7,50),np.float32))
    o = io.load_outputs(str(tmp_path), 1, 0)
    assert o["kp3d_mm"].shape == (2,50,3) and o["mesh_mm"].shape == (2,300,3)
    kp2d, conf = io.load_kp2d(str(tmp_path), 1, 0)
    assert kp2d.shape == (2,7,50,2) and conf.shape == (2,7,50)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `viz/core/io.py`:

```python
"""Artifact loaders + run-path resolution + frame reading for viz views."""
import os
import numpy as np
import cv2
import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.tracking.bout_masks import load_bout_masks

def fly_dir(run_root, bout, fly):
    return os.path.join(run_root, "bouts", f"bout_{int(bout):05d}", f"fly{int(fly)}")

def load_outputs(run_root, bout, fly):
    d = ioh5.load(os.path.join(fly_dir(run_root, bout, fly), "outputs.h5"))
    return {"kp3d_mm": np.asarray(d["kp3d_mm"]), "mesh_mm": np.asarray(d["mesh_mm"]),
            "kp_names": [str(n) for n in np.asarray(d["kp_names"]).tolist()]}

def load_kp2d(run_root, bout, fly):
    with np.load(os.path.join(fly_dir(run_root, bout, fly), "kp2d.npz")) as z:
        return np.asarray(z["kp2d"]), np.asarray(z["conf"])

def load_kp3d(run_root, bout, fly):
    with np.load(os.path.join(fly_dir(run_root, bout, fly), "kp3d.npz")) as z:
        return np.asarray(z["kp3d"]), np.asarray(z["conf3d"])

def load_masks(predictions_dir, bout, fly, cameras):
    npz = os.path.join(predictions_dir, f"bout_{int(bout):05d}", "sam3_masks.npz")
    return load_bout_masks(npz, fly, expected_cameras=list(cameras))

def read_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, bgr = cap.read(); cap.release()
    if not ok:
        raise IndexError(f"frame {frame_idx} not readable in {video_path}")
    return bgr

def read_frames(session_dir, cameras, start, count):
    caps = [cv2.VideoCapture(os.path.join(session_dir, f"{c}.mp4")) for c in cameras]
    try:
        for cap in caps: cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        for _ in range(int(count)):
            imgs = []
            for cap in caps:
                ok, bgr = cap.read()
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if ok else None)
            yield imgs
    finally:
        for cap in caps: cap.release()
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git add viz/core/io.py viz/tests/test_io.py && git commit -m "feat(viz): artifact loaders + frame reading core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 4: `core.layout` (montage / crop / banner)

**Files:**
- Create: `viz/core/layout.py`
- Test: `viz/tests/test_layout.py`

**Interfaces:**
- Produces:
  - `crop_to_points(img, uv, pad=55) -> (img_crop, (x0,y0))` — bounding box of finite `uv` + pad, clipped to image.
  - `montage(tiles, cols=2) -> np.ndarray` — pad tiles to a common size, arrange row-major into a grid (fills missing cells with black).
  - `banner(width, items, height=None) -> np.ndarray` — a black strip with a colour legend (uses `overlays.legend`).

- [ ] **Step 1: Write failing tests**

```python
# viz/tests/test_layout.py
import numpy as np
from viz.core import layout

def test_montage_grid_dims():
    tiles = [np.zeros((10,20,3),np.uint8), np.zeros((12,18,3),np.uint8), np.zeros((8,8,3),np.uint8)]
    m = layout.montage(tiles, cols=2)
    # 3 tiles, 2 cols -> 2 rows; cell = max h(12) x max w(20)
    assert m.shape == (2*12, 2*20, 3)

def test_crop_to_points_bounds():
    img = np.zeros((100,100,3),np.uint8)
    crop, (x0,y0) = layout.crop_to_points(img, np.array([[40.,40.],[60.,60.]]), pad=5)
    assert crop.shape[0] <= 100 and x0 >= 0 and y0 >= 0 and crop.shape[0] > 0
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `viz/core/layout.py`:

```python
"""Tile arrangement + cropping for multi-camera viz montages."""
import numpy as np
from viz.core import overlays

def crop_to_points(img, uv, pad=55):
    p = np.asarray(uv, float).reshape(-1, 2)
    p = p[np.isfinite(p).all(1)]
    if len(p) == 0:
        return img, (0, 0)
    x0, y0 = np.maximum(p.min(0) - pad, 0).astype(int)
    x1, y1 = (p.max(0) + pad).astype(int)
    x1 = min(img.shape[1], x1); y1 = min(img.shape[0], y1)
    return img[y0:y1, x0:x1].copy(), (int(x0), int(y0))

def montage(tiles, cols=2):
    if not tiles:
        return np.zeros((1, 1, 3), np.uint8)
    h = max(t.shape[0] for t in tiles); w = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        grid[r * h:r * h + t.shape[0], c * w:c * w + t.shape[1]] = t
    return grid

def banner(width, items, height=None):
    height = height or (len(items) * 18 + 10)
    b = np.zeros((height, width, 3), np.uint8)
    return overlays.legend(b, items)
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git add viz/core/layout.py viz/tests/test_layout.py && git commit -m "feat(viz): montage/crop/banner layout core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 5: CLI skeleton + dispatch

**Files:**
- Create: `viz/cli.py`, `viz/__main__.py`, `viz/views/__init__.py`
- Test: `viz/tests/test_cli.py`

**Interfaces:**
- Consumes: view entrypoints (added in later tasks) `viz.views.<name>.run(args)`. Until those exist, `cli.py` imports them lazily inside each subcommand handler so the CLI parses even with views stubbed.
- Produces:
  - `viz.cli.build_parser() -> argparse.ArgumentParser` with subcommands `overlay, legskel, kp-qc, fit-check, reproj-video, clip` and shared options `--run, --bout, --fly, --frame, --cams, --out` registered on the relevant subcommands.
  - `viz.cli.main(argv=None) -> int` — parse + dispatch to `args.func(args)`.

- [ ] **Step 1: Write failing test**

```python
# viz/tests/test_cli.py
from viz import cli

def test_parser_has_all_subcommands():
    p = cli.build_parser()
    sub = next(a for a in p._actions if a.__class__.__name__ == "_SubParsersAction")
    for name in ("overlay","legskel","kp-qc","fit-check","reproj-video","clip"):
        assert name in sub.choices

def test_overlay_parses_shared_flags():
    p = cli.build_parser()
    ns = p.parse_args(["overlay","--run","/r","--bout","1","--fly","0","--frame","250"])
    assert ns.run == "/r" and ns.bout == 1 and ns.fly == 0 and ns.frame == 250
    assert ns.func == "_overlay"   # name-based dispatch handle

def test_main_dispatches(monkeypatch):
    called = {}
    import viz.cli as c
    monkeypatch.setattr(c, "_overlay", lambda args: called.setdefault("frame", args.frame) or 0)
    assert c.main(["overlay","--run","/r","--bout","1","--fly","0","--frame","7"]) == 0
    assert called["frame"] == 7   # main resolved c._overlay at call time (name-based)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** `viz/cli.py` (each `_<name>` handler lazy-imports its view so this task stands alone; views land in Tasks 6-11):

```python
"""Unified viz CLI:  python -m viz <subcommand> [options]."""
import argparse

def _add_shared(sp):
    sp.add_argument("--run", help="run root (…/Session0_bouts_<date>)")
    sp.add_argument("--bout", type=int)
    sp.add_argument("--fly", type=int, default=0)
    sp.add_argument("--frame", type=int, default=0)
    sp.add_argument("--cams", nargs="*", default=None)
    sp.add_argument("--out", default=None)

def _overlay(args):     from viz.views import overlay;      return overlay.run(args)
def _legskel(args):     from viz.views import legskel;      return legskel.run(args)
def _kp_qc(args):       from viz.views import kp_qc;        return kp_qc.run(args)
def _fit_check(args):   from viz.views import fit_check;    return fit_check.run(args)
def _reproj_video(args):from viz.views import reproj_video; return reproj_video.run(args)
def _clip(args):        from viz.views import clip;         return clip.run(args)

def build_parser():
    p = argparse.ArgumentParser(prog="viz", description="Centralized 3d_tracking visualizations")
    sub = p.add_subparsers(dest="cmd", required=True)

    o = sub.add_parser("overlay", help="mesh+kp+mask reproj overlay (per-camera montage)")
    _add_shared(o); o.add_argument("--show", default="mesh,kp,mask,axis")
    o.add_argument("--compare", default=None); o.add_argument("--bodyalign", action="store_true")
    o.set_defaults(func="_overlay")

    l = sub.add_parser("legskel", help="leg-joint chains: detector vs fitted vs triangulated")
    _add_shared(l); l.add_argument("--compare", default=None); l.set_defaults(func="_legskel")

    k = sub.add_parser("kp-qc", help="detector pred-vs-GT keypoints + per-kp error")
    k.add_argument("--run-dir"); k.add_argument("--ckpt"); k.add_argument("--recording")
    k.add_argument("--n", type=int, default=8); k.add_argument("--female-vs-male", action="store_true")
    k.add_argument("--out", default=None); k.set_defaults(func="_kp_qc")

    f = sub.add_parser("fit-check", help="STAC-fit verification frames")
    f.add_argument("ik_h5"); f.add_argument("--start", type=int, default=0)
    f.add_argument("--n", type=int, default=10); f.add_argument("--camera", default=None)
    f.add_argument("--out", default=None); f.set_defaults(func="_fit_check")

    r = sub.add_parser("reproj-video", help="predicted 3D reprojected onto a bout video")
    r.add_argument("--session-dir", required=True); r.add_argument("--pred-dir", required=True)
    r.add_argument("--bout", type=int, required=True); r.add_argument("--cameras", nargs="*", default=None)
    r.add_argument("--with-masks", action="store_true"); r.add_argument("--out", default=None)
    r.set_defaults(func="_reproj_video")

    c = sub.add_parser("clip", help="multi-camera clip cut|stack|render")
    c.add_argument("mode", choices=["cut", "stack", "render"])
    c.add_argument("--session-dir"); c.add_argument("--start", type=int); c.add_argument("--end", type=int)
    c.add_argument("--cameras", nargs="*", default=None); c.add_argument("--bout-dir")
    c.add_argument("--out", default=None); c.set_defaults(func="_clip")
    return p

def main(argv=None):
    # func is the NAME of a module-level handler; resolve at call time via globals()
    # so tests (and future edits) can monkeypatch a handler by attribute.
    args = build_parser().parse_args(argv)
    return int(globals()[args.func](args) or 0)
```

```python
# viz/__main__.py
from viz.cli import main
raise SystemExit(main())
```

```python
# viz/views/__init__.py   (empty)
```

Note: `test_main_dispatches` monkeypatches `viz.cli._overlay`; keep the `_overlay`… handlers as module-level names.

- [ ] **Step 4: Run, verify pass** — `python -m pytest viz/tests/test_cli.py -q` → PASS.
- [ ] **Step 5: Commit** — `git add viz/cli.py viz/__main__.py viz/views/__init__.py viz/tests/test_cli.py && git commit -m "feat(viz): unified CLI skeleton + subcommand dispatch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 6: `overlay` view (+ `--bodyalign`)

**Files:**
- Create: `viz/views/overlay.py`
- Test: `viz/tests/test_views_smoke.py` (create; add an overlay smoke test)

**Reference sources** (behavior to reproduce, rewritten on the core): scratchpad `viz_both_flies.py` (both-fly mesh cloud, per-cam montage), `viz_mask_orient.py` (SAM mask + mesh + head/tail axis), `viz_detector_headtail.py` (detector head/tail dots), `viz_legmask_overlay.py` (mask + detector + fit, 2×2), `viz_bodyalign.py` (per-frame body-aligned fit vs raw vs detector).

**Interfaces:**
- Consumes: `core.reproject.{camera_matrices,reproject_all,project}`, `core.overlays.*`, `core.io.{load_outputs,load_kp2d,load_masks,read_frame}`, `core.layout.{crop_to_points,montage,banner}`, `core.colors.{PALETTE,keypoint_groups,leg_chains}`. Needs the recording's `calib_dir`, `session_dir`, `predictions_dir`, `cameras`, `KP_NAMES` — resolve via a Hydra compose of `pipeline` (see below).
- Produces: `viz.views.overlay.run(args) -> int` writing a PNG to `args.out` (default `overlay_bout{bout}_fly{fly}_f{frame}.png`).

- [ ] **Step 1:** Add a config-resolution helper (used by overlay + legskel). Create `viz/core/config.py`:

```python
# viz/core/config.py
"""Resolve recording paths + KP_NAMES for the courtship pipeline via Hydra."""
import os
from hydra import initialize_config_dir, compose

_CFG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")

def courtship_recording(config_name="pipeline", overrides=None):
    os.environ.setdefault("USER", "eabe")
    with initialize_config_dir(version_base=None, config_dir=os.path.abspath(_CFG_DIR)):
        c = compose(config_name=config_name, overrides=overrides or [])
    return {
        "calib_dir": str(c.recording.calib_dir),
        "session_dir": str(c.recording.session_dir),
        "predictions_dir": str(c.recording.predictions_dir),
        "cameras": list(c.recording.cameras),
        "kp_names": list(c.model.KP_NAMES),
    }
```

- [ ] **Step 2: Write the overlay view.** `run(args)` must: resolve recording; `cam_mats, names = camera_matrices(calib_dir)`; load `outputs` (mesh_mm, kp3d_mm), `kp2d`, `masks` for `(bout, fly)`; for each camera read the frame at `bout_start_frame + args.frame` (compute start via the recording's `bouts_csv` — reuse `scripts/run_bout.py::bout_start_frame`; import it: `from scripts.run_bout import bout_start_frame` and pass the composed cfg); draw per `--show` tokens: `mask` → `draw_mask(mask[fr,ci])`, `mesh` → `draw_cloud(project(cam_mats[ci], mesh_mm[fr]))` in `PALETTE["fly{fly}"]`, `kp` → detector `draw_points(kp2d[fr,ci])`, `axis` → head-group centroid vs tail via `draw_axis`; `--compare RUN2` overlays a second run's fitted mesh in `PALETTE["fit"]`; `--bodyalign` computes a per-frame Umeyama on the body group mapping `kp3d_mm→kp3d` and draws the aligned sites (port `viz_bodyalign.py`'s `umeyama`). Crop each tile with `crop_to_points`, `montage(tiles)`, append `banner`. Save PNG.

  (Complete code: adapt `docs/plans/viz-reference/{viz_both_flies,viz_mask_orient,viz_bodyalign}.py`, replacing their inline `project_points`/cv2 calls with `core.reproject`/`core.overlays`/`core.layout`. Those files are the line-by-line reference.)

- [ ] **Step 3: Smoke test** `viz/tests/test_views_smoke.py`:

```python
import os, numpy as np, pytest
from viz.views import overlay

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_07052026"
skip = pytest.mark.skipif(not os.path.isdir(ROOT), reason="courtship run not present")

@skip
def test_overlay_writes_png(tmp_path):
    class A: pass
    a = A(); a.run=ROOT; a.bout=1; a.fly=1; a.frame=250; a.cams=None
    a.show="mesh,kp,mask,axis"; a.compare=None; a.bodyalign=False; a.out=str(tmp_path/"o.png")
    assert overlay.run(a) == 0 and os.path.exists(a.out) and os.path.getsize(a.out) > 0
```

- [ ] **Step 4: Run** — `python -m pytest viz/tests/test_views_smoke.py -k overlay -q` (skips if the run dir is absent; run on a machine where it exists) → PASS/skip.
- [ ] **Step 5: Commit** — `git add viz/core/config.py viz/views/overlay.py viz/tests/test_views_smoke.py && git commit -m "feat(viz): overlay view (mesh/kp/mask + bodyalign) on core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 7: `legskel` view

**Files:**
- Create: `viz/views/legskel.py`; Modify: `viz/tests/test_views_smoke.py` (add legskel smoke)

**Reference sources:** scratchpad `viz_legskel.py` (detector-2D chains + fitted-3D chains + triangulated), `viz_legcompare.py`.

**Interfaces:**
- Consumes: same core as Task 6 + `core.colors.leg_chains`, `core.io.load_kp3d`.
- Produces: `viz.views.legskel.run(args) -> int` writing `legskel_bout{bout}_fly{fly}_f{frame}.png`. Draws, per camera: detector 2D leg chains (`kp2d[fr,ci]`, `PALETTE["detector"]`), fitted leg chains (`project(cam_mats[ci], kp3d_mm[fr])`, `PALETTE["fit"]`), and if `--compare RUN2` the compare run's fitted chains in a distinct colour. Chains from `leg_chains(kp_names)`.

- [ ] **Step 1: Write legskel.run** (adapt `viz_legskel.py`; use `core.colors.leg_chains`, `core.overlays.draw_chain`, `core.reproject.project`).
- [ ] **Step 2: Smoke test** (mirror Task 6's, `overlay`→`legskel`, assert PNG written).
- [ ] **Step 3: Run** — `python -m pytest viz/tests/test_views_smoke.py -k legskel -q`.
- [ ] **Step 4: Commit** — `git add viz/views/legskel.py viz/tests/test_views_smoke.py && git commit -m "feat(viz): legskel view on core

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 8: `kp-qc` view (port `viz_keypoints` + `viz_courtship_mf`)

**Files:**
- Create: `viz/views/kp_qc.py`; Modify: `viz/tests/test_views_smoke.py`
- Reference: `scripts/viz_keypoints.py` (single recording, pred-vs-GT + per-kp bars), `scripts/viz_courtship_mf.py` (female-vs-male).

**Interfaces:**
- Produces: `viz.views.kp_qc.run(args) -> int`. `--female-vs-male` selects the paired-recording layout (port `viz_courtship_mf`); otherwise the single-recording layout (port `viz_keypoints`). Writes `viz_frames.png` + `viz_perkp.png` (or the `_courtship_` variants) under `args.out`. This view is matplotlib + the ViTPose model load (`jarvis_jax.convert.build_checkpoint.load_vitpose`); it does not use the cv2 core (keep as-is but move under `viz/views/`, sharing `core.colors` for the per-group colouring only).

- [ ] **Step 1:** Move the two scripts' bodies into `kp_qc.run(args)`, dispatching on `--female-vs-male`. Keep their model-eval + matplotlib code; drop their `argparse main()` (the CLI provides args).
- [ ] **Step 2: Smoke test** — skipif no `run-dir`/ckpt available; assert a PNG is written. (May require GPU for the model; mark `@pytest.mark.skipif` on GPU/ckpt presence.)
- [ ] **Step 3: Run / skip.**
- [ ] **Step 4: Commit** — `git add viz/views/kp_qc.py viz/tests/test_views_smoke.py && git commit -m "feat(viz): kp-qc view (port viz_keypoints + viz_courtship_mf)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 9: `fit-check` view (port `render_fit_check`)

**Files:**
- Create: `viz/views/fit_check.py`; Modify: `viz/tests/test_views_smoke.py`
- Reference: `scripts/render_fit_check.py` (uses `stac_mjx.viz.viz_stac`; needs `MUJOCO_GL=egl`).

**Interfaces:**
- Produces: `viz.views.fit_check.run(args) -> int` from `args.ik_h5, args.start, args.n, args.camera`. Port verbatim behavior; set the `MUJOCO_GL`/`PYOPENGL_PLATFORM` env at import (as the original does). Writes the verification frames/video to `args.out`.

- [ ] **Step 1:** Move `render_fit_check.py`'s `main()` body into `fit_check.run(args)`, reading fields off `args` instead of its own argparse.
- [ ] **Step 2: Smoke test** — `@pytest.mark.skipif(no GPU)`; run 1 frame on a small `ik_h5` fixture if available, else skip.
- [ ] **Step 3: Run / skip.**
- [ ] **Step 4: Commit** — `git add viz/views/fit_check.py viz/tests/test_views_smoke.py && git commit -m "feat(viz): fit-check view (port render_fit_check)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 10: `reproj-video` view (reimplement on the core)

**Files:**
- Create: `viz/views/reproj_video.py`; Modify: `viz/tests/test_views_smoke.py`
- Reference: `scripts/viz_predictions_reproject.py` (was JARVIS `create_multi_animal_videos3D`); reimplement using `core.io.load_data3d_csv` + `core.reproject` + `core.overlays` + cv2 VideoWriter.

**Interfaces:**
- Consumes: add to `core.io`: `load_data3d_csv(csv_path) -> (kp3d (T,K,3), conf (T,K), names)` (parse the per-bout `fly{0,1}.csv` header used in `data3D_fly*.csv`: repeating `name,name,name,name` = `x,y,z,confidence`; dedupe to base names). Add `write_video(out_path, frames_iter, fps)` (cv2 VideoWriter mp4v).
- Produces: `viz.views.reproj_video.run(args) -> int` from `--session-dir, --pred-dir, --bout, --cameras, --with-masks`. For each selected camera, read the bout's frame range, reproject the per-bout per-fly 3D CSV onto each frame with `core.overlays.draw_points`/`draw_chain`, optionally overlay SAM masks, and write an mp4.

- [ ] **Step 1:** Add `load_data3d_csv` + `write_video` to `core.io` with unit tests in `test_io.py` (synthetic 2-frame CSV → shapes; write_video writes a nonzero mp4).
- [ ] **Step 2:** Write `reproj_video.run` on the core.
- [ ] **Step 3: Smoke test** — skipif no session/pred dir; else assert an mp4 is written.
- [ ] **Step 4: Run / skip.**
- [ ] **Step 5: Commit** — `git add viz/core/io.py viz/views/reproj_video.py viz/tests/test_io.py viz/tests/test_views_smoke.py && git commit -m "feat(viz): reproj-video view reimplemented on core (drop JARVIS dep)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 11: `clip` view (port `scripts/viz/`)

**Files:**
- Create: `viz/views/clip.py`; Modify: `viz/tests/test_views_smoke.py`
- Reference: `scripts/viz/cut_videos_by_frame.py` (→ `mode="cut"`), `scripts/viz/stack_clips.py` (→ `mode="stack"`), `scripts/viz/make_bout_clip.py` + `scripts/viz/render_bout_clips.py` (→ `mode="render"`).

**Interfaces:**
- Produces: `viz.views.clip.run(args) -> int` dispatching on `args.mode`:
  - `cut`: extract `[start,end]` per camera from `--session-dir` into per-camera clips (port `cut_videos_by_frame`).
  - `stack`: vstack pre-cut clips (port `stack_clips`).
  - `render`: build a stacked multi-camera bout clip / overlay clip (port `make_bout_clip` + `render_bout_clips`).

- [ ] **Step 1:** Move each script's core function into `clip.py`, dispatched by `mode`; reuse `core.io.read_frames`/`write_video` and `core.layout.montage` where they overlap (DRY the vstack).
- [ ] **Step 2: Smoke test** — skipif no sample video; else `mode=cut` on a tiny range writes a clip.
- [ ] **Step 3: Run / skip.**
- [ ] **Step 4: Commit** — `git add viz/views/clip.py viz/tests/test_views_smoke.py && git commit -m "feat(viz): clip view (port scripts/viz cut|stack|render)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

### Task 12: Migration cleanup — delete the old scattered scripts

**Files:**
- Delete: `scripts/viz_keypoints.py`, `scripts/viz_courtship_mf.py`, `scripts/viz_predictions_reproject.py`, `scripts/render_fit_check.py`, `scripts/viz/` (whole dir: `cut_videos_by_frame.py`, `make_bout_clip.py`, `render_bout_clips.py`, `stack_clips.py`)
- Modify: any doc/config referencing them (grep first).

- [ ] **Step 1: Verify nothing imports the deletion set.**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
grep -rnE "viz_keypoints|viz_courtship_mf|viz_predictions_reproject|render_fit_check|scripts/viz/" \
  scripts/ third_party/jarvis_jax/jarvis_jax/ configs/ README.md docs/ 2>/dev/null | grep -av "docs/plans/2026-07-06"
```
Expected: no hits from kept code (only this plan / historical docs). If a kept file references one, update it to the `viz` CLI equivalent before deleting.

- [ ] **Step 2: Delete.**

```bash
git rm scripts/viz_keypoints.py scripts/viz_courtship_mf.py scripts/viz_predictions_reproject.py scripts/render_fit_check.py
git rm -r scripts/viz/
```

- [ ] **Step 3: Verify the package still imports + core tests pass.**

```bash
OMP_NUM_THREADS=4 JAX_PLATFORMS=cpu python -m pytest viz/tests/ -q
python -m viz --help
```
Expected: core/layout/colors/reproject/cli tests PASS; `--help` lists all six subcommands.

- [ ] **Step 4: Add `viz/README.md`** — a short usage table (subcommand → one-line purpose → example invocation) so the package is self-documenting.

- [ ] **Step 5: Commit** — `git add -u scripts/ && git add viz/README.md && git commit -m "refactor(viz): remove scattered viz scripts superseded by the viz/ package

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"`

---

## Notes for the implementer

- The `overlay`/`legskel`/`kp-qc` views need the recording's `calib_dir`/`session_dir`/`predictions_dir`/`cameras`/`KP_NAMES`. Get them from `viz.core.config.courtship_recording()` (Task 6). For non-courtship/original-pipeline inputs, accept explicit `--calib-dir`/`--session-dir` overrides on those subcommands (add if a view needs them).
- `bout_start_frame(cfg, bout)` lives in `scripts/run_bout.py`; import it (`sys.path` has the repo root when running `python -m viz` from the repo). Do NOT duplicate the bouts_csv parsing.
- Keep views thin: parse args → load via `core.io` → reproject via `core.reproject` → draw via `core.overlays` → arrange via `core.layout` → save. Any logic worth testing belongs in the core, not a view.
- The `docs/plans/viz-reference/*.py` files are the exact behavioral references for the promoted overlays; open them side-by-side when writing Tasks 6–7.
