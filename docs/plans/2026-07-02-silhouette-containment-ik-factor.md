# Coverage-Asymmetric, Confidence-Weighted Silhouette IK Factor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a coverage-asymmetric, confidence-weighted, DOF-restricted silhouette factor to the fly IK so appendage kinematics can be recovered from SAM masks when keypoints are unreliable, robust to the mask's shadow halo.

**Architecture:** Two jaxls cost terms over the existing `SE3Var`+`JointVar`, both reusing the differentiable in-loop `fk_repose` + affine projection: (1) a NEW mesh→mask **containment** residual that bilinearly samples a precomputed cropped signed-distance field and penalizes verts outside the mask (`relu`), and (2) the EXISTING SAM→mesh **coverage** Chamfer, refit to use an eroded (halo-stripped) mask + per-point confidence. Per-frame data (eroded boundary points, cropped SDF, crop transforms, confidences) is closed over the cost factories as constants and selected inside the vmapped residual by a tiny per-frame `FrameVar` (retiring the large `SilVar`). A factor-specific `sil_qs_mask` restricts the silhouette gradient to appendage DOFs via `where(sil_qs_mask, full_q, stop_gradient(full_q))`.

**Tech Stack:** Python, JAX/MJX, jaxls, jaxlie, MuJoCo, NumPy, SciPy (`ndimage`), OpenCV (`cv2`); pytest.

## Global Constraints

- `stac-mjx/stac_mjx/stac_core_jaxls.py` stays **byte-identical** (an existing test runs `git diff` on it and asserts empty).
- New code under `third_party/jarvis_jax/jarvis_jax/cse/`, tests under `third_party/jarvis_jax/tests/`.
- Affine/telecentric projection everywhere: `proj = X @ M.T + t` where `M = P[:2,:3]`, `t = P[:2,3]` (3×4 DLT, 3rd row `[0,0,0,1]`, no perspective divide).
- Silhouette vertex indices are FULL-vertex-array indices (the mesh npz `fps_*` arrays are already full-array space; pass straight to `fk_repose(indices=...)`).
- Weight is a residual multiplier (LM cost ∝ weight²), matching the existing coverage convention (`res * silhouette_weight`).
- Baseline invariant: with `silhouette_weight=0` AND `containment_weight=0`, no silhouette/containment cost and no `FrameVar` are added → the solve reproduces `stac_core_jaxls.JaxlsBatchSolver` to `atol=1e-5`.
- Env: gpu-l40s compute node, `micromamba activate 3d_tracking`, `unset LD_LIBRARY_PATH`. Real fits run on GPU. **Unit tests CPU-only**: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/<file> -v`.
- Do not commit checkpoints/data/label-npz/masks/outputs/videos; commit by explicit path.
- Validation target: MALE recording `2026_03_18_15_31_22`, watertight visual mesh `fly_v1_visual_canonical_wings.npz`.

## Key existing interfaces (consumed, do NOT rebuild)

- `silhouette_ik.load_anatomy(model_xml, mesh_npz) -> anat`; `make_fk_repose(anat) -> fk_repose(qpos, scale=1.0, indices=None) -> (K,3)` (jitted, differentiable).
- `silhouette_chamfer.chamfer_residual(target_pts (N,2), proj_pts (M,2), *, beta=8.0, huber_delta=0.0, chunk_size=32) -> (N,)` (softmin, NaN-safe, chunk-invariant).
- `silhouette_boundary.mask_boundary_pixels(mask)`, `sample_boundary_points(mask, n_points, *, seed=0) -> (n_points,2)` (NaN rows for empty).
- `silhouette_targets.silhouette_fk_indices(mesh_npz, subset="fps_300", exclude_seg_ids=None) -> (M,) int32` (full-array).
- `silhouette_ik_solve._cam2img_for_frame(fs_row, id2file, cam_names) -> {cam_idx: image_id}`, `_ann_for_image(id2ann_multi, iid, ann_id_by_image) -> ann|None`, `_load_sam_mask(root, split, file_name, ann_id) -> mask|None`, `build_solver_inputs(ik_h5, model_xml) -> dict`, `_umeyama`, `_model_to_mm`, `_triangulate_kp_mm`.
- `geometry.reprojection_tool.ReprojectionTool(calib_dir)` → `.cameras` (dict), `.num_cameras`, `._camera_list[c].cameraMatrix` (3×4), `.reproject_point(p3d) -> (num_cam,2)`.
- `silhouette_joint_ik.SilhouetteJaxlsBatchSolver` + `make_silhouette_cost` (both MODIFIED here); `stac_mjx.utils.kinematics/com_pos/get_site_xpos`.

---

### Task 1: Cropped signed-distance-field precompute (`silhouette_sdf.py`)

Offline (NumPy/SciPy/cv2, no JAX): turn per-(frame,camera) SAM masks into a cropped, resized **signed distance field** stack + crop transforms for the containment residual.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_sdf.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_sdf.py`

**Interfaces:**
- Consumes: `silhouette_ik_solve._cam2img_for_frame/_ann_for_image/_load_sam_mask`, `ReprojectionTool`.
- Produces:
  - `_mask_bbox(mask, margin) -> (x0,y0,x1,y1) | None`
  - `_mask_to_sdf_crop(mask, bbox, out_hw) -> (sdf (H,W) f32, grid_scale (2,) f32, grid_offset (2,) f32) | None`
  - `build_sdf_stack(root, split, fs_imgids, calib_dir, *, out_hw=(128,128), bbox_margin=0.4, ann_id_by_image=None) -> dict(sdf (T,C,H,W) f32, grid_scale (T,C,2) f32, grid_offset (T,C,2) f32, present (T,C) bool, cam_Ms (C,2,3) f32, cam_ts (C,2) f32, cam_names)`. Grid convention: `grid_xy = (orig_xy - grid_offset) * grid_scale`; sdf is signed (− inside, + outside) in **original-image px**; absent/empty camera → `present=False`, sdf filled `+1e4`.

- [ ] **Step 1: Write the failing tests**

Create `third_party/jarvis_jax/tests/test_silhouette_sdf.py`:

```python
import numpy as np
from jarvis_jax.cse.silhouette_sdf import _mask_bbox, _mask_to_sdf_crop


def test_mask_bbox_expands_by_margin():
    mask = np.zeros((100, 200), bool)
    mask[40:60, 80:120] = True          # 20 tall x 40 wide, center (100,50)
    x0, y0, x1, y1 = _mask_bbox(mask, 0.5)
    # width 40 -> +/-20 margin -> x in [60,140]; height 20 -> +/-10 -> y in [30,70]
    assert (x0, x1) == (60.0, 140.0)
    assert (y0, y1) == (30.0, 70.0)


def test_mask_bbox_empty_returns_none():
    assert _mask_bbox(np.zeros((10, 10), bool), 0.4) is None


def test_sdf_crop_signs_and_transform():
    mask = np.zeros((100, 100), bool)
    mask[30:70, 30:70] = True           # 40x40 square
    bbox = (20.0, 20.0, 80.0, 80.0)     # 60x60 crop
    sdf, gs, go = _mask_to_sdf_crop(mask, bbox, (120, 120))
    assert sdf.shape == (120, 120)
    # grid_offset is the integer crop origin; grid_scale maps 60 orig px -> 120 grid px
    assert np.allclose(go, [20.0, 20.0])
    assert np.allclose(gs, [120.0 / 60.0, 120.0 / 60.0])
    # center of the square is deep inside -> negative; a point well outside -> positive
    cx = int((50 - go[0]) * gs[0]); cy = int((50 - go[1]) * gs[1])
    assert sdf[cy, cx] < 0
    corner = int((22 - go[0]) * gs[0])  # near crop corner, outside the square
    assert sdf[corner, corner] > 0
    # magnitude is in ORIGINAL px: center is ~20 orig px from the nearest edge
    assert abs(abs(sdf[cy, cx]) - 20.0) < 3.0


def test_sdf_crop_empty_returns_none():
    assert _mask_to_sdf_crop(np.zeros((50, 50), bool), (0, 0, 50, 50), (32, 32)) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_sdf.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.silhouette_sdf'`.

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/silhouette_sdf.py`:

```python
"""Offline cropped signed-distance-field precompute for the silhouette
containment residual (pure NumPy/SciPy/cv2, no JAX, no qpos).

For each (frame, camera) SAM mask we crop to the fly bbox + margin, resize to a
fixed grid, and compute a SIGNED distance field (negative inside, positive
outside) rescaled to ORIGINAL-image pixels. The containment residual bilinearly
samples this field at projected mesh verts; grid coords are
grid_xy = (orig_xy - grid_offset) * grid_scale. Cropping keeps the field small
(so the whole (T, n_cam, H, W) stack fits on device as a closed-over constant).
"""
from __future__ import annotations
import json
import os
import numpy as np
import cv2
from scipy import ndimage

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.silhouette_ik_solve import (
    _cam2img_for_frame, _ann_for_image, _load_sam_mask,
)

_BIG = 1e4  # sdf fill for absent cameras (present=False gates it anyway)


def _load_coco_index(root, split):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    return id2file, id2ann_multi


def _mask_bbox(mask, margin):
    """(x0,y0,x1,y1) fly bbox expanded by `margin` fraction of side; None if empty."""
    m = np.asarray(mask).astype(bool)
    if not m.any():
        return None
    ys, xs = np.where(m)
    x0, x1 = float(xs.min()), float(xs.max() + 1)
    y0, y1 = float(ys.min()), float(ys.max() + 1)
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return (x0 - mx, y0 - my, x1 + mx, y1 + my)


def _mask_to_sdf_crop(mask, bbox, out_hw):
    """Crop mask to bbox, resize to out_hw, signed distance in ORIGINAL px.

    Returns (sdf (H,W) f32, grid_scale (2,) f32, grid_offset (2,) f32) or None
    if the crop is empty. grid_xy = (orig_xy - grid_offset) * grid_scale.
    """
    m = np.asarray(mask).astype(np.uint8)
    H, W = out_hw
    x0, y0, x1, y1 = bbox
    x0i, y0i = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    x1i, y1i = min(m.shape[1], int(np.ceil(x1))), min(m.shape[0], int(np.ceil(y1)))
    if x1i <= x0i or y1i <= y0i:
        return None
    crop = m[y0i:y1i, x0i:x1i]
    if crop.sum() == 0:
        return None
    rc = cv2.resize(crop, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
    din = ndimage.distance_transform_edt(rc)     # >0 inside (dist to background)
    dout = ndimage.distance_transform_edt(~rc)   # >0 outside (dist to foreground)
    sdf_resized = dout - din                     # + outside, - inside (resized px)
    px_scale = ((x1i - x0i) / W + (y1i - y0i) / H) / 2.0   # resized px -> original px
    sdf = (sdf_resized * px_scale).astype(np.float32)
    grid_scale = np.array([W / (x1i - x0i), H / (y1i - y0i)], np.float32)
    grid_offset = np.array([x0i, y0i], np.float32)
    return sdf, grid_scale, grid_offset


def build_sdf_stack(root, split, fs_imgids, calib_dir, *, out_hw=(128, 128),
                    bbox_margin=0.4, ann_id_by_image=None):
    """Per-(frame,camera) cropped signed-distance stack + crop transforms."""
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    n_cam = rt.num_cameras
    cam_Ms = np.stack([rt._camera_list[c].cameraMatrix[:2, :3] for c in range(n_cam)]).astype(np.float32)
    cam_ts = np.stack([rt._camera_list[c].cameraMatrix[:2, 3] for c in range(n_cam)]).astype(np.float32)

    id2file, id2ann_multi = _load_coco_index(root, split)
    fs_list = list(fs_imgids)
    T = len(fs_list)
    H, W = out_hw
    sdf = np.full((T, n_cam, H, W), _BIG, np.float32)
    grid_scale = np.ones((T, n_cam, 2), np.float32)
    grid_offset = np.zeros((T, n_cam, 2), np.float32)
    present = np.zeros((T, n_cam), bool)

    for t in range(T):
        cam2img = _cam2img_for_frame(fs_list[t], id2file, cam_names)
        for c in range(n_cam):
            iid = cam2img.get(c)
            if iid is None:
                continue
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            mask = _load_sam_mask(root, split, id2file.get(int(iid), ""), ann["id"])
            if mask is None:
                continue
            mask = np.asarray(mask)
            bbox = _mask_bbox(mask, bbox_margin)
            if bbox is None:
                continue
            out = _mask_to_sdf_crop(mask, bbox, out_hw)
            if out is None:
                continue
            sdf[t, c], grid_scale[t, c], grid_offset[t, c] = out
            present[t, c] = True

    return dict(sdf=sdf, grid_scale=grid_scale, grid_offset=grid_offset,
                present=present, cam_Ms=cam_Ms, cam_ts=cam_ts, cam_names=cam_names)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_sdf.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_sdf.py third_party/jarvis_jax/tests/test_silhouette_sdf.py
git commit -m "feat(cse): cropped signed-distance-field precompute for silhouette containment

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Containment residual (`silhouette_containment.py`)

Pure-JAX differentiable residual: bilinear-sample the SDF at projected verts, `relu`-penalize outside-mask, with per-vertex confidence + per-camera present gate + out-of-crop overflow so far-outside verts still get inward gradient.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_containment.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_containment.py`

**Interfaces:**
- Produces:
  - `bilinear_sample(img (H,W), xy (M,2)) -> (M,)` (clamped bilinear; xy is (x=col, y=row) grid coords).
  - `containment_residual(proj_pts (M,2), sdf (H,W), grid_scale (2,), grid_offset (2,), conf (M,), *, margin=0.0, present=True) -> (M,)` — `conf * relu(d + margin)` gated by `present`, where `d = sdf_bilinear + out-of-crop overflow (original px)`.

- [ ] **Step 1: Write the failing tests**

Create `third_party/jarvis_jax/tests/test_silhouette_containment.py`:

```python
import jax
import jax.numpy as jnp
import numpy as np
from jarvis_jax.cse.silhouette_containment import bilinear_sample, containment_residual


def _ramp_sdf(H=20, W=20):
    # signed distance-like field: negative in the left half, positive in the right
    xs = np.arange(W)[None, :].repeat(H, 0).astype(np.float32)
    return jnp.asarray(xs - (W / 2))     # d = column - 10


def test_bilinear_matches_manual():
    img = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    # point (x=1.5, y=0.5): interp of rows 0..1, cols 1..2
    val = bilinear_sample(img, jnp.asarray([[1.5, 0.5]]))
    manual = (img[0, 1] + img[0, 2] + img[1, 1] + img[1, 2]) / 4.0
    assert float(abs(val[0] - manual)) < 1e-5


def test_bilinear_clamps_out_of_bounds():
    img = jnp.asarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    assert float(bilinear_sample(img, jnp.asarray([[-5.0, -5.0]]))[0]) == float(img[0, 0])
    assert float(bilinear_sample(img, jnp.asarray([[99.0, 99.0]]))[0]) == float(img[3, 3])


def test_inside_zero_outside_positive():
    sdf = _ramp_sdf()
    gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    # vert projecting to column 3 (d=-7, inside) -> 0 ; column 17 (d=+7, outside) -> +7
    r = containment_residual(jnp.asarray([[3.0, 10.0], [17.0, 10.0]]), sdf, gs, go,
                             conf=jnp.ones(2), margin=0.0)
    assert float(r[0]) == 0.0
    assert abs(float(r[1]) - 7.0) < 1e-4


def test_confidence_scales():
    sdf = _ramp_sdf(); gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    r = containment_residual(jnp.asarray([[17.0, 10.0]]), sdf, gs, go,
                             conf=jnp.array([0.5]), margin=0.0)
    assert abs(float(r[0]) - 3.5) < 1e-4


def test_present_gate_zeros():
    sdf = _ramp_sdf(); gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    r = containment_residual(jnp.asarray([[17.0, 10.0]]), sdf, gs, go,
                             conf=jnp.ones(1), margin=0.0, present=False)
    assert float(r[0]) == 0.0


def test_gradient_pulls_inward_and_is_finite():
    sdf = _ramp_sdf(); gs = jnp.array([1.0, 1.0]); go = jnp.array([0.0, 0.0])
    # a vert far outside the grid (column 40) should have finite grad pointing to -x (inward)
    def loss(px):
        p = jnp.stack([px, jnp.array(10.0)])[None, :]
        return containment_residual(p, sdf, gs, go, conf=jnp.ones(1)).sum()
    g = jax.grad(loss)(jnp.array(40.0))
    assert np.isfinite(float(g))
    assert float(g) > 0    # increasing x increases residual -> descent moves x inward
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_containment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.silhouette_containment'`.

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/silhouette_containment.py`:

```python
"""Differentiable mesh->mask CONTAINMENT residual (pure JAX).

For each projected mesh vertex, bilinearly sample the mask's cropped signed
distance field (negative inside, positive outside, original px) and penalize the
outside part: r = conf * relu(d + margin). This is the coverage-ASYMMETRIC term:
verts outside the mask are pulled in hard; verts inside contribute nothing (so
the SAM shadow halo, which only inflates the mask, cannot drag the fit outward).

An out-of-crop OVERFLOW distance (converted to original px via grid_scale) is
added so verts projecting beyond the cropped SDF still receive an inward
gradient instead of a clamped zero-gradient plateau.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp


def bilinear_sample(img, xy):
    """Clamped bilinear sample of img (H,W) at xy (M,2) grid coords (x=col,y=row)."""
    H, W = img.shape
    x = jnp.clip(xy[:, 0], 0.0, W - 1.0)
    y = jnp.clip(xy[:, 1], 0.0, H - 1.0)
    x0 = jnp.floor(x).astype(jnp.int32); y0 = jnp.floor(y).astype(jnp.int32)
    x1 = jnp.clip(x0 + 1, 0, W - 1); y1 = jnp.clip(y0 + 1, 0, H - 1)
    wx = x - x0; wy = y - y0
    Ia = img[y0, x0]; Ib = img[y0, x1]; Ic = img[y1, x0]; Id = img[y1, x1]
    return (Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy)
            + Ic * (1 - wx) * wy + Id * wx * wy)


def containment_residual(proj_pts, sdf, grid_scale, grid_offset, conf, *,
                         margin=0.0, present=True):
    """conf * relu(d + margin), gated by `present`. d = SDF(bilinear) + overflow.

    proj_pts (M,2) original px; grid_xy = (proj - grid_offset) * grid_scale.
    overflow = distance the point lies OUTSIDE the SDF grid box, back in original
    px, so far-outside verts keep a finite inward gradient (no clamp plateau).
    """
    H, W = sdf.shape
    grid = (proj_pts - grid_offset[None, :]) * grid_scale[None, :]        # (M,2)
    d_edge = bilinear_sample(sdf, grid)                                   # (M,) orig px
    ox = jnp.maximum(jnp.maximum(-grid[:, 0], grid[:, 0] - (W - 1.0)), 0.0)
    oy = jnp.maximum(jnp.maximum(-grid[:, 1], grid[:, 1] - (H - 1.0)), 0.0)
    overflow = jnp.sqrt((ox / grid_scale[0]) ** 2 + (oy / grid_scale[1]) ** 2 + 1e-12)
    d = d_edge + overflow
    r = conf * jax.nn.relu(d + margin)
    return jnp.where(present, r, jnp.zeros_like(r))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_containment.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_containment.py third_party/jarvis_jax/tests/test_silhouette_containment.py
git commit -m "feat(cse): differentiable mesh->mask containment residual (SDF relu, asymmetric)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Appendage DOF mask + vertex subset (`silhouette_dof.py`)

Name-matched helpers: which qpos DOFs the silhouette may move, and which mesh verts (appendage segments) the containment penalizes.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_dof.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_dof.py`

**Interfaces:**
- Consumes: `silhouette_targets.silhouette_fk_indices`.
- Produces:
  - `APPENDAGE_PATTERNS = {"wing": r"wing", "leg": r"coxa|femur|tibia|tarsus|claw|trochanter", "abdomen": r"abdomen"}`
  - `build_appendage_dof_mask(model, include=("wing","leg","abdomen")) -> (nq,) bool` (root/free joint always False).
  - `appendage_vertex_indices(mesh_npz, subset="fps_300", include=("wing","leg","abdomen")) -> (M,) int32` full-array indices.

- [ ] **Step 1: Write the failing tests**

Create `third_party/jarvis_jax/tests/test_silhouette_dof.py`:

```python
import os
import re
import numpy as np
import pytest
from jarvis_jax.cse.silhouette_dof import (
    build_appendage_dof_mask, appendage_vertex_indices, APPENDAGE_PATTERNS,
)

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip_xml = pytest.mark.skipif(not os.path.exists(XML), reason="model xml absent")
skip_mesh = pytest.mark.skipif(not os.path.exists(MESH), reason="mesh npz absent")


@skip_xml
def test_dof_mask_excludes_root_and_covers_all_hinges():
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    mask = build_appendage_dof_mask(m)
    assert mask.shape == (m.nq,)
    assert not mask[:7].any()               # free (root) joint DOFs excluded
    # V1 model: all 86 non-root hinge DOFs are wing/leg/abdomen
    assert int(mask.sum()) == m.nq - 7


@skip_xml
def test_dof_mask_wing_only_is_six():
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    mask = build_appendage_dof_mask(m, include=("wing",))
    assert int(mask.sum()) == 6             # yaw/roll/pitch x left/right
    assert not mask[:7].any()


@skip_xml
def test_dof_mask_abdomen_does_not_catch_coxa_abduct():
    import mujoco
    m = mujoco.MjModel.from_xml_path(XML)
    mask = build_appendage_dof_mask(m, include=("abdomen",))
    # 14 abdomen DOFs (abdomen_* + abdomen_abduct_*), NOT the leg coxa_abduct joints
    assert int(mask.sum()) == 14


@skip_mesh
def test_appendage_vertices_are_subset_of_fps_and_appendage_only():
    idx = appendage_vertex_indices(MESH, subset="fps_300")
    z = np.load(MESH, allow_pickle=True)
    fps = set(int(i) for i in z["fps_300"])
    assert set(int(i) for i in idx).issubset(fps)         # subset of fps_300
    assert len(idx) > 0 and len(idx) < len(fps)           # some excluded (body core)
    seg = np.asarray(z["vertex_segment"]); segids = np.asarray(z["seg_ids"])
    segname = {int(s): str(n) for s, n in zip(segids, z["seg_names"])}
    rex = re.compile("|".join(APPENDAGE_PATTERNS.values()))
    assert all(rex.search(segname[int(seg[i])].lower()) for i in idx)  # all appendage segs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_dof.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.silhouette_dof'`.

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/silhouette_dof.py`:

```python
"""Appendage DOF + vertex selection for the DOF-restricted silhouette factor.

The silhouette factor should move only the appendage DOFs the keypoints track
poorly (wings, legs, abdomen), leaving the root pose to keypoints. These helpers
name-match the model's joints and the mesh's segments. NOTE: the abdomen pattern
is `abdomen` (NOT a bare `abd`) so it does not also catch the leg `coxa_abduct`
joints.
"""
from __future__ import annotations
import re
import numpy as np

from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices

APPENDAGE_PATTERNS = {
    "wing": r"wing",
    "leg": r"coxa|femur|tibia|tarsus|claw|trochanter",
    "abdomen": r"abdomen",
}


def _pattern(include):
    return re.compile("|".join(APPENDAGE_PATTERNS[k] for k in include))


def build_appendage_dof_mask(model, include=("wing", "leg", "abdomen")):
    """(nq,) bool: True for appendage hinge qpos DOFs; free (root) joint False."""
    import mujoco
    rex = _pattern(include)
    mask = np.zeros(int(model.nq), bool)
    for j in range(int(model.njnt)):
        if int(model.jnt_type[j]) == int(mujoco.mjtJoint.FREE):
            continue
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if nm and rex.search(nm.lower()):
            mask[int(model.jnt_qposadr[j])] = True   # hinge/slide = 1 qpos each
    return mask


def appendage_vertex_indices(mesh_npz, subset="fps_300", include=("wing", "leg", "abdomen")):
    """Full-array vertex indices in `subset` that belong to appendage segments."""
    z = np.load(mesh_npz, allow_pickle=True)
    rex = _pattern(include)
    seg_ids = np.asarray(z["seg_ids"])
    seg_names = [str(s) for s in z["seg_names"]]
    exclude = [int(sid) for sid, nm in zip(seg_ids, seg_names) if not rex.search(nm.lower())]
    return silhouette_fk_indices(mesh_npz, subset=subset, exclude_seg_ids=exclude)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_dof.py -v`
Expected: 5 passed (0 skipped on the GPU node where XML+MESH exist).

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_dof.py third_party/jarvis_jax/tests/test_silhouette_dof.py
git commit -m "feat(cse): appendage DOF mask + vertex subset (name-matched, root-excluded)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Unpacked, halo-eroded silhouette targets (`silhouette_targets.py`)

Change `build_silhouette_targets` to return UNPACKED arrays (for the closed-over-constant `FrameVar` feeding), erode the mask by `erode_px` (halo strip) before boundary sampling, and emit per-point confidence.

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py:32-80`
- Test: `third_party/jarvis_jax/tests/test_silhouette_targets.py` (rewrite assertions for the new return)

**Interfaces:**
- Consumes: `silhouette_boundary.sample_boundary_points`, `silhouette_ik_solve._cam2img_for_frame/_ann_for_image/_load_sam_mask`, `ReprojectionTool`, `scipy.ndimage.binary_erosion`.
- Produces: `build_silhouette_targets(root, split, fs_imgids, calib_dir, *, n_points=128, erode_px=8, ann_id_by_image=None, seed=0) -> dict(boundary (T,C,n_points,2) f32, conf_p (T,C,n_points) f32, present (T,C) bool, cam_Ms (C,2,3) f32, cam_ts (C,2) f32, cam_names, n_pts)`. Absent camera → boundary NaN, conf_p 0, present False. `pack_sil_value`/`unpack_sil_value`/`SIL_PER_CAM` are removed in Task 5 (still referenced until then).

- [ ] **Step 1: Rewrite the test**

Replace `third_party/jarvis_jax/tests/test_silhouette_targets.py` with:

```python
import numpy as np
import jarvis_jax.cse.silhouette_targets as tgt


class _FakeCam:
    def __init__(self):
        self.cameraMatrix = np.array([[2.0, 0, 0, 5.0], [0, 3.0, 0, 7.0], [0, 0, 0, 1.0]])


class _FakeRT:
    def __init__(self, calib_dir):
        self.cameras = {"CamA": 1, "CamB": 2}
        self.num_cameras = 2
        self._camera_list = [_FakeCam(), _FakeCam()]


def _square_mask():
    m = np.zeros((60, 60), bool); m[20:40, 20:40] = True
    return m


def test_build_targets_shapes_and_erosion(monkeypatch):
    monkeypatch.setattr(tgt, "ReprojectionTool", _FakeRT)
    monkeypatch.setattr(tgt, "_load_coco_index",
                        lambda root, split: ({10: "f0", 11: "f1"}, {10: [{"id": 1}], 11: [{"id": 2}]}, None))
    monkeypatch.setattr(tgt, "_cam2img_for_frame", lambda row, id2f, cams: {0: 10, 1: 11})
    monkeypatch.setattr(tgt, "_ann_for_image", lambda m, iid, a: {"id": iid})
    monkeypatch.setattr(tgt, "_load_sam_mask", lambda root, split, fn, aid: _square_mask())

    out = tgt.build_silhouette_targets("r", "val", [None, None], "cd", n_points=32, erode_px=3)
    assert out["boundary"].shape == (2, 2, 32, 2)
    assert out["conf_p"].shape == (2, 2, 32)
    assert out["present"].all()
    assert out["cam_Ms"].shape == (2, 2, 3) and out["cam_ts"].shape == (2, 2)
    # eroded boundary lies strictly inside the raw 20..39 square (halo stripped inward)
    b = out["boundary"][0, 0]
    assert b[:, 0].min() >= 20 and b[:, 0].max() <= 39
    assert b[:, 0].min() > 20    # erosion pulled the boundary inward from the raw edge


def test_missing_camera_is_nan_absent(monkeypatch):
    monkeypatch.setattr(tgt, "ReprojectionTool", _FakeRT)
    monkeypatch.setattr(tgt, "_load_coco_index",
                        lambda root, split: ({10: "f0"}, {10: [{"id": 1}]}, None))
    monkeypatch.setattr(tgt, "_cam2img_for_frame", lambda row, id2f, cams: {0: 10})  # cam1 missing
    monkeypatch.setattr(tgt, "_ann_for_image", lambda m, iid, a: {"id": iid})
    monkeypatch.setattr(tgt, "_load_sam_mask", lambda root, split, fn, aid: _square_mask())

    out = tgt.build_silhouette_targets("r", "val", [None], n_points=16, erode_px=0, calib_dir="cd")
    assert out["present"][0, 0] and not out["present"][0, 1]
    assert np.isnan(out["boundary"][0, 1]).all()
    assert (out["conf_p"][0, 1] == 0).all()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_targets.py -v`
Expected: FAIL (new keys `boundary`/`conf_p`/`present` not returned; `build_silhouette_targets` still returns a tuple).

- [ ] **Step 3: Rewrite `build_silhouette_targets`**

In `third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py`, replace the import of `SIL_PER_CAM, pack_sil_value` (line 18) — remove it — and add `from scipy import ndimage` at the top. Replace the whole `build_silhouette_targets` function (lines 32-80) with:

```python
def build_silhouette_targets(
    root, split, fs_imgids, calib_dir, *,
    n_points: int = 128, erode_px: int = 8, ann_id_by_image=None, seed: int = 0,
):
    """Per-(frame,camera) eroded-mask boundary points + confidence, unpacked.

    Eroding by `erode_px` strips the SAM shadow/reflection halo so the coverage
    Chamfer pulls the mesh to the TRUE fly boundary, not the inflated ring.
    Absent camera -> boundary NaN, conf_p 0, present False.

    Returns dict(boundary (T,C,n_points,2) f32, conf_p (T,C,n_points) f32,
    present (T,C) bool, cam_Ms (C,2,3) f32, cam_ts (C,2) f32, cam_names, n_pts).
    """
    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    n_cam = rt.num_cameras
    cam_Ms = np.stack([rt._camera_list[c].cameraMatrix[:2, :3] for c in range(n_cam)]).astype(np.float32)
    cam_ts = np.stack([rt._camera_list[c].cameraMatrix[:2, 3] for c in range(n_cam)]).astype(np.float32)

    id2file, id2ann_multi, _ = _load_coco_index(root, split)
    fs_list = list(fs_imgids)
    T = len(fs_list)
    boundary = np.full((T, n_cam, n_points, 2), np.nan, np.float32)
    conf_p = np.zeros((T, n_cam, n_points), np.float32)
    present = np.zeros((T, n_cam), bool)

    struct = ndimage.generate_binary_structure(2, 1)
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_list[t], id2file, cam_names)
        for c in range(n_cam):
            iid = cam2img.get(c)
            if iid is None:
                continue
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            mask = _load_sam_mask(root, split, id2file.get(int(iid), ""), ann["id"])
            if mask is None:
                continue
            mask = np.asarray(mask).astype(bool)
            if erode_px > 0:
                eroded = ndimage.binary_erosion(mask, structure=struct, iterations=erode_px,
                                                border_value=0)
                if eroded.any():
                    mask = eroded
            bpts = sample_boundary_points(mask, n_points, seed=seed)
            boundary[t, c] = bpts
            conf_p[t, c] = np.where(np.isfinite(bpts).all(axis=-1), 1.0, 0.0)
            present[t, c] = bool(np.isfinite(bpts).any())

    return dict(boundary=boundary, conf_p=conf_p, present=present,
                cam_Ms=cam_Ms, cam_ts=cam_ts, cam_names=cam_names, n_pts=n_points)
```

Note: `_load_coco_index` already returns a 3-tuple `(id2file, id2ann_multi, None)`; the monkeypatch matches that. Keep `silhouette_fk_indices` unchanged below.

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_targets.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_targets.py third_party/jarvis_jax/tests/test_silhouette_targets.py
git commit -m "feat(cse): unpacked halo-eroded silhouette targets + per-point confidence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Coverage cost on `FrameVar` + retire `SilVar` (`silhouette_joint_ik.py`)

Refactor `make_silhouette_cost` and `SilhouetteJaxlsBatchSolver.solve_trajectory` so per-frame coverage data is closed over the factory as constants and selected by a tiny per-frame `FrameVar`; add the DOF-restricted `sil_q` assembly and per-point confidence. Preserve the baseline invariant.

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_joint_ik.py` (remove `SIL_PER_CAM`/`pack_sil_value`/`unpack_sil_value` lines 27-67; rewrite `make_silhouette_cost` lines 70-127; rewrite `solve_trajectory` lines 163-328)
- Test: `third_party/jarvis_jax/tests/test_silhouette_factor.py` (rewrite for the new signature), `third_party/jarvis_jax/tests/test_silhouette_joint_solve.py` (update solver-call kwargs)

**Interfaces:**
- Consumes: `silhouette_chamfer.chamfer_residual`, targets dict from Task 4.
- Produces:
  - `make_silhouette_cost(SE3Var, JointVar, FrameVar, *, fk_repose, vert_indices, cam_Ms, cam_ts, boundary_all, conf_p_all, sil_qs_mask, beta, huber_delta, silhouette_weight, qs_to_opt, template_qpos, scale=1.0, chunk_size=32) -> jaxls Cost.factory fn(root, joint, frame)`.
  - `SilhouetteJaxlsBatchSolver.solve_trajectory(..., *, fk_repose=None, cov_vert_indices=None, cont_vert_indices=None, cam_Ms=None, cam_ts=None, boundary_all=None, conf_p_all=None, sil_qs_mask=None, silhouette_weight=0.0, sdf_all=None, grid_scale_all=None, grid_offset_all=None, present_all=None, conf_v=None, containment_weight=0.0, margin=0.0)`. This task wires only the coverage path + `FrameVar`; the containment kwargs are added but unused until Task 6 (default 0/None → not added).

- [ ] **Step 1: Rewrite the factor test**

Replace `third_party/jarvis_jax/tests/test_silhouette_factor.py` with:

```python
import os
import numpy as np
import jax.numpy as jnp
import jaxls
import jaxlie
import pytest

from jarvis_jax.cse.silhouette_joint_ik import make_silhouette_cost

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip = pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)), reason="assets absent")


@skip
def test_silhouette_cost_builds_and_residual_shape():
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask
    from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
    import mujoco
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    vidx = silhouette_fk_indices(MESH, subset="fps_300")
    sil_qs = build_appendage_dof_mask(m)
    T, n_cam, n_pts = 3, 2, 16
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1))
    cam_ts = jnp.zeros((n_cam, 2))
    boundary_all = jnp.zeros((T, n_cam, n_pts, 2))
    conf_p_all = jnp.ones((T, n_cam, n_pts))
    qs_to_opt = jnp.ones((m.nq,), bool)

    class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                 retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
    class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((m.nq - 7,))): ...
    class FrameVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((1,))): ...

    cost = make_silhouette_cost(
        SE3Var, JointVar, FrameVar, fk_repose=fk, vert_indices=vidx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=boundary_all, conf_p_all=conf_p_all,
        sil_qs_mask=sil_qs, beta=8.0, huber_delta=0.0, silhouette_weight=1.0,
        qs_to_opt=qs_to_opt, template_qpos=jnp.asarray(anat["qpos0"]))
    root = SE3Var(jnp.arange(T)); joint = JointVar(jnp.arange(T)); frame = FrameVar(jnp.arange(T))
    prob = jaxls.LeastSquaresProblem(costs=[cost(root, joint, frame)],
                                     variables=[root, joint, frame]).analyze()
    vals = jaxls.VarValues.make([
        SE3Var(jnp.arange(T)).with_value(jaxlie.SE3.identity((T,))),
        JointVar(jnp.arange(T)).with_value(jnp.zeros((T, m.nq - 7))),
        FrameVar(jnp.arange(T)).with_value(jnp.arange(T).reshape(T, 1).astype(float)),
    ])
    r = prob.compute_residual_vector(vals)
    assert np.isfinite(np.asarray(r)).all()
    assert r.shape[0] == T * n_cam * n_pts
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_factor.py -v`
Expected: FAIL (`make_silhouette_cost()` got an unexpected keyword / missing `cam_Ms`; old signature took `SilVar`/`n_cam`/`n_pts`).

- [ ] **Step 3: Rewrite `make_silhouette_cost` and remove the pack helpers**

In `silhouette_joint_ik.py`: delete `SIL_PER_CAM`, `pack_sil_value`, `unpack_sil_value` (lines 27-67). Replace `make_silhouette_cost` (lines 70-127) with:

```python
def make_silhouette_cost(
    SE3Var, JointVar, FrameVar, *,
    fk_repose, vert_indices, cam_Ms, cam_ts, boundary_all, conf_p_all,
    sil_qs_mask, beta: float, huber_delta: float, silhouette_weight: float,
    qs_to_opt, template_qpos, scale: float = 1.0, chunk_size: int = 32,
):
    """Coverage cost: SAM(eroded)-boundary -> nearest projected mesh vertex.

    Per-frame data (boundary points, per-point confidence) is closed over as
    constants and selected by the integer FrameVar; cameras are iterated with
    jax.lax.scan (FK once per frame, no (C,N,M) tensor). The silhouette gradient
    is restricted to appendage DOFs via where(sil_qs_mask, full_q, stop_grad).
    """
    vert_indices = jnp.asarray(vert_indices)
    template_qpos = jnp.asarray(template_qpos)
    qs_to_opt = jnp.asarray(qs_to_opt)
    sil_qs_mask = jnp.asarray(sil_qs_mask)
    cam_Ms = jnp.asarray(cam_Ms); cam_ts = jnp.asarray(cam_ts)
    boundary_all = jnp.asarray(boundary_all); conf_p_all = jnp.asarray(conf_p_all)

    @jaxls.Cost.factory
    def silhouette_cost(var_values, root_var: SE3Var, joint_var: JointVar,
                        frame_var: FrameVar) -> jnp.ndarray:
        T_root = var_values[root_var]
        joints = var_values[joint_var]
        t = jax.lax.stop_gradient(var_values[frame_var])[0].astype(jnp.int32)
        xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
        q = jnp.concatenate([xyz, wxyz, joints])
        full_q = jnp.where(qs_to_opt, q, template_qpos)
        sil_q = jnp.where(sil_qs_mask, full_q, jax.lax.stop_gradient(full_q))
        verts3d = fk_repose(sil_q, scale, vert_indices)     # (M,3) FK once per frame

        tgt_all = boundary_all[t]; conf_all = conf_p_all[t]   # (C,n_pts,2),(C,n_pts)

        def scan_body(carry, cam):
            M, tt, tgt, cf = cam
            proj = verts3d @ M.T + tt
            r = chamfer_residual(tgt, proj, beta=beta, huber_delta=huber_delta,
                                 chunk_size=chunk_size)      # (n_pts,)
            return carry, r * cf
        _, res = jax.lax.scan(scan_body, None, (cam_Ms, cam_ts, tgt_all, conf_all))
        return (res * silhouette_weight).reshape(-1)         # (C*n_pts,)

    return silhouette_cost
```

- [ ] **Step 4: Rewrite `solve_trajectory` (coverage + FrameVar path)**

Replace `solve_trajectory` (lines 163-328) with the version below. It adds the new kwargs (containment kwargs present but only consumed in Task 6), builds a shared `FrameVar`, and appends the coverage cost via closed-over constants. The marker/reg/limit/smoothness blocks are unchanged.

```python
    def solve_trajectory(
        self, q_init, mjx_model, mjx_data_template, kp_data, qs_to_opt,
        kps_to_opt, lb, ub, site_idxs, q_reg_weights, *,
        fk_repose=None, cov_vert_indices=None, cont_vert_indices=None,
        cam_Ms=None, cam_ts=None, boundary_all=None, conf_p_all=None,
        sil_qs_mask=None, silhouette_weight=0.0,
        sdf_all=None, grid_scale_all=None, grid_offset_all=None, present_all=None,
        conf_v=None, containment_weight=0.0, margin=0.0,
    ):
        if kp_data.ndim == 3:
            kp_data = kp_data.reshape(kp_data.shape[0], -1)
        q_init = jnp.asarray(q_init); kp_data = jnp.asarray(kp_data)
        qs_to_opt = jnp.asarray(qs_to_opt); kps_to_opt = jnp.asarray(kps_to_opt)
        lb = jnp.asarray(lb); ub = jnp.asarray(ub)
        q_reg_weights = jnp.asarray(q_reg_weights)

        T = q_init.shape[0]
        nq = int(mjx_model.nq)
        n_kp_dim = int(kp_data.shape[-1])
        n_hinges = nq - _FREE_JOINT_NDOF
        smooth_weight = self.smooth_weight

        use_cov = (silhouette_weight != 0.0) and (boundary_all is not None)
        use_cont = (containment_weight != 0.0) and (sdf_all is not None)
        use_sil = use_cov or use_cont
        if sil_qs_mask is None:
            sil_qs_mask = jnp.ones((nq,), bool)

        dummy_joints = jnp.zeros((n_hinges,))
        dummy_kp = jnp.zeros((n_kp_dim,))

        class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                     retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
        class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_joints): ...
        class KpVar(jaxls.Var[jnp.ndarray], default_factory=lambda: dummy_kp): ...
        class FrameVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((1,))): ...

        root_all = SE3Var(jnp.arange(T))
        joint_all = JointVar(jnp.arange(T))
        kp_all = KpVar(jnp.arange(T))

        costs: list[jaxls.Cost] = []

        @jaxls.Cost.factory
        def marker_cost(var_values, root_var: SE3Var, joint_var: JointVar, kp_var: KpVar):
            T_root = var_values[root_var]; joints = var_values[joint_var]
            kp = jax.lax.stop_gradient(var_values[kp_var])
            xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
            q = jnp.concatenate([xyz, wxyz, joints])
            full_q = jnp.where(qs_to_opt, q, mjx_data_template.qpos)
            data = mjx_data_template.replace(qpos=full_q)
            data = utils.kinematics(mjx_model, data)
            data = utils.com_pos(mjx_model, data)
            markers = utils.get_site_xpos(data, site_idxs).flatten()
            finite = jnp.isfinite(kp)
            kp_clean = jnp.where(finite, kp, 0.0)
            return (kp_clean - markers) * kps_to_opt * finite
        costs.append(marker_cost(root_all, joint_all, kp_all))

        if jnp.any(q_reg_weights[_FREE_JOINT_NDOF:] > 0):
            hinge_regs = q_reg_weights[_FREE_JOINT_NDOF:]
            hinge_opt = qs_to_opt[_FREE_JOINT_NDOF:]

            @jaxls.Cost.factory
            def reg_cost(var_values, joint_var: JointVar):
                j = var_values[joint_var]
                return jnp.sqrt(hinge_regs * hinge_opt) * j
            costs.append(reg_cost(joint_all))

        hinge_lb = lb[_FREE_JOINT_NDOF:]; hinge_ub = ub[_FREE_JOINT_NDOF:]

        @jaxls.Cost.factory(kind="constraint_leq_zero")
        def limit_cost(var_values, joint_var: JointVar):
            j = var_values[joint_var]
            return jnp.concatenate([hinge_lb - j, j - hinge_ub])
        costs.append(limit_cost(joint_all))

        if smooth_weight > 0.0 and T > 1:
            @jaxls.Cost.factory
            def smoothness_cost(var_values, root_curr: SE3Var, root_prev: SE3Var,
                                joint_curr: JointVar, joint_prev: JointVar):
                root_diff = (var_values[root_prev].inverse() @ var_values[root_curr]).log()
                joint_diff = var_values[joint_curr] - var_values[joint_prev]
                return jnp.concatenate([root_diff, joint_diff]) * smooth_weight
            costs.append(smoothness_cost(
                SE3Var(jnp.arange(1, T)), SE3Var(jnp.arange(0, T - 1)),
                JointVar(jnp.arange(1, T)), JointVar(jnp.arange(0, T - 1))))

        variables = [root_all, joint_all, kp_all]

        frame_all = None
        if use_sil:
            frame_all = FrameVar(jnp.arange(T))
            variables.append(frame_all)

        if use_cov:
            cov_cost = make_silhouette_cost(
                SE3Var, JointVar, FrameVar,
                fk_repose=fk_repose, vert_indices=cov_vert_indices,
                cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=boundary_all,
                conf_p_all=conf_p_all, sil_qs_mask=sil_qs_mask, beta=self.beta,
                huber_delta=self.huber_delta, silhouette_weight=silhouette_weight,
                qs_to_opt=qs_to_opt, template_qpos=mjx_data_template.qpos, scale=1.0)
            costs.append(cov_cost(root_all, joint_all, frame_all))

        # containment cost is appended in Task 6 (guarded by use_cont)
        _ = (cont_vert_indices, sdf_all, grid_scale_all, grid_offset_all,
             present_all, conf_v, containment_weight, margin, use_cont)

        analyzed = jaxls.LeastSquaresProblem(costs=costs, variables=variables).analyze()

        xyz_init = q_init[:, :3]; wxyz_init = q_init[:, 3:7]
        hinges_init = q_init[:, _FREE_JOINT_NDOF:]
        qn = jnp.linalg.norm(wxyz_init, axis=-1, keepdims=True)
        wxyz_init = wxyz_init / jnp.where(qn > 0, qn, 1.0)
        roots_init = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz=wxyz_init), xyz_init)
        init_list = [
            SE3Var(jnp.arange(T)).with_value(roots_init),
            JointVar(jnp.arange(T)).with_value(hinges_init),
            KpVar(jnp.arange(T)).with_value(kp_data),
        ]
        if use_sil:
            init_list.append(FrameVar(jnp.arange(T)).with_value(
                jnp.arange(T).reshape(T, 1).astype(jnp.float32)))

        tangent_dim = 6 + n_hinges
        linear_solver = self._pick_linear_solver(T, tangent_dim)
        if use_sil and self.linear_solver == "auto":
            linear_solver = "conjugate_gradient"

        sol = analyzed.solve(
            verbose=False, linear_solver=linear_solver,
            trust_region=jaxls.TrustRegionConfig(lambda_initial=self.lambda_initial),
            termination=jaxls.TerminationConfig(max_iterations=self.n_iter),
            initial_vals=jaxls.VarValues.make(init_list))
        sol_roots = sol[SE3Var(jnp.arange(T))]
        sol_joints = sol[JointVar(jnp.arange(T))]
        xyz_sol = sol_roots.translation(); wxyz_sol = sol_roots.rotation().wxyz
        return jnp.concatenate([xyz_sol, wxyz_sol, sol_joints], axis=-1)
```

- [ ] **Step 5: Update the joint-solve test kwargs**

In `third_party/jarvis_jax/tests/test_silhouette_joint_solve.py`, update every `solve_trajectory(...)` call: the coverage-off baseline call passes no silhouette kwargs (or `silhouette_weight=0.0`), and any weighted call now passes `cov_vert_indices=<indices>`, `cam_Ms=`, `cam_ts=`, `boundary_all=`, `conf_p_all=`, `sil_qs_mask=` instead of the old `vert_indices=/sil_data=/n_pts=`. Concretely, replace the "weight moves qpos" test body's solver call with (using the Task-4 targets dict shapes and a far-away boundary):

```python
    # far-away boundary target so the coverage term must move qpos
    T = q_init.shape[0]
    n_cam, n_pts = cam_Ms.shape[0], 8
    boundary_all = jnp.full((T, n_cam, n_pts, 2), 1e4)         # unreachable -> nonzero grad
    conf_p_all = jnp.ones((T, n_cam, n_pts))
    q_w = solver.solve_trajectory(
        q_init=q_init, mjx_model=mjx_model, mjx_data_template=mjx_data,
        kp_data=kp_data, qs_to_opt=qs_to_opt, kps_to_opt=kps_to_opt, lb=lb, ub=ub,
        site_idxs=site_idxs, q_reg_weights=q_reg_weights,
        fk_repose=fk, cov_vert_indices=vidx, cam_Ms=cam_Ms, cam_ts=cam_ts,
        boundary_all=boundary_all, conf_p_all=conf_p_all, sil_qs_mask=sil_qs,
        silhouette_weight=0.3)
    assert float(jnp.linalg.norm(q_w - q0)) > 1e-4
```

Leave `test_stac_core_jaxls_is_byte_identical` and `test_no_silhouette_matches_jaxls_batch_solver` (baseline: call with no silhouette kwargs) unchanged in intent. Where those tests build `cam_Ms`/`cam_ts`/`vidx`/`sil_qs`/`fk`, add the imports (`silhouette_ik`, `silhouette_dof`, `silhouette_fk_indices`, `ReprojectionTool` or a small identity `cam_Ms=jnp.tile(jnp.eye(2,3)[None],(n_cam,1,1))`, `cam_ts=jnp.zeros((n_cam,2))`).

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_factor.py tests/test_silhouette_joint_solve.py -v`
Expected: pass (byte-identical + baseline-reproduces + weight-moves-qpos), skips only where heavy assets are absent.

- [ ] **Step 7: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_joint_ik.py third_party/jarvis_jax/tests/test_silhouette_factor.py third_party/jarvis_jax/tests/test_silhouette_joint_solve.py
git commit -m "refactor(cse): coverage silhouette cost on FrameVar constants; retire SilVar; DOF-restricted sil_q

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Containment cost wired into the solver (`silhouette_joint_ik.py`)

Add `make_containment_cost` and append it in `solve_trajectory` when `containment_weight != 0` and `sdf_all` is provided, sharing the `FrameVar`.

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_joint_ik.py` (add `make_containment_cost`; replace the Task-5 placeholder `_ = (...)` block with the real containment append)
- Test: `third_party/jarvis_jax/tests/test_silhouette_containment_solve.py` (new)

**Interfaces:**
- Consumes: `silhouette_containment.containment_residual`.
- Produces: `make_containment_cost(SE3Var, JointVar, FrameVar, *, fk_repose, vert_indices, cam_Ms, cam_ts, sdf_all, grid_scale_all, grid_offset_all, present_all, conf_v, sil_qs_mask, margin, containment_weight, qs_to_opt, template_qpos, scale=1.0) -> jaxls Cost.factory fn(root, joint, frame)`; residual per frame shape `(n_cam * M,)`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_silhouette_containment_solve.py`:

```python
import os
import numpy as np
import jax.numpy as jnp
import jaxls
import jaxlie
import pytest

from jarvis_jax.cse.silhouette_joint_ik import make_containment_cost

XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz"
skip = pytest.mark.skipif(not (os.path.exists(XML) and os.path.exists(MESH)), reason="assets absent")


@skip
def test_containment_cost_residual_shape_and_finite():
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices
    import mujoco
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    vidx = appendage_vertex_indices(MESH, subset="fps_300")
    M = len(vidx)
    sil_qs = build_appendage_dof_mask(m)
    T, n_cam, H, W = 2, 2, 32, 32
    # SDF all +5 (everything "outside") -> every appendage vert penalized -> positive residual
    sdf_all = jnp.full((T, n_cam, H, W), 5.0)
    grid_scale_all = jnp.ones((T, n_cam, 2))
    grid_offset_all = jnp.zeros((T, n_cam, 2))
    present_all = jnp.ones((T, n_cam), bool)
    conf_v = jnp.ones((M,))
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1)); cam_ts = jnp.zeros((n_cam, 2))

    class SE3Var(jaxls.Var[jaxlie.SE3], default_factory=jaxlie.SE3.identity,
                 retract_fn=jaxlie.manifold.rplus, tangent_dim=6): ...
    class JointVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((m.nq - 7,))): ...
    class FrameVar(jaxls.Var[jnp.ndarray], default_factory=lambda: jnp.zeros((1,))): ...

    cost = make_containment_cost(
        SE3Var, JointVar, FrameVar, fk_repose=fk, vert_indices=vidx,
        cam_Ms=cam_Ms, cam_ts=cam_ts, sdf_all=sdf_all, grid_scale_all=grid_scale_all,
        grid_offset_all=grid_offset_all, present_all=present_all, conf_v=conf_v,
        sil_qs_mask=sil_qs, margin=0.0, containment_weight=1.0,
        qs_to_opt=jnp.ones((m.nq,), bool), template_qpos=jnp.asarray(anat["qpos0"]))
    root = SE3Var(jnp.arange(T)); joint = JointVar(jnp.arange(T)); frame = FrameVar(jnp.arange(T))
    prob = jaxls.LeastSquaresProblem(costs=[cost(root, joint, frame)],
                                     variables=[root, joint, frame]).analyze()
    vals = jaxls.VarValues.make([
        SE3Var(jnp.arange(T)).with_value(jaxlie.SE3.identity((T,))),
        JointVar(jnp.arange(T)).with_value(jnp.zeros((T, m.nq - 7))),
        FrameVar(jnp.arange(T)).with_value(jnp.arange(T).reshape(T, 1).astype(float)),
    ])
    r = np.asarray(prob.compute_residual_vector(vals))
    assert r.shape[0] == T * n_cam * M
    assert np.isfinite(r).all()
    assert (r > 0).any()      # verts sit in "outside" SDF -> some positive penalty
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_containment_solve.py -v`
Expected: FAIL with `ImportError: cannot import name 'make_containment_cost'`.

- [ ] **Step 3: Add `make_containment_cost` and wire it in**

In `silhouette_joint_ik.py`, add the import near the top: `from jarvis_jax.cse.silhouette_containment import containment_residual`. Add this factory next to `make_silhouette_cost`:

```python
def make_containment_cost(
    SE3Var, JointVar, FrameVar, *,
    fk_repose, vert_indices, cam_Ms, cam_ts, sdf_all, grid_scale_all,
    grid_offset_all, present_all, conf_v, sil_qs_mask, margin: float,
    containment_weight: float, qs_to_opt, template_qpos, scale: float = 1.0,
):
    """Containment cost: appendage verts outside the mask -> relu(SDF) penalty.

    Per-frame SDF crops + transforms are closed over as constants and selected
    by the integer FrameVar; cameras iterated with jax.lax.scan (FK once/frame).
    Gradient restricted to appendage DOFs via where(sil_qs_mask, full_q, stop_grad).
    """
    vert_indices = jnp.asarray(vert_indices)
    template_qpos = jnp.asarray(template_qpos)
    qs_to_opt = jnp.asarray(qs_to_opt); sil_qs_mask = jnp.asarray(sil_qs_mask)
    cam_Ms = jnp.asarray(cam_Ms); cam_ts = jnp.asarray(cam_ts)
    sdf_all = jnp.asarray(sdf_all); grid_scale_all = jnp.asarray(grid_scale_all)
    grid_offset_all = jnp.asarray(grid_offset_all)
    present_all = jnp.asarray(present_all); conf_v = jnp.asarray(conf_v)

    @jaxls.Cost.factory
    def containment_cost(var_values, root_var: SE3Var, joint_var: JointVar,
                         frame_var: FrameVar) -> jnp.ndarray:
        T_root = var_values[root_var]; joints = var_values[joint_var]
        t = jax.lax.stop_gradient(var_values[frame_var])[0].astype(jnp.int32)
        xyz = T_root.translation(); wxyz = T_root.rotation().wxyz
        q = jnp.concatenate([xyz, wxyz, joints])
        full_q = jnp.where(qs_to_opt, q, template_qpos)
        sil_q = jnp.where(sil_qs_mask, full_q, jax.lax.stop_gradient(full_q))
        verts3d = fk_repose(sil_q, scale, vert_indices)     # (M,3)

        sdf_t = sdf_all[t]; gs_t = grid_scale_all[t]
        go_t = grid_offset_all[t]; pr_t = present_all[t]

        def scan_body(carry, cam):
            M, tt, sdf, gs, go, present = cam
            proj = verts3d @ M.T + tt                        # (M,2)
            r = containment_residual(proj, sdf, gs, go, conf_v,
                                     margin=margin, present=present)
            return carry, r                                  # (M,)
        _, res = jax.lax.scan(scan_body, None,
                              (cam_Ms, cam_ts, sdf_t, gs_t, go_t, pr_t))
        return (res * containment_weight).reshape(-1)        # (C*M,)

    return containment_cost
```

In `solve_trajectory`, replace the Task-5 placeholder block

```python
        # containment cost is appended in Task 6 (guarded by use_cont)
        _ = (cont_vert_indices, sdf_all, grid_scale_all, grid_offset_all,
             present_all, conf_v, containment_weight, margin, use_cont)
```

with:

```python
        if use_cont:
            cont_cost = make_containment_cost(
                SE3Var, JointVar, FrameVar,
                fk_repose=fk_repose, vert_indices=cont_vert_indices,
                cam_Ms=cam_Ms, cam_ts=cam_ts, sdf_all=sdf_all,
                grid_scale_all=grid_scale_all, grid_offset_all=grid_offset_all,
                present_all=present_all, conf_v=conf_v, sil_qs_mask=sil_qs_mask,
                margin=margin, containment_weight=containment_weight,
                qs_to_opt=qs_to_opt, template_qpos=mjx_data_template.qpos, scale=1.0)
            costs.append(cont_cost(root_all, joint_all, frame_all))
```

- [ ] **Step 4: Add a solver-level containment integration test**

Append to `third_party/jarvis_jax/tests/test_silhouette_containment_solve.py`:

```python
@skip
def test_containment_baseline_reproduces_and_moves_qpos():
    # containment_weight=0 -> identical to no-silhouette; weight>0 with an
    # all-"outside" SDF must move qpos (appendage DOFs) away from init.
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices
    from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver
    import mujoco
    IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
    if not os.path.exists(IK):
        pytest.skip("ik h5 absent")
    inp = build_solver_inputs(IK, XML)
    T = 2
    q_init = inp["q_init"][:T]; kp_data = inp["kp_data"][:T]
    anat = load_anatomy(XML, MESH); fk = make_fk_repose(anat)
    m = mujoco.MjModel.from_xml_path(XML)
    vidx = appendage_vertex_indices(MESH, subset="fps_300"); M = len(vidx)
    sil_qs = build_appendage_dof_mask(m)
    n_cam = 7
    cam_Ms = jnp.tile(jnp.eye(2, 3)[None], (n_cam, 1, 1)); cam_ts = jnp.zeros((n_cam, 2))
    sdf_all = jnp.full((T, n_cam, 32, 32), 5.0)
    gs = jnp.ones((T, n_cam, 2)); go = jnp.zeros((T, n_cam, 2)); pr = jnp.ones((T, n_cam), bool)
    solver = SilhouetteJaxlsBatchSolver(n_iter=5, smooth_weight=0.0)
    common = dict(q_init=q_init, mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
                  kp_data=kp_data, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
                  lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"],
                  q_reg_weights=inp["q_reg_weights"])
    q0 = np.asarray(solver.solve_trajectory(**common))          # no silhouette
    q_cont = np.asarray(solver.solve_trajectory(
        **common, fk_repose=fk, cont_vert_indices=vidx, cam_Ms=cam_Ms, cam_ts=cam_ts,
        sdf_all=sdf_all, grid_scale_all=gs, grid_offset_all=go, present_all=pr,
        conf_v=jnp.ones((M,)), sil_qs_mask=sil_qs, containment_weight=0.3))
    assert np.linalg.norm(q_cont - q0) > 1e-4
    # containment must NOT move the root translation far (DOF mask excludes root)
    assert np.linalg.norm(q_cont[:, :3] - q0[:, :3]) < np.linalg.norm(q_cont[:, 7:] - q0[:, 7:]) + 1e-6
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_containment_solve.py tests/test_silhouette_joint_solve.py -v`
Expected: pass (baseline-reproduces preserved; containment residual finite + positive; qpos moves; root barely moves), skips where assets absent.

- [ ] **Step 6: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_joint_ik.py third_party/jarvis_jax/tests/test_silhouette_containment_solve.py
git commit -m "feat(cse): wire mesh->mask containment cost into the silhouette solver (shared FrameVar)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: Validation driver — filled-tri IoU + do-no-harm (`run_silhouette_polish.py`)

Wire the new targets/SDF/DOF plumbing into the driver; add a filled-triangle IoU metric (the overlay methodology) as the primary success measure plus the do-no-harm guards (keypoint reproj RMSE, MPJPE-vs-STAC), reporting baseline vs treatment.

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/run_silhouette_polish.py`
- Test: `third_party/jarvis_jax/tests/test_run_silhouette_polish.py` (add a filled-tri IoU unit test; keep existing import/CLI smoke)

**Interfaces:**
- Consumes: Tasks 1/3/4/6 APIs, `fk_mesh_world_mm`/`fk` full-mesh FK, `ReprojectionTool`, `_umeyama`/`_model_to_mm`/`_triangulate_kp_mm`.
- Produces: `filled_tri_iou(verts2d (V,2), faces (F,3), mask_shape, ref_mask) -> float`; `run_polish(..., *, erode_px=8, sdf_hw=(128,128), bbox_margin=0.4, margin=0.0, containment_weight=0.3, silhouette_weight=0.3, mesh_subset="fps_300", appendage_include=("wing","leg","abdomen"))` returning a report dict with `iou_before/iou_after` (filled-tri), `reproj_px_before/after`, `mpjpe_stac_before/after`, `n_frames`.

- [ ] **Step 1: Write the filled-tri IoU unit test**

Add to `third_party/jarvis_jax/tests/test_run_silhouette_polish.py`:

```python
import numpy as np
from jarvis_jax.cse.run_silhouette_polish import filled_tri_iou


def test_filled_tri_iou_perfect_and_partial():
    # a single triangle covering a known region vs a matching mask
    verts = np.array([[10, 10], [40, 10], [10, 40]], float)
    faces = np.array([[0, 1, 2]])
    import cv2
    ref = np.zeros((50, 50), np.uint8)
    cv2.fillPoly(ref, [verts.astype(np.int32)], 1)
    ref = ref.astype(bool)
    assert filled_tri_iou(verts, faces, ref.shape, ref) > 0.99
    empty = np.zeros((50, 50), bool)
    assert filled_tri_iou(verts, faces, empty.shape, empty) == 0.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_silhouette_polish.py::test_filled_tri_iou_perfect_and_partial -v`
Expected: FAIL with `ImportError: cannot import name 'filled_tri_iou'`.

- [ ] **Step 3: Add `filled_tri_iou` and rewire `run_polish`**

In `run_silhouette_polish.py`, add near the top (after imports):

```python
def filled_tri_iou(verts2d, faces, mask_shape, ref_mask) -> float:
    """Filled-triangle silhouette hard-IoU: rasterize the projected mesh faces
    (cv2.fillPoly) and IoU vs ref_mask. The overlay methodology (a true filled
    silhouette, not a point splat)."""
    import cv2
    H, W = mask_shape
    sil = np.zeros((H, W), np.uint8)
    v = np.asarray(verts2d)
    tris = v[np.asarray(faces)].astype(np.int32)     # (F,3,2)
    cv2.fillPoly(sil, tris, 1)
    sil = sil > 0
    ref = np.asarray(ref_mask, bool)
    inter = np.logical_and(sil, ref).sum()
    union = np.logical_or(sil, ref).sum()
    return float(inter / union) if union > 0 else 0.0
```

Rewrite `run_polish` to: (a) build targets via the new dict API + `erode_px`; (b) build the SDF stack; (c) build the appendage DOF mask + containment vertex subset + `conf_v`; (d) call `solve_trajectory` with both terms; (e) compute filled-tri IoU (full-mesh FK per frame) + reproj + MPJPE-vs-STAC. Replace the body from the `build_silhouette_targets` call (line 120) through the `return` (line 237) with:

```python
    tg = build_silhouette_targets(root, split, fs_imgids, calib_dir,
                                  n_points=n_points, erode_px=erode_px)
    sdf = build_sdf_stack(root, split, fs_imgids, calib_dir,
                          out_hw=sdf_hw, bbox_margin=bbox_margin)

    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    faces = np.asarray(anat["faces"])
    full_idx = np.arange(len(anat["vlocal"]), dtype=np.int32)
    cov_idx = silhouette_fk_indices(mesh_npz, subset=mesh_subset)
    cont_idx = appendage_vertex_indices(mesh_npz, subset=mesh_subset, include=appendage_include)
    sil_qs = build_appendage_dof_mask(anat["m"], include=appendage_include)
    conf_v = jnp.ones((len(cont_idx),))

    cam_Ms = jnp.asarray(tg["cam_Ms"]); cam_ts = jnp.asarray(tg["cam_ts"])
    solver = SilhouetteJaxlsBatchSolver(
        n_iter=n_iter, smooth_weight=smooth_weight, beta=beta, huber_delta=huber_delta)

    def _solve(sw, cw):
        return np.asarray(solver.solve_trajectory(
            q_init=q_init, mjx_model=inputs["mjx_model"], mjx_data_template=inputs["mjx_data"],
            kp_data=kp_data, qs_to_opt=inputs["qs_to_opt"], kps_to_opt=inputs["kps_to_opt"],
            lb=inputs["lb"], ub=inputs["ub"], site_idxs=inputs["site_idxs"],
            q_reg_weights=inputs["q_reg_weights"], fk_repose=fk,
            cov_vert_indices=cov_idx, cont_vert_indices=cont_idx,
            cam_Ms=cam_Ms, cam_ts=cam_ts, boundary_all=jnp.asarray(tg["boundary"]),
            conf_p_all=jnp.asarray(tg["conf_p"]), sil_qs_mask=sil_qs, silhouette_weight=sw,
            sdf_all=jnp.asarray(sdf["sdf"]), grid_scale_all=jnp.asarray(sdf["grid_scale"]),
            grid_offset_all=jnp.asarray(sdf["grid_offset"]), present_all=jnp.asarray(sdf["present"]),
            conf_v=conf_v, containment_weight=cw, margin=margin))

    q_before = _solve(0.0, 0.0)                          # keypoint-only baseline
    q_after = _solve(silhouette_weight, containment_weight)

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    coco_kpnames = coco["keypoint_names"]
    kp_names = list(inputs["kp_names"])
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}

    def _metrics(qtraj):
        mjx_model = inputs["mjx_model"]; mjx_data_template = inputs["mjx_data"]
        site_idxs = inputs["site_idxs"]
        ious, reprojs, mpjpes = [], [], []
        for t in range(T):
            cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
            kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi)
            if kok.sum() < 3:
                continue
            data_t = mjx_data_template.replace(qpos=jnp.asarray(qtraj[t]))
            data_t = stac_utils.kinematics(mjx_model, data_t)
            data_t = stac_utils.com_pos(mjx_model, data_t)
            sites_model = np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))
            s, R, tr = _umeyama(sites_model[kok], kp_mm[kok])
            # MPJPE vs STAC init pose (both FK'd sites -> mm via same bridge)
            data_s = mjx_data_template.replace(qpos=jnp.asarray(q_init[t]))
            data_s = stac_utils.kinematics(mjx_model, data_s)
            data_s = stac_utils.com_pos(mjx_model, data_s)
            sites_stac = np.asarray(stac_utils.get_site_xpos(data_s, site_idxs))
            mpjpes.append(float(np.mean(np.linalg.norm(
                _model_to_mm(sites_model, s, R, tr) - _model_to_mm(sites_stac, s, R, tr), axis=1))))
            # filled-tri IoU on the FULL mesh (overlay methodology), affine projection
            verts_model = np.asarray(fk(jnp.asarray(qtraj[t].astype(np.float32)), 1.0, full_idx))
            verts_mm = _model_to_mm(verts_model, s, R, tr)
            for c, iid in cam2img.items():
                ann = _ann_for_image(id2ann_multi, iid, None)
                if ann is None:
                    continue
                mask = _load_sam_mask(root, split, id2file[iid], ann["id"])
                if mask is None:
                    continue
                mask = np.asarray(mask)
                uv = verts_mm @ np.asarray(tg["cam_Ms"])[c].T + np.asarray(tg["cam_ts"])[c]
                ious.append(filled_tri_iou(uv, faces, mask.shape, mask))
            for j, nm in enumerate(kp_names):
                ci = name2coco.get(nm)
                if ci is None:
                    continue
                for c, iid in cam2img.items():
                    ann = _ann_for_image(id2ann_multi, iid, None)
                    if ann is None:
                        continue
                    kp2d = np.asarray(ann["keypoints"], float).reshape(-1, 3)
                    if kp2d[ci, 2] > 0:
                        uvp = _model_to_mm(sites_model[j][None], s, R, tr)[0]
                        uvp = uvp @ np.asarray(tg["cam_Ms"])[c].T + np.asarray(tg["cam_ts"])[c]
                        reprojs.append(float(np.linalg.norm(uvp - kp2d[ci, :2])))
        return (float(np.mean(ious)) if ious else float("nan"),
                float(np.sqrt(np.mean(np.square(reprojs)))) if reprojs else float("nan"),
                float(np.mean(mpjpes)) if mpjpes else float("nan"))

    iou_b, reproj_b, mpjpe_b = _metrics(q_before)
    iou_a, reproj_a, mpjpe_a = _metrics(q_after)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, f"{recording}_polish_qpos.npz"),
             q_before=q_before, q_after=q_after)
    return dict(iou_before=iou_b, iou_after=iou_a,
                reproj_px_before=reproj_b, reproj_px_after=reproj_a,
                mpjpe_stac_before=mpjpe_b, mpjpe_stac_after=mpjpe_a, n_frames=T)
```

Update the `run_polish` signature (line 80) to add the new kwargs and the imports block (lines 90-99) to add `from jarvis_jax.cse.silhouette_sdf import build_sdf_stack` and `from jarvis_jax.cse.silhouette_dof import build_appendage_dof_mask, appendage_vertex_indices`. Update `main()` (lines 240-274) to add `--containment-weight` (default 0.3), `--erode-px` (default 8), `--margin` (default 0.0), `--sdf-hw` (default 128), and print the three before/after deltas with the honest do-no-harm framing (IoU up = good; reproj RMSE + MPJPE-vs-STAC must not worsen beyond tolerance).

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_silhouette_polish.py -v`
Expected: filled-tri IoU test passes; existing import/CLI/soft-IoU unit tests pass; the heavy real-recording smoke skips if the ik h5 is absent (CPU).

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/run_silhouette_polish.py third_party/jarvis_jax/tests/test_run_silhouette_polish.py
git commit -m "feat(cse): silhouette polish driver — containment+coverage, filled-tri IoU + do-no-harm guards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: GPU validation run on the male (acceptance)

Run the full driver on the male recording (GPU) and confirm the success criterion. NOT a pytest — a coordinator-run acceptance step; the deliverable is the truthfully-reported before/after table.

**Files:** none created; produces `<out_dir>/2026_03_18_15_31_22_polish_qpos.npz` + the printed report (out_dir under the scratchpad or cse_work, NOT committed).

- [ ] **Step 1: Run the full test suite (CPU) once more**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_sdf.py tests/test_silhouette_containment.py tests/test_silhouette_dof.py tests/test_silhouette_targets.py tests/test_silhouette_factor.py tests/test_silhouette_joint_solve.py tests/test_silhouette_containment_solve.py tests/test_run_silhouette_polish.py -v`
Expected: all pass (skips only where heavy assets are legitimately absent on CPU).

- [ ] **Step 2: Run the driver on the male (GPU)**

Run (on the gpu-l40s node, env activated, `unset LD_LIBRARY_PATH`):

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
python -m jarvis_jax.cse.run_silhouette_polish \
  --recording 2026_03_18_15_31_22 \
  --ik-h5 /gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5 \
  --xml /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml \
  --mesh /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_visual_canonical_wings.npz \
  --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \
  --split val --max-frames 60 \
  --silhouette-weight 0.3 --containment-weight 0.3 --erode-px 8 \
  --out-dir /gscratch/portia/eabe/data/Johnson_lab/cse_work/silhouette_polish_male
```

Expected: prints `iou_before/iou_after`, `reproj_px_before/after`, `mpjpe_stac_before/after`.

- [ ] **Step 3: Confirm the success criterion and report truthfully**

PASS if `iou_after > iou_before` (filled-tri IoU improves) AND `reproj_px_after <= reproj_px_before + 0.5` AND `mpjpe_stac_after <= mpjpe_stac_before * 1.02`. If IoU does not improve or a guard regresses, report it honestly (do NOT tune to pass); record the numbers and diagnose (weights, `erode_px`, `margin`, containment vertex density). Update the `female-keypoint-effort` memory with the validated outcome.

---

## Self-Review

**1. Spec coverage:** Containment residual (§3.1) → Task 2; halo-eroded coverage (§3.2) → Task 4 + Task 5; asymmetry via `relu` → Task 2; DOF mask (§4.1) with the `stop_gradient` assembly → Task 3 + Tasks 5/6; confidence plumbing (§4.2) `conf_v`/`conf_p`/`present` → Tasks 2/4/6; `FrameVar` data feeding (§5) → Tasks 5/6; SDF precompute (§6) → Task 1; file structure (§6) → Tasks 1-7; data flow (§7) → Task 7; testing (§8) → each task's tests; validation protocol (§2, §7 primary IoU + do-no-harm) → Tasks 7-8. No spec requirement is unmapped.

**2. Placeholder scan:** every code step contains complete code; commands have expected output; no "TBD"/"handle edge cases"/"similar to". Clear.

**3. Type consistency:** `build_silhouette_targets` returns a dict (Task 4) consumed by name in Tasks 5/7. `build_sdf_stack` keys (`sdf/grid_scale/grid_offset/present/cam_Ms/cam_ts`) match `make_containment_cost` args (Task 6) and the driver call (Task 7). `make_silhouette_cost`/`make_containment_cost` signatures match their `solve_trajectory` call sites. `cov_vert_indices`/`cont_vert_indices` are distinct and used consistently (coverage = `fps_300`; containment = appendage subset). `filled_tri_iou(verts2d, faces, mask_shape, ref_mask)` matches its call in `_metrics`. Weight convention (residual multiplier) is consistent across both terms and documented in Global Constraints.
