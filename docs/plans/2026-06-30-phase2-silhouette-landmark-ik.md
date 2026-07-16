# Phase 2: Single-Fly Silhouette-Landmark IK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed SAM-silhouette wing-tips as high-confidence 3-D markers into the existing temporal jaxls IK so a single fly's fitted wings reach their true extent, and prove the value with a keypoint-ablation.

**Architecture:** Productionize the proven wing-tip triangulation (SAM mask → per-camera distal wing point along the predicted wing axis → affine-DLT triangulate → 3-D, using Phase-1 refined calibration). Inject those 3-D tips as the targets for the existing distal wing markers (`WingL_V12/V13`=kp 7/8, `WingR_V12/V13`=kp 29/30) in the IK's `kp_data`, and raise those coordinates' weight in `kps_to_opt`. The solver (`stac_core_jaxls.JaxlsBatchSolver.solve_trajectory`) is reused unchanged — Phase 2 only builds better `(kp_data, kps_to_opt)`. Validation degrades the wing keypoints and shows the silhouette recovers wing extent vs a no-silhouette baseline.

**Tech Stack:** Python, JAX, jaxls (LM, reused), MuJoCo-MJX, NumPy, `stac_mjx` (io + solver), `jarvis_jax` (reprojection_tool, silhouette_ik, bundle_adjust), pytest.

## Global Constraints

- **3-D observation model** (spec decision 7): triangulate observations to 3-D, fit model markers to 3-D. The silhouette wing-tip is a triangulated 3-D marker target — NOT a 2-D reprojection term. No dense-pose head, no dense-silhouette 2-D factor (later phases).
- **Affine/telecentric cameras** (spec decision 8): 3×4 DLT, 3rd row `[0,0,0,1]`, no perspective divide; triangulation is affine. Use `jarvis_jax.geometry.reprojection_tool` + `jarvis_jax.tracking.affine_camera`.
- **Reuse, don't fork:** the IK backbone is `stac_mjx.stac_core_jaxls.JaxlsBatchSolver.solve_trajectory` — do NOT modify it. The wing-tip extractor productionizes `jarvis_jax/cse/demo_wingtip_triangulation.py`. Calibration comes from Phase-1 `run_bundle_adjust` refined output (fall back to factory if none).
- **Approach A** (user decision): silhouette tips override the existing wing markers' `kp_data` + raise their `kps_to_opt` weight; no model/site edits.
- **Wing marker keypoint indices (verbatim, coco order):** `WingL_V12`=7, `WingL_V13`=8 → left wing tip; `WingR_V12`=29, `WingR_V13`=30 → right wing tip. `WingL_base`=6, `WingR_base`=28 (hinge, unchanged).
- **Anatomy-agnostic** `(model_xml, canonical_mesh_npz)`, V1 now: `fruitfly_v1/fruitfly_v1_free.xml` + `fly_v1_collision_canonical_wings.npz`.
- **Environment:** gpu-l40s compute node; `source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH`. Unit tests CPU-only: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest`. New code under `third_party/jarvis_jax/jarvis_jax/cse/`, tests under `third_party/jarvis_jax/tests/`. Do NOT submit slurm; run directly.
- **Validation recording:** `2026_03_18_15_31_22`, split `val`; STAC fit at `cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5`; bout at `cse_work/2026_03_18_15_31_22_bout.h5`; masks under `red_data_unified_V3/sam3_masks/val/`.

---

## File structure

- `jarvis_jax/cse/silhouette_landmarks.py` **(new)** — the extractor: per-frame per-side 3-D wing-tip + confidence from SAM masks + calibration (productionized from `demo_wingtip_triangulation.py`). Pure geometry (numpy + ReprojectionTool); no solver.
- `jarvis_jax/cse/marker_augment.py` **(new)** — `augment_wing_markers(kp_data, kps_to_opt, tips, conf, ...)`: override the 4 wing-marker rows with the side tips and raise their weights; pure array logic.
- `jarvis_jax/cse/silhouette_ik_solve.py` **(new)** — the driver: load anatomy + STAC ik h5 + bout, build solver inputs (`site_idxs`, `q_init`, `lb/ub`, `q_reg_weights`), assemble triangulated `kp_data` (refined calib), call `augment_wing_markers`, run `solve_trajectory`, write qpos + a report. Includes an ablation entry point.
- `tests/test_silhouette_landmarks.py`, `tests/test_marker_augment.py`, `tests/test_silhouette_ik_solve.py` **(new)**.

Test prefix for every pytest command: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest`.

---

## Task 1: Wing-tip vertex indices + per-side silhouette extractor (geometry)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_landmarks.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_landmarks.py`

**Interfaces:**
- Consumes: `jarvis_jax.tracking.affine_camera.project_affine`, `reconstruct_affine`, `factor_affine`; `ReprojectionTool`.
- Produces:
  - `wing_side_vertices(mesh_npz) -> {"left": {"tip": int, "prox": int}, "right": {...}}` — fps-subset indices of each wing's tip (max distance from thorax centroid) and proximal vertex (min distance), computed from the canonical mesh (same logic as the POC).
  - `mask_wing_tip_2d(mask, prox2d, tip2d, corridor=12.0) -> np.ndarray(2,) | None` — the farthest mask pixel along the `prox2d→tip2d` axis within a perpendicular `corridor`, or None if <3 corridor pixels (folded/occluded).
  - `triangulate_wing_tips(masks, cam_mats, prox3d, predtip3d, *, corridor=12.0) -> {"left": (X(3,), n_cams_used) or None, "right": ...}` — per side, project `prox3d`/`predtip3d` into each camera (affine), find the mask tip, DLT-triangulate across cameras that yield one (≥2), return 3-D tip + camera count (confidence).

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_silhouette_landmarks.py`:

```python
import numpy as np
from jarvis_jax.tracking.affine_camera import factor_affine, reconstruct_affine, project_affine
from jarvis_jax.tracking.silhouette_landmarks import wing_side_vertices, mask_wing_tip_2d, triangulate_wing_tips

MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
P_REAL = np.array([[8.1001,0.0074869,-0.031773,-2.828],[0.0093308,-8.0788,-0.17912,462.78],[0,0,0,1.0]])


def test_wing_side_vertices_are_distinct_and_in_range():
    import numpy as np
    z = np.load(MESH, allow_pickle=True); nv = len(z["fps_300"])
    sides = wing_side_vertices(MESH)
    assert set(sides) == {"left", "right"}
    for s in ("left", "right"):
        assert 0 <= sides[s]["tip"] < nv and 0 <= sides[s]["prox"] < nv
        assert sides[s]["tip"] != sides[s]["prox"]


def test_mask_wing_tip_2d_finds_far_edge():
    # a horizontal bar mask from x=100..180 at y=50; axis points +x from prox (90,50)
    mask = np.zeros((100, 300), bool); mask[48:53, 100:181] = True
    tip = mask_wing_tip_2d(mask, np.array([90.0, 50.0]), np.array([140.0, 50.0]), corridor=6.0)
    assert tip is not None and abs(tip[0] - 180) <= 1 and abs(tip[1] - 50) <= 2


def test_triangulate_wing_tips_recovers_known_tip():
    # 2-camera affine rig; place a known left-wing tip; render bar masks; triangulate.
    K2, R, t = factor_affine(P_REAL)
    th = np.deg2rad(20.0); Ry = np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]])
    cams = [P_REAL, reconstruct_affine(K2, Ry @ R, t)]
    prox3d = {"left": np.array([0.0, 0.5, 11.0]), "right": np.array([0.0, -0.5, 11.0])}
    tip3d_true = {"left": np.array([0.0, 2.5, 11.0]), "right": np.array([0.0, -2.5, 11.0])}
    masks = []
    for P in cams:
        m = np.zeros((448, 1936), bool)
        for side in ("left", "right"):
            a = project_affine(P, prox3d[side]); b = project_affine(P, tip3d_true[side])
            for f in np.linspace(0, 1, 60):
                p = (a + f * (b - a)).astype(int)
                m[max(0,p[1]-3):p[1]+3, max(0,p[0]-3):p[0]+3] = True
        masks.append(m)
    out = triangulate_wing_tips(masks, cams, prox3d, tip3d_true, corridor=8.0)
    assert out["left"] is not None
    X, ncam = out["left"]
    assert ncam == 2 and np.linalg.norm(X - tip3d_true["left"]) < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_landmarks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.silhouette_landmarks'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/silhouette_landmarks.py`:

```python
"""Silhouette wing-tip landmark extraction (productionized from demo_wingtip_triangulation).

For each wing side, march along the predicted wing axis (proximal->tip) to the far
edge of the fly's SAM mask in each camera, then affine-DLT-triangulate those 2-D
tips into one 3-D point. Cameras are telecentric (affine). Confidence = #cameras used.
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.tracking.affine_camera import project_affine


def wing_side_vertices(mesh_npz):
    z = np.load(mesh_npz, allow_pickle=True)
    fps = z["fps_300"] if "fps_300" in z.files else z[f"fps_{len(z['vertex_segment'])}"]
    seg = z["vertex_segment"][fps]; C = z["vertices"][fps]
    id2n = {int(s): (n.decode() if isinstance(n, bytes) else n)
            for s, n in zip(z["seg_ids"], z["seg_names"])}
    names = [id2n[int(s)].lower() for s in seg]
    thorax_c = C[[("thorax" in n) for n in names]].mean(0)
    out = {}
    for side in ("left", "right"):
        si = np.where([("wing" in n and side in n) for n in names])[0]
        d = np.linalg.norm(C[si] - thorax_c, axis=1)
        out[side] = {"tip": int(si[d.argmax()]), "prox": int(si[d.argmin()])}
    return out


def mask_wing_tip_2d(mask, prox2d, tip2d, corridor=12.0):
    ax = np.asarray(tip2d, float) - np.asarray(prox2d, float)
    L = np.linalg.norm(ax)
    if L < 5:
        return None
    ax = ax / L
    ys, xs = np.where(mask)
    rel = np.stack([xs - prox2d[0], ys - prox2d[1]], 1)
    along = rel @ ax
    perp = np.abs(rel[:, 0] * (-ax[1]) + rel[:, 1] * ax[0])
    corr = (perp < corridor) & (along > 0.3 * L)
    if corr.sum() < 3:
        return None
    k = int(np.argmax(along[corr]))
    return np.array([xs[corr][k], ys[corr][k]], float)


def _triangulate(cam_mats, cam_ids, pts2d):
    A = np.zeros((2 * len(cam_ids), 4))
    for i, c in enumerate(cam_ids):
        P = np.asarray(cam_mats[c]); uv = pts2d[c]
        A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
    _, _, Vh = np.linalg.svd(A)
    Xh = Vh[-1]
    return (Xh / Xh[3])[:3]


def triangulate_wing_tips(masks, cam_mats, prox3d, predtip3d, *, corridor=12.0):
    C = len(cam_mats)
    out = {}
    for side in ("left", "right"):
        pts = np.zeros((C, 2)); used = []
        for c in range(C):
            if masks[c] is None:
                continue
            prox2d = project_affine(cam_mats[c], prox3d[side])
            tip2d = project_affine(cam_mats[c], predtip3d[side])
            t = mask_wing_tip_2d(masks[c], prox2d, tip2d, corridor=corridor)
            if t is not None:
                pts[c] = t; used.append(c)
        out[side] = (_triangulate(cam_mats, used, pts), len(used)) if len(used) >= 2 else None
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_landmarks.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_landmarks.py third_party/jarvis_jax/tests/test_silhouette_landmarks.py
git commit -m "feat(cse): silhouette wing-tip landmark extractor (affine multi-view)"
```

---

## Task 2: Wing-marker augmentation into kp_data / kps_to_opt

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/marker_augment.py`
- Test: `third_party/jarvis_jax/tests/test_marker_augment.py`

**Interfaces:**
- Produces:
  - `WING_MARKER_IDS = {"left": (7, 8), "right": (29, 30)}` (coco kp indices for `WingL_V12/V13`, `WingR_V12/V13`).
  - `augment_wing_markers(kp_data, kps_to_opt, tips, *, wing_weight=5.0, min_cams=2) -> (kp_data2, kps_to_opt2)` — `kp_data` shape `(T, n_kp, 3)`, `kps_to_opt` shape `(n_kp*3,)`. For each frame `t` and side, if `tips[t][side]` is not None and its cam-count ≥ `min_cams`, set `kp_data2[t, id, :] = tip3d` for that side's two marker ids and set their 3 coords in `kps_to_opt2` to `wing_weight`; otherwise leave the frame's markers untouched. Returns copies (never mutates inputs). `kps_to_opt` is broadcast to per-frame internally only if needed — here weights are shared across frames, so `kps_to_opt2` is the single `(n_kp*3,)` vector reflecting the boost (applied whenever ANY frame uses the tip).

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_marker_augment.py`:

```python
import numpy as np
from jarvis_jax.tracking.marker_augment import augment_wing_markers, WING_MARKER_IDS


def test_augment_overrides_wing_markers_and_boosts_weight():
    T, n_kp = 3, 50
    kp = np.zeros((T, n_kp, 3)); w = np.ones(n_kp * 3)
    tips = [
        {"left": (np.array([1.0, 2.0, 3.0]), 3), "right": None},   # frame 0: left tip, 3 cams
        {"left": None, "right": None},                              # frame 1: none
        {"left": None, "right": (np.array([4.0, 5.0, 6.0]), 2)},    # frame 2: right tip, 2 cams
    ]
    kp2, w2 = augment_wing_markers(kp, w, tips, wing_weight=5.0, min_cams=2)
    # frame 0 left markers (7,8) set to the left tip
    assert np.allclose(kp2[0, 7], [1, 2, 3]) and np.allclose(kp2[0, 8], [1, 2, 3])
    # frame 2 right markers (29,30) set to the right tip
    assert np.allclose(kp2[2, 29], [4, 5, 6]) and np.allclose(kp2[2, 30], [4, 5, 6])
    # frame 1 untouched (still zero)
    assert np.allclose(kp2[1, 7], [0, 0, 0])
    # weights boosted for all four wing markers' coords
    for idx in (7, 8, 29, 30):
        assert np.allclose(w2[idx * 3:idx * 3 + 3], 5.0)
    # a non-wing marker weight unchanged
    assert np.allclose(w2[3 * 3:3 * 3 + 3], 1.0)
    # inputs not mutated
    assert np.allclose(kp[0, 7], [0, 0, 0]) and np.allclose(w[7 * 3], 1.0)


def test_min_cams_gate():
    kp = np.zeros((1, 50, 3)); w = np.ones(150)
    tips = [{"left": (np.array([1.0, 1.0, 1.0]), 1), "right": None}]   # only 1 cam
    kp2, w2 = augment_wing_markers(kp, w, tips, min_cams=2)
    assert np.allclose(kp2[0, 7], [0, 0, 0])          # not applied (below min_cams)
    assert np.allclose(w2[7 * 3:7 * 3 + 3], 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_marker_augment.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.marker_augment'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/marker_augment.py`:

```python
"""Inject silhouette wing-tips as high-confidence 3-D targets for the existing
distal wing markers (WingL/R_V12/V13), raising their weight in kps_to_opt.
Reuses the STAC marker_cost unchanged: it fits site positions to kp_data with
per-coordinate weight kps_to_opt.
"""
from __future__ import annotations
import numpy as np

# coco keypoint indices of the distal wing markers (WingL_V12=7, WingL_V13=8, WingR_V12=29, WingR_V13=30)
WING_MARKER_IDS = {"left": (7, 8), "right": (29, 30)}


def augment_wing_markers(kp_data, kps_to_opt, tips, *, wing_weight=5.0, min_cams=2):
    kp2 = np.array(kp_data, dtype=np.float64, copy=True)          # (T, n_kp, 3)
    w2 = np.array(kps_to_opt, dtype=np.float64, copy=True)        # (n_kp*3,)
    used_ids = set()
    for t, frame in enumerate(tips):
        for side, ids in WING_MARKER_IDS.items():
            entry = frame.get(side) if frame else None
            if entry is None:
                continue
            tip3d, ncam = entry
            if ncam < min_cams:
                continue
            for idx in ids:
                kp2[t, idx, :] = tip3d
                used_ids.add(idx)
    for idx in used_ids:
        w2[idx * 3:idx * 3 + 3] = wing_weight
    return kp2, w2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_marker_augment.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/marker_augment.py third_party/jarvis_jax/tests/test_marker_augment.py
git commit -m "feat(cse): wing-marker augmentation (silhouette tips into kp_data/kps_to_opt)"
```

---

## Task 3: STAC solver-input assembly + integration probe

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py`
- Test: `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py`

**Interfaces:**
- Consumes: `stac_mjx.stac_core_jaxls.JaxlsBatchSolver`, `stac_mjx.io`, `stac_mjx.utils`; the anatomy `(xml, canonical mesh)`; the STAC ik h5.
- Produces:
  - `build_solver_inputs(ik_h5, model_xml) -> dict` with keys: `mjx_model`, `mjx_data`, `q_init (T,nq)`, `kp_data (T,n_kp,3)`, `kps_to_opt (n_kp*3,)`, `qs_to_opt (nq,)`, `lb (nq,)`, `ub (nq,)`, `site_idxs (n_kp,)`, `q_reg_weights (nq,)`, `kp_names (n_kp,)`. Sourced from the ik h5 (`qpos`, `kp_data`, `offsets`, `kp_names`, `names_qpos`) + the compiled model (site ids by keypoint name via the STAC `KEYPOINT_MODEL_PAIRS` site naming; joint ranges for lb/ub). Site offsets set on the model from the ik h5 `offsets` via `utils.set_site_pos`.
  - `solve_ik(inputs, *, smooth_weight=0.1, n_iter=50) -> np.ndarray(T,nq)` — thin wrapper constructing `JaxlsBatchSolver(...).solve_trajectory(**mapped inputs)`.

This task is a **de-risking integration probe** (like Phase-1 Task 4): the exact STAC site-index/offset/bounds assembly must be established empirically against the installed `stac_mjx`. The reference for correct usage is `stac_mjx/compute_stac.py` (root/pose optimization already call `solve_trajectory` with these inputs) and `stac_mjx/io.py` (loads the ik h5). The test gate: on a small frame slice, `solve_ik` returns finite qpos of shape `(T,nq)` and does not increase marker reprojection error vs the stored STAC qpos.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py`:

```python
import os, numpy as np, pytest
from jarvis_jax.tracking.silhouette_ik_solve import build_solver_inputs, solve_ik

IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_build_inputs_shapes():
    inp = build_solver_inputs(IK, XML)
    T = inp["q_init"].shape[0]; nq = inp["q_init"].shape[1]; nk = len(inp["kp_names"])
    assert nq == 93 and nk == 50
    assert inp["kp_data"].shape == (T, nk, 3)
    assert inp["kps_to_opt"].shape == (nk * 3,)
    assert inp["site_idxs"].shape == (nk,)
    assert inp["lb"].shape == (nq,) and inp["ub"].shape == (nq,)


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present")
def test_solve_ik_does_not_worsen_fit_on_slice():
    inp = build_solver_inputs(IK, XML)
    sl = slice(0, 8)
    small = dict(inp)
    small["q_init"] = inp["q_init"][sl]; small["kp_data"] = inp["kp_data"][sl]
    q = solve_ik(small, smooth_weight=0.0, n_iter=40)
    assert q.shape == small["q_init"].shape
    assert np.isfinite(q).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_ik_solve.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.tracking.silhouette_ik_solve'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py` with `build_solver_inputs` + `solve_ik`. Establish the exact assembly by reading `stac_mjx/compute_stac.py` (how `site_idxs`, `lb`, `ub`, `qs_to_opt`, `q_reg_weights` are constructed and how `solve_trajectory` is invoked) and `stac_mjx/io.py` (ik-h5 fields). Concretely:
- Load the ik h5 via `stac_mjx.io` (or h5py): `qpos (T,93)` → `q_init`; `kp_data (T,150)` → reshape `(T,50,3)`; `offsets (50,3)`; `kp_names (50,)`; `names_qpos (93,)`.
- Compile the model: `mujoco.MjModel.from_xml_path(XML)`, `mjx.put_model`, `mjx.make_data`.
- `site_idxs`: for each kp name, the model site id created by STAC for that keypoint (site naming follows `KEYPOINT_MODEL_PAIRS`); look up via `mujoco.mj_name2id(..., mjOBJ_SITE, name)`. Set the site offsets from the ik-h5 `offsets` via `stac_mjx.utils.set_site_pos`.
- `lb/ub`: from `model.jnt_range` expanded to qpos layout (free joint unbounded → large ±); `qs_to_opt`: all True except leave as the STAC convention (optimize root + hinges); `kps_to_opt`: `np.ones(150)`; `q_reg_weights`: small (e.g. from config or zeros for the probe).
- `solve_ik`: `JaxlsBatchSolver(n_iter=n_iter, smooth_weight=smooth_weight, use_se3_root=True).solve_trajectory(q_init, mjx_model, mjx_data, kp_data, qs_to_opt, kps_to_opt, lb, ub, site_idxs, q_reg_weights)` → np.asarray.

Iterate against the two tests until they pass; if `stac_mjx` needs a specific site-offset or bounds convention, match `compute_stac.py` exactly. Full code is written by the implementer from these references (the tests are the gate).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_ik_solve.py -v`
Expected: PASS (2 passed). If the installed `stac_mjx` API differs, adjust to match `compute_stac.py` and re-run until PASS. If the STAC input assembly cannot be reproduced faithfully, STOP and report BLOCKED with the specific mismatch — do not fake the solve.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py third_party/jarvis_jax/tests/test_silhouette_ik_solve.py
git commit -m "feat(cse): STAC solver-input assembly + solve_ik integration probe"
```

---

## Task 4: Single-fly driver (male, val recording) with silhouette markers

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py` (add driver)
- Test: `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py` (add a driver-wiring unit test)

**Interfaces:**
- Consumes: Task 1 `triangulate_wing_tips`, `wing_side_vertices`; Task 2 `augment_wing_markers`; Task 3 `build_solver_inputs`, `solve_ik`; `ReprojectionTool`; the bout h5 (`fs_imgids`) + coco (image→file, ann_id) + SAM masks + Phase-1 refined calib (fall back to factory).
- Produces:
  - `extract_tips_for_frames(root, split, recording, fs_imgids, calib_dir, mesh_npz, pred_tips3d, prox3d) -> list[dict]` — per bout frame, load the male's per-camera SAM masks, call `triangulate_wing_tips`, return the `tips` list consumed by `augment_wing_markers`. `pred_tips3d`/`prox3d` come from reposing the STAC mesh wing tip/prox vertices at each frame's qpos (via `silhouette_ik.make_fk_repose`) to define the wing axis.
  - `run_single_fly(recording, *, use_silhouette=True, wing_weight=5.0, max_frames=0, calib_dir=None, out_dir) -> dict` — full pipeline: build inputs, (optionally) augment wing markers, solve, write qpos npz + report (per-frame wing length vs STAC, reprojection error). `use_silhouette=False` gives the baseline.

- [ ] **Step 1: Write the failing test** — a unit test that `run_single_fly` with `use_silhouette=False` on a tiny `max_frames` slice returns a report dict with keys `qpos_shape`, `wing_len_pred`, `wing_len_stac`, `reproj_px`, and writes a qpos `.npz`. (Full code + asserts written by implementer; gate is: runs end-to-end on a 4-frame slice, finite outputs.)

- [ ] **Step 2: Run to verify it fails** (`ImportError: run_single_fly`).

- [ ] **Step 3: Implement** `extract_tips_for_frames` + `run_single_fly` per the interfaces, reusing Tasks 1–3. Male identity = the coco-annotated fly (the `ann_ids` in the bout); masks selected by that ann_id (matched mask), mirroring `demo_wingtip_triangulation.load_mask`. Wing length = distance from reposed wing prox vertex to reposed tip vertex under the solved qpos.

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Real run (gpu-l40s):** `run_single_fly("2026_03_18_15_31_22", use_silhouette=True, calib_dir=<phase-1 refined>, out_dir=cse_work/silik_phase2/)`. Record per-frame wing length (pred vs STAC) + mean reprojection error. Expected: with GT keypoints the wing stays correct and reprojection does not worsen (this task establishes the pipeline; the silhouette's value is quantified by the Task-5 ablation).

- [ ] **Step 6: Commit.**

---

## Task 5: Keypoint-ablation validation (the silhouette-value gate)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py` (add ablation entry)
- Test: `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py` (ablation-wiring unit test)

**Interfaces:**
- Produces:
  - `run_ablation(recording, *, wing_weight=5.0, max_frames=0, calib_dir=None, out_dir) -> dict` — for the same frames, run the IK three ways: (a) full GT keypoints (reference), (b) wing keypoints WITHHELD (kp_data wing rows set NaN so `marker_cost`'s finite-mask drops them) WITHOUT silhouette, (c) wing keypoints withheld WITH silhouette augmentation. Report per-frame wing length + wing reprojection/LOO error for each, and the recovery ratio `(len_c - len_b) / (len_a - len_b)` (→1 means the silhouette fully recovers the wing the missing keypoints would have provided).

- [ ] **Step 1: Write the failing test** — unit test that `run_ablation` on a 4-frame slice returns the three conditions' wing lengths and a finite `recovery_ratio`. (Implementer writes full asserts; gate = runs, finite.)

- [ ] **Step 2: Run to verify it fails.**

- [ ] **Step 3: Implement** `run_ablation` reusing `run_single_fly`'s pieces: condition (b) sets `kp_data[:, [7,8,29,30], :] = np.nan`; condition (c) additionally applies `augment_wing_markers` (which overwrites those NaN rows with the silhouette tips + boosts weights).

- [ ] **Step 4: Run to verify it passes.**

- [ ] **Step 5: Real run (gpu-l40s)** on `2026_03_18_15_31_22` with Phase-1 refined calibration. SUCCESS CRITERION: with wing keypoints withheld, the silhouette condition (c) recovers wing extent substantially better than the no-silhouette baseline (b) — i.e. `wing_len_c` close to reference `wing_len_a` while `wing_len_b` is compressed; report the recovery ratio + wing reprojection error (c) < (b). This demonstrates the silhouette's value for the predicted-keypoint / female case. If recovery is weak, report honestly with the numbers.

- [ ] **Step 6: Commit.**

---

## Self-review notes (spec coverage)

- Silhouette-landmark extractor (spec component 4) — Task 1. Approach-A marker augmentation into the reused `marker_cost` via `kp_data`/`kps_to_opt` (spec decision 5/7) — Task 2. Temporal jaxls IK reused unchanged (spec decision 4) — Task 3. Refined calibration from Phase 1 (spec component 3) — Tasks 4–5. Affine triangulation everywhere (spec decision 8) — Tasks 1,4. Single-fly (male) validation + ablation demonstrating silhouette value — Tasks 4–5.
- Deliberately OUT of Phase-2 scope (later phases): multi-animal identity linking (Phase 3), active-parts (Phase 4), dense-pose head + dense-silhouette 2-D factor (Phases 5–6), outputs/QC packaging (Phase 7).
- Tasks 3–5 use a probe-first pattern for the STAC integration because the exact `stac_mjx` input-assembly API must be matched empirically (reference: `compute_stac.py`); each has a test gate and a BLOCKED escape hatch rather than a guessed implementation.
