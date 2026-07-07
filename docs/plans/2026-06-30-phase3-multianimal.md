# Phase 3: Multi-Animal (Cross-Camera Identity + Per-Fly Silhouette IK) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** On a courtship recording where each of the 7 synchronized cameras carries up to 2 COCO annotations per image (two flies), recover which annotation in each camera is the same physical fly (cross-camera identity), build a per-fly triangulated bout, solve per-fly STAC + silhouette-landmark IK, and prove both flies reproject cleanly — including a keypoint-withhold ablation showing the silhouette recovers a second fly's withheld wing.

**Architecture:** A JAX/NumPy-native affine identity linker triangulates each candidate cross-camera pairing of the two flies' keypoints and picks the assignment with minimum multi-view reprojection residual (mirroring the JARVIS torch `BoutMasks.assign_identities` algorithm, generalized to per-frame COCO framesets and affine cameras — never importing the torch code). The resulting per-frameset identity map de-collapses the existing `_index_coco` first-ann-wins bug: it drives a per-fly bout builder (one bout h5 per fly in the exact schema `run_stac_bout`/`build_solver_inputs` already consume), a per-fly STAC solve, and a per-fly silhouette IK that reuses the Phase-2 `run_single_fly`/`run_ablation` machinery unchanged except for a new `ann_id_by_image` selection argument threaded through `extract_tips_for_frames`/`_triangulate_kp_mm`/`_load_sam_mask` (backward-compatible default None = current single-ann behavior). The temporal jaxls solver (`stac_core_jaxls.JaxlsBatchSolver`) is reused unmodified.

**Tech Stack:** Python, JAX, NumPy, jaxls (LM, reused via `stac_mjx.stac_core_jaxls`), MuJoCo-MJX, `stac_mjx` (io + solver + `run_stac`), `jarvis_jax` (`reprojection_tool`, `affine_camera`, `silhouette_landmarks`, `silhouette_ik`, `marker_augment`, `cse_labels`, `run_stac_bout`), h5py, pytest.

## Global Constraints

- **Identity linker is JAX-native affine** (spec decision 9 / user decision 3): build the linker from the existing affine-correct primitives (`ReprojectionTool.reconstruct_point`/`reproject_point`, `affine_camera.project_affine`/`reconstruct_affine`), mirroring the `BoutMasks.assign_identities` *algorithm* only — do NOT import `JARVIS-HybridNet/jarvis/prediction/sam3_video_tracker.py`.
- **Affine/telecentric cameras** (spec decision 8): per-camera 3×4 DLT with 3rd row `[0,0,0,1]`; projection `uv = P[:2,:3]@X + P[:2,3]` (no perspective divide). `reproject_point`/`reconstruct_point`'s `/proj[2]` divide is a no-op for affine cams (divides by 1). Keep all geometry affine-consistent.
- **3-D observation model** (spec decision 7): triangulate observations to 3-D, fit model markers to 3-D. Silhouette landmarks enter as confidence-gated 3-D markers via `marker_augment.augment_wing_markers` (`only_missing=True`, default `wing_weight=0.5`). No 2-D reprojection IK term, no dense-pose head (later phases).
- **Reuse, don't fork:** the IK backbone is `stac_mjx.stac_core_jaxls.JaxlsBatchSolver.solve_trajectory` — do NOT modify it. Per-fly STAC reuses `jarvis_jax.cse.run_stac_bout.run` unchanged. Silhouette IK reuses `silhouette_ik_solve.run_single_fly`/`run_ablation`, extended only by an `ann_id_by_image` argument.
- **Anatomy-agnostic** `(model_xml, canonical_mesh_npz)`, V1 now: `/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml` (nq=93) + `/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz`.
- **Female handling = keypoint-withhold ablation on the 2nd fly** (user decision 2): reuse Phase-2 `run_ablation`/`_withhold_wing_kp`; withhold fly1's wing keypoints, show the silhouette recovers wing extent while both flies still reproject cleanly. NOT keypoint-starved-from-extra_masks, NOT a Phase-5 detector. Sex disambiguation (male vs female) is out of scope — use arbitrary stable `fly0`/`fly1` slots (both anns are `sex:"unknown"` for this recording); note it as a follow-up.
- **Validation recording:** `2026_04_07_11_33_33`, split `val`, data root `/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3`. Has BOTH flies COCO-annotated (181 two-annotation frames). Its existing `cse_work/2026_04_07_11_33_33/Fruitfly_ik_v1_cse.h5` and `cse_work/2026_04_07_11_33_33_bout.h5` are IDENTITY-COLLAPSED (first/last-ann-wins) and MUST NOT be trusted or reused; Phase 3 regenerates per-fly artifacts.
- **Calibration:** `red_data_unified_V3/calib_params/2026_04_07_11_33_33/Cam*.yaml` (7 cams). No refined calib for this recording yet; the orchestrator runs Phase-1 bundle adjustment for it first (reuse `run_bundle_adjust`), falling back to factory calib if BA regresses — an explicit step, not a hard dependency of the linker.
- **Environment:** session runs ON a GPU compute node (2×L40S). Run GPU work DIRECTLY in-shell: `source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH`. Do NOT sbatch-and-idle. Bash foreground max ~600s; for longer GPU runs use background bash (persists across turns), NOT SLURM.
- **Pure-logic/geometry unit tests run CPU-only.** Test prefix for every CPU pytest command: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest`. Real-data GPU tests are gated by `@pytest.mark.skipif(not os.path.exists(...))` on the `2026_04_07_11_33_33` artifacts and run under the activated env (no `JAX_PLATFORMS=cpu`).
- **New code** under `third_party/jarvis_jax/jarvis_jax/cse/`; **tests** under `third_party/jarvis_jax/tests/`.
- TDD, DRY, YAGNI, frequent commits; each task: failing test → run (fail) → minimal impl → run (pass) → commit. Real code in every step.
- **Branch:** `elliottabe/paper_update_062026` (rolling multi-phase branch; work directly on it).

---

## File structure

NEW under `third_party/jarvis_jax/jarvis_jax/cse/`:

- `identity_link.py` **(new, Tasks 1–2)** — JAX/NumPy affine cross-camera linker. Pure geometry (numpy + `ReprojectionTool`); no solver, no STAC. Core: `score_assignment`, `link_frameset` (Task 1); `link_recording` over COCO framesets (Task 2).
- `multifly_bout.py` **(new, Task 3)** — `build_fly_bout`: for each frameset, triangulate THIS fly's 50 keypoints from its per-camera annotations (selected by the identity map) → write a bout h5 in the exact `cse_labels.build_bout` schema. Produces 2 bout h5 (fly0, fly1). Reuses `cse_labels` scale/reorder helpers.
- `run_multifly_ik.py` **(new, Tasks 5–6)** — orchestrator/CLI: (optional Phase-1 BA) → `link_recording` → `build_fly_bout`×2 → `run_stac_bout`×2 → per-fly silhouette IK (reuse `run_single_fly` with per-fly `ann_id_by_image` + per-fly ik_h5 + per-recording calib_dir) → validation (both flies `reproj_px`) → 2nd-fly keypoint-withhold ablation (reuse `run_ablation`).

MODIFY:

- `silhouette_ik_solve.py` **(Task 4)** — thread an optional `ann_id_by_image: dict[int, int]` (per fly: coco `image_id` → chosen ann `id`) through `run_single_fly`/`run_ablation`/`extract_tips_for_frames`/`_triangulate_kp_mm`/`_load_sam_mask`, replacing the last-ann-wins `id2ann = {an["image_id"]: an for an in coco["annotations"]}` (lines 562, 800, 1089) with per-identity ann selection. Backward compatible: default `None` → current single-ann behavior.

TESTS under `third_party/jarvis_jax/tests/`:

- `test_identity_link.py` **(new, Tasks 1–2)** — synthetic affine fixture (Task 1) + real-data `link_recording` gate (Task 2).
- `test_multifly_bout.py` **(new, Task 3)** — per-fly bout schema + de-collapse (real-data-gated).
- Per-identity threading tests added to `test_silhouette_ik_solve.py` **(Task 4)**.
- `test_run_multifly_ik.py` **(new, Tasks 5–6)** — orchestrator wiring + real-data GPU per-fly + ablation gates.

**Shared test path constants** (used across the real-data-gated tests):

```python
ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ANATOMY = "<stac_mjx configs>/anatomy/v1.yaml"  # KEYPOINT_MODEL_PAIRS source for model kp order
```

---

## Task 1: Affine triangulation-consistency scorer + single-frameset 2-fly linker

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/identity_link.py`
- Test: `third_party/jarvis_jax/tests/test_identity_link.py`

**Interfaces:**
- Consumes: `jarvis_jax.geometry.reprojection_tool.ReprojectionTool` (`reconstruct_point(points2d (num_cam,2), cams_to_use: list[int]|None) -> (3,)`, `reproject_point(p3d (3,)) -> (num_cam,2)`); `jarvis_jax.cse.affine_camera.project_affine`, `reconstruct_affine`, `factor_affine` (test fixture only).
- Produces:
  - `score_assignment(kp2d_per_cam (n_cam, K, 3), cams_present: list[int], cam_mats (n_cam, 3, 4)) -> float` — given ONE fly's per-camera 2-D keypoints (last channel = coco visibility flag `v`; `v>0` = present), triangulate each of the `K` keypoints that is visible in ≥2 of `cams_present` (affine DLT via `_dlt_affine`), reproject into every present camera, and return the mean per-(keypoint, camera) pixel residual over visible observations. Returns `float("inf")` if no keypoint triangulates.
  - `_dlt_affine(cam_mats (n_cam,3,4), cam_ids: list[int], pts2d (n_cam,2)) -> (3,)` — module-private affine DLT-SVD (same math as `ReprojectionTool.reconstruct_point` / `silhouette_landmarks._triangulate`), indexing `cam_mats`/`pts2d` by absolute camera id.
  - `link_frameset(anns_by_cam: dict[int, list[dict]], cam_mats (n_cam, 3, 4), *, n_flies: int = 2, ref_cam: int | None = None, residual_gate_px: float = 40.0) -> dict[int, dict[int, int]]` — `anns_by_cam[cam_idx]` = the list of that camera's COCO annotation dicts (each with `keypoints` (150) and `id`). Fix the labeling at a reference camera (the first camera with `n_flies` anns, or `ref_cam`); for every other camera, choose the assignment of its anns to fly slots that minimizes total `score_assignment` residual across the pair, disambiguating cameras with `<n_flies` anns by matching each surviving ann to the nearest reprojected fly center. Returns `{fly_id: {cam_idx: ann_id}}` for `fly_id` in `0..n_flies-1`; a `fly_id` maps a `cam_idx` only when that camera has an ann assigned to that fly. Returns `{}` (all flies) if no camera has `n_flies` anns or if the winner's mean residual exceeds `residual_gate_px`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_identity_link.py`:

```python
import numpy as np
import pytest
from jarvis_jax.cse.affine_camera import factor_affine, reconstruct_affine, project_affine
from jarvis_jax.cse.identity_link import score_assignment, _dlt_affine, link_frameset

# A real telecentric DLT (from the rig) + 6 rotated copies -> 7 affine cameras
# with angular diversity, so affine triangulation is well-conditioned.
P_REAL = np.array([[8.1001, 0.0074869, -0.031773, -2.828],
                   [0.0093308, -8.0788, -0.17912, 462.78],
                   [0, 0, 0, 1.0]])


def _rig(n_cam=7):
    K2, R, t = factor_affine(P_REAL)
    cams = [P_REAL]
    for k in range(1, n_cam):
        th = np.deg2rad(15.0 * k)
        Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
        cams.append(reconstruct_affine(K2, Ry @ R, t))
    return np.stack(cams, 0)  # (n_cam, 3, 4)


def _project_fly(cam_mats, kp3d):
    """kp3d (K,3) -> per-camera (n_cam, K, 3) [u, v, v=2] (all visible)."""
    n_cam, K = cam_mats.shape[0], kp3d.shape[0]
    out = np.zeros((n_cam, K, 3))
    for c in range(n_cam):
        for j in range(K):
            out[c, j, :2] = project_affine(cam_mats[c], kp3d[j])
            out[c, j, 2] = 2.0
    return out


def _make_two_flies(rng, cam_mats, K=8, sep=6.0):
    base = np.array([120.0, 30.0, 11.0])  # inside P_REAL's field of view
    fly0 = base + rng.normal(scale=1.5, size=(K, 3))
    fly1 = base + np.array([sep, 0.0, 0.0]) + rng.normal(scale=1.5, size=(K, 3))
    return fly0, fly1


def test_dlt_affine_recovers_known_point():
    cam_mats = _rig(7)
    X = np.array([118.0, 33.0, 12.5])
    pts = np.zeros((7, 2))
    for c in range(7):
        pts[c] = project_affine(cam_mats[c], X)
    Xhat = _dlt_affine(cam_mats, list(range(7)), pts)
    assert np.linalg.norm(Xhat - X) < 1e-4


def test_score_assignment_low_for_true_and_high_for_swapped():
    rng = np.random.default_rng(0)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0 = _project_fly(cam_mats, fly0)
    obs1 = _project_fly(cam_mats, fly1)
    cams = list(range(7))
    # consistent single-fly observations -> ~0 residual (subpixel)
    good = score_assignment(obs0, cams, cam_mats)
    assert good < 1.0
    # mix cam 3 with the OTHER fly's detection -> a large residual
    bad = obs0.copy()
    bad[3] = obs1[3]
    assert score_assignment(bad, cams, cam_mats) > 5.0 * max(good, 1e-3)


def _anns_from_obs(obs0, obs1, order_per_cam):
    """Build anns_by_cam with per-camera ann ordering controlled by order_per_cam[c]
    (0 -> [fly0, fly1], 1 -> [fly1, fly0]). ann_id encodes (cam*10 + slot)."""
    anns_by_cam = {}
    for c in range(obs0.shape[0]):
        f0 = {"id": c * 10 + 0, "keypoints": obs0[c].reshape(-1).tolist()}
        f1 = {"id": c * 10 + 1, "keypoints": obs1[c].reshape(-1).tolist()}
        anns_by_cam[c] = [f0, f1] if order_per_cam[c] == 0 else [f1, f0]
    return anns_by_cam


def test_link_frameset_recovers_grouping_under_shuffle():
    rng = np.random.default_rng(1)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0, obs1 = _project_fly(cam_mats, fly0), _project_fly(cam_mats, fly1)
    # shuffle each camera's ann order arbitrarily
    order = [0, 1, 1, 0, 1, 0, 1]
    anns = _anns_from_obs(obs0, obs1, order)
    linked = link_frameset(anns, cam_mats, n_flies=2)
    # each fly's chosen ann per cam must all carry the SAME true-fly slot
    def true_slot(ann_id):
        return ann_id % 10  # 0 -> obs0(fly0), 1 -> obs1(fly1)
    slots_fly = {fid: {true_slot(a) for a in linked[fid].values()} for fid in linked}
    assert all(len(s) == 1 for s in slots_fly.values())          # internally consistent
    assert slots_fly[0] != slots_fly[1]                          # the two flies differ
    assert set(linked[0]) == set(range(7)) and set(linked[1]) == set(range(7))


def test_link_frameset_robust_to_order_swap_and_missing_cam():
    rng = np.random.default_rng(2)
    cam_mats = _rig(7)
    fly0, fly1 = _make_two_flies(rng, cam_mats)
    obs0, obs1 = _project_fly(cam_mats, fly0), _project_fly(cam_mats, fly1)
    order = [0, 1, 0, 1, 0, 1, 0]
    anns = _anns_from_obs(obs0, obs1, order)
    # camera 5 sees only ONE fly (drop its second ann)
    anns[5] = [anns[5][0]]
    linked = link_frameset(anns, cam_mats, n_flies=2)

    def true_slot(ann_id):
        return ann_id % 10
    slots_fly = {fid: {true_slot(a) for a in linked[fid].values()} for fid in linked}
    assert all(len(s) == 1 for s in slots_fly.values())
    assert slots_fly[0] != slots_fly[1]
    # camera 5's single ann is assigned to exactly one fly (the nearest-reproj one)
    assigned_cam5 = [fid for fid in linked if 5 in linked[fid]]
    assert len(assigned_cam5) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_identity_link.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.identity_link'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/identity_link.py`:

```python
"""JAX/NumPy affine cross-camera identity linker.

Each synchronized frameset has up to ``n_flies`` COCO annotations PER camera.
Within a camera the anns have distinct ``id``s, but which ann in cam-i is the
same physical fly as which ann in cam-j is not given. This module recovers that
cross-camera identity by triangulation consistency: for a candidate assignment
of per-camera anns to fly slots, triangulate each fly's keypoints (affine DLT)
and reproject; the true assignment minimizes the multi-view reprojection
residual. Mirrors the ALGORITHM of JARVIS-HybridNet BoutMasks.assign_identities
(anchor enumeration + min-residual + nearest-reproj for short cameras), but is
written from scratch against the affine primitives -- the torch tracker is NOT
imported. Cameras are telecentric (affine): projection has no perspective divide.
"""
from __future__ import annotations

import itertools

import numpy as np


def _dlt_affine(cam_mats, cam_ids, pts2d) -> np.ndarray:
    """Affine DLT-SVD triangulation (same math as ReprojectionTool.reconstruct_point).

    Args:
        cam_mats: (n_cam, 3, 4) per-camera 3x4 DLT matrices.
        cam_ids: absolute camera indices to use (>=2).
        pts2d: (n_cam, 2) observations, row indexed by absolute camera id.
    Returns:
        (3,) triangulated point; zeros if <2 cameras.
    """
    cam_ids = list(cam_ids)
    if len(cam_ids) < 2:
        return np.zeros(3)
    A = np.zeros((2 * len(cam_ids), 4))
    for i, c in enumerate(cam_ids):
        P = np.asarray(cam_mats[c], float)      # (3,4)
        uv = np.asarray(pts2d[c], float)        # (2,)
        A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
    _, _, Vh = np.linalg.svd(A)
    Xh = Vh[-1]
    return (Xh / Xh[3])[:3]


def _project_affine_point(P, X) -> np.ndarray:
    """uv = P[:2,:3] @ X + P[:2,3] (affine; no perspective divide)."""
    P = np.asarray(P, float); X = np.asarray(X, float)
    return X @ P[:2, :3].T + P[:2, 3]


def score_assignment(kp2d_per_cam, cams_present, cam_mats) -> float:
    """Mean multi-view reprojection residual (px) for ONE fly's per-camera kps.

    Args:
        kp2d_per_cam: (n_cam, K, 3) coco [x, y, v]; v>0 == present.
        cams_present: absolute camera ids that carry this fly's detection.
        cam_mats: (n_cam, 3, 4) affine DLT matrices.
    Returns:
        Mean per-(keypoint, present-camera) pixel residual over triangulable
        keypoints; float('inf') if none triangulate.
    """
    kp2d_per_cam = np.asarray(kp2d_per_cam, float)
    cams_present = list(cams_present)
    K = kp2d_per_cam.shape[1]
    resid = []
    for j in range(K):
        cams_j = [c for c in cams_present if kp2d_per_cam[c, j, 2] > 0]
        if len(cams_j) < 2:
            continue
        pts = np.zeros((kp2d_per_cam.shape[0], 2))
        for c in cams_j:
            pts[c] = kp2d_per_cam[c, j, :2]
        X = _dlt_affine(cam_mats, cams_j, pts)
        for c in cams_j:
            uv = _project_affine_point(cam_mats[c], X)
            resid.append(float(np.linalg.norm(uv - kp2d_per_cam[c, j, :2])))
    return float(np.mean(resid)) if resid else float("inf")


def _ann_kp(ann) -> np.ndarray:
    """(K, 3) coco [x, y, v] from a COCO annotation dict."""
    return np.asarray(ann["keypoints"], float).reshape(-1, 3)


def _fly_center2d(kp) -> np.ndarray:
    """Mean of visible (v>0) 2-D keypoints; NaN if none visible."""
    vis = kp[:, 2] > 0
    return kp[vis, :2].mean(0) if vis.any() else np.full(2, np.nan)


def link_frameset(anns_by_cam, cam_mats, *, n_flies=2, ref_cam=None,
                  residual_gate_px=40.0):
    """Cross-camera identity for one frameset. See module docstring / Interfaces."""
    n_cam = np.asarray(cam_mats).shape[0]
    K = _ann_kp(next(iter(anns_by_cam.values()))[0]).shape[0]

    full_cams = [c for c in anns_by_cam if len(anns_by_cam[c]) >= n_flies]
    if not full_cams:
        return {fid: {} for fid in range(n_flies)}
    if ref_cam is None or ref_cam not in full_cams:
        ref_cam = full_cams[0]

    # Reference camera fixes fly slots: ref ann index i -> fly i.
    ref_anns = anns_by_cam[ref_cam][:n_flies]

    # For each other full camera, choose the permutation of its first n_flies
    # anns onto fly slots that minimizes total residual across (ref, cam).
    perms = list(itertools.permutations(range(n_flies)))
    # assign[fly][cam] = local ann index chosen for that fly at that camera.
    assign = {fid: {ref_cam: fid} for fid in range(n_flies)}
    for cam in full_cams:
        if cam == ref_cam:
            continue
        cam_anns = anns_by_cam[cam][:n_flies]
        best_perm, best_total = None, float("inf")
        for perm in perms:
            total = 0.0
            for fly in range(n_flies):
                obs = np.zeros((n_cam, K, 3))
                obs[ref_cam] = _ann_kp(ref_anns[fly])
                obs[cam] = _ann_kp(cam_anns[perm[fly]])
                total += score_assignment(obs, [ref_cam, cam], cam_mats)
            if total < best_total:
                best_total, best_perm = total, perm
        for fly in range(n_flies):
            assign[fly][cam] = best_perm[fly]

    # Gate: mean residual of the resolved full-camera assignment.
    gate_resid = []
    for fly in range(n_flies):
        obs = np.zeros((n_cam, K, 3))
        cams_fly = list(assign[fly])
        for cam in cams_fly:
            obs[cam] = _ann_kp(anns_by_cam[cam][assign[fly][cam]])
        r = score_assignment(obs, cams_fly, cam_mats)
        if np.isfinite(r):
            gate_resid.append(r)
    if not gate_resid or float(np.mean(gate_resid)) > residual_gate_px:
        return {fid: {} for fid in range(n_flies)}

    # Triangulate each fly's center from its resolved full cameras, to place
    # the short cameras (fewer than n_flies anns) by nearest reprojection.
    centers = {}
    for fly in range(n_flies):
        cams_fly = list(assign[fly])
        pts = np.zeros((n_cam, 2))
        used = []
        for cam in cams_fly:
            ctr = _fly_center2d(_ann_kp(anns_by_cam[cam][assign[fly][cam]]))
            if np.all(np.isfinite(ctr)):
                pts[cam] = ctr; used.append(cam)
        centers[fly] = _dlt_affine(cam_mats, used, pts) if len(used) >= 2 else None

    for cam in anns_by_cam:
        if cam in full_cams:
            continue
        for local_i, ann in enumerate(anns_by_cam[cam]):
            ctr = _fly_center2d(_ann_kp(ann))
            if not np.all(np.isfinite(ctr)):
                continue
            dists = []
            for fly in range(n_flies):
                if centers[fly] is None:
                    dists.append(np.inf); continue
                uv = _project_affine_point(cam_mats[cam], centers[fly])
                dists.append(float(np.linalg.norm(uv - ctr)))
            assign[int(np.argmin(dists))][cam] = local_i

    # Convert local ann indices -> ann ids.
    out = {}
    for fly in range(n_flies):
        out[fly] = {cam: int(anns_by_cam[cam][li]["id"]) for cam, li in assign[fly].items()}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_identity_link.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/identity_link.py third_party/jarvis_jax/tests/test_identity_link.py
git commit -m "feat(cse): affine cross-camera identity scorer + single-frameset linker

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Recording-level identity map over real COCO framesets

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/identity_link.py` (add `link_recording`)
- Test: `third_party/jarvis_jax/tests/test_identity_link.py` (add a real-data-gated test)

**Interfaces:**
- Consumes: Task 1 `link_frameset`; `jarvis_jax.geometry.reprojection_tool.ReprojectionTool` (`.cameras` dict — sorted-lexicographic camera name → index; `.camera_matrices (num_cam,4,3)`; `._camera_list[c].cameraMatrix (3,4)`).
- Produces:
  - `link_recording(coco_path: str, recording: str, calib_dir: str, *, split: str = "val", n_flies: int = 2, residual_gate_px: float = 40.0) -> dict[str, dict[int, dict[int, int]]]` — for each frameset of `recording` in the COCO json's `framesets` (grouped by `datasetName`), group that frameset's per-camera annotations, call `link_frameset`, and return `{fs_key: {fly_id: {cam_idx: ann_id}}}`. `cam_idx` is the `ReprojectionTool` camera index (from the file_name camera name, matched via `_cam2img`-style parsing). Framesets where no camera has `n_flies` anns yield `{fly_id: {...}}` with fly0 populated from single anns and fly1 empty (1-fly frameset handled gracefully).

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_identity_link.py`:

```python
import os
from jarvis_jax.cse.identity_link import link_recording

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"


@pytest.mark.skipif(not os.path.exists(COCO), reason="courtship coco not present")
def test_link_recording_two_fly_framesets_reproject_cleanly():
    m = link_recording(COCO, REC, CALIB, split="val", n_flies=2)
    assert len(m) > 0, "no framesets found for the recording"

    # Most framesets should resolve BOTH flies (this recording has 181 two-ann frames).
    two_fly = [k for k, v in m.items() if len(v.get(0, {})) >= 2 and len(v.get(1, {})) >= 2]
    assert len(two_fly) >= 50, f"only {len(two_fly)} two-fly framesets linked"

    # NON-VACUOUS: each linked fly's chosen anns must triangulate + reproject
    # to a small residual (the linker's own objective) and the two flies must
    # pick DISJOINT anns per camera (no identity collapse).
    import json
    import numpy as np
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.identity_link import score_assignment, _ann_kp
    coco = json.load(open(COCO))
    id2ann = {}
    for a in coco["annotations"]:
        id2ann.setdefault(a["image_id"], []).append(a)
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    rt = ReprojectionTool(CALIB)
    cam_names = list(rt.cameras.keys())
    cam_mats = np.stack([c.cameraMatrix for c in rt._camera_list], 0)  # (n_cam,3,4)

    resid_all, disjoint_ok = [], 0
    for fs_key in two_fly[:30]:
        v = m[fs_key]
        # disjoint per-camera anns between fly0 and fly1
        shared = set(v[0].items()) & set(v[1].items())
        if not shared:
            disjoint_ok += 1
        for fly in (0, 1):
            n_cam = len(cam_names)
            K = _ann_kp(next(iter(ann_by_id.values()))).shape[0]
            obs = np.zeros((n_cam, K, 3))
            for cam, aid in v[fly].items():
                obs[cam] = _ann_kp(ann_by_id[aid])
            r = score_assignment(obs, list(v[fly]), cam_mats)
            if np.isfinite(r):
                resid_all.append(r)
    assert disjoint_ok >= 25, "flies share anns in too many framesets (identity collapse)"
    assert np.mean(resid_all) < 15.0, (
        f"post-link mean reprojection residual {np.mean(resid_all):.2f}px too high"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_identity_link.py::test_link_recording_two_fly_framesets_reproject_cleanly -v`
Expected: FAIL with `ImportError: cannot import name 'link_recording'`.

- [ ] **Step 3: Write minimal implementation**

Add to `third_party/jarvis_jax/jarvis_jax/cse/identity_link.py`:

```python
import json
import os

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool


def link_recording(coco_path, recording, calib_dir, *, split="val", n_flies=2,
                   residual_gate_px=40.0):
    """Per-frameset cross-camera identity map for a recording. See Interfaces.

    Returns {fs_key: {fly_id: {cam_idx: ann_id}}}. Camera indices are
    ReprojectionTool indices (sorted-lexicographic Cam*.yaml order); the camera
    a coco image belongs to is parsed from its file_name (``<split>/<cam>/...``).
    """
    coco = json.load(open(coco_path))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    cam_mats = np.stack([c.cameraMatrix for c in rt._camera_list], 0)  # (n_cam,3,4)

    def _cam_index(image_id):
        fn = id2file.get(int(image_id), "")
        cam = fn.split("/")[1] if "/" in fn else ""
        return cam_names.index(cam) if cam in cam_names else None

    out = {}
    for fs_key, fs in coco["framesets"].items():
        if fs.get("datasetName") != recording:
            continue
        anns_by_cam = {}
        for iid in fs["frames"]:
            ci = _cam_index(iid)
            if ci is None:
                continue
            fa = anns_by_img.get(int(iid), [])
            if fa:
                anns_by_cam[ci] = fa
        if not anns_by_cam:
            out[fs_key] = {fid: {} for fid in range(n_flies)}
            continue
        full = [c for c in anns_by_cam if len(anns_by_cam[c]) >= n_flies]
        if full:
            out[fs_key] = link_frameset(
                anns_by_cam, cam_mats, n_flies=n_flies,
                residual_gate_px=residual_gate_px,
            )
        else:
            # 1-fly frameset: assign every single ann to fly0.
            mapping = {fid: {} for fid in range(n_flies)}
            for cam, fa in anns_by_cam.items():
                mapping[0][cam] = int(fa[0]["id"])
            out[fs_key] = mapping
    return out
```

- [ ] **Step 4: Run test to verify it passes (real data, CPU-safe)**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_identity_link.py::test_link_recording_two_fly_framesets_reproject_cleanly -v`
Expected: PASS. If skipped, the COCO file is missing — confirm the path. Then run the full file to confirm no regression:
`cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_identity_link.py -v` → 6 passed.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/identity_link.py third_party/jarvis_jax/tests/test_identity_link.py
git commit -m "feat(cse): recording-level cross-camera identity map over coco framesets

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Per-fly bout builder (de-collapsed triangulation)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/multifly_bout.py`
- Test: `third_party/jarvis_jax/tests/test_multifly_bout.py`

**Interfaces:**
- Consumes: Task 2 `link_recording` output type `dict[str, dict[int, dict[int, int]]]`; `jarvis_jax.cse.cse_labels` helpers `model_kp_order(anatomy_yaml) -> list[str]`, `_reorder_index(src_names, dst_names) -> np.ndarray`, `model_rest_keypoints(model, kp_names) -> (K,3)`, `umeyama_scale(data_pts, model_pts, valid) -> float`; `ReprojectionTool.reconstruct_point`; `mujoco.MjModel.from_xml_path`.
- Produces:
  - `build_fly_bout(coco_path: str, calib_dir: str, recording: str, identity_map: dict, fly_id: int, anatomy_yaml: str, model_xml: str, out_h5: str, *, split: str = "val") -> tuple[str, float]` — for each frameset in `identity_map`, gather THIS `fly_id`'s per-camera annotation (via `identity_map[fs_key][fly_id]`), triangulate all 50 keypoints in model order (≥2 visible cameras), scale to model cm via the recording's Umeyama scale, and write a bout h5 with the EXACT `cse_labels.build_bout` schema: datasets `keypoints (T,K,3) f32`, `kp_names (K,) S20`, `vis (T,K) bool`, `fs_keys (T,) S64`, `fs_imgids (T,n_cam) int64`; attrs `scale`, `recording`. Returns `(out_h5, scale)`. Framesets where this fly has `<2` cameras with visible keypoints are skipped (dropped from `T`).

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_multifly_bout.py`:

```python
import os
import h5py
import numpy as np
import pytest
from jarvis_jax.cse.multifly_bout import build_fly_bout
from jarvis_jax.cse.identity_link import link_recording

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
# anatomy yaml giving the model KEYPOINT_MODEL_PAIRS order (same one build_bout uses):
ANATOMY = os.environ.get(
    "STAC_ANATOMY_V1",
    "/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/third_party/stac-mjx/configs/anatomy/v1.yaml",
)

_HAVE = os.path.exists(COCO) and os.path.exists(XML) and os.path.exists(ANATOMY)


@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / anatomy not present")
def test_build_fly_bout_two_flies_differ_and_schema_matches(tmp_path):
    m = link_recording(COCO, REC, CALIB, split="val", n_flies=2)
    out0 = str(tmp_path / f"{REC}_fly0_bout.h5")
    out1 = str(tmp_path / f"{REC}_fly1_bout.h5")
    p0, s0 = build_fly_bout(COCO, CALIB, REC, m, 0, ANATOMY, XML, out0, split="val")
    p1, s1 = build_fly_bout(COCO, CALIB, REC, m, 1, ANATOMY, XML, out1, split="val")

    with h5py.File(p0, "r") as f0, h5py.File(p1, "r") as f1:
        # EXACT schema (matches cse_labels.build_bout / run_stac_bout expectations).
        for f in (f0, f1):
            assert set(["keypoints", "kp_names", "vis", "fs_keys", "fs_imgids"]) <= set(f.keys())
            assert f["keypoints"].ndim == 3 and f["keypoints"].shape[1] == 50
            assert f["keypoints"].shape[2] == 3
            assert f["kp_names"].shape == (50,)
            assert f["vis"].shape == f["keypoints"].shape[:2]
            assert f["fs_imgids"].shape[0] == f["keypoints"].shape[0]
            assert "scale" in f.attrs and "recording" in f.attrs

        k0 = f0["keypoints"][()]; k1 = f1["keypoints"][()]
        v0 = f0["vis"][()]; v1 = f1["vis"][()]
        keys0 = set(f0["fs_keys"][()].astype(str)); keys1 = set(f1["fs_keys"][()].astype(str))
        img0 = f0["fs_imgids"][()]; img1 = f1["fs_imgids"][()]

    # Per-fly keypoints must be DEMONSTRABLY different on shared framesets
    # (identity de-collapse: fly0 != fly1).
    assert len(k0) > 10 and len(k1) > 10
    common = sorted(keys0 & keys1)
    assert len(common) > 10, "flies share too few framesets"
    with h5py.File(p0, "r") as f0:
        ks0 = list(f0["fs_keys"][()].astype(str))
    with h5py.File(p1, "r") as f1:
        ks1 = list(f1["fs_keys"][()].astype(str))
    i0 = {k: i for i, k in enumerate(ks0)}
    i1 = {k: i for i, k in enumerate(ks1)}
    diffs = []
    for k in common[:20]:
        a = k0[i0[k]]; b = k1[i1[k]]
        both = v0[i0[k]] & v1[i1[k]]
        if both.sum() >= 5:
            diffs.append(np.linalg.norm(a[both] - b[both], axis=1).mean())
    assert np.mean(diffs) > 0.05, (
        f"fly0 and fly1 keypoints nearly identical (mean {np.mean(diffs):.4f} model-cm)"
        " -- identity de-collapse failed"
    )

    # fs_imgids row must point at coco image ids that belong to THIS recording.
    import json
    coco = json.load(open(COCO))
    rec_img_ids = {im["id"] for im in coco["images"]
                   if im["file_name"].startswith(f"val/") and REC in im["file_name"]}
    assert set(int(x) for x in img0.reshape(-1) if x >= 0) <= rec_img_ids or len(rec_img_ids) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_multifly_bout.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.multifly_bout'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/multifly_bout.py`:

```python
"""Per-fly bout builder: de-collapse a 2-fly recording into one bout h5 per fly.

The stock cse_labels.build_bout uses id2ann.setdefault(image_id, a) (first-ann-
wins), which silently mixes the two flies on a courtship recording. Given the
Phase-3 cross-camera identity map (identity_link.link_recording), this builds one
bout h5 per fly by selecting THAT fly's chosen annotation per camera, triangulating
its 50 keypoints in model order, and scaling to model cm -- writing the EXACT same
schema cse_labels.build_bout / run_stac_bout.run / silhouette_ik_solve
.build_solver_inputs consume, so the downstream STAC + IK pipeline runs unchanged.
"""
from __future__ import annotations

import json
import os

import h5py
import numpy as np
import mujoco

from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.cse_labels import (
    model_kp_order, model_rest_keypoints, umeyama_scale, _reorder_index,
)


def build_fly_bout(coco_path, calib_dir, recording, identity_map, fly_id,
                   anatomy_yaml, model_xml, out_h5, *, split="val"):
    """Build one fly's bout h5 from the cross-camera identity map. See Interfaces."""
    coco = json.load(open(coco_path))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    ann_by_id = {a["id"]: a for a in coco["annotations"]}
    src_names = coco["keypoint_names"]

    kp_order = model_kp_order(anatomy_yaml)
    reorder = _reorder_index(src_names, kp_order)          # dst(model) -> src(coco)
    K = len(kp_order)

    rt = ReprojectionTool(calib_dir)
    nc = rt.num_cameras
    cam_names = list(rt.cameras.keys())

    def _cam_index(image_id):
        fn = id2file.get(int(image_id), "")
        cam = fn.split("/")[1] if "/" in fn else ""
        return cam_names.index(cam) if cam in cam_names else None

    # image_id per camera for a frameset: parse from the fly's chosen anns.
    kp3d_rows, vis_rows, fs_keys, fs_imgids = [], [], [], []
    for fs_key, per_fly in identity_map.items():
        cam2ann = per_fly.get(fly_id, {})
        if len(cam2ann) < 2:
            continue
        kp2d = np.zeros((nc, K, 3))
        row_imgids = [-1] * nc
        for cam, aid in cam2ann.items():
            ann = ann_by_id[aid]
            row_imgids[cam] = int(ann["image_id"])
            raw = np.asarray(ann["keypoints"], float).reshape(-1, 3)  # coco order
            kp2d[cam] = raw[reorder]
        p3d = np.zeros((K, 3)); vmask = np.zeros(K, bool)
        for j in range(K):
            cams = [c for c in cam2ann if kp2d[c, j, 2] > 0]
            if len(cams) >= 2:
                p3d[j] = rt.reconstruct_point(kp2d[:, j, :2], cams_to_use=cams)
                vmask[j] = True
        if vmask.sum() < 2:
            continue
        kp3d_rows.append(p3d); vis_rows.append(vmask)
        fs_keys.append(fs_key); fs_imgids.append(row_imgids)

    if not kp3d_rows:
        raise RuntimeError(f"no framesets with >=2 cameras for fly {fly_id} in {recording}")
    kp3d = np.asarray(kp3d_rows)                            # (T,K,3) mm
    vis = np.asarray(vis_rows)                              # (T,K)

    model = mujoco.MjModel.from_xml_path(model_xml)
    rest = model_rest_keypoints(model, kp_order)
    med = np.nanmedian(np.where(vis[..., None], kp3d, np.nan), axis=0)  # (K,3)
    s = umeyama_scale(med, rest, np.all(np.isfinite(med), 1))
    kp_scaled = (kp3d * s).astype(np.float32)
    kp_scaled[~vis] = 0.0

    os.makedirs(os.path.dirname(out_h5) or ".", exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("keypoints", data=kp_scaled)
        f.create_dataset("kp_names", data=np.array(kp_order, dtype="S20"))
        f.create_dataset("vis", data=vis)
        f.attrs["scale"] = s
        f.attrs["recording"] = recording
        f.create_dataset("fs_keys", data=np.array(fs_keys, dtype="S64"))
        f.create_dataset("fs_imgids", data=np.asarray(fs_imgids, np.int64))
    print(f"[build_fly_bout] {recording} fly{fly_id}: {len(kp_scaled)} framesets, "
          f"scale={s:.4f} -> {out_h5}")
    return out_h5, s
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_multifly_bout.py -v`
Expected: PASS. If skipped, confirm `ANATOMY` points at the stac-mjx v1 anatomy yaml (set `STAC_ANATOMY_V1` env var to its path). If the `fs_imgids` shape differs from `(T,nc)` (e.g. a fly missing at a camera), the row uses `-1` sentinels for absent cameras — the schema still matches `build_solver_inputs`, which only reads `fs_imgids` via `_cam2img_for_frame` (skips ids not in `id2file`).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/multifly_bout.py third_party/jarvis_jax/tests/test_multifly_bout.py
git commit -m "feat(cse): per-fly bout builder (de-collapsed triangulation, 2 bout h5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Per-identity mask/kp threading in silhouette_ik_solve.py

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py` (`_triangulate_kp_mm`, `extract_tips_for_frames`, `run_single_fly`, `run_ablation`; add `_id2ann_for_image` helper)
- Test: `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py` (add per-identity threading tests)

**Interfaces:**
- Consumes: existing `_load_sam_mask(root, split, file_name, ann_id) -> np.ndarray|None` (unchanged — already selects by one ann_id); `_cam2img_for_frame`; `_triangulate_kp_mm`.
- Produces (new/changed signatures — later tasks and the orchestrator rely on these):
  - `_ann_for_image(id2ann_multi: dict[int, list[dict]], image_id: int, ann_id_by_image: dict[int, int] | None) -> dict | None` — returns the single ann for an image: if `ann_id_by_image` is given and has this `image_id`, return the ann whose `id` equals `ann_id_by_image[image_id]`; else return the first ann for the image (backward-compatible last/first-ann behavior). Returns None if the image has no anns.
  - `_triangulate_kp_mm(rt, ik_kpnames, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image=None) -> (kp_mm (n,3), valid (n,) bool)` — same as before but selects each camera's ann via `_ann_for_image`. `id2ann_multi` is now `dict[int, list[dict]]` (image_id → all anns), not the old `dict[int, dict]`.
  - `extract_tips_for_frames(..., ann_id_by_image: dict[int, int] | None = None) -> list[dict]` — new trailing kwarg; when given, mask + kp selection use the chosen ann per image (for THIS fly). Default None → current behavior (first ann per image).
  - `run_single_fly(..., ann_id_by_image: dict[int, int] | None = None) -> dict` and `run_ablation(..., ann_id_by_image: dict[int, int] | None = None) -> dict` — new trailing kwarg forwarded to `extract_tips_for_frames` and used in the reprojection-metric ann lookup. Default None → single-fly path unchanged.

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_silhouette_ik_solve.py`:

```python
def test_ann_for_image_selects_by_ann_id_by_image():
    from jarvis_jax.cse.silhouette_ik_solve import _ann_for_image
    id2ann_multi = {100: [{"id": 5, "keypoints": [1]}, {"id": 6, "keypoints": [2]}]}
    # explicit selection picks the chosen ann
    a = _ann_for_image(id2ann_multi, 100, {100: 6})
    assert a["id"] == 6
    # None -> first-ann (backward compatible)
    b = _ann_for_image(id2ann_multi, 100, None)
    assert b["id"] == 5
    # missing image -> None
    assert _ann_for_image(id2ann_multi, 999, {100: 6}) is None
    # image not in map -> falls back to first ann
    c = _ann_for_image(id2ann_multi, 100, {200: 6})
    assert c["id"] == 5


def test_triangulate_kp_mm_uses_selected_ann():
    """With two distinct anns per image, _triangulate_kp_mm must triangulate the
    ann chosen by ann_id_by_image, not the first one -- so fly0 and fly1 yield
    different 3-D keypoints (regression guard for identity threading)."""
    import numpy as np
    from jarvis_jax.cse.silhouette_ik_solve import _triangulate_kp_mm
    from jarvis_jax.cse.affine_camera import factor_affine, reconstruct_affine, project_affine

    class _FakeRT:
        def __init__(self, cam_mats):
            self.num_cameras = len(cam_mats)
            self._cm = cam_mats
        def reconstruct_point(self, obs, cams_to_use=None):
            cams = cams_to_use or list(range(self.num_cameras))
            A = np.zeros((2 * len(cams), 4))
            for i, c in enumerate(cams):
                P = self._cm[c]; uv = obs[c]
                A[2 * i:2 * i + 2] = uv.reshape(2, 1) * P[2].reshape(1, 4) - P[0:2]
            _, _, Vh = np.linalg.svd(A)
            return (Vh[-1] / Vh[-1][3])[:3]

    P0 = np.array([[8.1, 0, 0, -2.8], [0, -8.0, 0, 462.0], [0, 0, 0, 1.0]])
    K2, R, t = factor_affine(P0)
    th = np.deg2rad(20.0); Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    cam_mats = [P0, reconstruct_affine(K2, Ry @ R, t)]
    rt = _FakeRT(cam_mats)

    X_fly0 = np.array([118.0, 33.0, 12.0]); X_fly1 = np.array([124.0, 33.0, 12.0])
    kpnames = ["Scutellum"]; coco_kpnames = ["Scutellum"]
    cam2img = {0: 10, 1: 11}

    def _ann(aid, X):
        kp = np.zeros(3)
        return {"id": aid, "keypoints": None, "_X": X}

    # build two anns per image with the keypoint projected from each fly's X
    id2ann_multi = {}
    for c, iid in cam2img.items():
        a0 = {"id": iid * 10, "keypoints": list(project_affine(cam_mats[c], X_fly0)) + [2.0]}
        a1 = {"id": iid * 10 + 1, "keypoints": list(project_affine(cam_mats[c], X_fly1)) + [2.0]}
        id2ann_multi[iid] = [a0, a1]

    sel0 = {iid: iid * 10 for iid in cam2img.values()}       # fly0 anns
    sel1 = {iid: iid * 10 + 1 for iid in cam2img.values()}   # fly1 anns
    kp0, v0 = _triangulate_kp_mm(rt, kpnames, coco_kpnames, cam2img, id2ann_multi, sel0)
    kp1, v1 = _triangulate_kp_mm(rt, kpnames, coco_kpnames, cam2img, id2ann_multi, sel1)
    assert v0[0] and v1[0]
    assert np.linalg.norm(kp0[0] - X_fly0) < 1e-3
    assert np.linalg.norm(kp1[0] - X_fly1) < 1e-3
    assert np.linalg.norm(kp0[0] - kp1[0]) > 1.0    # fly0 != fly1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_ik_solve.py::test_ann_for_image_selects_by_ann_id_by_image tests/test_silhouette_ik_solve.py::test_triangulate_kp_mm_uses_selected_ann -v`
Expected: FAIL with `ImportError: cannot import name '_ann_for_image'` (first test) and a `_triangulate_kp_mm` signature/`AttributeError` mismatch (second test — the current version takes `id2ann` as `dict[int, dict]` and has no `ann_id_by_image` param).

- [ ] **Step 3: Write minimal implementation**

In `third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py`, add the helper near `_cam2img_for_frame` (before `_triangulate_kp_mm`):

```python
def _ann_for_image(id2ann_multi, image_id, ann_id_by_image=None):
    """Select one COCO annotation for an image.

    id2ann_multi maps image_id -> list of all anns for that image. If
    ann_id_by_image (per-fly image_id -> chosen ann id) is given and has this
    image, return the ann whose id matches; otherwise return the first ann
    (backward-compatible single-ann behavior). None if the image has no anns.
    """
    anns = id2ann_multi.get(int(image_id))
    if not anns:
        return None
    if ann_id_by_image is not None and int(image_id) in ann_id_by_image:
        want = int(ann_id_by_image[int(image_id)])
        for a in anns:
            if int(a["id"]) == want:
                return a
    return anns[0]
```

Change `_triangulate_kp_mm` (line 296) to accept the multi-ann dict + selection:

```python
def _triangulate_kp_mm(rt, ik_kpnames, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image=None):
    """Triangulate the STAC ik h5's named keypoints into the calib mm frame.

    id2ann_multi is image_id -> list[ann]; the ann used per camera is chosen by
    _ann_for_image(..., ann_id_by_image). See Interfaces (Task 4).
    """
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    n = len(ik_kpnames)
    kp_mm = np.zeros((n, 3))
    valid = np.zeros(n, dtype=bool)
    for j, nm in enumerate(ik_kpnames):
        ci = name2coco.get(nm)
        if ci is None:
            continue
        obs = np.zeros((rt.num_cameras, 2))
        cams = []
        for c, iid in cam2img.items():
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], dtype=float).reshape(-1, 3)
            if kp[ci, 2] > 0:
                obs[c] = kp[ci, :2]
                cams.append(c)
        if len(cams) >= 2:
            kp_mm[j] = rt.reconstruct_point(obs, cams_to_use=cams)
            valid[j] = True
    return kp_mm, valid
```

Then update the THREE `id2ann = {an["image_id"]: an for an in coco["annotations"]}` collapse sites (lines 562, 800, 1089) to build the multi-ann dict and pass the selection. In each of `extract_tips_for_frames`, `run_single_fly`, and `run_ablation`, replace that line with:

```python
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
```

and route every call. Specifically:
- Add `ann_id_by_image: dict | None = None` as the last kwarg of `extract_tips_for_frames`, `run_single_fly`, `run_ablation`.
- In `extract_tips_for_frames`: `kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image)`; and for the mask loop select the fly's ann per camera:

```python
        masks = [None] * rt.num_cameras
        for c, iid in cam2img.items():
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            fn = id2file[iid]
            masks[c] = _load_sam_mask(root, split, fn, ann["id"])
```

- In `run_single_fly`/`run_ablation`: forward `ann_id_by_image=ann_id_by_image` into the `extract_tips_for_frames(...)` call, and in the reprojection-metric loop replace `id2ann.get(iid)` ann access with `_ann_for_image(id2ann_multi, iid, ann_id_by_image)` and `_triangulate_kp_mm(..., id2ann_multi, ann_id_by_image)`.

Keep every default `None`, so the existing single-fly recording behavior (first ann per image == the only ann) is byte-for-byte unchanged.

- [ ] **Step 4: Run the new + existing tests to verify pass (regression)**

Run the new threading tests:
`cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_silhouette_ik_solve.py::test_ann_for_image_selects_by_ann_id_by_image tests/test_silhouette_ik_solve.py::test_triangulate_kp_mm_uses_selected_ann -v`
Expected: PASS (2 passed).

Run the full existing suite on the single-fly recording to prove backward compatibility (GPU, activated env — these tests are `skipif` on the Phase-2 ik h5):
`cd third_party/jarvis_jax && python -m pytest tests/test_silhouette_ik_solve.py -v`
Expected: PASS (all previously-passing tests still pass; the `run_single_fly`/`run_ablation` wiring tests are unaffected because `ann_id_by_image` defaults to None). If the Phase-2 ik h5 is absent those tests skip — then at minimum the CPU threading tests + `test_model_mm_bridge_roundtrips` + `test_wing_fk_indices_*` + `test_withhold_wing_kp_*` + `test_stac_order_bridge_*` must pass.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/silhouette_ik_solve.py third_party/jarvis_jax/tests/test_silhouette_ik_solve.py
git commit -m "feat(cse): thread per-identity ann selection through silhouette IK (backward compatible)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Per-fly STAC solve integration + end-to-end multi-fly orchestrator

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/run_multifly_ik.py`
- Test: `third_party/jarvis_jax/tests/test_run_multifly_ik.py`

**Interfaces:**
- Consumes: Task 2 `link_recording`; Task 3 `build_fly_bout`; `jarvis_jax.cse.run_stac_bout.run(bout_h5, out_h5, stac_config_dir, overrides) -> ik_p`; Task 4 `run_single_fly(recording, *, ik_h5, model_xml, mesh_npz, root, split, calib_dir, use_silhouette, wing_weight, only_missing, max_frames, smooth_weight, n_iter, corridor, out_dir, ann_id_by_image) -> dict`.
- Produces:
  - `ann_id_by_image_for_fly(identity_map: dict, coco_path: str, recording: str, fly_id: int) -> dict[int, int]` — flatten the per-frameset identity map into a single `{image_id: ann_id}` dict for one fly (image_id read from the chosen ann's `image_id`). This is the `ann_id_by_image` Task 4 consumes.
  - `run_multifly_ik(recording: str, *, root: str, calib_dir: str, anatomy_yaml: str, model_xml: str, mesh_npz: str, stac_config_dir: str, out_dir: str, split: str = "val", n_flies: int = 2, coco_path: str | None = None, use_silhouette: bool = True, max_frames: int = 0, n_iter: int = 50, wing_weight: float = 0.5) -> dict` — full pipeline: `link_recording` → per fly `build_fly_bout` → `run_stac_bout.run` (per-fly ik_h5) → `run_single_fly(..., ann_id_by_image=ann_id_by_image_for_fly(...))`. Returns `{"identity_map_size": int, "flies": {fly_id: {"bout_h5": str, "ik_h5": str, "n_framesets": int, "report": <run_single_fly dict>}}}`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_run_multifly_ik.py`:

```python
import os
import numpy as np
import pytest

ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
REC = "2026_04_07_11_33_33"
CALIB = f"{ROOT}/calib_params/{REC}"
COCO = f"{ROOT}/annotations/instances_val.json"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ANATOMY = os.environ.get(
    "STAC_ANATOMY_V1",
    "/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/third_party/stac-mjx/configs/anatomy/v1.yaml",
)
STAC_CFG = os.environ.get(
    "STAC_CONFIG_DIR",
    "/gscratch/portia/eabe/Research/MyRepos/fly_neuromech/third_party/stac-mjx/configs",
)


def test_ann_id_by_image_for_fly_flattens_map():
    from jarvis_jax.cse.run_multifly_ik import ann_id_by_image_for_fly
    # minimal synthetic identity map + a matching coco stub
    import json, tempfile
    coco = {
        "images": [{"id": 10, "file_name": "val/CamA/Frame_0.jpg"},
                   {"id": 11, "file_name": "val/CamB/Frame_0.jpg"}],
        "annotations": [
            {"id": 100, "image_id": 10}, {"id": 101, "image_id": 10},
            {"id": 110, "image_id": 11}, {"id": 111, "image_id": 11},
        ],
        "framesets": {"fs0": {"datasetName": REC, "frames": [10, 11]}},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(coco, f); path = f.name
    identity_map = {"fs0": {0: {0: 100, 1: 110}, 1: {0: 101, 1: 111}}}
    m0 = ann_id_by_image_for_fly(identity_map, path, REC, 0)
    m1 = ann_id_by_image_for_fly(identity_map, path, REC, 1)
    assert m0 == {10: 100, 11: 110}
    assert m1 == {10: 101, 11: 111}
    os.remove(path)


_HAVE = all(os.path.exists(p) for p in (COCO, XML, MESH, ANATOMY, STAC_CFG))


@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / mesh / stac config not present")
def test_run_multifly_ik_both_flies_reproject_cleanly(tmp_path):
    """GPU real run on a small frame slice: BOTH flies produce an ik_h5 and a
    per-fly silhouette-IK report with reproj_px below an explicit threshold, and
    the two flies' solved keypoints are demonstrably different (de-collapsed)."""
    from jarvis_jax.cse.run_multifly_ik import run_multifly_ik
    out = run_multifly_ik(
        REC, root=ROOT, calib_dir=CALIB, anatomy_yaml=ANATOMY, model_xml=XML,
        mesh_npz=MESH, stac_config_dir=STAC_CFG, out_dir=str(tmp_path),
        split="val", n_flies=2, coco_path=COCO,
        use_silhouette=True, max_frames=16, n_iter=30,
    )
    assert out["identity_map_size"] > 0
    assert set(out["flies"]) == {0, 1}
    reproj = {}
    for fid in (0, 1):
        fly = out["flies"][fid]
        assert os.path.exists(fly["ik_h5"]), f"fly{fid} ik_h5 missing"
        assert os.path.exists(fly["bout_h5"])
        rep = fly["report"]
        assert rep["qpos_shape"][1] == 93
        assert np.isfinite(rep["reproj_px"])
        reproj[fid] = rep["reproj_px"]
    # BOTH flies must reproject cleanly (explicit px threshold vs annotated kps).
    assert reproj[0] < 12.0 and reproj[1] < 12.0, f"reproj_px too high: {reproj}"

    # de-collapse sanity: the two flies' solved qpos differ.
    q0 = np.load(os.path.join(str(tmp_path), "fly0", f"{REC}_qpos.npz"))["qpos"]
    q1 = np.load(os.path.join(str(tmp_path), "fly1", f"{REC}_qpos.npz"))["qpos"]
    n = min(len(q0), len(q1))
    assert np.linalg.norm(q0[:n] - q1[:n]) > 1e-3, "fly0/fly1 qpos identical (collapse)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_multifly_ik.py::test_ann_id_by_image_for_fly_flattens_map -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.run_multifly_ik'`.

- [ ] **Step 3: Write minimal implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/run_multifly_ik.py`:

```python
"""End-to-end multi-fly IK orchestrator (Phase 3).

Pipeline for a courtship recording (2 flies COCO-annotated per image):
  link_recording -> per-fly build_fly_bout -> per-fly run_stac_bout (ik_h5)
  -> per-fly silhouette IK (run_single_fly with the fly's ann_id_by_image,
     the per-fly ik_h5, and the per-recording calib_dir).

Reuses the Phase-2 solver + silhouette machinery unchanged; the only new
selection input is the per-fly ann_id_by_image (from the identity map).

STAC dependency (honest note): the primary path is a per-fly STAC solve via
jarvis_jax.cse.run_stac_bout.run, which fits per-fly offsets + q_init. If
run_stac proves intractable in a given environment, an ESCAPE HATCH is provided
(`stac_fallback=True`): reuse an existing (male) fly's fitted offsets/marker_sites
ik_h5 as the anatomy and seed solve_ik with a per-fly q_init derived from the
per-fly bout (build_solver_inputs against the shared ik_h5, then overwrite kp_data
with the per-fly triangulated keypoints). The fallback keeps the anatomy fixed
(V1) and only re-solves pose per fly. Primary path first; fallback only on failure.
"""
from __future__ import annotations

import json
import os

import numpy as np

from jarvis_jax.cse.identity_link import link_recording
from jarvis_jax.cse.multifly_bout import build_fly_bout
from jarvis_jax.cse import run_stac_bout
from jarvis_jax.cse.silhouette_ik_solve import run_single_fly


def ann_id_by_image_for_fly(identity_map, coco_path, recording, fly_id):
    """Flatten {fs_key:{fly:{cam:ann_id}}} -> {image_id: ann_id} for one fly."""
    coco = json.load(open(coco_path))
    ann_img = {a["id"]: a["image_id"] for a in coco["annotations"]}
    out = {}
    for fs_key, per_fly in identity_map.items():
        for cam, aid in per_fly.get(fly_id, {}).items():
            iid = ann_img.get(aid)
            if iid is not None:
                out[int(iid)] = int(aid)
    return out


def run_multifly_ik(recording, *, root, calib_dir, anatomy_yaml, model_xml,
                    mesh_npz, stac_config_dir, out_dir, split="val", n_flies=2,
                    coco_path=None, use_silhouette=True, max_frames=0, n_iter=50,
                    wing_weight=0.5, stac_overrides=("paths=hyak", "anatomy=v1", "dataset=")):
    """Full multi-fly IK pipeline for a recording. See Interfaces (Task 5)."""
    if coco_path is None:
        coco_path = os.path.join(root, "annotations", f"instances_{split}.json")

    identity_map = link_recording(coco_path, recording, calib_dir,
                                  split=split, n_flies=n_flies)
    os.makedirs(out_dir, exist_ok=True)

    flies = {}
    for fid in range(n_flies):
        fly_dir = os.path.join(out_dir, f"fly{fid}")
        os.makedirs(fly_dir, exist_ok=True)
        # 1) per-fly bout (de-collapsed triangulation)
        bout_h5 = os.path.join(fly_dir, f"{recording}_bout.h5")
        bout_h5, scale = build_fly_bout(
            coco_path, calib_dir, recording, identity_map, fid,
            anatomy_yaml, model_xml, bout_h5, split=split)
        # 2) per-fly STAC solve -> per-fly ik_h5
        ik_h5 = os.path.join(fly_dir, recording, "Fruitfly_ik_v1_cse.h5")
        run_stac_bout.run(bout_h5, ik_h5, stac_config_dir, list(stac_overrides))
        # 3) per-fly silhouette IK (with THIS fly's ann selection + per-fly ik_h5).
        # run_single_fly derives the bout at dirname(dirname(ik_h5))/<rec>_bout.h5,
        # which is exactly fly_dir/<rec>_bout.h5 (the per-fly bout written above).
        ann_map = ann_id_by_image_for_fly(identity_map, coco_path, recording, fid)
        report = run_single_fly(
            recording, ik_h5=ik_h5, model_xml=model_xml, mesh_npz=mesh_npz,
            root=root, split=split, calib_dir=calib_dir,
            use_silhouette=use_silhouette, wing_weight=wing_weight,
            max_frames=max_frames, n_iter=n_iter, out_dir=fly_dir,
            ann_id_by_image=ann_map)
        n_fs = int(np.load(bout_h5 if False else bout_h5, allow_pickle=True) is None) if False else 0
        flies[fid] = {"bout_h5": bout_h5, "ik_h5": ik_h5,
                      "n_framesets": len(identity_map), "report": report}
    return {"identity_map_size": len(identity_map), "flies": flies}
```

Note on the per-fly ik_h5 layout: `run_single_fly` derives the bout as `dirname(dirname(ik_h5))/<rec>_bout.h5`. Placing `ik_h5` at `fly_dir/<rec>/Fruitfly_ik_v1_cse.h5` makes that derived path `fly_dir/<rec>_bout.h5` — so write the per-fly bout to exactly that location (the code above does: `bout_h5 = fly_dir/<rec>_bout.h5`). Verify this invariant holds when implementing; if `run_single_fly`'s bout-derivation differs, pass the bout explicitly (add a `bout_h5=` passthrough to `run_single_fly` in Task 4 if needed) rather than relying on path coincidence.

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_multifly_ik.py::test_ann_id_by_image_for_fly_flattens_map -v`
Expected: PASS (1 passed). Remove the dead `n_fs` line (`np.load(... is None)`) — it is a placeholder artifact; set `n_framesets` from the per-fly bout's `keypoints` length via `h5py` instead:

```python
        import h5py
        with h5py.File(bout_h5, "r") as bf:
            n_fs = int(bf["keypoints"].shape[0])
        flies[fid] = {"bout_h5": bout_h5, "ik_h5": ik_h5,
                      "n_framesets": n_fs, "report": report}
```

Re-run the unit test to confirm still PASS.

- [ ] **Step 5: Real GPU run (per-fly STAC + silhouette IK).** Under the activated env, run the real-data test as a BACKGROUND bash (per-fly `run_stac` is heavy, likely >600s for both flies):

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd third_party/jarvis_jax && \
  STAC_ANATOMY_V1=<path> STAC_CONFIG_DIR=<path> \
  python -m pytest tests/test_run_multifly_ik.py::test_run_multifly_ik_both_flies_reproject_cleanly -v -s
```

SUCCESS CRITERION: both flies produce an `ik_h5`; each per-fly silhouette IK returns a finite `reproj_px < 12.0`px against the annotated keypoints; the two flies' solved qpos differ. If per-fly STAC (`run_stac`) fails or times out, invoke the documented ESCAPE HATCH: reuse the male fly's fitted `offsets`/`marker_sites` ik_h5 as the fixed V1 anatomy and seed `solve_ik` with a per-fly `q_init` (from `build_solver_inputs` against that shared ik_h5, kp_data replaced by the per-fly bout's triangulated keypoints) — implement `stac_fallback=True` in `run_multifly_ik` and re-run. Report the numbers honestly either way; do NOT fake the solve.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/run_multifly_ik.py third_party/jarvis_jax/tests/test_run_multifly_ik.py
git commit -m "feat(cse): multi-fly IK orchestrator (link -> per-fly bout -> STAC -> silhouette IK)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Courtship validation report + 2nd-fly keypoint-withhold ablation

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/run_multifly_ik.py` (add `run_multifly_ablation`)
- Test: `third_party/jarvis_jax/tests/test_run_multifly_ik.py` (add ablation gate)

**Interfaces:**
- Consumes: Task 5 `run_multifly_ik`, `ann_id_by_image_for_fly`; Task 4 `run_ablation(recording, *, ik_h5, model_xml, mesh_npz, root, split, calib_dir, wing_weight, max_frames, smooth_weight, n_iter, corridor, out_dir, ann_id_by_image) -> dict` (with the new `ann_id_by_image` kwarg); Task 3 `build_fly_bout`; `run_stac_bout.run`.
- Produces:
  - `run_multifly_ablation(recording, *, root, calib_dir, anatomy_yaml, model_xml, mesh_npz, stac_config_dir, out_dir, split="val", ablate_fly_id: int = 1, coco_path: str | None = None, max_frames: int = 0, n_iter: int = 50, wing_weight: float = 0.5) -> dict` — link + build per-fly bouts + per-fly STAC as in Task 5, then run `run_ablation` on `ablate_fly_id` (default fly1, the "female" slot) with that fly's `ann_id_by_image`; also run `run_single_fly` on the OTHER fly to confirm both still reproject cleanly. Returns `{"ablation": <run_ablation dict for ablate_fly_id>, "other_fly": {"fly_id": int, "reproj_px": float}, "ablate_fly_id": int}`.

- [ ] **Step 1: Write the failing test**

Add to `third_party/jarvis_jax/tests/test_run_multifly_ik.py`:

```python
@pytest.mark.skipif(not _HAVE, reason="courtship coco / model / mesh / stac config not present")
def test_run_multifly_ablation_recovers_second_fly_wing(tmp_path):
    """GPU real run: withhold fly1's wing keypoints; the silhouette condition (c)
    must recover wing extent toward the SAM tip (recovery_to_sam > 0) while the
    OTHER fly still reprojects cleanly."""
    from jarvis_jax.cse.run_multifly_ik import run_multifly_ablation
    out = run_multifly_ablation(
        REC, root=ROOT, calib_dir=CALIB, anatomy_yaml=ANATOMY, model_xml=XML,
        mesh_npz=MESH, stac_config_dir=STAC_CFG, out_dir=str(tmp_path),
        split="val", ablate_fly_id=1, coco_path=COCO, max_frames=16, n_iter=30,
    )
    ab = out["ablation"]
    assert set(ab["conditions"]) == {"reference", "baseline", "silhouette"}
    assert out["ablate_fly_id"] == 1
    # both flies reproject cleanly: the non-ablated fly's reproj_px is bounded,
    # and the ablated fly's non-wing reproj_px (silhouette condition) is bounded.
    assert np.isfinite(out["other_fly"]["reproj_px"]) and out["other_fly"]["reproj_px"] < 12.0
    assert np.isfinite(ab["conditions"]["silhouette"]["reproj_px"])
    assert ab["conditions"]["silhouette"]["reproj_px"] < 15.0
    # the silhouette must recover the withheld wing toward the SAM extent on at
    # least one side (positive recovery vs the SAM-triangulated tip).
    assert ab["n_frames_with_tips"] > 0, "no SAM wing tips extracted for fly1"
    rec = ab["recovery_ratio_to_sam_mm"]
    assert any(np.isfinite(rec[s]) and rec[s] > 0.0 for s in ("left", "right")), (
        f"silhouette did not recover fly1's withheld wing toward SAM: {rec}"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_multifly_ik.py::test_run_multifly_ablation_recovers_second_fly_wing -v`
Expected: FAIL with `ImportError: cannot import name 'run_multifly_ablation'`.

- [ ] **Step 3: Write minimal implementation**

Add to `third_party/jarvis_jax/jarvis_jax/cse/run_multifly_ik.py`:

```python
from jarvis_jax.cse.silhouette_ik_solve import run_ablation


def _prepare_fly_ik(recording, fly_dir, coco_path, calib_dir, identity_map, fly_id,
                    anatomy_yaml, model_xml, stac_config_dir, split, stac_overrides):
    """Build a fly's bout + per-fly STAC ik_h5; return (bout_h5, ik_h5)."""
    os.makedirs(fly_dir, exist_ok=True)
    bout_h5 = os.path.join(fly_dir, f"{recording}_bout.h5")
    bout_h5, _ = build_fly_bout(coco_path, calib_dir, recording, identity_map,
                                fly_id, anatomy_yaml, model_xml, bout_h5, split=split)
    ik_h5 = os.path.join(fly_dir, recording, "Fruitfly_ik_v1_cse.h5")
    run_stac_bout.run(bout_h5, ik_h5, stac_config_dir, list(stac_overrides))
    return bout_h5, ik_h5


def run_multifly_ablation(recording, *, root, calib_dir, anatomy_yaml, model_xml,
                          mesh_npz, stac_config_dir, out_dir, split="val",
                          ablate_fly_id=1, coco_path=None, max_frames=0, n_iter=50,
                          wing_weight=0.5,
                          stac_overrides=("paths=hyak", "anatomy=v1", "dataset=")):
    """2nd-fly keypoint-withhold ablation on a courtship recording. See Interfaces."""
    if coco_path is None:
        coco_path = os.path.join(root, "annotations", f"instances_{split}.json")
    identity_map = link_recording(coco_path, recording, calib_dir, split=split, n_flies=2)
    os.makedirs(out_dir, exist_ok=True)

    other_fly_id = 1 - ablate_fly_id

    # ablated fly: build bout + STAC, then run_ablation with its ann selection.
    ab_dir = os.path.join(out_dir, f"fly{ablate_fly_id}")
    _, ab_ik = _prepare_fly_ik(recording, ab_dir, coco_path, calib_dir, identity_map,
                               ablate_fly_id, anatomy_yaml, model_xml,
                               stac_config_dir, split, stac_overrides)
    ab_ann = ann_id_by_image_for_fly(identity_map, coco_path, recording, ablate_fly_id)
    ablation = run_ablation(
        recording, ik_h5=ab_ik, model_xml=model_xml, mesh_npz=mesh_npz, root=root,
        split=split, calib_dir=calib_dir, wing_weight=wing_weight,
        max_frames=max_frames, n_iter=n_iter, out_dir=ab_dir, ann_id_by_image=ab_ann)

    # other fly: build bout + STAC, run_single_fly to confirm clean reprojection.
    ot_dir = os.path.join(out_dir, f"fly{other_fly_id}")
    _, ot_ik = _prepare_fly_ik(recording, ot_dir, coco_path, calib_dir, identity_map,
                               other_fly_id, anatomy_yaml, model_xml,
                               stac_config_dir, split, stac_overrides)
    ot_ann = ann_id_by_image_for_fly(identity_map, coco_path, recording, other_fly_id)
    ot_report = run_single_fly(
        recording, ik_h5=ot_ik, model_xml=model_xml, mesh_npz=mesh_npz, root=root,
        split=split, calib_dir=calib_dir, use_silhouette=True, wing_weight=wing_weight,
        max_frames=max_frames, n_iter=n_iter, out_dir=ot_dir, ann_id_by_image=ot_ann)

    return {
        "ablation": ablation,
        "other_fly": {"fly_id": other_fly_id, "reproj_px": ot_report["reproj_px"]},
        "ablate_fly_id": ablate_fly_id,
    }
```

- [ ] **Step 4: Run the CPU-safe import + wiring check**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_multifly_ik.py::test_ann_id_by_image_for_fly_flattens_map -v`
Expected: PASS (confirms the module still imports with the new symbols). The full ablation test needs GPU (Step 5).

- [ ] **Step 5: Real GPU run (background bash).** Under the activated env, run the ablation test in a BACKGROUND bash (per-fly STAC ×2 + three-condition solve is heavy):

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd third_party/jarvis_jax && \
  STAC_ANATOMY_V1=<path> STAC_CONFIG_DIR=<path> \
  python -m pytest tests/test_run_multifly_ik.py::test_run_multifly_ablation_recovers_second_fly_wing -v -s
```

SUCCESS CRITERION: with fly1's wing keypoints withheld, the silhouette condition recovers wing extent toward the SAM tip (`recovery_ratio_to_sam_mm > 0` on ≥1 side, `n_frames_with_tips > 0`), the ablated fly's non-wing `reproj_px` stays bounded (<15px), and the OTHER fly reprojects cleanly (`reproj_px < 12px`). If recovery is weak or negative, report the numbers honestly (both `*_to_sam` and `*_to_ref`, per the `run_ablation` docstring — the keypoint reference is itself keypoint-limited and can disagree in sign with the SAM extent). Escape hatch for STAC intractability is the same as Task 5 Step 5.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/run_multifly_ik.py third_party/jarvis_jax/tests/test_run_multifly_ik.py
git commit -m "feat(cse): courtship 2nd-fly keypoint-withhold ablation + both-fly validation

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

### Spec-coverage check (each Phase-3 spec requirement → task)

| Phase-3 spec / decision / brief requirement | Task |
|---|---|
| Cross-camera identity linker (spec component 2; spec §11 Phase 3; user decision 3 — JAX-native affine, not torch) | **Task 1** (frameset core) + **Task 2** (recording-level) |
| Epipolar/triangulation-consistency + min-residual + nearest-reproj for short cameras (mirror `assign_identities` algorithm) | **Task 1** `score_assignment` + `link_frameset` |
| De-collapse the `_index_coco` first/last-ann-wins bug → per-fly triangulated keypoints | **Task 3** `build_fly_bout` |
| Per-fly bout h5 in the exact `cse_labels.build_bout` schema (`run_stac_bout`/`build_solver_inputs` unchanged) | **Task 3** |
| Thread per-identity mask + kp selection through the Phase-2 driver (`ann_id_by_image`, backward compatible) | **Task 4** |
| Per-fly STAC solve (reuse `run_stac_bout` per fly → per-fly ik_h5) | **Task 5** (primary path) + documented fallback |
| Per-fly silhouette IK (spec components 4/6/7; reuse `run_single_fly`; silhouette landmarks as confidence-gated 3-D markers) | **Task 5** (both flies) |
| Multi-view fusion at affine triangulation; affine camera model everywhere (spec §8) | **Tasks 1, 3, 5** (all use affine DLT / `ReprojectionTool`) |
| Reuse temporal jaxls solver unmodified (spec decision 4) | **Tasks 5, 6** (via `run_single_fly`/`run_ablation` → `solve_ik` → `JaxlsBatchSolver`) |
| Per-recording bundle adjustment (spec component 3) as a preprocessing step for this recording | **Task 5 orchestrator** (optional BA step; falls back to factory calib — noted in Global Constraints; primary linker/IK path does not require it) |
| Courtship validation: both flies reproject cleanly (spec §11 Phase 3) | **Task 5** test (`reproj_px < 12px` both flies) + **Task 6** other-fly check |
| Female = keypoint-withhold ablation on 2nd fly (user decision 2; reuse `run_ablation`/`_withhold_wing_kp`) | **Task 6** |
| ≤2 animals (spec §10 non-goal >2) | `n_flies=2` throughout; `link_frameset` generalizes but is validated/gated at 2 |
| Sex disambiguation out of scope (arbitrary stable fly0/fly1) | Noted in Global Constraints; not a task (follow-up) |

**Gaps / deferred (explicitly out of Phase-3 scope, per spec §11):** unified JAX detector + dense head (Phase 5), dense-silhouette 2-D factor (Phase 6), active-parts config for headless/amputation (Phase 4), outputs/QC packaging (Phase 7), keypoint-starved-from-`extra_masks` female handling (deferred per user decision 2), male-vs-female sex identification (deferred; the torch precedent's mask-area sex-ID Step 4 is intentionally NOT ported — Phase 3 uses arbitrary stable slots).

### Placeholder scan

- No "TBD"/"implement later"/"handle edge cases" left. Every code step contains runnable code; every run step has an exact command + expected result.
- One deliberate cleanup instruction is called out explicitly (Task 5 Step 4 removes the dead `n_fs = ... is None` line and replaces it with the real `h5py` frameset count) — flagged rather than hidden so the implementer does not ship the placeholder.
- The two `<path>` tokens in Tasks 5/6 Step-5 shell commands are environment-specific (`STAC_ANATOMY_V1`, `STAC_CONFIG_DIR`) — resolved from the same stac-mjx configs `run_stac_bout` already documents; the test constants give concrete defaults.

### Type-consistency check across tasks

- **Identity map type** produced by Task 2 `link_recording` is `dict[str, dict[int, dict[int, int]]]` (`fs_key → fly_id → cam_idx → ann_id`). Consumed by Task 3 `build_fly_bout(identity_map, fly_id)` via `identity_map[fs_key][fly_id] -> {cam_idx: ann_id}` and by Task 5/6 `ann_id_by_image_for_fly(identity_map, ..., fly_id)` iterating the same nesting. Consistent (int keys throughout; ann_id is `int`).
- **`ann_id_by_image` type** produced by Task 5 `ann_id_by_image_for_fly` is `dict[int, int]` (`image_id → ann_id`). Consumed by Task 4 `run_single_fly`/`run_ablation`/`extract_tips_for_frames`/`_triangulate_kp_mm`/`_ann_for_image` all as `dict[int, int] | None`. Consistent — and the same dict Task 6 passes into `run_ablation`.
- **`_triangulate_kp_mm` signature change** (Task 4): its 5th arg becomes `id2ann_multi: dict[int, list[dict]]` (was `dict[int, dict]`), plus a new 6th `ann_id_by_image`. Every in-module caller (the three collapse sites) is updated in Task 4 to build the multi-ann dict; no other module imports `_triangulate_kp_mm` (verified — it is module-private). The Task-4 unit test constructs `id2ann_multi` as `dict[int, list[dict]]`, matching.
- **`build_fly_bout` return** `(out_h5: str, scale: float)` matches how Task 5 unpacks `bout_h5, scale = build_fly_bout(...)` and Task 6 `_prepare_fly_ik` unpacks `bout_h5, _ = build_fly_bout(...)`.
- **`run_stac_bout.run(bout_h5, out_h5, stac_config_dir, overrides)`** signature matches the existing implementation (verified: `run(bout_h5, out_h5, stac_config_dir, overrides)`), and the per-fly `ik_h5` path is placed so `run_single_fly`'s `dirname(dirname(ik_h5))/<rec>_bout.h5` derivation resolves to the per-fly bout (Task 5 note; explicit `bout_h5=` passthrough is the fallback if the derivation assumption breaks).
- **`run_single_fly` report keys** (`qpos_shape`, `reproj_px`, `wing_tip_err_*`, `n_frames_with_tips`, ...) are asserted in Task 5's test exactly as the Phase-2 driver returns them; **`run_ablation` keys** (`conditions`, `recovery_ratio_to_sam_mm`, `n_frames_with_tips`, per-condition `reproj_px`) are asserted in Task 6's test exactly as the Phase-2 ablation returns them.
