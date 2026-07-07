# Phase 7 — Outputs/QC + deploy (V1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build component 8 of the multi-view silhouette+keypoint IK pipeline — serialize per-fly IK outputs (qpos/root/scale + posed mesh + 3-D keypoints), compute a bundled QC report (per-camera reprojection error, silhouette IoU, leave-one-out reprojection), render reprojection overlay videos, and deploy this outputs+QC layer over the existing scenario solvers' qpos across all four scenarios (single / multi-courtship / amputation / headless), on V1 anatomy only.

**Architecture:** Four new modules under `third_party/jarvis_jax/jarvis_jax/cse/`, each with a pure-function core and a thin driver. `outputs.py` FK-reposes a stored qpos trajectory (batched over frames) into world-mm mesh vertices (a documented vertex SUBSET by default) + 50 FK sites, using the exact per-frame Umeyama model→mm bridge already used by `run_single_fly`, and writes one h5 per fly. `qc.py` generalizes the main()-only patterns in `reproj_validate.py` into importable functions (all 50 keypoints, not just wing-tips) and reuses the Phase-6 IoU helpers over the posed mesh, bundling everything into one json/npz. `reproj_video.py` streams each camera's raw `Frame_*.jpg` through `imageio.get_writer` — one decoded frame in memory at a time — drawing projected mesh+keypoints (+optional SAM contour). `run_outputs_qc.py` is the CLI deploy driver that resolves calibration PER RECORDING, loads a scenario's `{recording}_qpos.npz` (or a passed npz), and calls the three modules. Phase 7 CONSUMES scenario qpos and never changes any solver.

**Tech Stack:** Python 3, JAX/MJX (FK; run FK/site tests on GPU), NumPy (QC math, CPU), `stac_mjx.utils` (mjx kinematics/site xpos), `stac_mjx.io_dict_to_hdf5` (h5 I/O, read-only import from the frozen submodule), `stac_mjx.io` (ik-h5 config/data load), `jarvis_jax.geometry.reprojection_tool.ReprojectionTool` (DLT reproj/triangulation), `imageio` (video, v2.37 present), `matplotlib` (rasterize overlays), `pytest`.

## Global Constraints

- Write things in JAX when possible for speed, and prioritize computational efficiency in BOTH speed and memory. (Much of outputs/QC is numpy + I/O; the FK/projection over T frames × 7 cams × verts is the hot part — keep it vectorized/batched; for VIDEO, STREAM frames one at a time via imageio, never hold all decoded frames in memory.)
- Env: gpu-l40s compute node; `source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH`; GPU on-node `XLA_PYTHON_CLIENT_PREALLOCATE=false` (or MEM_FRACTION=0.9); coordinator runs the heavy GPU/video deploy runs (subagents orphan on long jobs). RUN MJX/JAXLS-INVOLVING TESTS ON GPU (CPU mjx is far too slow) — drop JAX_PLATFORMS=cpu for those; pure-numpy unit tests stay CPU.
- New code under `third_party/jarvis_jax/jarvis_jax/cse/`; tests under `third_party/jarvis_jax/tests/`. stac-mjx is a FROZEN submodule — do NOT modify it (import from it read-only).
- TDD: failing test w/ REAL asserts -> run(fail) -> minimal impl -> run(pass) -> commit per task by EXPLICIT path (never git add -A). Commits end "Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>".
- Do NOT commit outputs/videos/QC artifacts/checkpoints/data. Branch elliottabe/paper_update_062026 (rolling).
- PER-RECORDING CALIBRATION: the deploy/QC driver MUST resolve calib per recording (refined cse_work/calib_refined/<rec> if present else red_data_unified_V3/calib_params/<rec>); do NOT rely on the hardcoded single-recording _DEFAULT_*_CALIB_DIR (silhouette_ik_solve.py:235-241 default to ONE recording — a real deploy blocker).

## Verified reused interfaces (do NOT rebuild — import read-only)

- `stac_mjx.io_dict_to_hdf5.save(filename, dic, compression='gzip', compression_opts=5, chunked_datasets=None)` and `.load(filename, ASLIST=False, enable_jax=False, auto_convert_lists=True)` — nested-dict h5 write/read.
- `stac_mjx.io.load_stac_data(ik_h5) -> (config, stac_data)`; `stac_data` has `.qpos`, `.kp_data`, `.marker_sites`, `.kp_names`, `.names_qpos`, `.offsets`. (Used inside `build_solver_inputs`; we also read `marker_sites` directly via `io_dict_to_hdf5.load(ik_h5)["marker_sites"]` as `run_single_fly` does.)
- `stac_mjx.utils`: `kinematics(mjx_model, data)`, `com_pos(mjx_model, data)`, `get_site_xpos(data, site_idxs) -> (n_site,3)`.
- `jarvis_jax.cse.silhouette_ik.load_anatomy(model_xml, mesh_npz) -> dict(m, mx, dx, vlocal, vgeom, faces, fps, seg_names, seg_ids, vertex_segment, nq, qpos0)`.
- `jarvis_jax.cse.silhouette_ik.make_fk_repose(anat) -> fk_repose(qpos, scale=1.0, indices=None) -> (K,3)` MODEL-frame world verts (jitted; `indices` selects into the FULL vertex array).
- `jarvis_jax.cse.silhouette_ik_solve`: `build_solver_inputs(ik_h5, model_xml) -> dict(mjx_model, mjx_data, q_init(T,nq), kp_data(T,n_kp,3), kps_to_opt, qs_to_opt, lb, ub, site_idxs(n_kp,), q_reg_weights, kp_names)`; `_umeyama(src,dst)->(s,R,t)` (dst=s*(R@src.T).T+t); `_model_to_mm(pts,s,R,t)`; `_mm_to_model(pts,s,R,t)`; `_triangulate_kp_mm(rt, ik_kpnames, coco_kpnames, cam2img, id2ann_multi, ann_id_by_image=None) -> (kp_mm(n,3), valid(n,) bool)`; `_cam2img_for_frame(fs_imgids_row, id2file, cam_names) -> {cam_idx:image_id}`; `_ann_for_image(id2ann_multi, image_id, ann_id_by_image=None) -> ann|None`; `_load_sam_mask(root, split, file_name, ann_id) -> mask(H,W) bool | None`.
- `jarvis_jax.cse.run_silhouette_polish.iou_of_projected_verts(verts2d, mask_shape, ref_mask) -> float` (hard); `soft_iou_of_verts(verts2d, mask, *, sigma=1.3, splat_k=2) -> float` (soft, eval-only).
- `jarvis_jax.geometry.reprojection_tool.ReprojectionTool(calib_dir)`: `.cameras` (dict name->Camera), `.num_cameras`, `._camera_list[c].cameraMatrix` (3,4 DLT), `.reproject_point(X_mm) -> (n_cam,2)` (DLT + perspective divide — the correct reporting projection), `.reconstruct_point(points2d(n_cam,2), cams_to_use=None) -> X_mm(3,)`.
- `imageio.get_writer(path, fps=FPS)` context manager; `video.append_data(frame_uint8)` (mirror `stac-mjx/stac_mjx/stac.py:735,745` — do NOT import stac's helper; use imageio directly).
- Scenario qpos outputs (CONSUME, do not change): single & active-parts write `{out_dir}/{recording}_qpos.npz` (key `qpos` (T,nq)); multifly writes per fly `{out_dir}/fly{fid}/{recording}_qpos.npz`.

## Data layout (verified; all under `/gscratch/portia/eabe/data/Johnson_lab/`)

- ik h5: `cse_work/<rec>/Fruitfly_ik_v1_cse.h5` (qpos(T,93), kp_data, marker_sites(T,50,3), kp_names, names_qpos, config...).
- bout h5: `cse_work/<rec>_bout.h5` (fs_imgids(T,7), keypoints(T,50,3), kp_names, vis(T,50)).
- coco: `red_data/red_data_unified_V3/annotations/instances_{split}.json` (keypoint_names, images[i].file_name = `<rec>/<cam>/Frame_*.jpg`, annotations, framesets).
- SAM masks: `root/sam3_masks/<split>/<file_name-no-ext>.npz` (via `_load_sam_mask`).
- calib: factory `red_data/red_data_unified_V3/calib_params/<rec>`; refined `cse_work/calib_refined/<rec>`.
- raw frames: `red_data/red_data_unified_V3/<split>/<rec>/<cam>/Frame_<n>.jpg`.
- V1 model `fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml` (nq=93); V1 mesh `fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz` (~61666 verts; `fps_300`, `fps_500` subsets available via `load_anatomy(...)["fps"]`).

## Efficiency / memory decisions baked in (spec §8/§9 mandate)

- **mesh_mm size.** Full posed mesh is `T × 61666 × 3 × 4 bytes` (≈ 0.74 MB/frame; ≈ 22 GB for a 30k-frame recording) — never the default. The output writer defaults to a **vertex SUBSET** (`fps_500`, from `anat["fps"][500]`, 500 verts → ≈ 6 KB/frame) stored **float32**, with the subset name recorded in the h5 (`mesh_subset` attr) and the subset vertex indices stored (`mesh_vert_idx`). A `mesh_subset="full"` option writes the whole mesh (documented as memory-heavy; only for short clips). This is the single biggest memory lever and is called out in the writer docstring.
- **Batched FK.** FK over T frames is done with a single `jax.vmap` over `qpos` (mesh subset) so the T×K repose is one XLA call, not a Python loop; sites (needing `com_pos`) reuse the existing per-frame `kinematics→com_pos→get_site_xpos` pattern from `run_single_fly` (kept as-is; that path is not vmap-friendly across mjx `com_pos`, so it stays a loop but only over the 50 sites, cheap).
- **Video streaming.** `reproj_video.py` opens ONE `imageio.get_writer` per camera and appends one rasterized frame at a time; the raw jpg for frame t is read, drawn on, rasterized to uint8, appended, then dropped. No all-frames buffer ever exists.

---

### Task 1: Output writer (`cse/outputs.py`)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/outputs.py`
- Test: `third_party/jarvis_jax/tests/test_outputs.py`

**Interfaces:**
- Consumes: `silhouette_ik.load_anatomy`, `silhouette_ik.make_fk_repose`, `stac_mjx.io_dict_to_hdf5.save`, `stac_mjx.utils.{kinematics,com_pos,get_site_xpos}`, and the per-frame mm bridge tuple `(s, R, t)` produced by `silhouette_ik_solve._umeyama` (caller supplies bridges; the writer does not fit them).
- Produces:
  - `mesh_subset_indices(anat, subset="fps_500") -> np.ndarray(int32)` — full-vertex-array indices for a named fps subset, or `np.arange(nverts)` when `subset=="full"`.
  - `fk_mesh_world_mm(fk_repose, qpos(T,nq), vert_idx(K,), bridges) -> np.ndarray(T,K,3) float32` — vmapped model-frame FK of the subset verts per frame, each frame mapped model→mm by that frame's bridge `(s,R,t)`; frames whose bridge is `None` get `np.nan`.
  - `fk_sites_world_mm(mjx_model, mjx_data, site_idxs, qpos(T,nq), bridges) -> np.ndarray(T,n_site,3) float32` — per-frame `kinematics→com_pos→get_site_xpos` then model→mm; `None` bridge -> `np.nan`.
  - `write_outputs_h5(out_path, *, qpos(T,nq), root_se3(T,7), scale(T,), mesh_mm(T,K,3), kp3d_mm(T,n_site,3), mesh_vert_idx(K,), mesh_subset:str, kp_names:list[str]) -> str` — writes an h5 (via `io_dict_to_hdf5.save`) with those datasets under a flat dict and returns `out_path`.
  - `build_fly_outputs(recording, *, ik_h5, model_xml, mesh_npz, qpos, bridges, out_path, mesh_subset="fps_500") -> dict` — glue: loads anatomy + solver inputs, derives root/scale from qpos+bridges, calls the three above, writes the h5, returns a small schema dict `{out_path, mesh_subset, shapes:{...}}`.

- [ ] **Step 1: Write the failing test (subset-index + schema on synthetic qpos, CPU)**

```python
# tests/test_outputs.py
import os
import numpy as np
import pytest


def test_mesh_subset_indices_full_and_named():
    from jarvis_jax.cse import outputs
    fake_anat = {"fps": {500: np.arange(0, 61666, 123, dtype=np.int64)[:500]},
                 "vlocal": np.zeros((61666, 3), np.float32)}
    idx = outputs.mesh_subset_indices(fake_anat, subset="fps_500")
    assert idx.dtype == np.int32
    assert idx.shape == (500,)
    full = outputs.mesh_subset_indices(fake_anat, subset="full")
    assert full.shape == (61666,)
    assert full[0] == 0 and full[-1] == 61665


def test_write_outputs_h5_roundtrip(tmp_path):
    from jarvis_jax.cse import outputs
    import stac_mjx.io_dict_to_hdf5 as ioh5
    T, nq, K, nkp = 3, 93, 5, 50
    qpos = np.random.default_rng(0).normal(size=(T, nq)).astype(np.float32)
    root = qpos[:, :7].copy()
    scale = np.full((T,), 0.9, np.float32)
    mesh_mm = np.zeros((T, K, 3), np.float32)
    kp3d = np.zeros((T, nkp, 3), np.float32)
    vidx = np.arange(K, dtype=np.int32)
    out = str(tmp_path / "fly_out.h5")
    p = outputs.write_outputs_h5(
        out, qpos=qpos, root_se3=root, scale=scale, mesh_mm=mesh_mm,
        kp3d_mm=kp3d, mesh_vert_idx=vidx, mesh_subset="fps_500",
        kp_names=[f"kp{i}" for i in range(nkp)])
    assert p == out and os.path.exists(out)
    d = ioh5.load(out)
    assert np.asarray(d["qpos"]).shape == (T, nq)
    assert np.asarray(d["mesh_mm"]).shape == (T, K, 3)
    assert np.asarray(d["kp3d_mm"]).shape == (T, nkp, 3)
    assert np.asarray(d["root_se3"]).shape == (T, 7)
    assert np.asarray(d["scale"]).shape == (T,)
    assert np.asarray(d["mesh_vert_idx"]).shape == (K,)
    assert str(np.asarray(d["mesh_subset"]).astype(str)) == "fps_500"


def test_fk_mesh_world_mm_applies_bridge_and_nan(tmp_path):
    from jarvis_jax.cse import outputs
    # fk stub: identity model verts independent of qpos (K fixed points)
    K = 4
    base = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)

    def fk_stub(qpos, scale=1.0, indices=None):
        return base[indices] if indices is not None else base

    qpos = np.zeros((2, 7), np.float32)
    vidx = np.arange(K, dtype=np.int32)
    # frame0 bridge: scale 2, identity R, translate +10 in x; frame1: None
    bridges = [(2.0, np.eye(3), np.array([10.0, 0, 0])), None]
    out = outputs.fk_mesh_world_mm(fk_stub, qpos, vidx, bridges)
    assert out.shape == (2, K, 3)
    # frame0 vertex1 (1,0,0) -> 2*(1,0,0)+ (10,0,0) = (12,0,0)
    assert np.allclose(out[0, 1], [12.0, 0.0, 0.0])
    # frame1 all nan
    assert np.isnan(out[1]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_outputs.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.outputs'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/outputs.py
"""Phase 7 / component 8: serialize per-fly IK outputs to h5.

Given a solved qpos trajectory + the V1 anatomy + a per-frame model->mm
Umeyama bridge (fit exactly as run_single_fly does, from the STAC-fitted
MODEL-frame marker_sites vs. the recording's triangulated coco keypoints),
FK-repose the posed mesh (a documented vertex SUBSET by default) and the 50
keypoint sites, map both to calibrated world mm, and write an h5 per fly:
{qpos, root_se3, scale, mesh_mm, kp3d_mm, mesh_vert_idx, mesh_subset, kp_names}.

MEMORY: the full posed mesh is T*61666*3*4 bytes (~0.74 MB/frame, ~22 GB for
a 30k-frame recording). The default is a `fps_500` vertex SUBSET (500 verts,
~6 KB/frame, float32). `mesh_subset="full"` writes the whole mesh and is only
for short clips -- documented tradeoff per the efficiency mandate. FK over T
frames is a single jax.vmap for the mesh subset (one XLA call).
"""
from __future__ import annotations
import numpy as np


def mesh_subset_indices(anat, subset="fps_500"):
    """Full-vertex-array indices for a named fps subset (or all verts).

    subset="full" -> np.arange(nverts). subset="fps_<N>" -> anat["fps"][N]
    (the FK-repose `indices` argument indexes the FULL vertex array, so the
    fps arrays -- which already hold full-array indices -- are used directly).
    """
    if subset == "full":
        nverts = len(anat["vlocal"])
        return np.arange(nverts, dtype=np.int32)
    if not subset.startswith("fps_"):
        raise ValueError(f"subset must be 'full' or 'fps_<N>', got {subset!r}")
    n = int(subset.split("_")[1])
    if n not in anat["fps"]:
        raise ValueError(f"fps subset {n} not in anatomy (have {sorted(anat['fps'])})")
    return np.asarray(anat["fps"][n], dtype=np.int32)


def fk_mesh_world_mm(fk_repose, qpos, vert_idx, bridges):
    """(T,K,3) world-mm mesh verts: vmapped model-frame FK of the subset, then
    per-frame model->mm via `bridges[t]=(s,R,t)`. None bridge -> nan frame."""
    import jax
    import jax.numpy as jnp
    qpos = np.asarray(qpos, np.float32)
    T = qpos.shape[0]
    vert_idx = np.asarray(vert_idx, np.int32)
    K = vert_idx.shape[0]

    def _one(q):
        return fk_repose(q, 1.0, vert_idx)  # (K,3) model frame

    verts_model = np.asarray(jax.vmap(_one)(jnp.asarray(qpos)))  # (T,K,3)
    out = np.full((T, K, 3), np.nan, np.float32)
    for t in range(T):
        br = bridges[t]
        if br is None:
            continue
        s, R, tr = br
        vm = s * (np.asarray(R) @ verts_model[t].T).T + np.asarray(tr)
        out[t] = vm.astype(np.float32)
    return out


def fk_sites_world_mm(mjx_model, mjx_data, site_idxs, qpos, bridges):
    """(T,n_site,3) world-mm keypoint sites: per-frame kinematics->com_pos->
    get_site_xpos (MODEL) then model->mm via bridges[t]. None -> nan frame."""
    import stac_mjx.utils as stac_utils
    qpos = np.asarray(qpos, np.float32)
    T = qpos.shape[0]
    site_idxs = np.asarray(site_idxs)
    n_site = site_idxs.shape[0]
    out = np.full((T, n_site, 3), np.nan, np.float32)
    for t in range(T):
        br = bridges[t]
        if br is None:
            continue
        data_t = mjx_data.replace(qpos=qpos[t])
        data_t = stac_utils.kinematics(mjx_model, data_t)
        data_t = stac_utils.com_pos(mjx_model, data_t)
        sites_model = np.asarray(stac_utils.get_site_xpos(data_t, site_idxs))
        s, R, tr = br
        out[t] = (s * (np.asarray(R) @ sites_model.T).T + np.asarray(tr)).astype(np.float32)
    return out


def write_outputs_h5(out_path, *, qpos, root_se3, scale, mesh_mm, kp3d_mm,
                     mesh_vert_idx, mesh_subset, kp_names):
    """Write the per-fly outputs h5 via stac_mjx.io_dict_to_hdf5.save."""
    import os
    import stac_mjx.io_dict_to_hdf5 as ioh5
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    dic = {
        "qpos": np.asarray(qpos, np.float32),
        "root_se3": np.asarray(root_se3, np.float32),
        "scale": np.asarray(scale, np.float32),
        "mesh_mm": np.asarray(mesh_mm, np.float32),
        "kp3d_mm": np.asarray(kp3d_mm, np.float32),
        "mesh_vert_idx": np.asarray(mesh_vert_idx, np.int32),
        "mesh_subset": np.asarray(str(mesh_subset)),
        "kp_names": np.asarray([str(n) for n in kp_names]),
    }
    ioh5.save(out_path, dic)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_outputs.py -q`
Expected: PASS (3 passed). If `mesh_subset` roundtrips as bytes, the test coerces via `.astype(str)`; the assert compares the string.

- [ ] **Step 5: Add the `build_fly_outputs` glue + its test**

Add to `tests/test_outputs.py`:

```python
def test_build_fly_outputs_derives_root_scale(monkeypatch, tmp_path):
    """build_fly_outputs must set root_se3 = qpos[:, :7] and scale from the
    per-frame bridge s (nan where bridge is None), and write the h5."""
    from jarvis_jax.cse import outputs
    import stac_mjx.io_dict_to_hdf5 as ioh5
    T, nq, nkp, K = 3, 93, 50, 4

    qpos = np.random.default_rng(1).normal(size=(T, nq)).astype(np.float32)
    bridges = [(1.5, np.eye(3), np.zeros(3)), None, (2.0, np.eye(3), np.zeros(3))]
    fake_anat = {"fps": {500: np.arange(K, dtype=np.int64)},
                 "vlocal": np.zeros((K, 3), np.float32)}
    base = np.eye(3, dtype=np.float32)[:K] if K <= 3 else np.zeros((K, 3), np.float32)

    def fk_stub(q, scale=1.0, indices=None):
        return base[indices] if indices is not None else base

    monkeypatch.setattr(outputs, "load_anatomy", lambda xml, npz: fake_anat)
    monkeypatch.setattr(outputs, "make_fk_repose", lambda anat: fk_stub)
    monkeypatch.setattr(outputs, "_load_solver_bits",
                        lambda ik_h5, xml: (None, None, np.arange(nkp), [f"kp{i}" for i in range(nkp)]))
    monkeypatch.setattr(outputs, "fk_sites_world_mm",
                        lambda *a, **k: np.zeros((T, nkp, 3), np.float32))

    out = str(tmp_path / "fly0.h5")
    rep = outputs.build_fly_outputs(
        "rec", ik_h5="x", model_xml="x", mesh_npz="x", qpos=qpos,
        bridges=bridges, out_path=out, mesh_subset="fps_500")
    assert rep["out_path"] == out
    d = ioh5.load(out)
    assert np.allclose(np.asarray(d["root_se3"]), qpos[:, :7])
    sc = np.asarray(d["scale"])
    assert np.isclose(sc[0], 1.5) and np.isnan(sc[1]) and np.isclose(sc[2], 2.0)
```

- [ ] **Step 6: Run to verify the new test fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_outputs.py::test_build_fly_outputs_derives_root_scale -q`
Expected: FAIL with `AttributeError: module 'jarvis_jax.cse.outputs' has no attribute 'build_fly_outputs'`

- [ ] **Step 7: Implement `build_fly_outputs` + helper**

Append to `jarvis_jax/cse/outputs.py`:

```python
from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose


def _load_solver_bits(ik_h5, model_xml):
    """Return (mjx_model, mjx_data, site_idxs, kp_names) from build_solver_inputs."""
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs
    inp = build_solver_inputs(ik_h5, model_xml)
    return inp["mjx_model"], inp["mjx_data"], inp["site_idxs"], list(inp["kp_names"])


def build_fly_outputs(recording, *, ik_h5, model_xml, mesh_npz, qpos, bridges,
                      out_path, mesh_subset="fps_500"):
    """Glue: FK the mesh subset + sites to world mm and write the per-fly h5.

    root_se3 = qpos[:, :7] (free-joint SE3: xyz + wxyz quat, the model-frame
    root as stored by the solver). scale[t] = the per-frame bridge scale `s`
    (nan where bridges[t] is None). Efficiency: the mesh FK is one jax.vmap.
    """
    anat = load_anatomy(model_xml, mesh_npz)
    fk = make_fk_repose(anat)
    vert_idx = mesh_subset_indices(anat, subset=mesh_subset)
    mjx_model, mjx_data, site_idxs, kp_names = _load_solver_bits(ik_h5, model_xml)

    qpos = np.asarray(qpos, np.float32)
    T = qpos.shape[0]
    mesh_mm = fk_mesh_world_mm(fk, qpos, vert_idx, bridges)
    kp3d_mm = fk_sites_world_mm(mjx_model, mjx_data, site_idxs, qpos, bridges)

    root_se3 = qpos[:, :7].copy()
    scale = np.array([br[0] if br is not None else np.nan for br in bridges],
                     dtype=np.float32)
    assert scale.shape[0] == T, "one bridge per frame required"

    write_outputs_h5(
        out_path, qpos=qpos, root_se3=root_se3, scale=scale, mesh_mm=mesh_mm,
        kp3d_mm=kp3d_mm, mesh_vert_idx=vert_idx, mesh_subset=mesh_subset,
        kp_names=kp_names)
    return {"out_path": out_path, "mesh_subset": mesh_subset,
            "shapes": {"qpos": tuple(qpos.shape), "mesh_mm": tuple(mesh_mm.shape),
                       "kp3d_mm": tuple(kp3d_mm.shape)}}
```

- [ ] **Step 8: Run the full outputs test file**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_outputs.py -q`
Expected: PASS (4 passed). Note the monkeypatch of `outputs.load_anatomy`/`make_fk_repose` requires they are module-level names imported into `outputs` (Step 7 imports them at module top — keep that import).

- [ ] **Step 9: Commit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/outputs.py third_party/jarvis_jax/tests/test_outputs.py
git commit -m "feat(cse): Phase 7 output writer (qpos/root/scale + posed mesh subset + 3D kp to h5)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: QC metrics module (`cse/qc.py`)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/qc.py`
- Test: `third_party/jarvis_jax/tests/test_qc.py`

**Interfaces:**
- Consumes: `ReprojectionTool.reproject_point` / `.reconstruct_point`; `iou_of_projected_verts`, `soft_iou_of_verts` (Task-independent, Phase 6). Reprojection/LOO math is pure NumPy + `rt` (no FK inside qc.py — callers pass already-computed 3-D points).
- Produces:
  - `per_camera_reproj_error(rt, kp3d_mm(n_kp,3), kp2d_by_cam:{cam_idx:(n_kp,2)}, vis_by_cam:{cam_idx:(n_kp,) bool}) -> dict{cam_idx: median_px}` — project each 3-D kp into each cam, per-kp pixel error vs annotated 2-D (only where visible), median per camera.
  - `loo_reproj(rt, kp2d_by_cam:{cam_idx:(n_kp,2)}, vis_by_cam:{cam_idx:(n_kp,) bool}) -> dict{"per_kp":(n_kp,) float, "median": float, "n": int}` — for each keypoint, triangulate from all visible cams; then for each held-out visible cam, re-triangulate from the OTHER visible cams (≥2), reproject into the held-out cam, pixel error vs that cam's own 2-D; aggregate.
  - `silhouette_iou_report(rt, mesh_mm(K,3), masks_by_cam:{cam_idx: mask(H,W) bool}) -> dict{"hard":{cam:iou}, "soft":{cam:iou}, "hard_mean":float, "soft_mean":float}` — project the posed mesh subset into each cam, IoU vs that cam's SAM mask (both helpers).
  - `qc_report(rt, *, kp3d_by_frame, mesh_by_frame, kp2d_by_frame, vis_by_frame, masks_by_frame, out_json:str|None) -> dict` — loop frames, aggregate the three metrics into one dict, optionally `json.dump` it (numpy-safe) and return it.

- [ ] **Step 1: Write the failing tests (synthetic multi-cam geometry, CPU)**

```python
# tests/test_qc.py
import numpy as np
import pytest


class _FakeRT:
    """Minimal ReprojectionTool stand-in with real DLT-ish 3x4 matrices.

    Two orthographic-ish cameras looking down different axes so a 3-D point is
    recoverable and reprojects back exactly (consistent cameras -> LOO ~ 0)."""
    def __init__(self):
        # cam0: image = (x, y); cam1: image = (x, z). Both affine (3rd row homog).
        P0 = np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        P1 = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        self._P = [P0, P1]
        self.num_cameras = 2

        class _C:
            def __init__(s, M): s.cameraMatrix = M
        self._camera_list = [_C(P0), _C(P1)]

    def reproject_point(self, X):
        X = np.asarray(X, float)
        Xh = np.concatenate([X, [1.0]])
        out = np.zeros((self.num_cameras, 2))
        for i, P in enumerate(self._P):
            p = P @ Xh
            out[i] = (p / p[2])[:2]
        return out

    def reconstruct_point(self, points2d, cams_to_use=None):
        cams = cams_to_use if cams_to_use is not None else list(range(self.num_cameras))
        A = []
        for c in cams:
            P = self._P[c]; u, v = points2d[c]
            A.append(u * P[2] - P[0]); A.append(v * P[2] - P[1])
        A = np.asarray(A)
        _, _, Vh = np.linalg.svd(A)
        Xh = Vh[-1] / Vh[-1][3]
        return Xh[:3]


def test_per_camera_reproj_error_zero_on_consistent():
    from jarvis_jax.cse.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    kp2d = {0: np.array([[1., 2.], [4., 5.]]),   # (x,y)
            1: np.array([[1., 3.], [4., 6.]])}   # (x,z)
    vis = {0: np.array([True, True]), 1: np.array([True, True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert set(err) == {0, 1}
    assert err[0] < 1e-9 and err[1] < 1e-9


def test_per_camera_reproj_error_offset():
    from jarvis_jax.cse.qc import per_camera_reproj_error
    rt = _FakeRT()
    kp3d = np.array([[0.0, 0.0, 0.0]])
    kp2d = {0: np.array([[3.0, 4.0]])}   # true proj is (0,0); err = 5
    vis = {0: np.array([True])}
    err = per_camera_reproj_error(rt, kp3d, kp2d, vis)
    assert abs(err[0] - 5.0) < 1e-9


def test_loo_reproj_near_zero_when_consistent():
    from jarvis_jax.cse.qc import loo_reproj
    rt = _FakeRT()
    # one keypoint at (1,2,3); its two 2-D obs are the exact projections.
    kp2d = {0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}
    vis = {0: np.array([True]), 1: np.array([True])}
    rep = loo_reproj(rt, kp2d, vis)
    # only 2 cams -> each held-out leaves 1 other (<2) -> no LOO error terms.
    assert rep["n"] == 0
    assert np.isnan(rep["median"])


def test_loo_reproj_three_cams_consistent():
    from jarvis_jax.cse.qc import loo_reproj

    class _RT3(_FakeRT):
        def __init__(s):
            P0 = np.array([[1., 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            P1 = np.array([[1., 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            P2 = np.array([[0., 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            s._P = [P0, P1, P2]; s.num_cameras = 3

            class _C:
                def __init__(c, M): c.cameraMatrix = M
            s._camera_list = [_C(P0), _C(P1), _C(P2)]

    rt = _RT3()
    X = np.array([1.0, 2.0, 3.0])
    proj = rt.reproject_point(X)
    kp2d = {c: proj[c][None] for c in range(3)}
    vis = {c: np.array([True]) for c in range(3)}
    rep = loo_reproj(rt, kp2d, vis)
    assert rep["n"] == 3          # each of 3 cams held out, 2 others triangulate
    assert rep["median"] < 1e-6   # consistent cams -> ~0 held-out error


def test_qc_report_bundles_keys():
    from jarvis_jax.cse.qc import qc_report
    rt = _FakeRT()
    kp3d_by_frame = [np.array([[1., 2., 3.]])]
    mesh_by_frame = [np.zeros((0, 3))]              # no mesh -> empty iou
    kp2d_by_frame = [{0: np.array([[1., 2.]]), 1: np.array([[1., 3.]])}]
    vis_by_frame = [{0: np.array([True]), 1: np.array([True])}]
    masks_by_frame = [{}]                           # no masks
    rep = qc_report(rt, kp3d_by_frame=kp3d_by_frame, mesh_by_frame=mesh_by_frame,
                    kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                    masks_by_frame=masks_by_frame, out_json=None)
    for k in ("per_camera_reproj_px", "loo_reproj_px", "silhouette_iou", "n_frames"):
        assert k in rep
    assert rep["n_frames"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_qc.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.qc'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/qc.py
"""Phase 7 / component 8: bundled QC for a fitted fly trajectory.

Pure-ish functions (no FK -- callers pass already-FK'd 3-D points/mesh):
  * per_camera_reproj_error -- all 50 kp; generalizes reproj_validate.py's
    per-cam reprojection-error to every keypoint (was wing-tip-only there).
  * loo_reproj -- generalizes reproj_validate.py's leave-one-out wing-tip test
    to all 50 keypoints: per kp, triangulate from all visible cams; per
    held-out cam re-triangulate from the OTHER visible cams (>=2), reproject
    into the held-out cam, pixel error vs its own 2-D.
  * silhouette_iou_report -- reuses Phase-6 iou_of_projected_verts /
    soft_iou_of_verts over the posed mesh subset vs SAM masks.
  * qc_report -- bundles all three across frames into one dict, saves json.
"""
from __future__ import annotations
import json
import numpy as np


def per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam):
    """Median per-camera pixel reprojection error over visible keypoints."""
    kp3d_mm = np.asarray(kp3d_mm, float)
    out = {}
    for c, kp2d in kp2d_by_cam.items():
        vis = np.asarray(vis_by_cam.get(c, np.ones(len(kp3d_mm), bool)), bool)
        errs = []
        for j in range(len(kp3d_mm)):
            if not vis[j]:
                continue
            uv = rt.reproject_point(kp3d_mm[j])[c]
            errs.append(float(np.linalg.norm(uv - np.asarray(kp2d[j], float))))
        if errs:
            out[c] = float(np.median(errs))
    return out


def loo_reproj(rt, kp2d_by_cam, vis_by_cam):
    """Leave-one-out reprojection error over all keypoints.

    For each kp: gather cams where it is visible; for each held-out visible
    cam with >=2 OTHER visible cams, triangulate from the others, reproject
    into the held-out cam, pixel error vs that cam's own observation.
    """
    cams = sorted(kp2d_by_cam)
    if not cams:
        return {"per_kp": np.zeros(0), "median": float("nan"), "n": 0}
    n_kp = len(next(iter(kp2d_by_cam.values())))
    per_kp = np.full(n_kp, np.nan)
    all_errs = []
    for j in range(n_kp):
        vis_cams = [c for c in cams
                    if np.asarray(vis_by_cam.get(c, np.ones(n_kp, bool)))[j]]
        if len(vis_cams) < 3:
            continue
        obs = np.zeros((rt.num_cameras, 2))
        for c in vis_cams:
            obs[c] = np.asarray(kp2d_by_cam[c][j], float)
        kp_errs = []
        for held in vis_cams:
            others = [c for c in vis_cams if c != held]
            if len(others) < 2:
                continue
            X = rt.reconstruct_point(obs, cams_to_use=others)
            proj = rt.reproject_point(X)[held]
            kp_errs.append(float(np.linalg.norm(proj - obs[held])))
        if kp_errs:
            per_kp[j] = float(np.mean(kp_errs))
            all_errs.extend(kp_errs)
    return {"per_kp": per_kp,
            "median": float(np.median(all_errs)) if all_errs else float("nan"),
            "n": len(all_errs)}


def silhouette_iou_report(rt, mesh_mm, masks_by_cam):
    """Hard+soft IoU of the projected posed mesh subset vs SAM masks, per cam."""
    from jarvis_jax.cse.run_silhouette_polish import (
        iou_of_projected_verts, soft_iou_of_verts)
    mesh_mm = np.asarray(mesh_mm, float)
    hard, soft = {}, {}
    for c, mask in masks_by_cam.items():
        if mask is None or len(mesh_mm) == 0:
            continue
        mask = np.asarray(mask)
        uv = np.stack([rt.reproject_point(mesh_mm[k])[c] for k in range(len(mesh_mm))], 0)
        hard[c] = iou_of_projected_verts(uv, mask.shape, mask)
        soft[c] = soft_iou_of_verts(uv, mask)
    hm = float(np.mean(list(hard.values()))) if hard else float("nan")
    sm = float(np.mean(list(soft.values()))) if soft else float("nan")
    return {"hard": hard, "soft": soft, "hard_mean": hm, "soft_mean": sm}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def qc_report(rt, *, kp3d_by_frame, mesh_by_frame, kp2d_by_frame,
              vis_by_frame, masks_by_frame, out_json=None):
    """Bundle per-camera reproj, LOO reproj, and silhouette IoU over frames."""
    T = len(kp3d_by_frame)
    per_cam_all, loo_all, iou_hard_all, iou_soft_all = [], [], [], []
    for t in range(T):
        pc = per_camera_reproj_error(rt, kp3d_by_frame[t], kp2d_by_frame[t], vis_by_frame[t])
        per_cam_all.extend(pc.values())
        lo = loo_reproj(rt, kp2d_by_frame[t], vis_by_frame[t])
        if lo["n"]:
            loo_all.append(lo["median"])
        sr = silhouette_iou_report(rt, mesh_by_frame[t], masks_by_frame[t])
        if np.isfinite(sr["hard_mean"]):
            iou_hard_all.append(sr["hard_mean"])
        if np.isfinite(sr["soft_mean"]):
            iou_soft_all.append(sr["soft_mean"])

    def _agg(xs):
        return float(np.median(xs)) if xs else float("nan")

    report = {
        "per_camera_reproj_px": {"median": _agg(per_cam_all), "n": len(per_cam_all)},
        "loo_reproj_px": {"median": _agg(loo_all), "n_frames": len(loo_all)},
        "silhouette_iou": {"hard_median": _agg(iou_hard_all),
                           "soft_median": _agg(iou_soft_all),
                           "n_frames": len(iou_hard_all)},
        "n_frames": T,
    }
    if out_json is not None:
        import os
        os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
    return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_qc.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/qc.py third_party/jarvis_jax/tests/test_qc.py
git commit -m "feat(cse): Phase 7 QC module (per-cam reproj + LOO all-50-kp + silhouette IoU + bundled json)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Reprojection overlay video (`cse/reproj_video.py`)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/reproj_video.py`
- Test: `third_party/jarvis_jax/tests/test_reproj_video.py`

**Interfaces:**
- Consumes: `ReprojectionTool.reproject_point`; `imageio.get_writer` (streaming). Reads raw `Frame_*.jpg` via `matplotlib.image.imread` (mirrors `silhouette_render_demo.py:130`).
- Produces:
  - `draw_overlay_frame(raw_rgb(H,W,3) uint8|float, mesh2d(K,2), kp2d(n_kp,2), *, contour=None) -> np.ndarray(H,W,3) uint8` — rasterize the raw frame with cyan mesh dots + magenta kp dots (+ optional yellow SAM contour) via an Agg matplotlib figure, return a uint8 RGB array.
  - `write_camera_video(out_path, *, frames_rgb_iter, mesh2d_by_frame, kp2d_by_frame, contour_by_frame=None, fps=30) -> str` — stream: for each frame pull the raw RGB from `frames_rgb_iter`, draw, `append_data`, one in memory at a time; return `out_path`.

- [ ] **Step 1: Write the failing tests (tiny synthetic frames, CPU)**

```python
# tests/test_reproj_video.py
import os
import numpy as np
import pytest


def test_draw_overlay_frame_returns_uint8_same_size():
    from jarvis_jax.cse.reproj_video import draw_overlay_frame
    raw = np.zeros((32, 40, 3), np.uint8)
    mesh2d = np.array([[5.0, 5.0], [10.0, 8.0], [20.0, 15.0]])
    kp2d = np.array([[6.0, 6.0], [25.0, 20.0]])
    out = draw_overlay_frame(raw, mesh2d, kp2d)
    assert out.dtype == np.uint8
    assert out.shape == (32, 40, 3)
    # something was drawn (not all-black anymore)
    assert out.max() > 0


def test_write_camera_video_streams_and_counts(tmp_path):
    import imageio
    from jarvis_jax.cse.reproj_video import write_camera_video
    T = 2
    frames = [np.zeros((32, 40, 3), np.uint8) for _ in range(T)]
    mesh2d = [np.array([[5.0, 5.0], [10.0, 8.0]]) for _ in range(T)]
    kp2d = [np.array([[6.0, 6.0]]) for _ in range(T)]
    out = str(tmp_path / "cam0.mp4")
    p = write_camera_video(out, frames_rgb_iter=iter(frames),
                           mesh2d_by_frame=mesh2d, kp2d_by_frame=kp2d, fps=10)
    assert p == out
    assert os.path.exists(out) and os.path.getsize(out) > 0
    rdr = imageio.get_reader(out)
    n = rdr.count_frames() if hasattr(rdr, "count_frames") else sum(1 for _ in rdr)
    rdr.close()
    assert n == T
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_reproj_video.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.reproj_video'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/reproj_video.py
"""Phase 7 / component 8: per-camera reprojection overlay video.

Streams each camera's raw Frame_*.jpg through imageio.get_writer -- ONE decoded
frame in memory at a time (mirrors stac-mjx/stac_mjx/stac.py:735,745) -- drawing
the projected posed mesh (subset) + keypoints (+ optional SAM mask contour).
Never buffers all frames.
"""
from __future__ import annotations
import numpy as np


def draw_overlay_frame(raw_rgb, mesh2d, kp2d, *, contour=None):
    """Rasterize raw_rgb with mesh (cyan) + kp (magenta) [+ contour (yellow)]
    overlays to a uint8 RGB array via an Agg matplotlib figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = np.asarray(raw_rgb)
    if raw.dtype != np.uint8:
        raw = np.clip(raw * (255.0 if raw.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    H, W = raw.shape[:2]
    dpi = 100.0
    fig = plt.figure(figsize=(W / dpi, H / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(raw)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
    if contour is not None and len(contour):
        cc = np.asarray(contour)
        ax.plot(cc[:, 0], cc[:, 1], c="yellow", lw=0.8)
    if len(mesh2d):
        m = np.asarray(mesh2d)
        ax.scatter(m[:, 0], m[:, 1], s=2, c="cyan", linewidths=0)
    if len(kp2d):
        k = np.asarray(kp2d)
        ax.scatter(k[:, 0], k[:, 1], s=10, c="magenta", linewidths=0)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    # resize-safe: crop/pad to (H,W) if the Agg canvas rounded differently
    bh, bw = buf.shape[:2]
    if (bh, bw) != (H, W):
        out = np.zeros((H, W, 3), np.uint8)
        out[:min(H, bh), :min(W, bw)] = buf[:min(H, bh), :min(W, bw)]
        buf = out
    return buf.astype(np.uint8)


def write_camera_video(out_path, *, frames_rgb_iter, mesh2d_by_frame,
                       kp2d_by_frame, contour_by_frame=None, fps=30):
    """Stream overlay frames to an mp4 via imageio (one frame in memory)."""
    import os
    import imageio
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with imageio.get_writer(out_path, fps=fps) as video:
        for t, raw in enumerate(frames_rgb_iter):
            contour = None if contour_by_frame is None else contour_by_frame[t]
            frame = draw_overlay_frame(raw, mesh2d_by_frame[t], kp2d_by_frame[t],
                                       contour=contour)
            video.append_data(frame)
    return out_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_reproj_video.py -q`
Expected: PASS (2 passed). If imageio's ffmpeg mp4 plugin is unavailable in the test env, the writer raises on `.mp4`; the test will surface that — the coordinator has ffmpeg on the compute node. (If the CI env lacks ffmpeg, temporarily assert on a `.gif` path instead; note this in the commit message. Do NOT weaken the streaming contract.)

- [ ] **Step 5: Commit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/reproj_video.py third_party/jarvis_jax/tests/test_reproj_video.py
git commit -m "feat(cse): Phase 7 reprojection overlay video (streaming imageio, one frame in memory)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Deploy/QC driver (`cse/run_outputs_qc.py`)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/run_outputs_qc.py`
- Test: `third_party/jarvis_jax/tests/test_run_outputs_qc.py`

**Interfaces:**
- Consumes: Task 1 (`outputs.build_fly_outputs`), Task 2 (`qc.qc_report`), Task 3 (`reproj_video.write_camera_video`); the mm-bridge helpers from `silhouette_ik_solve` (`build_solver_inputs`, `_umeyama`, `_triangulate_kp_mm`, `_cam2img_for_frame`, `_ann_for_image`, `_load_sam_mask`); `ReprojectionTool`; `stac_mjx.io_dict_to_hdf5.load` (marker_sites); `h5py` (bout fs_imgids).
- Produces:
  - `resolve_calib_dir(recording, *, root, cse_work_dir) -> str` — refined `cse_work/calib_refined/<rec>` if present else factory `<root>/calib_params/<rec>`; raises if neither exists.
  - `compute_bridges(recording, *, ik_h5, model_xml, root, split, calib_dir, fs_imgids, T, ann_id_by_image=None) -> list[(s,R,t)|None]` — per-frame Umeyama bridges (marker_sites→triangulated coco kp), exactly as `run_single_fly`.
  - `gather_qc_frame_inputs(recording, *, root, split, calib_dir, fs_imgids, T, ann_id_by_image=None) -> dict{kp2d_by_frame, vis_by_frame, masks_by_frame}` — per-frame {cam_idx: (n_kp,2)} annotated 2-D, {cam_idx:(n_kp,) bool} visibility, {cam_idx: mask} SAM masks (using the coco keypoint order aligned to the ik kp_names).
  - `run_outputs_qc(recording, *, ik_h5, model_xml, mesh_npz, root, split="val", qpos_npz=None, calib_dir=None, cse_work_dir, out_dir, mesh_subset="fps_500", max_frames=0, make_video=False, video_fps=30, ann_id_by_image=None) -> dict` — the deploy entrypoint; writes `{out_dir}/{recording}_outputs.h5` + `{out_dir}/{recording}_qc.json` (+ optional per-cam mp4s), returns a schema dict.
  - `main()` — argparse CLI.

- [ ] **Step 1: Write the failing tests (importability/CLI + calib resolution, CPU)**

```python
# tests/test_run_outputs_qc.py
import os
import numpy as np
import pytest


def test_module_importable_and_has_cli():
    import jarvis_jax.cse.run_outputs_qc as m
    assert hasattr(m, "run_outputs_qc") and callable(m.run_outputs_qc)
    assert hasattr(m, "main") and callable(m.main)
    assert hasattr(m, "resolve_calib_dir") and callable(m.resolve_calib_dir)


def test_resolve_calib_dir_prefers_refined(tmp_path):
    from jarvis_jax.cse.run_outputs_qc import resolve_calib_dir
    rec = "2026_03_18_15_31_22"
    root = tmp_path / "root"
    cse = tmp_path / "cse_work"
    refined = cse / "calib_refined" / rec
    factory = root / "calib_params" / rec
    factory.mkdir(parents=True)
    (factory / "Cam0.yaml").write_text("x")
    # no refined yet -> factory
    assert resolve_calib_dir(rec, root=str(root), cse_work_dir=str(cse)) == str(factory)
    # refined present -> refined wins
    refined.mkdir(parents=True)
    (refined / "Cam0.yaml").write_text("x")
    assert resolve_calib_dir(rec, root=str(root), cse_work_dir=str(cse)) == str(refined)


def test_resolve_calib_dir_raises_when_missing(tmp_path):
    from jarvis_jax.cse.run_outputs_qc import resolve_calib_dir
    with pytest.raises(FileNotFoundError):
        resolve_calib_dir("nope", root=str(tmp_path / "r"), cse_work_dir=str(tmp_path / "c"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_outputs_qc.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'jarvis_jax.cse.run_outputs_qc'`

- [ ] **Step 3: Write minimal implementation**

```python
# jarvis_jax/cse/run_outputs_qc.py
"""Phase 7 / component 8 DEPLOY driver: wrap a scenario's solved qpos into
outputs (Task 1) + QC report (Task 2) + optional overlay videos (Task 3).

Consumes scenario qpos (single/multi/amputation/headless) -- NEVER changes any
solver. Resolves calibration PER RECORDING (refined cse_work/calib_refined/<rec>
if present, else factory <root>/calib_params/<rec>): the hardcoded single-recording
_DEFAULT_*_CALIB_DIR in silhouette_ik_solve is a deploy blocker and is not used
here.
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np


def resolve_calib_dir(recording, *, root, cse_work_dir):
    """Refined calib if present, else factory; raise if neither exists."""
    refined = os.path.join(cse_work_dir, "calib_refined", recording)
    factory = os.path.join(root, "calib_params", recording)
    if os.path.isdir(refined):
        return refined
    if os.path.isdir(factory):
        return factory
    raise FileNotFoundError(
        f"no calib for {recording}: neither {refined} nor {factory} exists")


def _load_coco(root, split):
    coco = json.load(open(os.path.join(root, "annotations", f"instances_{split}.json")))
    id2file = {im["id"]: im["file_name"] for im in coco["images"]}
    id2ann_multi = {}
    for an in coco["annotations"]:
        id2ann_multi.setdefault(an["image_id"], []).append(an)
    return coco, id2file, id2ann_multi


def compute_bridges(recording, *, ik_h5, model_xml, root, split, calib_dir,
                    fs_imgids, T, ann_id_by_image=None):
    """Per-frame model->mm Umeyama bridges, exactly as run_single_fly."""
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik_solve import (
        build_solver_inputs, _umeyama, _triangulate_kp_mm, _cam2img_for_frame)

    inputs = build_solver_inputs(ik_h5, model_xml)
    kp_names = list(inputs["kp_names"])
    marker_sites = np.asarray(ioh5.load(ik_h5)["marker_sites"])[:T]

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco, id2file, id2ann_multi = _load_coco(root, split)
    coco_kpnames = coco["keypoint_names"]

    bridges = []
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp_mm, kok = _triangulate_kp_mm(rt, kp_names, coco_kpnames, cam2img,
                                        id2ann_multi, ann_id_by_image)
        if kok.sum() < 3:
            bridges.append(None)
            continue
        s, R, tr = _umeyama(marker_sites[t][kok], kp_mm[kok])
        bridges.append((s, R, tr))
    return bridges


def gather_qc_frame_inputs(recording, *, root, split, calib_dir, fs_imgids, T,
                           ik_kpnames, ann_id_by_image=None):
    """Per-frame QC inputs aligned to ik_kpnames order: kp2d/vis/masks by cam."""
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik_solve import (
        _cam2img_for_frame, _ann_for_image, _load_sam_mask)

    rt = ReprojectionTool(calib_dir)
    cam_names = list(rt.cameras.keys())
    coco, id2file, id2ann_multi = _load_coco(root, split)
    coco_kpnames = coco["keypoint_names"]
    name2coco = {n: i for i, n in enumerate(coco_kpnames)}
    # map each ik keypoint slot -> its coco index (or -1 if absent)
    coco_idx = np.array([name2coco.get(n, -1) for n in ik_kpnames], int)
    n_kp = len(ik_kpnames)

    kp2d_by_frame, vis_by_frame, masks_by_frame = [], [], []
    for t in range(T):
        cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
        kp2d, vis, masks = {}, {}, {}
        for c, iid in cam2img.items():
            ann = _ann_for_image(id2ann_multi, iid, ann_id_by_image)
            if ann is None:
                continue
            kp = np.asarray(ann["keypoints"], float).reshape(-1, 3)
            uv = np.zeros((n_kp, 2)); vv = np.zeros(n_kp, bool)
            for j, ci in enumerate(coco_idx):
                if ci >= 0 and kp[ci, 2] > 0:
                    uv[j] = kp[ci, :2]; vv[j] = True
            kp2d[c] = uv; vis[c] = vv
            m = _load_sam_mask(root, split, id2file[iid], ann["id"])
            if m is not None:
                masks[c] = m
        kp2d_by_frame.append(kp2d); vis_by_frame.append(vis); masks_by_frame.append(masks)
    return {"kp2d_by_frame": kp2d_by_frame, "vis_by_frame": vis_by_frame,
            "masks_by_frame": masks_by_frame}


def run_outputs_qc(recording, *, ik_h5, model_xml, mesh_npz, root, split="val",
                   qpos_npz=None, calib_dir=None, cse_work_dir, out_dir,
                   mesh_subset="fps_500", max_frames=0, make_video=False,
                   video_fps=30, ann_id_by_image=None):
    """Deploy entrypoint: outputs h5 + QC json (+ optional per-cam mp4s)."""
    import h5py
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
    from jarvis_jax.cse import outputs as outmod
    from jarvis_jax.cse import qc as qcmod
    from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs

    if calib_dir is None:
        calib_dir = resolve_calib_dir(recording, root=root, cse_work_dir=cse_work_dir)

    # qpos: from the passed npz else the ik h5's stored trajectory.
    if qpos_npz is not None:
        qpos = np.asarray(np.load(qpos_npz)["qpos"], np.float32)
    else:
        qpos = np.asarray(ioh5.load(ik_h5)["qpos"], np.float32)
    T_full = qpos.shape[0]
    T = T_full if max_frames <= 0 else min(max_frames, T_full)
    qpos = qpos[:T]

    bout_h5 = os.path.join(os.path.dirname(os.path.dirname(ik_h5)), f"{recording}_bout.h5")
    with h5py.File(bout_h5, "r") as f:
        fs_imgids = f["fs_imgids"][()][:T]

    inputs = build_solver_inputs(ik_h5, model_xml)
    ik_kpnames = list(inputs["kp_names"])

    bridges = compute_bridges(
        recording, ik_h5=ik_h5, model_xml=model_xml, root=root, split=split,
        calib_dir=calib_dir, fs_imgids=fs_imgids, T=T, ann_id_by_image=ann_id_by_image)

    os.makedirs(out_dir, exist_ok=True)
    out_h5 = os.path.join(out_dir, f"{recording}_outputs.h5")
    schema = outmod.build_fly_outputs(
        recording, ik_h5=ik_h5, model_xml=model_xml, mesh_npz=mesh_npz,
        qpos=qpos, bridges=bridges, out_path=out_h5, mesh_subset=mesh_subset)

    # QC: read back the FK'd mesh/kp from the h5 (world mm) for the report.
    d = ioh5.load(out_h5)
    mesh_mm = np.asarray(d["mesh_mm"]); kp3d_mm = np.asarray(d["kp3d_mm"])
    qi = gather_qc_frame_inputs(
        recording, root=root, split=split, calib_dir=calib_dir,
        fs_imgids=fs_imgids, T=T, ik_kpnames=ik_kpnames,
        ann_id_by_image=ann_id_by_image)

    rt = ReprojectionTool(calib_dir)
    out_json = os.path.join(out_dir, f"{recording}_qc.json")
    report = qcmod.qc_report(
        rt, kp3d_by_frame=[kp3d_mm[t] for t in range(T)],
        mesh_by_frame=[mesh_mm[t] for t in range(T)],
        kp2d_by_frame=qi["kp2d_by_frame"], vis_by_frame=qi["vis_by_frame"],
        masks_by_frame=qi["masks_by_frame"], out_json=out_json)

    videos = []
    if make_video:
        from jarvis_jax.cse.reproj_video import write_camera_video
        import matplotlib.image as mpimg
        cam_names = list(rt.cameras.keys())
        _coco, id2file, _m = _load_coco(root, split)
        from jarvis_jax.cse.silhouette_ik_solve import _cam2img_for_frame
        for c, cam in enumerate(cam_names):
            # per-frame raw path + projected mesh/kp for this camera
            def _frames():
                for t in range(T):
                    cam2img = _cam2img_for_frame(fs_imgids[t], id2file, cam_names)
                    iid = cam2img.get(c)
                    if iid is None:
                        yield np.zeros((8, 8, 3), np.uint8)
                    else:
                        yield mpimg.imread(os.path.join(root, split, id2file[iid]))
            mesh2d = [np.stack([rt.reproject_point(mesh_mm[t][k])[c]
                                for k in range(mesh_mm.shape[1])], 0)
                      if not np.isnan(mesh_mm[t]).all() else np.zeros((0, 2))
                      for t in range(T)]
            kp2d = [np.stack([rt.reproject_point(kp3d_mm[t][j])[c]
                              for j in range(kp3d_mm.shape[1])], 0)
                    if not np.isnan(kp3d_mm[t]).all() else np.zeros((0, 2))
                    for t in range(T)]
            vpath = os.path.join(out_dir, f"{recording}_{cam}_reproj.mp4")
            write_camera_video(vpath, frames_rgb_iter=_frames(),
                               mesh2d_by_frame=mesh2d, kp2d_by_frame=kp2d, fps=video_fps)
            videos.append(vpath)

    return {"out_h5": out_h5, "qc_json": out_json, "calib_dir": calib_dir,
            "videos": videos, "n_frames": T, "qc": report, "schema": schema}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recording", required=True)
    ap.add_argument("--ik-h5", required=True)
    ap.add_argument("--xml", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--qpos-npz", default=None)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--cse-work-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--mesh-subset", default="fps_500")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--make-video", action="store_true")
    ap.add_argument("--video-fps", type=int, default=30)
    a = ap.parse_args()
    rep = run_outputs_qc(
        a.recording, ik_h5=a.ik_h5, model_xml=a.xml, mesh_npz=a.mesh, root=a.root,
        split=a.split, qpos_npz=a.qpos_npz, calib_dir=a.calib_dir,
        cse_work_dir=a.cse_work_dir, out_dir=a.out_dir, mesh_subset=a.mesh_subset,
        max_frames=a.max_frames, make_video=a.make_video, video_fps=a.video_fps)
    print("PHASE7 OUTPUTS+QC REPORT")
    print(f"  out_h5: {rep['out_h5']}")
    print(f"  qc_json: {rep['qc_json']}")
    print(f"  calib_dir: {rep['calib_dir']}")
    print(f"  n_frames: {rep['n_frames']}")
    print(f"  qc: {json.dumps(rep['qc'], indent=2)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_outputs_qc.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Add the skipif-gated real-data smoke test**

Append to `tests/test_run_outputs_qc.py`:

```python
IK = "/gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5"
XML = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml"
MESH = "/gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz"
ROOT = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
CSE = "/gscratch/portia/eabe/data/Johnson_lab/cse_work"


@pytest.mark.skipif(not os.path.exists(IK), reason="STAC ik h5 not present (GPU-only heavy run)")
def test_run_outputs_qc_smoke_keys(tmp_path):
    """Tiny 2-frame real-data smoke (coordinator GPU): asserts the output h5
    keys + QC json keys exist. NOT a scientific magnitude check."""
    import stac_mjx.io_dict_to_hdf5 as ioh5
    from jarvis_jax.cse.run_outputs_qc import run_outputs_qc
    rep = run_outputs_qc(
        "2026_03_18_15_31_22", ik_h5=IK, model_xml=XML, mesh_npz=MESH, root=ROOT,
        split="val", cse_work_dir=CSE, out_dir=str(tmp_path), mesh_subset="fps_500",
        max_frames=2, make_video=False)
    assert os.path.exists(rep["out_h5"]) and os.path.exists(rep["qc_json"])
    d = ioh5.load(rep["out_h5"])
    for k in ("qpos", "root_se3", "scale", "mesh_mm", "kp3d_mm",
              "mesh_vert_idx", "mesh_subset", "kp_names"):
        assert k in d
    assert np.asarray(d["mesh_mm"]).shape[0] == 2
    for k in ("per_camera_reproj_px", "loo_reproj_px", "silhouette_iou", "n_frames"):
        assert k in rep["qc"]
    assert rep["qc"]["n_frames"] == 2
```

- [ ] **Step 6: Run the CPU (non-skipped) tests again to confirm no regression**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_run_outputs_qc.py -q`
Expected: PASS (3 passed, 1 skipped) when the ik h5 is absent on the CPU box. The skipped smoke runs GPU-side (Step 7).

- [ ] **Step 7: Coordinator GPU verification of the smoke (NOT pytest-in-subagent)**

Run (coordinator, on gpu-l40s node):
```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 \
  python -m pytest tests/test_run_outputs_qc.py::test_run_outputs_qc_smoke_keys -q
```
Expected: `1 passed` (writes into a pytest tmp_path; nothing to clean up).

- [ ] **Step 8: Commit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/run_outputs_qc.py third_party/jarvis_jax/tests/test_run_outputs_qc.py
git commit -m "feat(cse): Phase 7 deploy/QC driver (per-recording calib; outputs+QC+videos over scenario qpos)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Coordinator deploy across the 4 scenarios (GPU; NOT pytest)

**Files:** none (operational). This task is the heavy deploy the coordinator runs on one representative recording per scenario; subagents must NOT run these (they orphan on long GPU jobs).

**Interfaces:**
- Consumes: `run_outputs_qc.run_outputs_qc` / its CLI `python -m jarvis_jax.cse.run_outputs_qc`.
- Produces: per scenario, `{out_dir}/{rec}_outputs.h5`, `{out_dir}/{rec}_qc.json`, and (with `--make-video`) `{out_dir}/{rec}_{cam}_reproj.mp4`.

- [ ] **Step 1: Discover a representative recording per scenario (coordinator)**

For each scenario, confirm the recording has an ik h5 + bout h5 + coco anns + SAM masks present. Candidates from the brief:
- single: `2026_03_18_15_31_22` under `red_data/red_data_unified_V3` (split `val`), `cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5`.
- multi (courtship): a recording under `merge_courtship_V3` / `courtship` with per-fly qpos at `.../fly{0,1}/{rec}_qpos.npz` (run `run_multifly_ik` first if absent).
- amputation: a recording under `amputation` (split typically `train`); qpos from `run_active_parts_ik`.
- headless: a recording under `BDN2_headless` (split typically `train`); qpos from `run_active_parts_ik`.

Run (coordinator) to sanity-check presence, e.g.:
```bash
ls -1 /gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5 \
      /gscratch/portia/eabe/data/Johnson_lab/cse_work/2026_03_18_15_31_22_bout.h5
```
Expected: both paths listed (no "No such file").

- [ ] **Step 2: Deploy — single (male)**

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
REC=2026_03_18_15_31_22
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m jarvis_jax.cse.run_outputs_qc \
  --recording $REC \
  --ik-h5 /gscratch/portia/eabe/data/Johnson_lab/cse_work/$REC/Fruitfly_ik_v1_cse.h5 \
  --qpos-npz /gscratch/portia/eabe/data/Johnson_lab/cse_work/$REC/${REC}_qpos.npz \
  --xml /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml \
  --mesh /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz \
  --root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3 \
  --split val \
  --cse-work-dir /gscratch/portia/eabe/data/Johnson_lab/cse_work \
  --out-dir /gscratch/portia/eabe/data/Johnson_lab/cse_work/phase7/$REC \
  --mesh-subset fps_500 --make-video --video-fps 30
```
Expected: prints `PHASE7 OUTPUTS+QC REPORT` with `out_h5`, `qc_json`, and a QC block; `calib_dir` should be the refined dir if it exists for this recording, else factory. If `${REC}_qpos.npz` is absent, drop `--qpos-npz` (falls back to the ik h5's stored qpos).
HONEST ACCEPTANCE: `per_camera_reproj_px.median` within the expected ~3-5 px band (spec §2: predicted kp reproject ~3.5 px); `loo_reproj_px.median` reported (wing-tip LOO ~6 px in spec §2, all-kp LOO expected in a similar single-digit band; report the number, do not gate hard); `silhouette_iou.hard_median`/`soft_median` reported (collision-mesh cap ~0.76 per spec §12 — a lower value is acceptable and reported truthfully). One mp4 per camera exists and is non-empty.

- [ ] **Step 3: Deploy — multi (courtship), per fly**

Ensure per-fly qpos exists (run `run_multifly_ik` first if needed). Then for each fly:
```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
REC=<courtship_rec>; COND_ROOT=/gscratch/portia/eabe/data/Johnson_lab/merge_courtship_V3
for FID in 0 1; do
  XLA_PYTHON_CLIENT_PREALLOCATE=false python -m jarvis_jax.cse.run_outputs_qc \
    --recording $REC \
    --ik-h5 $COND_ROOT/cse_work/$REC/fly$FID/Fruitfly_ik_v1_cse.h5 \
    --qpos-npz $COND_ROOT/cse_work/$REC/fly$FID/${REC}_qpos.npz \
    --xml /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml \
    --mesh /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz \
    --root $COND_ROOT --split val \
    --cse-work-dir $COND_ROOT/cse_work \
    --out-dir $COND_ROOT/cse_work/phase7/$REC/fly$FID \
    --mesh-subset fps_500 --make-video
done
```
Expected: two per-fly output h5 + qc json. NOTE: the multi-fly `ann_id_by_image` per-identity selection is needed for the bridge/QC to pick the correct fly on 2-fly images. If the multifly driver persisted a per-fly `ann_id_by_image` map, the coordinator must extend the CLI to pass it (see Open Risk R2); otherwise the QC for the female may mix annotations. HONEST ACCEPTANCE: male reprojects in the single-fly band; the female's per-camera reproj may be worse (keypoint-hard case per spec §2/§4) — report both, do not gate the female hard; her silhouette IoU is the meaningful metric.

- [ ] **Step 4: Deploy — amputation**

```bash
source ~/.bashrc && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/jarvis_jax
REC=<amputation_rec>; COND_ROOT=/gscratch/portia/eabe/data/Johnson_lab/amputation
XLA_PYTHON_CLIENT_PREALLOCATE=false python -m jarvis_jax.cse.run_outputs_qc \
  --recording $REC \
  --ik-h5 $COND_ROOT/cse_work/$REC/Fruitfly_ik_v1_cse.h5 \
  --qpos-npz $COND_ROOT/cse_work/$REC/${REC}_qpos.npz \
  --xml /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_v1/fruitfly_v1_free.xml \
  --mesh /gscratch/portia/eabe/Research/MyRepos/fruitfly_body_models/fruitfly_cse/fly_v1_collision_canonical_wings.npz \
  --root $COND_ROOT --split train \
  --cse-work-dir $COND_ROOT/cse_work \
  --out-dir $COND_ROOT/cse_work/phase7/$REC \
  --mesh-subset fps_500 --make-video
```
Expected: output h5 + qc json. The off-leg keypoints are absent from the coco schema, so `gather_qc_frame_inputs` naturally marks them not-visible and they drop out of the reproj/LOO metrics (no phantom residual). HONEST ACCEPTANCE: present-part reproj in the single-fly band; report which parts were off (visible via the absent kp names). Silhouette IoU reported; the amputated geom is still in the collision mesh subset, so IoU may be slightly depressed — report truthfully.

- [ ] **Step 5: Deploy — headless**

Same as Step 4 with `COND_ROOT=/gscratch/portia/eabe/data/Johnson_lab/BDN2_headless` and `--split train`. HONEST ACCEPTANCE: body/leg reproj in-band; head keypoints absent from schema so they drop out; silhouette IoU reported (head still in the mesh subset — expect a small IoU depression around the head; report truthfully).

- [ ] **Step 6: Summarize the 4-scenario QC (coordinator)**

Collect the four `*_qc.json` files and produce a short table (median per-camera reproj px, LOO reproj px, hard/soft silhouette IoU, n_frames) per scenario. Confirm each output h5 has the 8 datasets (`qpos, root_se3, scale, mesh_mm, kp3d_mm, mesh_vert_idx, mesh_subset, kp_names`) and each requested mp4 is non-empty. Do NOT commit any of these artifacts (Global Constraints).

---

## Self-Review

### Spec §8/§9 coverage table

| Spec item (§8/§9 + §11 Phase 7) | Task |
|---|---|
| Multi-view fusion at triangulation is the reporting basis (§8) | Tasks 2, 4 use `rt.reconstruct_point`/`reproject_point` (DLT) consistently for QC |
| Per fly per frame: qpos + root (model frame + scale) (§9) | Task 1 `build_fly_outputs` (`qpos`, `root_se3=qpos[:, :7]`, `scale` from bridge) |
| Posed mesh vertices (world mm) (§9) | Task 1 `fk_mesh_world_mm` + `mesh_mm` dataset (subset default, full option) |
| 3-D keypoints (world mm) (§9) | Task 1 `fk_sites_world_mm` + `kp3d_mm` dataset |
| Reprojection overlay videos on every camera (§9) | Task 3 `write_camera_video` + Task 4 `--make-video` (one mp4/cam) |
| QC: per-camera reprojection error (§9) | Task 2 `per_camera_reproj_error` (all 50 kp) |
| QC: silhouette IoU (§9) | Task 2 `silhouette_iou_report` (Phase-6 hard+soft helpers) |
| QC: leave-one-out reprojection (§9) | Task 2 `loo_reproj` (generalized to all 50 kp) |
| Combined into one QC artifact for outlier flagging (§9) | Task 2 `qc_report` -> one json (+ per_kp array kept for flagging) |
| Deploy across 4 scenarios (§11 Phase 7) | Task 5 (single/multi/amputation/headless) |
| Consume scenario solvers' qpos, do not change them | Task 4 loads `{rec}_qpos.npz` / ik h5 qpos; no solver edits |
| Per-recording calibration (Global Constraint) | Task 4 `resolve_calib_dir` (refined else factory; raises otherwise) |
| Efficiency: batched FK; streaming video; mesh size (§8 mandate) | Task 1 vmap FK + fps_500 default; Task 3 streaming imageio |

Non-goals correctly excluded: V2/V2.1 anatomy port (§10 / brief NON-GOALS) — not touched; no stac-mjx edits; no scenario-solver edits.

### Placeholder scan

No `TODO`/`TBD`/"implement later"/"similar to Task N"/"add error handling" placeholders. Every code step contains complete runnable code; every run step gives the exact command + expected output. The only conditional-language items are HONEST ACCEPTANCE bands in Task 5 (operational thresholds the coordinator reports, not code) and the two flagged Open Risks below.

### Type consistency across tasks

- Bridges are a `list[tuple(s: float, R: (3,3), t: (3,)) | None]` of length T everywhere: produced by `run_outputs_qc.compute_bridges`, consumed by `outputs.fk_mesh_world_mm` / `fk_sites_world_mm` / `build_fly_outputs`. Consistent.
- `mesh_subset` is a `str` ("fps_500" default / "full" / "fps_<N>") in `outputs.mesh_subset_indices`, `build_fly_outputs`, `write_outputs_h5`, and the CLI `--mesh-subset`. Consistent.
- QC frame inputs: `kp2d_by_cam: {cam_idx:int -> (n_kp,2)}`, `vis_by_cam: {cam_idx -> (n_kp,) bool}`, `masks_by_cam: {cam_idx -> (H,W) bool}` — produced by `gather_qc_frame_inputs`, consumed by `qc.per_camera_reproj_error`/`loo_reproj`/`silhouette_iou_report`/`qc_report`. `n_kp` is aligned to `ik_kpnames` order in both producer and the FK'd `kp3d_mm` (both from `build_solver_inputs`'s `kp_names`/`site_idxs`), so the j-th kp3d matches the j-th kp2d slot. Consistent.
- h5 dataset names match between `write_outputs_h5` and the Task 4 read-back + the smoke test asserts (`qpos, root_se3, scale, mesh_mm, kp3d_mm, mesh_vert_idx, mesh_subset, kp_names`). Consistent.
- `qc_report` return keys (`per_camera_reproj_px, loo_reproj_px, silhouette_iou, n_frames`) match Task 2 test asserts and Task 4 smoke asserts. Consistent.

### Flagged spec items with no code / efficiency tradeoffs / open risks

- **mesh_mm size (efficiency, baked in):** default `fps_500` float32 subset (~6 KB/frame) vs. full mesh (~0.74 MB/frame, ~22 GB for 30k frames). Full is opt-in (`mesh_subset="full"`) and documented as short-clips-only in the `outputs.py` module docstring.
- **Video streaming (efficiency, baked in):** Task 3 streams one frame at a time via `imageio.get_writer`/`append_data`; no all-frames buffer. Test `test_write_camera_video_streams_and_counts` passes an `iter(...)` to enforce single-pass consumption.
- **R1 — ffmpeg availability:** `write_camera_video` writes `.mp4` via imageio's ffmpeg plugin. If the CPU test env lacks ffmpeg, Task 3 Step 4 notes a `.gif` fallback for the test only (compute node has ffmpeg for the real deploy). Coordinator to confirm ffmpeg on the gpu-l40s node before Task 5 `--make-video` runs.
- **R2 — multi-fly per-identity ann selection (OPEN, coordinator to resolve before Task 5 Step 3):** `run_outputs_qc` accepts `ann_id_by_image` but the CLI does not yet load a persisted per-fly map. For courtship, the female's bridge + QC must use her `ann_id_by_image` (as `run_multifly_ik` does via `ann_id_by_image_for_fly`), else the female's QC may pull the male's annotations on 2-fly images. RESOLUTION OPTIONS: (a) have the multifly driver persist each fly's `ann_id_by_image` json next to its qpos and add a `--ann-id-map <json>` CLI arg to `run_outputs_qc` that loads it; or (b) reconstruct it in the driver via `run_multifly_ik.ann_id_by_image_for_fly(identity_map, coco_path, recording, fid)`. This is a small, additive change (does not alter any solver); flagged so the coordinator wires it before the multi-fly deploy rather than silently reporting mixed-identity female QC.
- **R3 — root frame semantics:** `root_se3 = qpos[:, :7]` is the MODEL-frame free-joint (xyz + wxyz quat) as stored by the solver, NOT world-mm; the world-mm root is recoverable via the per-frame `scale`+bridge if needed. Spec §9 says "root (model frame, + scale)", which matches this choice. Documented in `build_fly_outputs`. No action needed; noted for the coordinator's downstream consumers.
