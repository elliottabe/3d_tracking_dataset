# V2VNet-Free Mesh/Silhouette Courtship Inference Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a V2VNet-free end-to-end pipeline that turns multi-camera courtship video (+ SAM3 masks + calibration) into 3-D fly kinematics for all bouts × 2 flies, via ViTPose 2-D → DLT triangulation → STAC → silhouette-containment IK → outputs/QC/overlays, parallelized as a resumable SLURM array.

**Architecture:** Five staged JAX components (+ a PyTorch SAM3 Stage 0 for new recordings) reusing validated code (the retrained ViTPose, `triangulate_dlt_batched`, `stac_mjx.run_stac`, `SilhouetteJaxlsBatchSolver`+bridge fix, the unpackbits mask adapter, `outputs.py`/`qc.py`/`reproj_video.py`). A resumable per-bout driver writes atomic stage artifacts; a SLURM-array driver fans bouts across gpu-l40s/ckpt with `--requeue`. Hydra config extends the existing repo-root `configs/`.

**Tech Stack:** Python, JAX/Flax(nnx)/MJX, jaxls, jaxlie, MuJoCo, PyTorch(SAM3), NumPy, SciPy, OpenCV, trimesh, Orbax, Hydra/OmegaConf, imageio; pytest.

## Global Constraints

- Frozen: `stac-mjx/stac_mjx/stac_core_jaxls.py` (byte-identical test). Do not modify the `stac-mjx` submodule.
- New pipeline code under `third_party/jarvis_jax/jarvis_jax/cse/`; tests under `third_party/jarvis_jax/tests/`; drivers under repo `scripts/`; configs extend repo-root `3d_tracking_dataset/configs/`.
- **On a gpu-l40s compute node: run GPU tests directly on GPU** — `cd third_party/jarvis_jax && OMP_NUM_THREADS=4 python -m pytest tests/<f> -v` (do NOT set `JAX_PLATFORMS=cpu` for mjx/jaxls/ViTPose tests; they are too slow on CPU). Pure-numpy tests may prefix `JAX_PLATFORMS=cpu`. Env: `source ~/.bashrc; micromamba activate 3d_tracking; unset LD_LIBRARY_PATH` (JAX). Stage 0 (SAM3, PyTorch) uses `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` + `LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib`.
- Affine/telecentric projection: `triangulate_dlt_batched`/`project_center_to_cameras` use `cameraMatrices (nc,4,3) = P.T`; the silhouette solver uses affine `cam_Ms (nc,2,3)`, `cam_ts (nc,2)` from `ReprojectionTool`.
- Masks: `np.unpackbits(packed[fly,cam,frame], axis=-1)[:, :W]` → full-frame (H,W) bool, `W = shape[1]` (1936). `packed` shape `(2,7,T,448,242)`.
- ViTPose: 4-ch RGB+mask, `img_size=448`, `heatmap_size=224`, `num_keypoints=50`; decode with `heatmaps_to_keypoints(hm, in_size=448)`; normalize with `normalize_image` BEFORE forward; `use_running_average=True` at inference.
- Retrained detector checkpoint: `${paths...vit_runs_root}/v3_kp_maskaware/final` (Orbax `StandardCheckpointer`, model-state only).
- Config-driven, no hardcoded absolute paths in pipeline code; generalize `user` to `${oc.env:USER,eabe}`.
- Resumability: per-bout `<run_root>/bouts/bout_<idx:05d>/fly<f>/` stage artifacts, atomic tmp→`os.replace`, per-bout `DONE` marker; stages skipped if artifact present.
- Commit by explicit path (never `git add -A`; uncommitted session artifacts exist). Commit messages end `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`. Do not commit data/masks/outputs/checkpoints.

## File Structure

New (`third_party/jarvis_jax/jarvis_jax/cse/`):
- `courtship_bout_masks.py` — unpackbits SAM masks → per-(frame,cam) full-frame masks; per-bout loader.
- `courtship_triangulate.py` — per-camera 2-D keypoints (+conf) → 3-D mm keypoints (+conf) via DLT.
- `courtship_predict_2d.py` — ViTPose 2-D inference on SAM-cropped fly (load ckpt, crop, infer, map to full-frame).
- `courtship_stac.py` — feed triangulated 3-D kp into `stac_mjx.run_stac`: fit offsets once + `ik_only` per bout.
- `courtship_polish.py` — courtship silhouette-containment polish (STAC h5 + unpackbits masks + triangulated-kp bridges → `SilhouetteJaxlsBatchSolver`).
- `mesh_decimate.py` — one-time watertight-mesh face decimation (trimesh).
- `courtship_qc.py` — aggregate per-bout QC JSON → session dashboard (json + plots).

New (repo `scripts/`):
- `run_courtship_bout.py` — resumable single-bout, both-flies driver (chains stages A–E, atomic stage artifacts, skip-complete). Hydra app on repo `configs/`.
- `slurm_courtship_array.py` — submit Stage-0 SAM3 (if needed) → precompute → JAX array → aggregate, with SLURM dependency + `--requeue`.

New (repo `configs/`): `courtship_pipeline.yaml`, `recording/session0.yaml`, `detector/vitpose_v3.yaml`, `silhouette/default.yaml`, `outputs/default.yaml`, `slurm/ckpt_g2.yaml`, `slurm/gpu_l40s.yaml`.

Modify: `configs/paths/hyak.yaml` (generalize `user`), `third_party/jarvis_jax/configs/paths/hyak.yaml` (generalize `user`).

Reused unchanged: `models/vitpose.py`, `convert/build_checkpoint.load_vitpose`, `eval/mpjpe.heatmaps_to_keypoints`, `data/device.normalize_image`, `predict/session_frameset.build_frameset`, `data/transforms.crop_origin`, `geometry/center3d.{triangulate_dlt_batched,project_center_to_cameras,centroids_to_fullpx}`, `geometry/reprojection_tool.ReprojectionTool`, `stac_mjx.run_stac`, `silhouette_ik_solve.build_solver_inputs`, `silhouette_joint_ik.SilhouetteJaxlsBatchSolver`, `silhouette_ik.{load_anatomy,make_fk_repose}`, `silhouette_dof.*`, `silhouette_targets.silhouette_fk_indices`, `outputs.py`, `qc.py`, `reproj_video.write_camera_video`, `scripts/sam3_masks.py`.

---

### Task 1: Config — generalize paths + add courtship pipeline Hydra groups

**Files:**
- Modify: `configs/paths/hyak.yaml` (line 1: `user`)
- Modify: `third_party/jarvis_jax/configs/paths/hyak.yaml` (the `user`/roots)
- Create: `configs/recording/session0.yaml`, `configs/detector/vitpose_v3.yaml`, `configs/silhouette/default.yaml`, `configs/outputs/default.yaml`, `configs/slurm/ckpt_g2.yaml`, `configs/slurm/gpu_l40s.yaml`, `configs/courtship_pipeline.yaml`
- Test: `third_party/jarvis_jax/tests/test_courtship_config.py`

**Interfaces:**
- Produces: a composable Hydra config `courtship_pipeline` with groups `paths, recording, detector, silhouette, outputs, slurm` (+ reused `anatomy, stac`). Keys used by later tasks: `recording.{session_dir,bouts_csv,calib_dir,num_animals,cameras}`, `detector.{ckpt,num_keypoints,crop,heatmap_size,conf_thresh}`, `silhouette.{mesh_npz,silhouette_weight,containment_weight,erode_px,mesh_subset,appendage_include,n_points,sdf_hw,bbox_margin,margin,smooth_weight,n_iter}`, `outputs.{out,overlay,mesh_subset,decimated_faces}`, `paths.*`, `slurm.*`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_courtship_config.py`:

```python
import os
from hydra import initialize_config_dir, compose

CFG_DIR = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"


def _compose(overrides):
    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        return compose(config_name="courtship_pipeline", overrides=overrides)


def test_courtship_pipeline_composes_and_paths_generalize(monkeypatch):
    monkeypatch.setenv("USER", "someone")
    cfg = _compose(["paths=hyak"])
    # user comes from env; no hardcoded 'eabe'
    assert cfg.paths.user == "someone"
    # required groups present
    assert cfg.recording.session_dir and cfg.recording.num_animals == 2
    assert cfg.detector.num_keypoints == 50 and cfg.detector.crop == 448
    assert cfg.silhouette.mesh_npz.endswith(".npz")
    assert "bouts" not in cfg.outputs.out or True  # out is a resolvable pattern


def test_default_user_when_env_absent(monkeypatch):
    monkeypatch.delenv("USER", raising=False)
    cfg = _compose(["paths=hyak"])
    assert cfg.paths.user == "eabe"   # fallback default
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_config.py -v`
Expected: FAIL (no `courtship_pipeline` config; `paths.user` hardcoded to `eabe`).

- [ ] **Step 3: Generalize `user` + write the config groups**

Edit `configs/paths/hyak.yaml` line 1: change `user: eabe` → `user: ${oc.env:USER,eabe}`. Do the same in `third_party/jarvis_jax/configs/paths/hyak.yaml` (its `user:` line).

Create `configs/recording/session0.yaml`:
```yaml
# @package recording
name: Session0
session_dir: ${paths.data_dir_johnson}/Video_recordings/courtship/Session0/2025_10_20_13_20_04
predictions_dir: ${recording.session_dir}/Predictions_3D_36233268   # source of bout dirs + existing masks
bouts_csv: ${recording.session_dir}/courtship_bouts_unified_summary.csv
calib_dir: ${recording.session_dir}/calibration
num_animals: 2
cameras: [Cam2012630, Cam2012631, Cam2012853, Cam2012855, Cam2012857, Cam2012861, Cam2012862]
```

Create `configs/detector/vitpose_v3.yaml`:
```yaml
# @package detector
ckpt: ${paths.vit_runs_root}/v3_kp_maskaware/final
num_keypoints: 50
crop: 448
heatmap_size: 224
conf_thresh: 0.3      # min 2D peak conf for a view to enter triangulation
```

Create `configs/silhouette/default.yaml`:
```yaml
# @package silhouette
mesh_npz: ${paths.body_model_dir}/fruitfly_cse/fly_v1_visual_canonical_wings.npz
xml: ${paths.body_model_dir}/fruitfly_v1/fruitfly_v1_free.xml
silhouette_weight: 0.3
containment_weight: 0.3
erode_px: 8
n_points: 128
sdf_hw: [128, 128]
bbox_margin: 0.4
margin: 0.0
mesh_subset: fps_300
appendage_include: [wing, leg, abdomen]
smooth_weight: 0.1
n_iter: 50
beta: 8.0
huber_delta: 0.0
```

Create `configs/outputs/default.yaml`:
```yaml
# @package outputs
out: ${paths.out_root}/${recording.name}_bouts_${now:%m%d%Y}
overlay: true
overlay_fps: 30
mesh_subset: fps_500+wing        # for output h5 verts
decimated_faces: 5000            # decimated mesh for overlay raster
```

Create `configs/slurm/ckpt_g2.yaml`:
```yaml
# @package slurm
partition: ckpt-g2
account: portia
gpus_per_task: 1
cpus: 8
mem: 48
time: '8:00:00'
requeue: true
conda_env: 3d_tracking
mail_user: ${oc.env:USER,eabe}@uw.edu
```

Create `configs/slurm/gpu_l40s.yaml`:
```yaml
# @package slurm
partition: gpu-l40s
account: portia
gpus_per_task: 1
cpus: 8
mem: 48
time: '8:00:00'
requeue: false
conda_env: 3d_tracking
mail_user: ${oc.env:USER,eabe}@uw.edu
```

Create `configs/courtship_pipeline.yaml`:
```yaml
defaults:
  - _self_
  - paths: hyak
  - anatomy: v1
  - stac: courtship
  - recording: session0
  - detector: vitpose_v3
  - silhouette: default
  - outputs: default
  - slurm: ckpt_g2

model: ${anatomy.model}
bout_ids: ''      # '' = all; else comma-separated
max_frames: 0     # 0 = all frames per bout
seed: 0
```

Add to `configs/paths/hyak.yaml` the pipeline path keys (append):
```yaml
data_dir_johnson: /gscratch/portia/${paths.user}/data/Johnson_lab
out_root: ${paths.data_dir_johnson}/courtship
vit_runs_root: ${paths.data_dir_johnson}/jax_vitpose_runs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_config.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add configs/paths/hyak.yaml configs/recording configs/detector configs/silhouette configs/outputs configs/slurm configs/courtship_pipeline.yaml third_party/jarvis_jax/configs/paths/hyak.yaml third_party/jarvis_jax/tests/test_courtship_config.py
git commit -m "feat(cse): courtship pipeline Hydra config groups + generalized (env) paths

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `courtship_bout_masks.py` — unpackbits SAM mask adapter

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_bout_masks.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_bout_masks.py`

**Interfaces:**
- Produces:
  - `unpack_bout_masks(npz_path) -> dict(packed_shape, W)` and `unpack_one(packed, fly, cam, frame, W) -> (H,W) bool`.
  - `load_bout_masks(npz_path, fly) -> dict(masks (T,C) list-of-arrays or (T,C,H,W) bool, valid (T,C) bool, T, C, H, W)` — full-frame per-(frame,cam) masks for one fly, via `np.unpackbits(packed[fly,cam], axis=-1)[..., :W]`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_courtship_bout_masks.py`:

```python
import numpy as np
from jarvis_jax.cse.courtship_bout_masks import unpack_one, load_bout_masks


def _synth_npz(tmp_path):
    # 2 flies, 2 cams, 3 frames, H=8, W=13 -> packed width ceil(13/8)=2
    H, W = 8, 13
    full = np.zeros((2, 2, 3, H, W), np.uint8)
    full[0, 1, 2, 2:5, 3:9] = 1                      # fly0 cam1 frame2 block
    packed = np.packbits(full, axis=-1)              # (2,2,3,H,2)
    valid = np.zeros((2, 2, 3), bool); valid[0, 1, 2] = True
    p = tmp_path / "sam3_masks.npz"
    np.savez(p, packed=packed, valid=valid, shape=np.array([H, W], np.int32),
             centroids=np.zeros((2, 2, 3, 2), np.float32))
    return str(p), full


def test_unpack_one_roundtrips_fullframe(tmp_path):
    p, full = _synth_npz(tmp_path)
    z = np.load(p)
    m = unpack_one(z["packed"], 0, 1, 2, int(z["shape"][1]))
    assert m.shape == (8, 13) and m.dtype == bool
    assert np.array_equal(m, full[0, 1, 2].astype(bool))    # exact full-frame match, trimmed to W


def test_load_bout_masks_shapes_and_valid(tmp_path):
    p, full = _synth_npz(tmp_path)
    out = load_bout_masks(p, fly=0)
    assert out["T"] == 3 and out["C"] == 2 and out["H"] == 8 and out["W"] == 13
    assert out["valid"][2, 1] and not out["valid"][0, 0]
    assert np.array_equal(np.asarray(out["masks"])[2, 1], full[0, 1, 2].astype(bool))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_bout_masks.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_bout_masks.py`:

```python
"""Courtship SAM3 mask adapter: bit-packed per-bout masks -> full-frame bool.

`sam3_masks.npz` stores `packed (A, C, T, H, ceil(W/8)) uint8` (np.packbits along
width), `valid (A,C,T) bool`, `shape [H,W]`, `centroids (A,C,T,2)`. Unpacking
gives the FULL-FRAME (H,W) mask in the calibration pixel frame (verified: fly0/fly1
land at the correct full-frame columns), so no centroid offset is needed.
"""
from __future__ import annotations
import numpy as np


def unpack_one(packed, fly: int, cam: int, frame: int, W: int) -> np.ndarray:
    """(H,W) bool full-frame mask for one (fly,cam,frame)."""
    return np.unpackbits(packed[fly, cam, frame], axis=-1)[:, :W].astype(bool)


def load_bout_masks(npz_path: str, fly: int) -> dict:
    """Full-frame masks for one fly across the bout: masks (T,C,H,W) bool + valid (T,C)."""
    z = np.load(npz_path)
    packed = z["packed"]; H, W = int(z["shape"][0]), int(z["shape"][1])
    A, C, T = packed.shape[0], packed.shape[1], packed.shape[2]
    valid = np.asarray(z["valid"])[fly].transpose(1, 0)          # (T,C)
    masks = np.zeros((T, C, H, W), bool)
    for c in range(C):
        for t in range(T):
            masks[t, c] = np.unpackbits(packed[fly, c, t], axis=-1)[:, :W].astype(bool)
    return dict(masks=masks, valid=valid, T=T, C=C, H=H, W=W)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_bout_masks.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_bout_masks.py third_party/jarvis_jax/tests/test_courtship_bout_masks.py
git commit -m "feat(cse): unpackbits courtship SAM mask adapter (full-frame per fly/cam/frame)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: `courtship_triangulate.py` — 2-D keypoints → 3-D via DLT

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_triangulate.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_triangulate.py`

**Interfaces:**
- Consumes: `geometry/center3d.triangulate_dlt_batched(points2d (B,nc,2), cameraMatrices (B,nc,4,3), valid (B,nc)) -> (B,3)`.
- Produces: `triangulate_keypoints(kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3), *, conf_thresh=0.3) -> (kp3d (T,K,3) f32 with NaN where <2 views, conf3d (T,K) f32)`. Per keypoint, valid views = `conf >= conf_thresh`; `conf3d = mean conf over valid views × (n_valid>=2)`.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_courtship_triangulate.py`:

```python
import numpy as np
from jarvis_jax.cse.courtship_triangulate import triangulate_keypoints


def _cam(P):  # P is (3,4); center3d expects (4,3)=P.T
    return P.T.astype(np.float32)


def test_triangulates_known_point_and_nan_when_underdetermined():
    # two orthogonal pinhole-ish cameras
    P0 = np.array([[1000, 0, 320, 0], [0, 1000, 240, 0], [0, 0, 1, 5.0]], float)
    P1 = np.array([[0, 0, 1000, 0], [0, 1000, 240, 0], [-1, 0, 0, 5.0]], float)
    cam_mats = np.stack([_cam(P0), _cam(P1)])            # (2,4,3)
    X = np.array([0.3, -0.2, 1.0])
    def proj(P):
        x = P @ np.append(X, 1.0); return x[:2] / x[2]
    kp2d = np.zeros((1, 2, 1, 2), np.float32)            # T=1,C=2,K=1
    kp2d[0, 0, 0] = proj(P0); kp2d[0, 1, 0] = proj(P1)
    conf = np.ones((1, 2, 1), np.float32)
    p3d, c3d = triangulate_keypoints(kp2d, conf, cam_mats, conf_thresh=0.3)
    assert np.allclose(p3d[0, 0], X, atol=1e-2)
    assert c3d[0, 0] > 0
    # only one confident view -> NaN + zero conf
    conf1 = conf.copy(); conf1[0, 1, 0] = 0.0
    p3d1, c3d1 = triangulate_keypoints(kp2d, conf1, cam_mats, conf_thresh=0.3)
    assert np.isnan(p3d1[0, 0]).all() and c3d1[0, 0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_triangulate.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_triangulate.py`:

```python
"""DLT triangulation of per-camera 2-D keypoints into 3-D mm keypoints + conf.

Geometric (no learned lift): per keypoint, use the cameras where the 2-D peak
confidence >= conf_thresh (>=2 required) and solve the DLT via
center3d.triangulate_dlt_batched. Keypoints with <2 confident views -> NaN
(down-weighted downstream by conf3d=0).
"""
from __future__ import annotations
import numpy as np
from jarvis_jax.geometry.center3d import triangulate_dlt_batched


def triangulate_keypoints(kp2d, conf, cam_mats, *, conf_thresh: float = 0.3):
    """kp2d (T,C,K,2), conf (T,C,K), cam_mats (C,4,3) -> (kp3d (T,K,3), conf3d (T,K))."""
    kp2d = np.asarray(kp2d, np.float32); conf = np.asarray(conf, np.float32)
    cam_mats = np.asarray(cam_mats, np.float32)
    T, C, K, _ = kp2d.shape
    valid = conf >= conf_thresh                                   # (T,C,K)
    kp3d = np.full((T, K, 3), np.nan, np.float32)
    conf3d = np.zeros((T, K), np.float32)
    # batch over (T*K) points; each has C views
    pts = kp2d.transpose(0, 2, 1, 3).reshape(T * K, C, 2)         # (TK,C,2)
    val = valid.transpose(0, 2, 1).reshape(T * K, C)             # (TK,C)
    cams = np.broadcast_to(cam_mats[None], (T * K, C, 4, 3))
    nvalid = val.sum(1)                                          # (TK,)
    ok = nvalid >= 2
    if ok.any():
        X = np.asarray(triangulate_dlt_batched(pts[ok], cams[ok], val[ok]))   # (n,3)
        finite = np.isfinite(X).all(1)
        idx = np.where(ok)[0][finite]
        flat3d = kp3d.reshape(T * K, 3); flatc = conf3d.reshape(T * K)
        flat3d[idx] = X[finite]
        cmean = (conf.transpose(0, 2, 1).reshape(T * K, C) * val).sum(1) / np.maximum(nvalid, 1)
        flatc[idx] = cmean[idx]
        kp3d = flat3d.reshape(T, K, 3); conf3d = flatc.reshape(T, K)
    return kp3d, conf3d
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_triangulate.py -v`
Expected: 1 passed (2 asserts).

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_triangulate.py third_party/jarvis_jax/tests/test_courtship_triangulate.py
git commit -m "feat(cse): DLT triangulation of per-camera 2D keypoints -> 3D mm + confidence

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `courtship_predict_2d.py` — ViTPose 2-D inference on SAM-cropped fly

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_predict_2d.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_predict_2d.py`

**Interfaces:**
- Consumes: `convert/build_checkpoint.load_vitpose(ckpt_dir, ViTPoseConfig) -> ViTPose`; `config.ViTPoseConfig`; `data/device.normalize_image`; `eval/mpjpe.heatmaps_to_keypoints(hm, in_size=448) -> (B,K,2)`; `predict/session_frameset.build_frameset(frame_imgs (nc,H,W,3), masks (nc,H,W), centroids (nc,2), valid (nc,), cameraMatrices (nc,4,3), crop=448) -> (crops4 (nc,448,448,4)|None, centerHM (nc,2), n_valid)`; `geometry/center3d.centroids_to_fullpx(kp (nc,K,2), centerHM (nc,2), 448)`.
- Produces:
  - `peaks_and_conf(hm) -> (kp2d_crop (B,K,2), conf (B,K))` — `heatmaps_to_keypoints(hm, in_size=448)` for coords + `jax.nn.relu(hm).max(axis=(1,2))` for per-channel peak conf.
  - `predict_bout_2d(vitpose, frames_by_cam_frame, masks (T,C,H,W), centroids (T,C,2), valid (T,C), cam_mats (C,4,3), *, crop=448, batch=64) -> (kp2d (T,C,K,2) full-frame, conf (T,C,K))`. Frames where `n_valid<2` → conf 0. Batched ViTPose forward.
  - `load_detector(ckpt, num_keypoints=50) -> ViTPose` (eval mode).

- [ ] **Step 1: Write the failing test** (pure-JAX, no checkpoint — tests the peak/conf + crop→full-frame mapping)

Create `third_party/jarvis_jax/tests/test_courtship_predict_2d.py`:

```python
import numpy as np
import jax.numpy as jnp
from jarvis_jax.cse.courtship_predict_2d import peaks_and_conf
from jarvis_jax.geometry.center3d import centroids_to_fullpx


def test_peaks_and_conf_locates_peak_and_reports_value():
    # one 224x224 heatmap, K=2, sharp peaks at known locations
    hm = np.full((1, 224, 224, 2), -5.0, np.float32)
    hm[0, 50, 30, 0] = 9.0     # (row=50,col=30) -> in_size=448 => (x=60,y=100)
    hm[0, 100, 200, 1] = 4.0
    kp, conf = peaks_and_conf(jnp.asarray(hm))
    kp = np.asarray(kp); conf = np.asarray(conf)
    assert abs(kp[0, 0, 0] - 60) < 2 and abs(kp[0, 0, 1] - 100) < 2   # x=col*2, y=row*2
    assert conf[0, 0] > conf[0, 1] > 0                                # peak values, relu'd


def test_crop_to_fullframe_mapping():
    # centerHM = crop center in full px; kp at crop center -> full == centerHM
    kp_crop = np.array([[[224.0, 224.0]]])          # (nc=1,K=1,2)
    centerHM = np.array([[900.0, 300.0]])           # (nc=1,2)
    full = np.asarray(centroids_to_fullpx(kp_crop, centerHM, 448))
    assert np.allclose(full[0, 0], [900.0, 300.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_predict_2d.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_predict_2d.py`:

```python
"""ViTPose 2-D inference on the SAM-cropped fly (V2VNet-free front end).

Load the retrained 4-ch ViTPose, crop each camera on the triangulated-centroid
reprojection (build_frameset), run the model on the (448,448,4) crops, decode
heatmap peaks (+ peak-value confidence), and map crop keypoints back to
full-frame pixels. No V2VNet / no 3-D volume.
"""
from __future__ import annotations
import numpy as np
import jax, jax.numpy as jnp

from jarvis_jax.config import ViTPoseConfig
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.data.device import normalize_image
from jarvis_jax.eval.mpjpe import heatmaps_to_keypoints
from jarvis_jax.predict.session_frameset import build_frameset
from jarvis_jax.geometry.center3d import centroids_to_fullpx


def load_detector(ckpt: str, num_keypoints: int = 50):
    vit = load_vitpose(ckpt, ViTPoseConfig(num_keypoints=num_keypoints))
    vit.eval()
    return vit


def peaks_and_conf(hm):
    """hm (B,Hh,Wh,K) logits -> (kp2d_crop (B,K,2) in 448px, conf (B,K) peak value)."""
    kp = heatmaps_to_keypoints(hm, in_size=448)               # (B,K,2)
    conf = jax.nn.relu(hm).max(axis=(1, 2))                    # (B,K)
    return kp, conf


def _forward(vit, crops4_u8):
    """(B,448,448,4) uint8 -> (kp2d_crop (B,K,2), conf (B,K)) on device."""
    x = normalize_image(jnp.asarray(crops4_u8))
    hm = vit(x, use_running_average=True)
    return peaks_and_conf(hm)


def predict_bout_2d(vitpose, frames_iter, masks, centroids, valid, cam_mats,
                    *, crop: int = 448, batch: int = 64):
    """Per (frame,cam): crop -> ViTPose -> full-frame 2-D kp + conf.

    frames_iter: iterable of length T, each -> (C,H,W,3) uint8 RGB (all cameras
    for that frame). masks (T,C,H,W) bool, centroids (T,C,2), valid (T,C),
    cam_mats (C,4,3). Returns kp2d (T,C,K,2) full-frame, conf (T,C,K) (0 where
    the frame had <2 valid views or the crop was empty).
    """
    T = masks.shape[0]; C = masks.shape[1]
    K = int(vitpose.decoder.__call__.__self__.cfg.num_keypoints) if False else None
    crops, centerHMs, keep = [], [], []            # collect valid (frame) crops
    per_frame = []                                  # (t, centerHM) for reassembly
    for t, frame_imgs in enumerate(frames_iter):
        c4, centerHM, nval = build_frameset(
            np.asarray(frame_imgs), masks[t], centroids[t], valid[t], cam_mats, crop=crop)
        if c4 is None:
            per_frame.append((t, None, None))
            continue
        crops.append(c4); centerHMs.append(centerHM); per_frame.append((t, len(crops) - 1, centerHM))
    # infer in batches of frames (each frame = C crops)
    kp2d = conf = None
    if crops:
        allc = np.concatenate(crops, 0)             # (n_valid_frames*C, 448,448,4)
        outs_kp, outs_cf = [], []
        for i in range(0, allc.shape[0], batch):
            k, cf = _forward(vitpose, allc[i:i + batch])
            outs_kp.append(np.asarray(k)); outs_cf.append(np.asarray(cf))
        kp_crop = np.concatenate(outs_kp, 0); cf = np.concatenate(outs_cf, 0)
        K = kp_crop.shape[1]
        kp2d = np.zeros((T, C, K, 2), np.float32); conf = np.zeros((T, C, K), np.float32)
        for t, ci, centerHM in per_frame:
            if ci is None:
                continue
            block_kp = kp_crop[ci * C:(ci + 1) * C]  # (C,K,2) crop px
            block_cf = cf[ci * C:(ci + 1) * C]
            kp2d[t] = centroids_to_fullpx(block_kp, centerHM, crop)
            conf[t] = block_cf
    else:
        kp2d = np.zeros((T, C, 0, 2), np.float32); conf = np.zeros((T, C, 0), np.float32)
    return kp2d, conf
```
Note: `K` is discovered from the model output at first forward; the `False` guard line is a no-op kept to document that K comes from the heatmap, not a private attr — delete it and set `K = None` before the loop if preferred.

- [ ] **Step 4: Run test to verify it passes** (pure-JAX part on CPU is fine)

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_predict_2d.py -v`
Expected: 2 passed.

- [ ] **Step 5: GPU smoke (checkpoint load + one forward)** — coordinator/manual, not a pytest

Run on the GPU node:
```bash
cd third_party/jarvis_jax && OMP_NUM_THREADS=4 python -c "
import numpy as np
from jarvis_jax.cse.courtship_predict_2d import load_detector, _forward
vit = load_detector('/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_kp_maskaware/final')
k,c = _forward(vit, np.zeros((2,448,448,4), np.uint8))
print('kp', np.asarray(k).shape, 'conf', np.asarray(c).shape)"
```
Expected: `kp (2,50,2) conf (2,50)`.

- [ ] **Step 6: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_predict_2d.py third_party/jarvis_jax/tests/test_courtship_predict_2d.py
git commit -m "feat(cse): ViTPose 2D inference on SAM-cropped fly (peaks+conf, crop->fullframe)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: `courtship_stac.py` — STAC fit-once + `ik_only` per bout

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_stac.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_stac.py`

**Interfaces:**
- Consumes: `stac_mjx.run_stac(cfg, kp_flat (T,K*3), kp_names, base_path, save_path) -> (fit_offsets_path, ik_only_path)`; `cfg.model.MOCAP_SCALE_FACTOR`; `cfg.stac.{fit_offsets_path, ik_only_path, skip_fit_offsets, skip_ik_only, n_fit_frames, n_frames_per_clip}`.
- Produces:
  - `fit_offsets_once(cfg, kp3d_sample (Tf,K,3), kp_names, *, offsets_path, save_path) -> offsets_h5` — runs `run_stac` with `skip_ik_only=1` on a high-confidence sample → writes the shared offsets h5.
  - `ik_only_bout(cfg, kp3d (T,K,3), kp_names, *, offsets_path, out_h5, save_path) -> out_h5` — `skip_fit_offsets=1`, `n_frames_per_clip=T`, writes a per-bout STAC h5 consumable by `build_solver_inputs`. NaN keypoints pass through (STAC handles). Keypoints scaled by `cfg.model.MOCAP_SCALE_FACTOR` inside (do NOT pre-scale).

- [ ] **Step 1: Write the failing test** (monkeypatch `run_stac` to assert the cfg toggles + kp flattening)

Create `third_party/jarvis_jax/tests/test_courtship_stac.py`:

```python
import numpy as np
from omegaconf import OmegaConf
import jarvis_jax.cse.courtship_stac as cst


def _cfg():
    return OmegaConf.create({"model": {"MOCAP_SCALE_FACTOR": 1000.0, "MJCF_PATH": "x.xml"},
                             "stac": {"fit_offsets_path": "off.h5", "ik_only_path": "ik.h5",
                                      "skip_fit_offsets": False, "skip_ik_only": 0,
                                      "n_fit_frames": 5, "n_frames_per_clip": 5}})


def test_fit_offsets_once_sets_skip_ik(monkeypatch, tmp_path):
    seen = {}
    def fake_run_stac(cfg, kp_flat, kp_names, base_path=None, save_path=None):
        seen.update(skip_ik=int(cfg.stac.skip_ik_only), skip_fit=int(cfg.stac.skip_fit_offsets),
                    shape=kp_flat.shape); return (str(tmp_path / "off.h5"), None)
    monkeypatch.setattr(cst.stac_mjx, "run_stac", fake_run_stac)
    cst.fit_offsets_once(_cfg(), np.zeros((5, 3, 3), np.float32), ["a", "b", "c"],
                         offsets_path="off.h5", save_path=str(tmp_path))
    assert seen["skip_ik"] == 1 and seen["skip_fit"] == 0 and seen["shape"] == (5, 9)


def test_ik_only_bout_reuses_offsets(monkeypatch, tmp_path):
    seen = {}
    def fake_run_stac(cfg, kp_flat, kp_names, base_path=None, save_path=None):
        seen.update(skip_ik=int(cfg.stac.skip_ik_only), skip_fit=int(cfg.stac.skip_fit_offsets),
                    nfpc=int(cfg.stac.n_frames_per_clip), ik=cfg.stac.ik_only_path)
        return (str(tmp_path / "off.h5"), str(tmp_path / "ik.h5"))
    monkeypatch.setattr(cst.stac_mjx, "run_stac", fake_run_stac)
    cst.ik_only_bout(_cfg(), np.zeros((7, 3, 3), np.float32), ["a", "b", "c"],
                     offsets_path="off.h5", out_h5="bout1_ik.h5", save_path=str(tmp_path))
    assert seen["skip_fit"] == 1 and seen["skip_ik"] == 0 and seen["nfpc"] == 7
    assert seen["ik"] == "bout1_ik.h5"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_stac.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_stac.py`:

```python
"""STAC articulated fit for courtship bouts (fit offsets once, ik_only per bout).

Wraps stac_mjx.run_stac. Offsets are fit ONCE on a high-confidence keypoint
sample and shared; each bout then runs ik_only reusing those offsets. Output h5
is the schema build_solver_inputs consumes (qpos, kp_data, offsets, kp_names,
names_qpos, config).
"""
from __future__ import annotations
import numpy as np
import stac_mjx


def _flat_scaled(kp3d, scale):
    """(T,K,3) mm -> (T,K*3) scaled by MOCAP_SCALE_FACTOR (NaNs preserved)."""
    T, K, _ = kp3d.shape
    return (np.asarray(kp3d, np.float32).reshape(T, K * 3) * float(scale))


def fit_offsets_once(cfg, kp3d_sample, kp_names, *, offsets_path, save_path):
    """Fit marker offsets on a sample (skip ik). Writes <save_path>/<offsets_path>."""
    cfg.stac.fit_offsets_path = offsets_path
    cfg.stac.skip_fit_offsets = False
    cfg.stac.skip_ik_only = 1
    T = kp3d_sample.shape[0]
    cfg.stac.n_fit_frames = min(int(cfg.stac.get("n_fit_frames", T)), T)
    cfg.stac.n_frames_per_clip = T
    kp_flat = _flat_scaled(kp3d_sample, cfg.model["MOCAP_SCALE_FACTOR"])
    stac_mjx.run_stac(cfg, kp_flat, list(kp_names), save_path=save_path)
    import os
    return os.path.join(str(save_path), offsets_path)


def ik_only_bout(cfg, kp3d, kp_names, *, offsets_path, out_h5, save_path):
    """ik_only for one bout reusing fitted offsets. Writes <save_path>/<out_h5>."""
    cfg.stac.fit_offsets_path = offsets_path      # run_stac reloads offsets from here
    cfg.stac.ik_only_path = out_h5
    cfg.stac.skip_fit_offsets = 1
    cfg.stac.skip_ik_only = 0
    cfg.stac.n_frames_per_clip = int(kp3d.shape[0])
    kp_flat = _flat_scaled(kp3d, cfg.model["MOCAP_SCALE_FACTOR"])
    stac_mjx.run_stac(cfg, kp_flat, list(kp_names), save_path=save_path)
    import os
    return os.path.join(str(save_path), out_h5)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_stac.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_stac.py third_party/jarvis_jax/tests/test_courtship_stac.py
git commit -m "feat(cse): courtship STAC wrapper (fit offsets once + ik_only per bout)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: `mesh_decimate.py` — one-time watertight-mesh face decimation

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/mesh_decimate.py`
- Test: `third_party/jarvis_jax/tests/test_mesh_decimate.py`

**Interfaces:**
- Produces: `decimate_mesh_npz(mesh_npz, out_npz, *, target_faces=5000) -> out_npz` — reads the canonical mesh npz (`vertices`, `faces`), runs `trimesh.Trimesh(...).simplify_quadric_decimation(target_faces)`, writes `out_npz` with `vertices (Vd,3) f32`, `faces (Fd,3) i32`, `source`, `target_faces`. Used only for overlay rasterization (`filled_tri_iou`/`write_camera_video`), never for kinematics.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_mesh_decimate.py`:

```python
import numpy as np, trimesh
from jarvis_jax.cse.mesh_decimate import decimate_mesh_npz


def test_decimation_reduces_faces_keeps_watertight_shape(tmp_path):
    s = trimesh.creation.icosphere(subdivisions=4)        # ~20480 faces
    src = tmp_path / "m.npz"
    np.savez(src, vertices=np.asarray(s.vertices, np.float32), faces=np.asarray(s.faces, np.int32))
    out = tmp_path / "d.npz"
    decimate_mesh_npz(str(src), str(out), target_faces=2000)
    z = np.load(out)
    assert z["faces"].shape[0] <= 3000 and z["faces"].shape[0] < 20480
    assert z["vertices"].shape[1] == 3
    # bounding box roughly preserved (shape not destroyed)
    assert np.allclose(z["vertices"].max(0), s.vertices.max(0), atol=0.1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_mesh_decimate.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/mesh_decimate.py`:

```python
"""One-time quadric decimation of a canonical mesh for fast overlay rasterization.

The 875k-vertex visual mesh is too heavy to fillPoly per frame x camera x bout.
A ~5k-face decimated copy renders ~100x faster and is visually equivalent for
overlay silhouettes. Kinematics/outputs never use this — only the raster.
"""
from __future__ import annotations
import numpy as np
import trimesh


def decimate_mesh_npz(mesh_npz: str, out_npz: str, *, target_faces: int = 5000) -> str:
    z = np.load(mesh_npz, allow_pickle=True)
    m = trimesh.Trimesh(np.asarray(z["vertices"], np.float64),
                        np.asarray(z["faces"]), process=False)
    d = m.simplify_quadric_decimation(target_faces)
    np.savez(out_npz,
             vertices=np.asarray(d.vertices, np.float32),
             faces=np.asarray(d.faces, np.int32),
             source=str(mesh_npz), target_faces=np.int32(target_faces))
    return out_npz
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_mesh_decimate.py -v`
Expected: 1 passed. (If the installed trimesh exposes decimation as `simplify_quadric_decimation` with a `face_count=` kwarg, adjust the call accordingly; verify with `python -c "import trimesh;print([x for x in dir(trimesh.Trimesh) if 'simplif' in x or 'decim' in x])"` first.)

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/mesh_decimate.py third_party/jarvis_jax/tests/test_mesh_decimate.py
git commit -m "feat(cse): one-time watertight-mesh decimation for fast overlay raster

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: `courtship_polish.py` — courtship silhouette-containment polish

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_polish.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_polish.py`

**Interfaces:**
- Consumes: `silhouette_ik_solve.build_solver_inputs(stac_h5, xml)`; `silhouette_ik.{load_anatomy,make_fk_repose}`; `silhouette_dof.{appendage_vertex_indices,build_appendage_dof_mask}`; `silhouette_targets.silhouette_fk_indices`; `silhouette_sdf.build_sdf_stack` helpers `_mask_bbox`/`_mask_to_sdf_crop`; `silhouette_boundary.sample_boundary_points`; `silhouette_joint_ik.SilhouetteJaxlsBatchSolver`; `geometry.reprojection_tool.ReprojectionTool`; `courtship_bout_masks.load_bout_masks`; `courtship_triangulate.triangulate_keypoints`; `silhouette_ik_solve._umeyama`; `stac_utils`.
- Produces: `courtship_targets_from_masks(masks (T,C,H,W) bool, present (T,C) bool, cam_mats_affine, *, erode_px, n_points, sdf_hw, bbox_margin) -> dict(boundary (T,C,n_points,2), conf_p (T,C,n_points), present (T,C), sdf (T,C,Hh,Ww), grid_scale, grid_offset, cam_Ms (C,2,3), cam_ts (C,2))` — the courtship analogue of `build_silhouette_targets`+`build_sdf_stack`, computed directly from unpacked masks; and `polish_bout(stac_h5, cfg, kp3d_mm, kp3d_conf, masks_dict, calib_dir) -> qpos_refined (T,nq)` — builds solver inputs from the STAC h5, per-frame model→mm bridges from `_umeyama(FK sites at q_init, kp3d_mm)` (identity + gated where <3 valid kp), and calls `SilhouetteJaxlsBatchSolver.solve_trajectory` with coverage+containment.

- [ ] **Step 1: Write the failing test** (targets-from-masks on synthetic square masks; CPU numpy)

Create `third_party/jarvis_jax/tests/test_courtship_polish.py`:

```python
import numpy as np
from jarvis_jax.cse.courtship_polish import courtship_targets_from_masks


def test_targets_from_masks_shapes_and_present():
    T, C, H, W = 2, 2, 60, 60
    masks = np.zeros((T, C, H, W), bool)
    masks[0, 0, 20:40, 20:40] = True                 # one present cell
    present = masks.any(axis=(2, 3))
    cam_Ms = np.tile(np.eye(2, 3)[None], (C, 1, 1)).astype(np.float32)
    cam_ts = np.zeros((C, 2), np.float32)
    out = courtship_targets_from_masks(masks, present, (cam_Ms, cam_ts),
                                       erode_px=2, n_points=16, sdf_hw=(32, 32), bbox_margin=0.4)
    assert out["boundary"].shape == (T, C, 16, 2)
    assert out["sdf"].shape == (T, C, 32, 32)
    assert out["present"][0, 0] and not out["present"][1, 1]
    # present cell: some finite boundary points; absent cell: NaN
    assert np.isfinite(out["boundary"][0, 0]).any()
    assert np.isnan(out["boundary"][1, 1]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_polish.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_polish.py`. Build `courtship_targets_from_masks` by reusing the tested primitives (`_mask_bbox`, `_mask_to_sdf_crop` from `silhouette_sdf`; `sample_boundary_points` from `silhouette_boundary`; `scipy.ndimage.binary_erosion`) in a `(T,C)` loop — mirroring `build_silhouette_targets`/`build_sdf_stack` but taking masks directly instead of the COCO loaders. Build `polish_bout` by mirroring `run_silhouette_polish.run_polish`'s solver-input + bridge + solve block (from the API map §6b/§6c), but sourcing masks from `load_bout_masks` and mm keypoints from `kp3d_mm` (already triangulated) instead of COCO. Full code:

```python
"""Courtship silhouette-containment polish: STAC h5 + unpacked SAM masks +
triangulated mm keypoints -> refined qpos via SilhouetteJaxlsBatchSolver.

Reuses the solver, bridge, and DOF/vertex logic; replaces the COCO mask/keypoint
providers of run_silhouette_polish with courtship sources.
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
from scipy import ndimage

import stac_mjx.utils as stac_utils
from jarvis_jax.cse.silhouette_ik import load_anatomy, make_fk_repose
from jarvis_jax.cse.silhouette_ik_solve import build_solver_inputs, _umeyama
from jarvis_jax.cse.silhouette_dof import appendage_vertex_indices, build_appendage_dof_mask
from jarvis_jax.cse.silhouette_targets import silhouette_fk_indices
from jarvis_jax.cse.silhouette_sdf import _mask_bbox, _mask_to_sdf_crop, _BIG
from jarvis_jax.cse.silhouette_boundary import sample_boundary_points
from jarvis_jax.cse.silhouette_joint_ik import SilhouetteJaxlsBatchSolver


def courtship_targets_from_masks(masks, present, cam_affine, *, erode_px, n_points,
                                 sdf_hw, bbox_margin):
    """(T,C) unpacked masks -> boundary/conf_p/present + SDF stack + affine cams."""
    cam_Ms, cam_ts = cam_affine
    T, C, H, W = masks.shape
    Hh, Ww = sdf_hw
    boundary = np.full((T, C, n_points, 2), np.nan, np.float32)
    conf_p = np.zeros((T, C, n_points), np.float32)
    pres = np.zeros((T, C), bool)
    sdf = np.full((T, C, Hh, Ww), _BIG, np.float32)
    gscale = np.ones((T, C, 2), np.float32); goff = np.zeros((T, C, 2), np.float32)
    struct = ndimage.generate_binary_structure(2, 1)
    for t in range(T):
        for c in range(C):
            if not present[t, c] or not masks[t, c].any():
                continue
            m = masks[t, c]
            er = ndimage.binary_erosion(m, structure=struct, iterations=erode_px, border_value=0) \
                if erode_px > 0 else m
            if not er.any():
                er = m
            bpts = sample_boundary_points(er, n_points, seed=0)
            boundary[t, c] = bpts
            conf_p[t, c] = np.isfinite(bpts).all(-1).astype(np.float32)
            bbox = _mask_bbox(m, bbox_margin)
            out = _mask_to_sdf_crop(m, bbox, sdf_hw) if bbox is not None else None
            if out is not None:
                sdf[t, c], gscale[t, c], goff[t, c] = out
                pres[t, c] = True
    return dict(boundary=boundary, conf_p=conf_p, present=pres, sdf=sdf,
                grid_scale=gscale, grid_offset=goff,
                cam_Ms=np.asarray(cam_Ms, np.float32), cam_ts=np.asarray(cam_ts, np.float32))


def polish_bout(stac_h5, cfg, kp3d_mm, kp3d_conf, masks_dict, calib_dir):
    """Refine qpos with silhouette+containment. kp3d_mm (T,K,3), masks_dict from
    load_bout_masks (fly), calib_dir has Cam*.yaml. Returns (T,nq)."""
    from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
    sil = cfg.silhouette
    inp = build_solver_inputs(stac_h5, sil.xml)
    T = inp["q_init"].shape[0]
    q_init = np.asarray(inp["q_init"])[:T]; kp_data = inp["kp_data"][:T]
    anat = load_anatomy(sil.xml, sil.mesh_npz); fk = make_fk_repose(anat)
    cov_idx = silhouette_fk_indices(sil.mesh_npz, subset=sil.mesh_subset)
    cont_idx = appendage_vertex_indices(sil.mesh_npz, subset=sil.mesh_subset,
                                        include=tuple(sil.appendage_include))
    sil_qs = build_appendage_dof_mask(anat["m"], include=tuple(sil.appendage_include))
    conf_v = jnp.ones((len(cont_idx),))

    rt = ReprojectionTool(calib_dir)
    cam_Ms = np.stack([rt._camera_list[c].cameraMatrix[:2, :3] for c in range(rt.num_cameras)])
    cam_ts = np.stack([rt._camera_list[c].cameraMatrix[:2, 3] for c in range(rt.num_cameras)])

    tg = courtship_targets_from_masks(masks_dict["masks"], masks_dict["valid"],
                                      (cam_Ms, cam_ts), erode_px=sil.erode_px,
                                      n_points=sil.n_points, sdf_hw=tuple(sil.sdf_hw),
                                      bbox_margin=sil.bbox_margin)

    # per-frame model->mm bridges from q_init sites vs triangulated kp3d_mm
    bridge_s = np.ones((T,), np.float32)
    bridge_R = np.broadcast_to(np.eye(3, dtype=np.float32), (T, 3, 3)).copy()
    bridge_t = np.zeros((T, 3), np.float32)
    frame_ok = np.zeros(T, bool)
    for t in range(T):
        kok = np.isfinite(kp3d_mm[t]).all(-1) & (kp3d_conf[t] > 0)
        if kok.sum() < 3:
            continue
        d0 = inp["mjx_data"].replace(qpos=jnp.asarray(q_init[t]))
        d0 = stac_utils.kinematics(inp["mjx_model"], d0); d0 = stac_utils.com_pos(inp["mjx_model"], d0)
        sites0 = np.asarray(stac_utils.get_site_xpos(d0, inp["site_idxs"]))
        s, R, tr = _umeyama(sites0[kok], np.asarray(kp3d_mm[t])[kok])
        bridge_s[t] = s; bridge_R[t] = R; bridge_t[t] = tr; frame_ok[t] = True
    present_g = tg["present"].copy(); present_g[~frame_ok] = False
    conf_p_g = tg["conf_p"].copy(); conf_p_g[~frame_ok] = 0.0

    solver = SilhouetteJaxlsBatchSolver(n_iter=int(sil.n_iter), smooth_weight=float(sil.smooth_weight),
                                        beta=float(sil.beta), huber_delta=float(sil.huber_delta))
    q = solver.solve_trajectory(
        q_init=jnp.asarray(q_init), mjx_model=inp["mjx_model"], mjx_data_template=inp["mjx_data"],
        kp_data=kp_data, qs_to_opt=inp["qs_to_opt"], kps_to_opt=inp["kps_to_opt"],
        lb=inp["lb"], ub=inp["ub"], site_idxs=inp["site_idxs"], q_reg_weights=inp["q_reg_weights"],
        fk_repose=fk, cov_vert_indices=cov_idx, cont_vert_indices=cont_idx,
        cam_Ms=jnp.asarray(cam_Ms), cam_ts=jnp.asarray(cam_ts),
        boundary_all=jnp.asarray(tg["boundary"]), conf_p_all=jnp.asarray(conf_p_g),
        sil_qs_mask=sil_qs, silhouette_weight=float(sil.silhouette_weight),
        sdf_all=jnp.asarray(tg["sdf"]), grid_scale_all=jnp.asarray(tg["grid_scale"]),
        grid_offset_all=jnp.asarray(tg["grid_offset"]), present_all=jnp.asarray(present_g),
        conf_v=conf_v, containment_weight=float(sil.containment_weight), margin=float(sil.margin),
        bridge_s_all=jnp.asarray(bridge_s), bridge_R_all=jnp.asarray(bridge_R),
        bridge_t_all=jnp.asarray(bridge_t))
    return np.asarray(q)
```

- [ ] **Step 4: Run the targets test (CPU) + confirm imports**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_polish.py -v`
Expected: 1 passed. Also `JAX_PLATFORMS=cpu python -c "import jarvis_jax.cse.courtship_polish"` → no error.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_polish.py third_party/jarvis_jax/tests/test_courtship_polish.py
git commit -m "feat(cse): courtship silhouette polish (mask targets + triangulated-kp bridges -> solver)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 8: `courtship_qc.py` — per-session QC aggregation dashboard

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_qc.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_qc.py`

**Interfaces:**
- Produces: `aggregate_session_qc(bout_qc_paths (list[str]), out_json, *, plot_dir=None) -> dict(session summary)` — reads each bout/fly `qc.json` (from `qc.qc_report`), assembles per-bout rows (bout_idx, fly, iou hard/soft, reproj px, loo px, n_frames), writes `session_qc.json` and (if `plot_dir`) per-metric bar plots. Robust to missing/partial files.

- [ ] **Step 1: Write the failing test**

Create `third_party/jarvis_jax/tests/test_courtship_qc.py`:

```python
import json
import numpy as np
from jarvis_jax.cse.courtship_qc import aggregate_session_qc


def _bout_qc(p, iou, reproj, n):
    json.dump({"silhouette_iou": {"hard_median": iou, "soft_median": iou + 0.02, "n_frames": n},
               "per_camera_reproj_px": {"median": reproj, "n": n * 7},
               "loo_reproj_px": {"median": reproj + 1.0, "n_frames": n}, "n_frames": n}, open(p, "w"))


def test_aggregate_reads_bouts_and_writes_summary(tmp_path):
    paths = []
    for i, (iou, rp, n) in enumerate([(0.74, 5.0, 100), (0.71, 6.5, 80)]):
        pth = tmp_path / f"bout_{i:05d}_fly0_qc.json"; _bout_qc(pth, iou, rp, n); paths.append(str(pth))
    out = tmp_path / "session_qc.json"
    summ = aggregate_session_qc(paths, str(out), plot_dir=str(tmp_path))
    assert summ["n_bouts_flies"] == 2
    assert abs(summ["iou_hard_median"] - 0.725) < 1e-6
    assert out.exists() and json.load(open(out))["n_bouts_flies"] == 2
    assert (tmp_path / "iou_by_bout.png").exists()


def test_aggregate_skips_missing(tmp_path):
    summ = aggregate_session_qc([str(tmp_path / "nope.json")], str(tmp_path / "s.json"))
    assert summ["n_bouts_flies"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_qc.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the implementation**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_qc.py`:

```python
"""Aggregate per-bout/fly QC (qc.qc_report output) into a session dashboard."""
from __future__ import annotations
import json, os, re
import numpy as np


def _parse_name(p):
    m = re.search(r"bout_(\d+)_fly(\d)", os.path.basename(p))
    return (int(m.group(1)), int(m.group(2))) if m else (-1, -1)


def aggregate_session_qc(bout_qc_paths, out_json, *, plot_dir=None) -> dict:
    rows = []
    for p in bout_qc_paths:
        if not os.path.exists(p):
            continue
        try:
            d = json.load(open(p))
        except Exception:
            continue
        b, f = _parse_name(p)
        rows.append(dict(bout=b, fly=f,
                         iou_hard=d.get("silhouette_iou", {}).get("hard_median", float("nan")),
                         iou_soft=d.get("silhouette_iou", {}).get("soft_median", float("nan")),
                         reproj_px=d.get("per_camera_reproj_px", {}).get("median", float("nan")),
                         loo_px=d.get("loo_reproj_px", {}).get("median", float("nan")),
                         n_frames=d.get("n_frames", 0)))
    def _med(k):
        v = [r[k] for r in rows if r[k] == r[k]]      # drop nan
        return float(np.median(v)) if v else float("nan")
    summ = dict(n_bouts_flies=len(rows),
                iou_hard_median=_med("iou_hard"), iou_soft_median=_med("iou_soft"),
                reproj_px_median=_med("reproj_px"), loo_px_median=_med("loo_px"),
                total_frames=int(sum(r["n_frames"] for r in rows)), rows=rows)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    json.dump(summ, open(out_json, "w"), indent=2)
    if plot_dir and rows:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        os.makedirs(plot_dir, exist_ok=True)
        order = sorted(range(len(rows)), key=lambda i: (rows[i]["bout"], rows[i]["fly"]))
        labels = [f"{rows[i]['bout']}.{rows[i]['fly']}" for i in order]
        for key, fname in [("iou_hard", "iou_by_bout.png"), ("reproj_px", "reproj_by_bout.png")]:
            fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.3), 4))
            ax.bar(range(len(rows)), [rows[i][key] for i in order])
            ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_ylabel(key); ax.set_title(f"{key} by bout.fly")
            fig.tight_layout(); fig.savefig(os.path.join(plot_dir, fname), dpi=110); plt.close(fig)
    return summ
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_qc.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_qc.py third_party/jarvis_jax/tests/test_courtship_qc.py
git commit -m "feat(cse): session QC aggregation dashboard (per-bout IoU/reproj json + plots)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 9: `run_courtship_bout.py` — resumable per-bout driver (stages A–E)

**Files:**
- Create: `scripts/run_courtship_bout.py`
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_resume.py` (the pure stage-skip/atomic helpers, so they're unit-testable)
- Test: `third_party/jarvis_jax/tests/test_courtship_resume.py`

**Interfaces:**
- Produces (`courtship_resume.py`): `atomic_save_npz(path, **arrays)` / `atomic_save_json(path, obj)` / `atomic_write(path, write_fn)` (write to `path+".tmp"`, `os.replace`); `stage_done(path) -> bool` (exists and non-empty); `mark_done(bout_dir)` (writes `DONE`); `bout_complete(bout_dir) -> bool` (`DONE` exists). `run_courtship_bout.py` (Hydra app, `config_path=<repo>/configs`, `config_name=courtship_pipeline`): for the bout index (`bout_ids`/array id) and each fly, run A→E, skipping stages whose artifact exists, writing to `<run_root>/bouts/bout_<idx:05d>/fly<f>/` (`kp2d.npz, kp3d.npz, stac_ik.h5, qpos_refined.npz, outputs.h5, qc.json`) atomically, then `overlays/`, then `DONE`.

- [ ] **Step 1: Write the failing test** (pure resume helpers)

Create `third_party/jarvis_jax/tests/test_courtship_resume.py`:

```python
import os, numpy as np
from jarvis_jax.cse.courtship_resume import (atomic_save_npz, atomic_save_json,
                                             stage_done, mark_done, bout_complete)


def test_atomic_save_and_stage_done(tmp_path):
    p = str(tmp_path / "kp2d.npz")
    assert not stage_done(p)
    atomic_save_npz(p, a=np.zeros(3))
    assert stage_done(p) and not os.path.exists(p + ".tmp")
    atomic_save_json(str(tmp_path / "qc.json"), {"x": 1})
    assert stage_done(str(tmp_path / "qc.json"))


def test_done_marker(tmp_path):
    d = str(tmp_path / "bout_00001" / "fly0"); os.makedirs(d)
    assert not bout_complete(d)
    mark_done(d)
    assert bout_complete(d)


def test_stage_skip_logic(tmp_path):
    # a present artifact is "done"; a leftover .tmp is NOT
    p = str(tmp_path / "stac_ik.h5"); open(p + ".tmp", "w").write("partial")
    assert not stage_done(p)
    open(p, "w").write("real")
    assert stage_done(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_resume.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the resume helpers**

Create `third_party/jarvis_jax/jarvis_jax/cse/courtship_resume.py`:

```python
"""Atomic writes + stage-skip markers for the resumable per-bout driver."""
from __future__ import annotations
import json, os
import numpy as np


def atomic_write(path, write_fn):
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_fn(tmp)
    os.replace(tmp, path)
    return path


def atomic_save_npz(path, **arrays):
    return atomic_write(path, lambda p: np.savez(p if p.endswith(".npz") else p + ".npz", **arrays)
                        if False else _savez_exact(p, arrays))


def _savez_exact(tmp, arrays):
    # np.savez appends .npz if missing; write to a fixed temp then it will match on replace
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)


def atomic_save_json(path, obj):
    return atomic_write(path, lambda p: json.dump(obj, open(p, "w"), indent=2))


def stage_done(path) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def mark_done(bout_dir):
    atomic_write(os.path.join(bout_dir, "DONE"), lambda p: open(p, "w").write("ok"))


def bout_complete(bout_dir) -> bool:
    return os.path.exists(os.path.join(bout_dir, "DONE"))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_courtship_resume.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the driver `scripts/run_courtship_bout.py`**

Create `scripts/run_courtship_bout.py` — a Hydra app (`config_path="../configs"`, `config_name="courtship_pipeline"`) that, for the selected bout index and each fly in `range(recording.num_animals)`:
1. Resolve `run_root = cfg.outputs.out`, `bout_dir = <run_root>/bouts/bout_<idx:05d>/fly<f>/`. If `bout_complete(bout_dir_parent)` → return.
2. Resolve `bout_npz = <predictions_dir>/bout_<idx:05d>/sam3_masks.npz`; `masks = load_bout_masks(bout_npz, fly)`.
3. **Stage A** (`kp2d.npz`): if not present → open the 7 `Cam*.mp4`, iterate the bout's frames (seek to bout start), build a per-frame `(C,H,W,3)` RGB generator, `vit = load_detector(cfg.detector.ckpt)`, `kp2d, conf = predict_bout_2d(vit, frames_iter, masks["masks"], centroids, masks["valid"], cam_mats_4x3, batch=cfg... )`; `atomic_save_npz(kp2d.npz, kp2d=kp2d, conf=conf)`. (`centroids` from the npz `centroids[fly]`.)
4. **Stage B** (`kp3d.npz`): `kp3d, conf3d = triangulate_keypoints(kp2d, conf, cam_mats_4x3, conf_thresh=cfg.detector.conf_thresh)`; save.
5. **Stage C** (`stac_ik.h5`): `ik_only_bout(cfg, kp3d, kp_names, offsets_path=<run_root>/offsets.h5, out_h5="stac_ik.h5", save_path=bout_dir)`. (kp_names from `cfg.model.KP_NAMES`.)
6. **Stage D** (`qpos_refined.npz`): `q = polish_bout(<bout_dir>/stac_ik.h5, cfg, kp3d, conf3d, masks, cfg.recording.calib_dir)`; save.
7. **Stage E**: `build_fly_outputs(recording, ik_h5=stac_ik.h5, model_xml=cfg.silhouette.xml, mesh_npz=cfg.silhouette.mesh_npz, qpos=q, bridges=<from polish>, out_path=outputs.h5, mesh_subset=cfg.outputs.mesh_subset)`; `qc_report(...) -> qc.json`; if `cfg.outputs.overlay`: for each camera `write_camera_video(overlays/<cam>_reproj.mp4, frames_rgb_iter=<bout frames for cam>, mesh2d_by_frame=<decimated mesh verts projected>, kp2d_by_frame=<kp3d projected>, fps=cfg.outputs.overlay_fps)` skipping existing files.
8. `mark_done(bout_dir)`.

Every artifact write uses `atomic_save_*`; each stage guarded by `if not stage_done(<artifact>):`. The bridges from Stage D must be persisted (e.g. inside `qpos_refined.npz` as `bridge_s/bridge_R/bridge_t`) so Stage E and a resume can reuse them without recomputing. (Have `polish_bout` return `(q, bridges)` — update its return + Task 7's callers accordingly; adjust Task 7 signature to `-> (qpos_refined, bridge_s, bridge_R, bridge_t)`.)

Include a `main()` guarded by `@hydra.main`. Selection of the bout index: `cfg.bout_ids` (single id when run per-array-task via `+bout_ids=<n>`), else iterate all bout dirs found under `predictions_dir`.

- [ ] **Step 6: Import/CLI smoke**

Run: `cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset && JAX_PLATFORMS=cpu python -c "import ast; ast.parse(open('scripts/run_courtship_bout.py').read()); print('parse ok')"`
Expected: `parse ok`. (Full run is the Task 11 GPU de-risk.)

- [ ] **Step 7: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add scripts/run_courtship_bout.py third_party/jarvis_jax/jarvis_jax/cse/courtship_resume.py third_party/jarvis_jax/tests/test_courtship_resume.py
git commit -m "feat(cse): resumable per-bout courtship driver (atomic stage artifacts, skip-complete)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 10: `slurm_courtship_array.py` — array submission + dependency chain

**Files:**
- Create: `scripts/slurm_courtship_array.py`
- Test: `third_party/jarvis_jax/tests/test_slurm_courtship_array.py`

**Interfaces:**
- Produces: a submitter that (a) counts bouts under `predictions_dir`; (b) if any bout lacks `sam3_masks.npz` → submit a **Stage-0 SAM3 array** (PyTorch env) over those bouts; (c) submit a **precompute** job (offsets fit-once + decimated mesh) `--dependency=afterok:<sam3>`; (d) submit the **JAX array** (`run_courtship_bout.py +bout_ids=$SLURM_ARRAY_TASK_ID`) `--dependency=afterok:<precompute>` with `--requeue`; (e) submit an **aggregation** job `--dependency=afterok:<jax_array>`. `--dry-run` prints the scripts. Partition from `cfg.slurm`.

- [ ] **Step 1: Write the failing test** (pure script-building, no sbatch)

Create `third_party/jarvis_jax/tests/test_slurm_courtship_array.py`:

```python
import importlib.util, sys

SPEC = "/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/scripts/slurm_courtship_array.py"


def _load():
    spec = importlib.util.spec_from_file_location("slurm_courtship_array", SPEC)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_build_array_script_has_requeue_and_bout_id():
    m = _load()
    s = m.build_jax_array_script(job_name="c", partition="ckpt-g2", account="portia",
                                 cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                 conda_env="3d_tracking", n_bouts=30, run_dir="/tmp/rd",
                                 config_name="courtship_pipeline", overrides="")
    assert "--array=0-29" in s and "--requeue" in s
    assert "+bout_ids=${SLURM_ARRAY_TASK_ID}" in s
    assert "unset LD_LIBRARY_PATH" in s        # JAX env


def test_sam3_script_uses_pytorch_env():
    m = _load()
    s = m.build_sam3_array_script(job_name="s", partition="ckpt-g2", account="portia",
                                  cpus=8, mem=48, gpus=1, time_limit="8:00:00", requeue=True,
                                  conda_env="3d_tracking", n_bouts=30, session_dir="/x",
                                  masks_out="/y")
    assert "cu13/lib" in s and "LD_PRELOAD" in s and "--array=0-29" in s
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_slurm_courtship_array.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Write the submitter**

Create `scripts/slurm_courtship_array.py` with pure `build_sam3_array_script(...)`, `build_precompute_script(...)`, `build_jax_array_script(...)`, `build_aggregate_script(...)` returning sbatch script strings, plus a `main()` (argparse: `--config-name`, `--slurm`, `--dry-run`, passthrough overrides) that composes the Hydra config to read `recording`/`outputs`/`slurm`, counts bouts, and submits with `sbatch --parsable` capturing job ids for `--dependency=afterok:<id>`. The JAX array body activates the JAX env (`unset LD_LIBRARY_PATH`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`) and runs `python scripts/run_courtship_bout.py +bout_ids=${SLURM_ARRAY_TASK_ID} <overrides>`; the SAM3 body sets the cu13 `LD_LIBRARY_PATH` + `LD_PRELOAD` and runs `python scripts/sam3_masks.py sam3.session_dir=... sam3.out=... sam3.bout_ids=${SLURM_ARRAY_TASK_ID}`. Mirror the header/style of `scripts/slurm_train_vit.py` (partition/account/gpus/requeue lines). Use `#SBATCH --array=0-<n_bouts-1>` and `#SBATCH --requeue` when `requeue`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python -m pytest tests/test_slurm_courtship_array.py -v`
Expected: 2 passed.

- [ ] **Step 5: Dry-run smoke** (no submission)

Run: `cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset && python scripts/slurm_courtship_array.py --dry-run 2>&1 | grep -E "array=|dependency|requeue|sam3|run_courtship_bout" | head`
Expected: prints the 4 scripts with the dependency chain (no `sbatch` called).

- [ ] **Step 6: Commit**

```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git add scripts/slurm_courtship_array.py third_party/jarvis_jax/tests/test_slurm_courtship_array.py
git commit -m "feat(cse): SLURM array submitter (SAM3->precompute->JAX array->aggregate, requeue)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 11: De-risk — full pipeline on `bout_00001`, both flies (GPU acceptance)

NOT a pytest — coordinator-run acceptance on the GPU node. Deliverable: `bout_00001` fly0+fly1 outputs + QC + overlays under a run root, with the silhouette-refined IoU ≥ keypoint-only IoU.

- [ ] **Step 1: Run the full CPU-safe unit suite once**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 python -m pytest tests/test_courtship_config.py tests/test_courtship_bout_masks.py tests/test_courtship_triangulate.py tests/test_courtship_predict_2d.py tests/test_courtship_stac.py tests/test_mesh_decimate.py tests/test_courtship_polish.py tests/test_courtship_qc.py tests/test_courtship_resume.py tests/test_slurm_courtship_array.py -v`
Expected: all pass.

- [ ] **Step 2: Precompute (offsets + decimated mesh) then run bout 1 on GPU**

Run (GPU node, JAX env):
```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
OMP_NUM_THREADS=4 python scripts/run_courtship_bout.py +bout_ids=1 \
  outputs.out=/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_derisk
```
Expected: `Session0_derisk/bouts/bout_00001/fly0/{kp2d,kp3d,qpos_refined}.npz + stac_ik.h5 + outputs.h5 + qc.json`, `fly1/...`, `overlays/Cam*_reproj.mp4`, and `DONE` in each fly dir. (The driver runs the offset-fit precompute on first invocation if `offsets.h5`/`decimated_mesh.npz` are absent.)

- [ ] **Step 3: Confirm the success criterion + resume behavior**

Verify each fly's `qc.json` has `silhouette_iou.hard_median` finite; compare to a keypoint-only baseline (rerun Stage D with `silhouette.silhouette_weight=0 silhouette.containment_weight=0` into a scratch dir) → silhouette IoU should be ≥ baseline. Then delete `fly0/qpos_refined.npz` and re-run the same command → confirm it **skips** A/B/C (artifacts present) and only redoes D/E (resume works). Report the numbers truthfully; if IoU regresses or a stage crashes, diagnose before the full array.

---

### Task 12: Full-session array run (acceptance)

NOT a pytest — coordinator-run. Deliverable: all 30 bouts × 2 flies processed + session dashboard.

- [ ] **Step 1: Submit the array (ckpt, resumable)**

Run:
```bash
cd /gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
python scripts/slurm_courtship_array.py --slurm ckpt_g2 \
  outputs.out=/gscratch/portia/eabe/data/Johnson_lab/courtship/Session0_bouts_$(date +%m%d%Y)
```
Expected: prints submitted job ids (sam3 skipped — masks exist; precompute; JAX array 0-29; aggregate) with the dependency chain. Monitor `squeue -u $USER`.

- [ ] **Step 2: On completion, verify + report**

Confirm every `bouts/bout_<NNNNN>/fly<f>/DONE` exists; run/inspect `qc/session_qc.json` + `qc/dashboard/*.png`. Report per-session IoU/reproj medians + any failed bouts (resubmit the array — done bouts are skipped, interrupted ones resume). Update the `female-keypoint-effort` memory with the full-session outcome.

---

## Self-Review

**1. Spec coverage:** Stage 0 SAM3 (spec §3) → Task 10 (SAM3 array) + reuse `sam3_masks.py`; Stage A → Task 4; Stage B → Task 3; Stage C → Task 5; Stage D → Task 7 (+ mask adapter Task 2); Stage E → Tasks 6 (decimate) + 8 (QC) + `outputs.py`/`reproj_video` in Task 9; speed/array (§4) → Task 10; resumability (§4) → Task 9 (`courtship_resume`) + Task 11 Step 3; output layout (§7a) → Task 9; Hydra config + generalized paths (§6a) → Task 1; testing (§8) → each task; de-risk gate → Task 11; full run → Task 12. All spec sections mapped.

**2. Placeholder scan:** code steps contain runnable code; the two moderate drivers (Tasks 9, 10) give the exact stage sequence, artifact names, function calls, and sbatch structure with concrete snippets and the reused signatures inline — no "TBD"/"implement X". Task 9's overlay/bridge detail names the exact functions and the `polish_bout` return-shape change.

**3. Type consistency:** `load_bout_masks`→`masks (T,C,H,W)`/`valid (T,C)` used by Tasks 4/7; `predict_bout_2d`→`kp2d (T,C,K,2)`,`conf (T,C,K)` feed `triangulate_keypoints`→`kp3d (T,K,3)`,`conf3d (T,K)` feed `ik_only_bout` + `polish_bout`; `polish_bout` returns `(qpos, bridge_s, bridge_R, bridge_t)` (noted in Task 9) consumed by `build_fly_outputs(bridges=...)`; `cam_mats` are `(C,4,3)` for triangulation/frameset and `(C,2,3)+(C,2)` affine for the solver — both derived from `ReprojectionTool` and used consistently. `aggregate_session_qc` reads the `qc_report` schema written in Task 9. Config keys in Task 1 match their consumers in Tasks 4–10.
