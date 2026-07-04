# Silhouette-Bootstrapped Detector Finetuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working courtship mesh/silhouette pipeline into a label factory — reproject the fitted mesh's keypoint-sites to all cameras as multiview-consistent, quality-gated pseudo-labels, and finetune the ViTPose detector on them to fix courtship/female 2D keypoints.

**Architecture:** For each pipeline-processed bout, `outputs.h5` holds `kp3d_mm` (FK'd 50 keypoint-sites in world mm). Reproject those to each camera → 2D labels that are multiview-consistent by construction. Gate per-frame (fit quality) and per-keypoint (mesh↔detector consensus), assemble into the existing V3 COCO dataset format merged with real annotations, and continue-finetune from `v3_kp_maskaware` into a new `v4` checkpoint. Validate on the ground-truth val-split courtship recordings.

**Tech Stack:** Python, NumPy, JAX/Flax-nnx (ViTPose), MuJoCo/mjx, OpenCV (frame extraction), Hydra, Orbax (checkpoints), pytest.

## Global Constraints

- Frozen: `stac-mjx/stac_mjx/stac_core_jaxls.py` (byte-identical; do not touch).
- NEVER overwrite the `v3_kp_maskaware` checkpoint — `v4` is a NEW dir; all new artifacts additive/reversible.
- Do NOT commit checkpoints/data/label-npz/masks/outputs/videos/jpg. Commit by explicit path; never `git add -A`.
- New code under `third_party/jarvis_jax/jarvis_jax/cse/` (+ `data/` helpers); tests under `third_party/jarvis_jax/tests/`; drivers + configs at repo root; extend the existing Hydra `configs/` (do not fork).
- Heavy work on GPU compute nodes only. GPU tests (model/finetune): `cd third_party/jarvis_jax && OMP_NUM_THREADS=4 python -m pytest tests/<f> -v` (source ~/.bashrc; micromamba activate 3d_tracking; unset LD_LIBRARY_PATH). Pure-numpy tests (gating, geometry, dataset-format, config) prefix `JAX_PLATFORMS=cpu`.
- Sex-agnostic: courtship fly identity (male/female) is unreliable (sex_swaps).
- Gate thresholds config-driven with defaults; tuned later on a small check set.

## Reused interfaces (already exist — do not reimplement)

- `jarvis_jax.geometry.reprojection_tool.ReprojectionTool(calib_dir)`: `.camera_matrices` (C,4,3), `.num_cameras`, `.reproject_point(X3)->(C,2)`, `.reconstruct_point(pts (C,2), cams_to_use=list)->(3,)`.
- `outputs.h5` (per bout/fly, written by `build_fly_outputs`): `kp3d_mm` (T,50,3) FK'd sites world-mm (NaN where bridge None), `mesh_mm` (T,Kmesh,3), `kp_names`.
- `kp2d.npz` (per bout/fly): `kp2d` (T,C,50,2) full-px, `conf` (T,C,50).
- `jarvis_jax.cse.courtship_bout_masks.load_bout_masks(npz, fly) -> {masks (T,C,H,W) bool, valid (T,C), T,C,H,W}`.
- `jarvis_jax.cse.qc` / `courtship_qc`: `silhouette_iou_report(rt, mesh_mm, masks_by_cam)->{"hard":{c:..},"soft":{c:..}}`, `per_camera_reproj_error(rt, kp3d_mm, kp2d_by_cam, vis_by_cam)`, `loo_reproj(rt, kp2d_by_cam, vis_by_cam)`.
- `jarvis_jax.data.v3.V3Dataset(root, split, *, crop=448, heatmap_size=224, sigma=7.0, recordings=None)`; `V3Dataset.__getitem__ -> (img4 (448,448,4) u8, kp_xy (50,2) hm-coords, vis (50,) bool)`; `batches(ds, batch_size, *, shuffle, seed, drop_last)`; COCO at `root/annotations/instances_{split}.json` (images: id/file_name/width/height; annotations: id/image_id/bbox[x,y,w,h]/keypoints[x,y,v]*50); images at `root/<file_name>`; masks at `root/sam3_masks/{split}/<file_noext>.npz` (`masks` (N,H,W), `ann_ids` (N,), `matched` (N,) bool).
- `jarvis_jax.data.transforms.crop_origin(bbox[x,y,w,h], img_w, img_h, crop=448)->(x0,y0)`; `transform_keypoints(kps (50,3), x0, y0, crop, heatmap_size)->(hm_xy (50,2), vis (50,))`.
- `jarvis_jax.convert.build_checkpoint.load_vitpose(ckpt_dir, ViTPoseConfig())->ViTPose`; `jarvis_jax.models.vitpose.ViTPose`, `ViTPoseConfig`.
- `jarvis_jax.train.train`: `make_optimizer(model, tcfg)`, `make_train_step(mask_weight, aug, lr_swap, heatmap_size, mask_dilate)`, `eval_mpjpe(model, ds, batch_size, in_size=448)`, `TrainConfig`.
- `jarvis_jax.data.device.normalize_image`, `render_heatmaps`; `jarvis_jax.eval.mpjpe.heatmaps_to_keypoints`, `mpjpe`.
- Pipeline driver `scripts/run_courtship_bout.py`; SLURM array `scripts/slurm_courtship_array.py`; recording config `configs/recording/*.yaml`.

---

### Task 1: Per-frame QC metrics (Gate A inputs)

The QC path emits median metrics; Gate A needs **per-frame** values. Add a pure function that returns per-frame arrays and have the driver persist them next to `qc.json`.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/qc_perframe.py`
- Test: `third_party/jarvis_jax/tests/test_qc_perframe.py`
- Modify: `scripts/run_courtship_bout.py` (write `qc_perframe.npz` in Stage E, next to `qc.json`)

**Interfaces:**
- Consumes: `ReprojectionTool`, `silhouette_iou_report`, `per_camera_reproj_error` (from `jarvis_jax.cse.qc`).
- Produces: `per_frame_qc(rt, *, mesh_by_frame, kp3d_by_frame, kp2d_by_frame, vis_by_frame, masks_by_frame) -> dict{"soft_iou":(T,), "hard_iou":(T,), "reproj_px":(T,), "n_cams":(T,)}` (NaN where a frame has no usable cam).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_qc_perframe.py
import numpy as np, pytest
from jarvis_jax.cse.qc_perframe import per_frame_qc

class FakeRT:
    num_cameras = 2
    def reproject_point(self, X):            # identity-ish 2 cams
        return np.array([[X[0], X[1]], [X[0]+1, X[1]]], float)

def test_per_frame_qc_shapes_and_nan():
    T = 3
    mesh = [np.zeros((4,3)) for _ in range(T)]
    kp3d = [np.zeros((2,3)) for _ in range(T)]
    kp2d = [{0: np.zeros((2,2)), 1: np.zeros((2,2))} for _ in range(T)]
    vis  = [{0: np.ones(2,bool), 1: np.ones(2,bool)} for _ in range(T)]
    masks = [{0: np.ones((5,5),bool), 1: np.ones((5,5),bool)} for _ in range(T)]
    masks[1] = {0: None, 1: None}            # frame 1 has no masks -> NaN row
    out = per_frame_qc(FakeRT(), mesh_by_frame=mesh, kp3d_by_frame=kp3d,
                       kp2d_by_frame=kp2d, vis_by_frame=vis, masks_by_frame=masks)
    for k in ("soft_iou","hard_iou","reproj_px","n_cams"):
        assert out[k].shape == (T,)
    assert np.isnan(out["soft_iou"][1])      # no-mask frame -> NaN
    assert out["n_cams"][0] == 2
```

- [ ] **Step 2: Run it, verify it fails** — `JAX_PLATFORMS=cpu python -m pytest tests/test_qc_perframe.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# jarvis_jax/cse/qc_perframe.py
"""Per-frame QC metrics feeding the pseudo-label Gate A (soft/hard IoU, marker reproj)."""
import numpy as np
from jarvis_jax.cse.qc import silhouette_iou_report, per_camera_reproj_error


def per_frame_qc(rt, *, mesh_by_frame, kp3d_by_frame, kp2d_by_frame,
                 vis_by_frame, masks_by_frame):
    T = len(mesh_by_frame)
    soft = np.full(T, np.nan); hard = np.full(T, np.nan)
    reproj = np.full(T, np.nan); ncam = np.zeros(T, int)
    for t in range(T):
        masks_c = {c: m for c, m in masks_by_frame[t].items() if m is not None}
        ncam[t] = len(masks_c)
        if masks_c:
            rep = silhouette_iou_report(rt, mesh_by_frame[t], masks_c)  # {"hard":{c:},"soft":{c:}}
            hv = [v for v in rep["hard"].values() if v is not None]
            sv = [v for v in rep["soft"].values() if v is not None]
            if hv: hard[t] = float(np.median(hv))
            if sv: soft[t] = float(np.median(sv))
        errs = per_camera_reproj_error(rt, kp3d_by_frame[t],
                                       kp2d_by_frame[t], vis_by_frame[t])
        if errs is not None and len(errs):
            reproj[t] = float(np.median(errs))
    return {"soft_iou": soft, "hard_iou": hard, "reproj_px": reproj,
            "n_cams": ncam.astype(np.float32)}
```

If `per_camera_reproj_error` returns a per-camera dict rather than a flat list, adapt the aggregation to `np.median([e for c in errs for e in np.atleast_1d(errs[c])])`; check its actual return in `jarvis_jax/cse/qc.py` before finalizing and match it.

- [ ] **Step 4: Run tests** — `JAX_PLATFORMS=cpu python -m pytest tests/test_qc_perframe.py -v` → PASS.

- [ ] **Step 5: Wire into the driver.** In `scripts/run_courtship_bout.py` Stage E (right after `qc_report(...)` writes `qc.json`, ~line 353), add — reusing the `kp3d_by_frame/mesh_by_frame/kp2d_by_frame/vis_by_frame/masks_by_frame` locals already built there:

```python
    qc_perframe_path = os.path.join(bout_dir, f"fly{fly}", "qc_perframe.npz")
    if not stage_done(qc_perframe_path):
        from jarvis_jax.cse.qc_perframe import per_frame_qc
        pf = per_frame_qc(rt, mesh_by_frame=mesh_by_frame, kp3d_by_frame=kp3d_by_frame,
                          kp2d_by_frame=kp2d_by_frame, vis_by_frame=vis_by_frame,
                          masks_by_frame=masks_by_frame)
        atomic_save_npz(qc_perframe_path, **pf)
```
(Use the existing `atomic_save_npz`, `stage_done`. Match the exact `fly{fly}` dir construction already used for `qc_json_path`.)

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/qc_perframe.py third_party/jarvis_jax/tests/test_qc_perframe.py scripts/run_courtship_bout.py
git commit -m "feat(cse): per-frame QC metrics (Gate A inputs) + driver wiring"
```

---

### Task 2: Pseudo-label extraction + gating (`courtship_pseudolabel.py`)

Reproject `kp3d_mm` to each camera and apply Gate A (per-frame) + Gate B (per-keypoint consensus).

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/courtship_pseudolabel.py`
- Test: `third_party/jarvis_jax/tests/test_courtship_pseudolabel.py`

**Interfaces:**
- Consumes: `ReprojectionTool` (`.reproject_point`, `.num_cameras`), Task-1 `qc_perframe.npz`, `outputs.h5` `kp3d_mm`, `kp2d.npz` (`kp2d`,`conf`), `masks_dict`.
- Produces:
  - `reproject_sites(rt, kp3d_mm) -> mesh2d (T,C,K,2)` (NaN where a site is NaN).
  - `GateCfg` dataclass: `tau_iou=0.05, tau_cont=8.0, tau_reproj=30.0, consensus_px=25.0, tau_conf=0.5` (soft-IoU floor is low because the metric is a sparse-vertex splat; defaults are starting points).
  - `gate_pseudolabels(mesh2d, det_kp2d, det_conf, qc_pf, masks_valid, *, cfg) -> labels (T,C,K,3), frame_keep (T,) bool` where `labels[...,:2]` = mesh 2D (full-px), `labels[...,2]` = visibility ∈ {0,1}: 1 iff frame passes Gate A AND the site is finite AND Gate B consensus holds for that (t,c,k) AND `masks_valid[t,c]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_courtship_pseudolabel.py
import numpy as np
from jarvis_jax.cse.courtship_pseudolabel import gate_pseudolabels, GateCfg

def test_gates_frame_and_consensus():
    T,C,K = 2,2,3
    mesh2d = np.zeros((T,C,K,2), float)            # all at origin
    det = np.zeros((T,C,K,2), float)
    det[0,0,2] = [100,100]                          # kp2 in (t0,c0) far from mesh -> consensus fail
    conf = np.ones((T,C,K), float)
    conf[0,1,1] = 0.1                               # low conf -> Gate B drop
    qc_pf = {"soft_iou": np.array([0.5, 0.001]),    # frame1 fails Gate A (iou<tau)
             "reproj_px": np.array([5.0, 5.0]),
             "cont_px": np.array([1.0, 1.0])}
    mv = np.ones((T,C), bool)
    cfg = GateCfg(tau_iou=0.05, consensus_px=25.0, tau_conf=0.5, tau_reproj=30.0, tau_cont=8.0)
    labels, keep = gate_pseudolabels(mesh2d, det, conf, qc_pf, mv, cfg=cfg)
    assert keep.tolist() == [True, False]           # frame1 dropped by Gate A
    assert labels[0,0,2,2] == 0                      # consensus fail -> vis 0
    assert labels[0,1,1,2] == 0                      # low conf -> vis 0
    assert labels[0,0,0,2] == 1                      # agree + conf + frame ok -> vis 1
    assert (labels[1,:,:,2] == 0).all()             # dropped frame -> all vis 0
```

- [ ] **Step 2: Run it, verify it fails** — FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# jarvis_jax/cse/courtship_pseudolabel.py
"""Silhouette-mesh pseudo-labels: reproject FK'd sites -> 2D, gate per-frame + per-kp."""
from dataclasses import dataclass
import numpy as np


@dataclass
class GateCfg:
    tau_iou: float = 0.05        # per-frame soft-IoU floor (sparse-splat metric)
    tau_cont: float = 8.0        # per-frame containment residual ceiling (px)
    tau_reproj: float = 30.0     # per-frame marker reproj ceiling (px)
    consensus_px: float = 25.0   # Gate B: max mesh<->detector distance (px)
    tau_conf: float = 0.5        # Gate B: min detector confidence


def reproject_sites(rt, kp3d_mm):
    """kp3d_mm (T,K,3) world-mm -> mesh2d (T,C,K,2) full-px (NaN where site NaN)."""
    T, K, _ = kp3d_mm.shape
    C = rt.num_cameras
    out = np.full((T, K, C, 2), np.nan, np.float32)
    for t in range(T):
        for k in range(K):
            X = kp3d_mm[t, k]
            if np.isfinite(X).all():
                out[t, k] = np.asarray(rt.reproject_point(X))   # (C,2)
    return np.transpose(out, (0, 2, 1, 3))                       # (T,C,K,2)


def gate_pseudolabels(mesh2d, det_kp2d, det_conf, qc_pf, masks_valid, *, cfg):
    """mesh2d (T,C,K,2), det_kp2d (T,C,K,2), det_conf (T,C,K), qc_pf dict of (T,),
    masks_valid (T,C). Returns labels (T,C,K,3) [x,y,vis] and frame_keep (T,)."""
    T, C, K, _ = mesh2d.shape
    soft = qc_pf["soft_iou"]; reproj = qc_pf["reproj_px"]
    frame_keep = (np.nan_to_num(soft, nan=-1.0) >= cfg.tau_iou) \
        & (np.nan_to_num(reproj, nan=1e9) <= cfg.tau_reproj)
    if "cont_px" in qc_pf:                        # containment residual is optional
        frame_keep &= (np.nan_to_num(qc_pf["cont_px"], nan=1e9) <= cfg.tau_cont)
    labels = np.zeros((T, C, K, 3), np.float32)
    labels[..., :2] = mesh2d
    finite = np.isfinite(mesh2d).all(-1)                                # (T,C,K)
    dist = np.linalg.norm(mesh2d - det_kp2d, axis=-1)                   # (T,C,K)
    consensus = np.isfinite(dist) & (dist <= cfg.consensus_px) & (det_conf >= cfg.tau_conf)
    vis = (finite & consensus & masks_valid[:, :, None]
           & frame_keep[:, None, None])
    labels[..., 2] = vis.astype(np.float32)
    labels[~np.isfinite(labels[..., :2])] = 0.0                        # scrub NaN coords (vis already 0)
    return labels, frame_keep
```

Containment residual (`cont_px`) is an **optional** frame gate: Task 1's `qc_perframe.npz` emits `soft_iou`/`hard_iou`/`reproj_px`/`n_cams` but not `cont_px`, so in production the containment gate is skipped and Gate A = IoU + reproj. The test passes an explicit `cont_px` to exercise the optional path. This keeps Task 1 and Task 2 consistent with no extra containment plumbing.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_pseudolabel.py third_party/jarvis_jax/tests/test_courtship_pseudolabel.py
git commit -m "feat(cse): silhouette-mesh pseudo-label reprojection + per-frame/per-kp gating"
```

---

### Task 3: V3-format dataset builder (`build_pseudolabel_dataset.py`)

Turn gated labels into the V3 COCO dataset (frame jpg + SAM mask npz + COCO annotations) the trainer reads.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/build_pseudolabel_dataset.py`
- Test: `third_party/jarvis_jax/tests/test_build_pseudolabel_dataset.py`

**Interfaces:**
- Consumes: Task-2 `labels (T,C,K,3)`, per-bout `masks_dict`, the camera video files, `ReprojectionTool` (for nothing here — labels already 2D), recording camera names.
- Produces: `write_pseudolabel_coco(out_root, records, *, split="train") -> ann_path`, where each `record` = `{"file_name": "<rec>/<cam>/Frame_<f>.jpg", "img_w","img_h", "rgb": (H,W,3) u8, "mask": (H,W) bool, "keypoints": (K,3) full-px [x,y,vis], "bbox":[x,y,w,h]}`. Writes: `out_root/<file_name>` (jpg), `out_root/sam3_masks/<split>/<file_noext>.npz` (`masks`(1,H,W),`ann_ids`(1,),`matched`(1,)=True), and `out_root/annotations/instances_<split>.json` (COCO). A record with **no visible keypoints** is skipped.
- Also: `bbox_from_mask(mask) -> [x,y,w,h]` (tight bbox of True pixels).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_pseudolabel_dataset.py
import numpy as np, json, os
from jarvis_jax.cse.build_pseudolabel_dataset import write_pseudolabel_coco, bbox_from_mask
from jarvis_jax.data.v3 import V3Dataset

def test_roundtrip_loads_in_v3(tmp_path):
    H=W=64; m=np.zeros((H,W),bool); m[20:40,25:45]=True
    kp=np.zeros((50,3),float); kp[:5,0]=30; kp[:5,1]=30; kp[:5,2]=1  # 5 visible
    rec={"file_name":"recA/Cam0/Frame_1.jpg","img_w":W,"img_h":H,
         "rgb":np.zeros((H,W,3),np.uint8),"mask":m,"keypoints":kp,"bbox":bbox_from_mask(m)}
    ann=write_pseudolabel_coco(str(tmp_path),[rec],split="train")
    assert os.path.exists(ann)
    assert os.path.exists(os.path.join(tmp_path,"recA/Cam0/Frame_1.jpg"))
    assert os.path.exists(os.path.join(tmp_path,"sam3_masks/train/recA/Cam0/Frame_1.npz"))
    ds=V3Dataset(str(tmp_path),"train")
    assert len(ds)==1
    img4,kpxy,vis=ds[0]
    assert img4.shape==(448,448,4) and kpxy.shape==(50,2) and vis.shape==(50,)
    assert vis[:5].all() and not vis[5:].any()

def test_bbox_from_mask():
    m=np.zeros((10,10),bool); m[2:6,3:8]=True
    assert bbox_from_mask(m)==[3,2,5,4]
```

- [ ] **Step 2: Run it, verify it fails** — FAIL.

- [ ] **Step 3: Implement**

```python
# jarvis_jax/cse/build_pseudolabel_dataset.py
"""Assemble gated silhouette pseudo-labels into the V3 COCO dataset format."""
import os, json
import numpy as np
import cv2


def bbox_from_mask(mask):
    ys, xs = np.where(np.asarray(mask, bool))
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()),
            int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]


def write_pseudolabel_coco(out_root, records, *, split="train"):
    os.makedirs(os.path.join(out_root, "annotations"), exist_ok=True)
    images, annotations = [], []
    for i, r in enumerate(records):
        kp = np.asarray(r["keypoints"], float)
        if not (kp[:, 2] > 0).any():
            continue                                    # no visible label -> skip
        fn = r["file_name"]
        img_path = os.path.join(out_root, fn)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        cv2.imwrite(img_path, cv2.cvtColor(r["rgb"], cv2.COLOR_RGB2BGR))
        npz_path = os.path.join(out_root, "sam3_masks", split,
                                os.path.splitext(fn)[0] + ".npz")
        os.makedirs(os.path.dirname(npz_path), exist_ok=True)
        np.savez(npz_path, masks=np.asarray(r["mask"], bool)[None],
                 ann_ids=np.array([i], int), matched=np.array([True], bool))
        images.append({"id": i, "file_name": fn,
                       "width": int(r["img_w"]), "height": int(r["img_h"])})
        annotations.append({"id": i, "image_id": i, "category_id": 1,
                            "bbox": [float(x) for x in r["bbox"]],
                            "keypoints": kp.reshape(-1).tolist()})
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": 1, "name": "fly"}]}
    ann_path = os.path.join(out_root, "annotations", f"instances_{split}.json")
    with open(ann_path, "w") as f:
        json.dump(coco, f)
    return ann_path
```

Note: `V3Dataset._load_mask` matches `ann_ids == ann_id & matched`; we set `ann_ids=[i]` and the annotation `id=i`, so the mask matches. `np.savez` appends `.npz` — the path already ends in `.npz`, so it is exact.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/build_pseudolabel_dataset.py third_party/jarvis_jax/tests/test_build_pseudolabel_dataset.py
git commit -m "feat(cse): V3 COCO pseudo-label dataset builder (frame jpg + mask + annotations)"
```

---

### Task 4: Bout → records bridge (`pseudolabel_records_for_bout`)

Glue Task-2 gated labels + the source video frames + masks into Task-3 records for one bout/fly. This is the per-bout end-to-end pseudo-label producer.

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/cse/courtship_pseudolabel.py` (add `records_for_bout`)
- Test: `third_party/jarvis_jax/tests/test_courtship_pseudolabel.py` (add a case)

**Interfaces:**
- Produces: `records_for_bout(labels (T,C,K,3), masks (T,C,H,W), frames_iter, cam_names, rec_tag, start_frame) -> list[record]` where `frames_iter` yields `(t, imgs (C,H,W,3) u8)` per frame (the driver supplies it, reusing `all_cams_frames`); `rec_tag` = recording dir name; `file_name = f"{rec_tag}/{cam}/Frame_{start_frame+t}.jpg"`. One record per (t,c) with ≥1 visible keypoint; `keypoints` = `labels[t,c]`, `mask` = `masks[t,c]`, `bbox` = `bbox_from_mask(mask)`.

- [ ] **Step 1: Write the failing test**

```python
def test_records_for_bout_emits_only_visible():
    from jarvis_jax.cse.courtship_pseudolabel import records_for_bout
    T,C,K,H,W = 1,2,3,16,16
    labels=np.zeros((T,C,K,3)); labels[0,0,:,2]=1; labels[0,0,:,:2]=5  # cam0 visible
    masks=np.zeros((T,C,H,W),bool); masks[0,0,4:8,4:8]=True; masks[0,1,4:8,4:8]=True
    frames=[(0, np.zeros((C,H,W,3),np.uint8))]
    recs=records_for_bout(labels, masks, iter(frames), ["Cam0","Cam1"], "recA", 100)
    assert len(recs)==1                                   # only cam0 had visible kps
    assert recs[0]["file_name"]=="recA/Cam0/Frame_100.jpg"
    assert (np.asarray(recs[0]["keypoints"])[:,2]==1).all()
```

- [ ] **Step 2: Run it, verify it fails** — FAIL (function missing).

- [ ] **Step 3: Implement** (append to `courtship_pseudolabel.py`)

```python
from jarvis_jax.cse.build_pseudolabel_dataset import bbox_from_mask

def records_for_bout(labels, masks, frames_iter, cam_names, rec_tag, start_frame):
    T, C, K, _ = labels.shape
    recs = []
    for t, imgs in frames_iter:
        for c in range(C):
            if not (labels[t, c, :, 2] > 0).any():
                continue
            m = np.asarray(masks[t, c], bool)
            rgb = np.asarray(imgs[c])
            recs.append({
                "file_name": f"{rec_tag}/{cam_names[c]}/Frame_{start_frame + t}.jpg",
                "img_w": rgb.shape[1], "img_h": rgb.shape[0],
                "rgb": rgb, "mask": m,
                "keypoints": labels[t, c].astype(np.float32),
                "bbox": bbox_from_mask(m)})
    return recs
```

- [ ] **Step 4: Run tests** → PASS (whole `test_courtship_pseudolabel.py`).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/courtship_pseudolabel.py third_party/jarvis_jax/tests/test_courtship_pseudolabel.py
git commit -m "feat(cse): per-bout pseudo-label record builder (labels+frames+masks -> V3 records)"
```

---

### Task 5: Mixed dataset wrapper (`ConcatV3`)

Combine real + pseudo V3 datasets at a configurable ratio for the trainer's `batches()`.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/data/concat.py`
- Test: `third_party/jarvis_jax/tests/test_concat_dataset.py`

**Interfaces:**
- Produces: `ConcatV3(datasets: list, weights: list[int]|None=None)` — `__len__` = sum of `len(d)*w`; `__getitem__(i)` maps a flat index to `(dataset, local_index)` honoring integer oversample `weights` (default all 1). Exposes `.heatmap_size` (from `datasets[0]`) so `eval_mpjpe`/`batches` work unchanged. Compatible with `batches(ds, batch_size)` (needs `__len__`+`__getitem__`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_concat_dataset.py
from jarvis_jax.data.concat import ConcatV3

class Fake:
    heatmap_size=224
    def __init__(self,n,tag): self.n=n; self.tag=tag
    def __len__(self): return self.n
    def __getitem__(self,i): return (self.tag, i)

def test_concat_len_and_oversample():
    a=Fake(3,"real"); b=Fake(2,"pseudo")
    ds=ConcatV3([a,b], weights=[1,2])
    assert len(ds)==3*1+2*2                    # 7
    tags=[ds[i][0] for i in range(len(ds))]
    assert tags.count("real")==3 and tags.count("pseudo")==4
    assert ds.heatmap_size==224
```

- [ ] **Step 2: Run it, verify it fails** — FAIL.

- [ ] **Step 3: Implement**

```python
# jarvis_jax/data/concat.py
"""Concatenate V3 datasets with integer oversampling for real+pseudo mixing."""


class ConcatV3:
    def __init__(self, datasets, weights=None):
        self.datasets = list(datasets)
        self.weights = list(weights) if weights else [1] * len(self.datasets)
        assert len(self.weights) == len(self.datasets)
        self.heatmap_size = self.datasets[0].heatmap_size
        self._index = []                       # (ds_idx, local_idx)
        for di, (d, w) in enumerate(zip(self.datasets, self.weights)):
            for _ in range(int(w)):
                self._index.extend((di, li) for li in range(len(d)))

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        di, li = self._index[int(i)]
        return self.datasets[di][li]
```

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/concat.py third_party/jarvis_jax/tests/test_concat_dataset.py
git commit -m "feat(data): ConcatV3 mixed-dataset wrapper for real+pseudo finetune"
```

---

### Task 6: Finetune driver (`finetune_detector.py`)

Continue-train from `v3` on the mixed dataset, early-stop on held-out courtship MPJPE, save a NEW `v4` checkpoint.

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/cse/finetune_detector.py`
- Test: `third_party/jarvis_jax/tests/test_finetune_detector.py` (GPU smoke)

**Interfaces:**
- Consumes: `load_vitpose`, `ViTPoseConfig`, `make_optimizer`, `make_train_step`, `eval_mpjpe`, `TrainConfig`, `V3Dataset`, `ConcatV3`, `batches`, `ocp.StandardCheckpointer`.
- Produces: `finetune(*, v3_ckpt, real_root, pseudo_root, out_dir, val_recordings, tcfg, pseudo_weight=1, total_steps=..., eval_every=..., patience=...) -> dict{"best_step","best_female_mpjpe","full_val_mpjpe"}`. Loads `v3`, builds `ConcatV3([V3Dataset(real_root,"train"), V3Dataset(pseudo_root,"train")], [1, pseudo_weight])`, trains with `make_train_step`, evaluates `eval_mpjpe` on `V3Dataset(real_root,"val", recordings=val_recordings)` (held-out courtship) every `eval_every`, keeps the best, and at the end saves the best model to `out_dir` via `StandardCheckpointer` (NEVER `v3_ckpt`).

- [ ] **Step 1: Write the failing test** (GPU smoke; tiny fake dataset via Task-3 builder into `tmp_path`)

```python
# tests/test_finetune_detector.py  (GPU)
import numpy as np, os
from jarvis_jax.cse.build_pseudolabel_dataset import write_pseudolabel_coco, bbox_from_mask

def _tiny_root(tmp, split, n=2):
    recs=[]
    for i in range(n):
        H=W=200; m=np.zeros((H,W),bool); m[60:140,60:140]=True
        kp=np.zeros((50,3)); kp[:,0]=100; kp[:,1]=100; kp[:,2]=1
        recs.append({"file_name":f"rec/Cam/Frame_{i}.jpg","img_w":W,"img_h":H,
                     "rgb":(np.random.rand(H,W,3)*255).astype(np.uint8),
                     "mask":m,"keypoints":kp,"bbox":bbox_from_mask(m)})
    write_pseudolabel_coco(tmp, recs, split=split)

def test_finetune_smoke(tmp_path):
    from jarvis_jax.cse.finetune_detector import finetune
    real=str(tmp_path/"real"); pseudo=str(tmp_path/"pseudo"); out=str(tmp_path/"v4")
    os.makedirs(real); os.makedirs(pseudo)
    _tiny_root(real,"train"); _tiny_root(real,"val"); _tiny_root(pseudo,"train")
    r=finetune(v3_ckpt=os.environ["V3_CKPT"], real_root=real, pseudo_root=pseudo,
               out_dir=out, val_recordings=["rec"], total_steps=2, eval_every=2,
               patience=1, batch_size=2)
    assert os.path.exists(out) and "best_female_mpjpe" in r
```

Run with `V3_CKPT=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v3_kp_maskaware/final`.

- [ ] **Step 2: Run it, verify it fails** — FAIL (module missing).

- [ ] **Step 3: Implement** (mirror `scripts/train_keypoints.py`'s `train()` loop, but load `v3` + mixed ds + early-stop + save-best; reuse `make_optimizer/make_train_step/eval_mpjpe`).

```python
# jarvis_jax/cse/finetune_detector.py
"""Finetune the ViTPose detector from v3 on real+pseudo mix; save a NEW v4 ckpt."""
import jax, numpy as np
from flax import nnx
import orbax.checkpoint as ocp
from jarvis_jax.convert.build_checkpoint import load_vitpose
from jarvis_jax.models.vitpose import ViTPoseConfig
from jarvis_jax.data.v3 import V3Dataset, batches
from jarvis_jax.data.concat import ConcatV3
from jarvis_jax.train.train import make_optimizer, make_train_step, eval_mpjpe, TrainConfig


def finetune(*, v3_ckpt, real_root, pseudo_root, out_dir, val_recordings,
             total_steps=4000, eval_every=250, patience=6, batch_size=None,
             pseudo_weight=1, lr=2e-5, seed=0):
    model = load_vitpose(v3_ckpt, ViTPoseConfig())
    tcfg = TrainConfig(lr=lr, total_steps=total_steps, batch_size=batch_size or 8, seed=seed)
    opt = make_optimizer(model, tcfg)
    train_ds = ConcatV3([V3Dataset(real_root, "train"), V3Dataset(pseudo_root, "train")],
                        [1, pseudo_weight])
    fem_ds = V3Dataset(real_root, "val", recordings=list(val_recordings))
    full_ds = V3Dataset(real_root, "val")
    step = make_train_step(tcfg.mask_weight, None, None,
                           heatmap_size=model.cfg.heatmap_size if hasattr(model, "cfg") else 224)
    key = jax.random.PRNGKey(seed)
    best = {"best_step": -1, "best_female_mpjpe": float("inf"), "full_val_mpjpe": float("nan")}
    best_state = None
    it = _epochs(train_ds, tcfg.batch_size, seed)
    bad = 0
    for i in range(total_steps):
        img4, kpxy, vis = next(it)
        step(model, opt, jax.random.fold_in(key, i), img4, kpxy, vis)
        if (i + 1) % eval_every == 0:
            fem = float(eval_mpjpe(model, fem_ds, tcfg.batch_size))
            if fem < best["best_female_mpjpe"]:
                best.update(best_step=i + 1, best_female_mpjpe=fem)
                best_state = nnx.split(model)[1]
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is None:
        best_state = nnx.split(model)[1]
    best["full_val_mpjpe"] = float(eval_mpjpe(model, full_ds, tcfg.batch_size))
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(out_dir, best_state, force=True); ckptr.wait_until_finished()
    return best


def _epochs(ds, bs, seed):
    ep = 0
    while True:
        yield from batches(ds, bs, shuffle=True, seed=seed + ep); ep += 1
```

Before finalizing: open `scripts/train_keypoints.py` and `jarvis_jax/train/train.py` and match `TrainConfig` field names (`lr`, `total_steps`, `batch_size`, `seed`, `mask_weight`, `weight_decay`, warmup) and the `make_train_step` signature EXACTLY (adjust the call if fields differ). `nnx.split(model)[1]` is the saved state (matches `train_keypoints.py`'s `ckptr.save(out_dir, nnx.split(model)[1])`). Deep-copy `best_state` if `nnx.split` returns a live reference (use `jax.tree.map(lambda x: x.copy(), state)` if needed so a later step doesn't mutate the saved best).

- [ ] **Step 4: Run the GPU smoke test** — `V3_CKPT=... OMP_NUM_THREADS=4 python -m pytest tests/test_finetune_detector.py -v` → PASS (writes a checkpoint to tmp).

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/cse/finetune_detector.py third_party/jarvis_jax/tests/test_finetune_detector.py
git commit -m "feat(cse): detector finetune driver (v3->v4, real+pseudo mix, early-stop on held-out)"
```

---

### Task 7: Orchestration + config + validation (`run_pseudolabel_finetune.py`, configs)

End-to-end driver + Hydra config that: (a) reads which bouts/recordings have pipeline outputs, (b) builds gated pseudo-labels for each, (c) writes the merged dataset, (d) calls `finetune`, (e) prints the three-level validation. Ties Tasks 1–6 together and adds the config group.

**Files:**
- Create: `scripts/run_pseudolabel_finetune.py`
- Create: `configs/detector_finetune.yaml` (+ reuse `configs/paths`, `configs/recording`)
- Test: `third_party/jarvis_jax/tests/test_detector_finetune_config.py` (config composes + resolves)

**Interfaces:**
- Consumes: all prior tasks; `all_cams_frames`/`bout_start_frame`/`parse_bouts` from `scripts/run_courtship_bout.py`; `slurm_courtship_array.py` (pseudo-label generation = the pipeline run, already resumable).
- Config `configs/detector_finetune.yaml`:

```yaml
defaults:
  - _self_
  - paths: hyak
  - recording: session0

v3_ckpt: ${paths.vit_runs_root}/v3_kp_maskaware/final
out_dir: ${paths.vit_runs_root}/v4_kp_silbootstrap/final
real_root: ${paths.red_data_v3_root}      # real annotations mixed in (add red_data_v3_root to paths/hyak.yaml)
pseudo_root: ${paths.data_dir_johnson}/courtship/pseudolabels
# recordings whose pipeline outputs feed pseudo-labels (train-split courtship + Session0)
label_sources: []            # list of run-root dirs with bouts/bout_*/fly*/outputs.h5
# held-out courtship val recordings (have GT) — never pseudo-labeled
val_recordings: [2026_05_27_11_56_05, 2026_04_07_11_33_33]
gate:
  tau_iou: 0.05
  tau_cont: 8.0
  tau_reproj: 30.0
  consensus_px: 25.0
  tau_conf: 0.5
finetune:
  total_steps: 4000
  eval_every: 250
  patience: 6
  batch_size: 8
  pseudo_weight: 1
  lr: 2.0e-5
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_detector_finetune_config.py
import os
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf

CFG="/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/configs"

def test_detector_finetune_config_resolves():
    os.environ.setdefault("USER","eabe")
    import stac_mjx  # register resolvers
    with initialize_config_dir(version_base=None, config_dir=CFG):
        c=compose(config_name="detector_finetune")
        d=OmegaConf.to_container(c, resolve=True)
    assert d["gate"]["consensus_px"]==25.0
    assert d["val_recordings"]
    assert d["out_dir"].endswith("v4_kp_silbootstrap/final")
    assert d["real_root"].endswith("red_data_unified_V3")   # resolves via paths.red_data_v3_root
```

- [ ] **Step 2: Run it, verify it fails** — FAIL (config missing).

- [ ] **Step 3: Implement the config** (above) and the driver `scripts/run_pseudolabel_finetune.py`:

```python
#!/usr/bin/env python3
"""End-to-end: gated silhouette pseudo-labels from pipeline outputs -> finetune v3->v4."""
import os, glob
import numpy as np
import hydra
from omegaconf import OmegaConf

import stac_mjx.io_dict_to_hdf5 as ioh5
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
from jarvis_jax.cse.courtship_bout_masks import load_bout_masks
from jarvis_jax.cse.courtship_pseudolabel import (
    reproject_sites, gate_pseudolabels, records_for_bout, GateCfg)
from jarvis_jax.cse.build_pseudolabel_dataset import write_pseudolabel_coco
from jarvis_jax.cse.finetune_detector import finetune
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_courtship_bout import all_cams_frames, open_video_captures, bout_start_frame


@hydra.main(version_base=None, config_path="../configs", config_name="detector_finetune")
def main(cfg):
    rt = ReprojectionTool(cfg.recording.calib_dir)
    cams = list(cfg.recording.cameras)
    gcfg = GateCfg(**OmegaConf.to_container(cfg.gate, resolve=True))
    all_records = []
    for src in cfg.label_sources:                          # each src = a run-root
        for fly_dir in sorted(glob.glob(os.path.join(src, "bouts", "bout_*", "fly*"))):
            out_h5 = os.path.join(fly_dir, "outputs.h5")
            kp2d_npz = os.path.join(fly_dir, "kp2d.npz")
            qc_pf = os.path.join(fly_dir, "qc_perframe.npz")
            if not (os.path.exists(out_h5) and os.path.exists(kp2d_npz) and os.path.exists(qc_pf)):
                continue
            kp3d_mm = np.asarray(ioh5.load(out_h5)["kp3d_mm"])
            z = np.load(kp2d_npz); det, conf = z["kp2d"], z["conf"]
            pf = dict(np.load(qc_pf))
            bout_idx = int(os.path.basename(os.path.dirname(fly_dir)).split("_")[-1])
            fly = int(os.path.basename(fly_dir).replace("fly", ""))
            masks = load_bout_masks(_mask_npz(cfg, src, bout_idx), fly)
            mesh2d = reproject_sites(rt, kp3d_mm)
            labels, _ = gate_pseudolabels(mesh2d, det, conf, pf, masks["valid"], cfg=gcfg)
            start = bout_start_frame(cfg, bout_idx)
            caps = open_video_captures(str(cfg.recording.session_dir), cams)
            frames = ((t, f) for t, f in enumerate(all_cams_frames(caps, start, kp3d_mm.shape[0])))
            rec_tag = os.path.basename(str(cfg.recording.session_dir))
            all_records += records_for_bout(labels, masks["masks"], frames, cams, rec_tag, start)
    write_pseudolabel_coco(cfg.pseudo_root, all_records, split="train")
    res = finetune(v3_ckpt=cfg.v3_ckpt, real_root=cfg.real_root,
                   pseudo_root=cfg.pseudo_root, out_dir=cfg.out_dir,
                   val_recordings=list(cfg.val_recordings),
                   **OmegaConf.to_container(cfg.finetune, resolve=True))
    print("FINETUNE RESULT:", res)


def _mask_npz(cfg, src, bout_idx):
    return os.path.join(cfg.recording.predictions_dir, f"bout_{bout_idx:05d}", "sam3_masks.npz")


if __name__ == "__main__":
    main()
```

Add `red_data_v3_root: ${paths.data_dir_johnson}/red_data/red_data_unified_V3` to `configs/paths/hyak.yaml` if absent (it is the real-annotation root the finetune mixes in). Reuse `all_cams_frames`/`open_video_captures`/`bout_start_frame` exactly as defined in `run_courtship_bout.py`; if their signatures differ, adapt the call.

- [ ] **Step 4: Run tests** — `JAX_PLATFORMS=cpu python -m pytest tests/test_detector_finetune_config.py -v` → PASS. Also `python -c "import ast; ast.parse(open('scripts/run_pseudolabel_finetune.py').read())"`.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_pseudolabel_finetune.py configs/detector_finetune.yaml third_party/jarvis_jax/tests/test_detector_finetune_config.py configs/paths/hyak.yaml
git commit -m "feat: pseudo-label finetune orchestration + config (v3->v4 bootstrap)"
```

---

## Coordinator acceptance (after Tasks 1–7, run by the coordinator on GPU)

Not a task — the coordinator executes these to validate end-to-end:

1. **Pseudo-label generation = the deferred full run.** Ensure pipeline outputs (+ `qc_perframe.npz` from Task 1) exist for the train-split courtship recordings + Session0 by running `scripts/slurm_courtship_array.py` (resumable). This is the expensive step; it doubles as the full-session run.
2. **Bootstrap round 1.** Run `scripts/run_pseudolabel_finetune.py label_sources=[...]` on GPU → `v4` checkpoint + printed validation.
3. **Adoption gate (spec §Validation):** accept `v4` only if held-out courtship female MPJPE improves vs. v3's 26px AND full-val MPJPE stays ~6–7px AND a de-risk bout re-run with `v4` (point `configs/detector/*.yaml` at `v4`) beats the recorded v3 baseline (fly0 soft-IoU 0.102, per-cam reproj 55px, LOO 46px). Record numbers in the ledger.
4. **Optional round 2:** if round 1 improves held-out LOO, refit the pipeline with `v4`, regenerate gated pseudo-labels, finetune again; stop when held-out stops improving or after 2 rounds.
