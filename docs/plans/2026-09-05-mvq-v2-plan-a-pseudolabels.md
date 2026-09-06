# MVQ v2 — Plan A: gated pseudo-labels from the P3b campaign

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the masked P3b campaign (160 courtship bouts / 11 recordings), the two single-fly sessions and the coarse pass's empty windows into a human-reviewed, v12-format pseudo-label export (`red_data_3d_v12_pseudo_p3b_20260905/`) that Plan B's loader can read beside the real v12 root: ~10,000 courtship framesets (each carrying its T=2 partners at Delta 1/4/16), ~2,000 single-fly framesets, ~2,000 empty-window existence negatives, every frame passing the spec §3.2 admission gates and a ~300-frameset accept/reject review.

**Architecture:** Gates are pure numpy over ONE bout's on-disk arrays (`jarvis_jax/data/pseudo_gates.py`) — no model, no GPU, so they are unit-testable on a fake run root. A writer module (`jarvis_jax/data/pseudo_export.py`) turns admitted (recording, frame, fly) records into the v12 on-disk layout (`annotations/instances_*.json` + `images/` JPEGs + `masks/` npz + `calibrations/<group>/` + `manifest.json`), permuting the keypoint axis from the campaign's MODEL order back into the export's own order BY NAME. Three CLIs sit on top: the extractor (walk + gate + stratify + write), the review gallery (+ `--apply-review`), and the empty-window sampler. The single-fly wrapper produces P3b-shaped bout dirs so the extractor runs on them unchanged.

**Tech Stack:** numpy, OpenCV (`cv2`), PIL, matplotlib (Agg), JAX 0.11 only where `MVQRunner` is invoked (single-fly pass); pytest from `third_party/jarvis_jax/` with `JAX_PLATFORMS=cpu`.

**Spec:** `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` — §3.2 gates and target size, §3.3 the review gate, §3.4 single-fly, §3.5 negatives, §6 items 1-3. Read it first. Facts it rests on: `docs/benchmark/2026-09-mvq/{p3b-notes.md, p4-maskfree-notes.md, gate-bouts-probe-2026-09-05.md}`.

## Global Constraints

- Package root `third_party/jarvis_jax/jarvis_jax/`; tests in `third_party/jarvis_jax/tests/` (bare `from pseudo_fixtures import ...`, no `tests/__init__.py`); repo-root CLIs under `scripts/pseudo_labels/`. Run pytest from `third_party/jarvis_jax/` with `JAX_PLATFORMS=cpu`.
- **World units are 0.1 mm** (`MM_PER_UNIT = 0.1`). Spec thresholds in mm become: per-keypoint step `<= 0.3 mm` = **3.0 units**; contact `< 1.5 mm` = **15 units**; empty-window clearance `>= 6 mm` = **60 units**.
- Admission gates (spec §3.2), all per frame, encoded ONCE in `GateThresholds`: `exist >= 0.8` per present fly; `identity_source == "mask"`, zero containment drops on the frame, `collapsed == False`; per-keypoint step `<= 3.0 units` against BOTH neighbours; per-view median (over keypoints) 2D-head-vs-reprojected-3D `<= 3 px` in `>= 5` cameras; mask containment (reprojected keypoints inside the own mask, dilated 6 px) `>= 0.9` in `>= 5` cameras; `>= 16` frames from any other admitted ANCHOR of the same bout unless the frame is a T=2 partner.
- Target composition (spec §3.2): ~10,000 courtship framesets, 50 % female-host / 50 % male-host, `>= 25 %` contact (inter-fly centroid `< 15 units`) and `>= 25 %` separated, all 11 recordings present, no bout `> 2 %` of the set, wall frames at their pool proportion. Weight **0.3** (real labels 1.0). Partners at Delta in **{1, 4, 16}**.
- **Never index a keypoint or camera axis by integer.** Campaign npz files are in `configs/anatomy/v1.yaml` `model.KP_NAMES` order (`lift_mvq.to_pipeline` permuted them); the v12 export's own order is `annotations/keypoint_names.json` (the detector order) — the pseudo export MUST be written in the export order, permuted with `detector_to_model_perm(written_model_names, export_names)`. Cameras come from each npz's own `cameras` array and are matched BY NAME against `configs/recording/session0.yaml` `recording.cameras`.
- Figures under `figures/2026-09-mvq/v2_pseudo/` (gitignored), JSON beside each PNG, the expectation in the generating script's docstring, and **every PNG read back with the Read tool before any claim**. Notes and small JSON in `docs/benchmark/2026-09-mvq/v2-pseudolabel-notes.md` (force-add: the directory is gitignored).
- Commits: only the task's files (`git add <paths>`, never `-A`), trailers `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01Npo4HiYC4t5M2xYjKUJsFP`.
- Compute: gating, sampling and JPEG extraction are CPU; the single-fly pass (Task 4) needs a GPU — run directly if the node's GPUs are idle, else `scripts/slurm/submit_task.sh` (ckpt-all). Never on the login node.

## File map

| file | responsibility |
|---|---|
| `jarvis_jax/data/pseudo_gates.py` (new) | `GateThresholds`, per-bout gate functions, `admit_bout`, decorrelation, partner masks |
| `jarvis_jax/data/pseudo_export.py` (new) | `PseudoRecord`, `write_pseudo_export`, keypoint/camera permutation, manifest/frameset schema |
| `scripts/pseudo_labels/extract_p3b_pseudolabels.py` (new) | walk run roots -> gates -> stratified sample -> export + `gate_histograms.json` |
| `scripts/pseudo_labels/pseudolabel_gallery.py` (new) | ~300-frameset gallery + `review.csv` + `--apply-review` |
| `scripts/pseudo_labels/singlefly_p3b_pass.py` (new) | mask-free P3b pass over free_running Session11 / Clip Session6 -> P3b-shaped bout dirs |
| `scripts/pseudo_labels/extract_empty_windows.py` (new) | ~2,000 empty-window negatives from coarse tracks |
| `jarvis_jax/data/v12_windows.py` | expose `source(i)` / `weight(i)`; `sample_weight` in `WINDOW_KEYS` |
| tests | `pseudo_fixtures.py`, `test_pseudo_gates.py`, `test_pseudo_export.py`, `test_extract_pseudolabels.py`, `test_pseudolabel_gallery.py`, `test_empty_windows.py`, `test_v12_windows.py` (+2) |

---

### Task 1: `pseudo_gates.py` — the admission gates over one bout

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/data/pseudo_gates.py`
- Create: `third_party/jarvis_jax/tests/pseudo_fixtures.py`
- Create: `third_party/jarvis_jax/tests/test_pseudo_gates.py`

**Interfaces:**
- Produces: `GateThresholds` (frozen dataclass, fields below); `BoutArrays` (NamedTuple `kp3d (F,T,K,3)`, `kp2d (F,T,C,K,2)`, `conf (F,T,C,K)`, `kp_names`, `cameras`, `meta`); `load_bout_arrays(bout_dir, *, cameras, n_flies=2) -> BoutArrays`; `existence_gate`, `identity_gate`, `step_gate`, `reprojection_gate`, `containment_gate`, `decorrelate`, `partner_masks`; `admit_bout(arrays, store, cam_mats, thr, *, use_identity=True) -> GateResult` with `GateResult(frame (T,) bool, fly (F,T) bool, reasons dict, quant dict, partners dict[int, (T,) bool])`.
- Consumes: `jarvis_jax.tracking.lift_mvq.{BoutMaskStore, project_points, _inside_mask_dilated, _disk_offsets}`.

- [ ] **Step 1: Fixture — a tiny fake P3b run root** (`tests/pseudo_fixtures.py`)

```python
"""A miniature `pose_mvq_p3b/bouts/bout_XXXXX/` tree plus its SAM3 mask npz,
small enough for CPU tests: 2 flies, 3 cameras, T frames, 6 keypoints, an
80x120 frame. Geometry is exact: kp2d is the projection of kp3d through
`cam_mats`, so the reprojection gate passes unless a test perturbs it."""
import json
import numpy as np

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853"]
KP_NAMES = ["Antenna_Base", "EyeL", "EyeR", "Scutellum", "Abd_tip", "WingL_base"]
H, W = 80, 120


def cam_mats(n_cam=3):
    """(C,4,3) `ReprojectionTool.camera_matrices` convention (p_h @ M)."""
    out = np.zeros((n_cam, 4, 3), np.float64)
    for c in range(n_cam):
        th = 2 * np.pi * c / n_cam
        out[c, :3, 0] = [2.0 * np.cos(th), -2.0 * np.sin(th), 0.0]
        out[c, :3, 1] = [0.0, 0.0, -2.0]
        out[c, 3] = [60.0, 40.0, 1.0]
    return out


def project(cm, xyz):
    from jarvis_jax.tracking.lift_mvq import project_points
    return project_points(cm, xyz)                       # (C,K,2)


def make_bout(tmp_path, *, T=40, sep_units=40.0, name="bout_00004", exist=0.95,
              identity="mask", drop_frames=(), spike_frames=()):
    """Returns (bout_dir, mask_npz_path, cam_mats). fly0 sits at the origin,
    fly1 `sep_units` away in +x; both drift 0.2 units/frame."""
    cm = cam_mats()
    root = tmp_path / "rec" / "pose_mvq_p3b" / "bouts" / name
    K, C = len(KP_NAMES), len(CAMS)
    kp3d = np.zeros((2, T, K, 3), np.float32)
    rng = np.random.default_rng(0)
    body = rng.normal(size=(K, 3)) * np.array([3.0, 1.5, 1.0])
    for f in range(2):
        base = np.array([f * sep_units, 0.0, 0.0])
        for t in range(T):
            kp3d[f, t] = body + base + np.array([0.2 * t, 0.0, 0.0])
    for t in spike_frames:                                # 8 units in one frame = 0.8 mm
        kp3d[1, t, 0] += np.array([8.0, 0.0, 0.0])
    kp2d = np.zeros((2, T, C, K, 2), np.float32)
    for f in range(2):
        for t in range(T):
            kp2d[f, t] = project(cm, kp3d[f, t])
    conf = np.full((2, T, C, K), 0.9, np.float32)
    per_frame = {
        "exist": [[exist] * T, [exist] * T],
        "collapsed": [0] * T,
        "slot_used": [[1] * T, [2] * T],
        "identity_source": [identity] * T,
        "assign_reason": [["typed_preferred"] * T, ["typed_preferred"] * T],
    }
    for t in drop_frames:                                  # a containment drop on fly1
        per_frame["exist"][0][t] = 0.2
    meta = {"n_frames": T, "identity_resolved": identity, "containment": True,
            "keypoint_names_written": KP_NAMES, "cameras": CAMS,
            "checkpoint": "/fake/final", "step": "final", "frame_start": 1000,
            "bout": int(name.split("_")[-1]), "session_dir": str(tmp_path / "video"),
            "containment_report": {"enabled": True, "per_frame": {
                "n_dropped": [[0] * T, [len(drop_frames) and 1 or 0] * T]}},
            "per_frame": per_frame}
    for f in range(2):
        d = root / f"fly{f}"; d.mkdir(parents=True, exist_ok=True)
        np.savez(d / "kp3d.npz", kp3d=kp3d[f], conf3d=conf[f].mean(1),
                 conf3d_mvq_raw=conf[f].mean(1), kp_names=np.array(KP_NAMES),
                 gates=np.asarray("{}"))
        np.savez(d / "kp2d.npz", kp2d=kp2d[f], conf=conf[f],
                 cameras=np.array(CAMS), kp_names=np.array(KP_NAMES))
    (root / "mvq_meta.json").write_text(json.dumps(meta))
    npz = make_masks(tmp_path, kp3d, cm, T)
    return str(root), npz, cm


def make_masks(tmp_path, kp3d, cm, T):
    """A `BoutMaskStore`-readable npz: a filled box around each fly's
    reprojected keypoints in every camera."""
    F, _, K, _ = kp3d.shape; C = len(CAMS)
    full = np.zeros((F, C, T, H, W), bool)
    cent = np.zeros((F, C, T, 2), np.float32)
    for f in range(F):
        for t in range(T):
            uv = project(cm, kp3d[f, t])
            for c in range(C):
                x0, y0 = np.nanmin(uv[c], 0) - 4; x1, y1 = np.nanmax(uv[c], 0) + 4
                xs = slice(max(int(x0), 0), min(int(x1) + 1, W))
                ys = slice(max(int(y0), 0), min(int(y1) + 1, H))
                full[f, c, t, ys, xs] = True
                cent[f, c, t] = np.nanmean(uv[c], 0)
    p = tmp_path / "sam3_masks.npz"
    np.savez(p, packed=np.packbits(full, axis=-1), valid=np.ones((F, C, T), bool),
             centroids=cent, shape=np.array([H, W], np.int32), cameras=np.array(CAMS))
    return str(p)
```

- [ ] **Step 2: Write the failing tests** (`tests/test_pseudo_gates.py`)

```python
import numpy as np
import pytest
from pseudo_fixtures import CAMS, make_bout


def _gates(tmp_path, **kw):
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, **kw)
    a = load_bout_arrays(d, cameras=CAMS)
    return admit_bout(a, BoutMaskStore(npz, CAMS), cm, GateThresholds()), a


def test_clean_bout_admits_every_frame_except_the_decorrelation_thinning(tmp_path):
    r, a = _gates(tmp_path, T=40)
    assert r.fly.all()                                   # both flies pass every per-fly gate
    # decorrelation keeps one anchor every 16 frames: frames 0, 16, 32
    assert np.flatnonzero(r.frame).tolist() == [0, 16, 32]
    assert set(r.partners) == {1, 4, 16}
    assert r.partners[1][0] and r.partners[4][0] and r.partners[16][0]


def test_low_existence_frame_is_a_negative_for_that_slot_not_a_guess(tmp_path):
    r, _ = _gates(tmp_path, T=20, drop_frames=(5,))
    assert not r.fly[0, 5] and r.fly[1, 5]               # fly0 absent, fly1 still admitted
    assert r.reasons["exist"][0, 5]
    assert 5 not in np.flatnonzero(r.frame).tolist() or r.frame[5]   # frame usable, fly0 absent


def test_step_spike_rejects_both_neighbouring_frames(tmp_path):
    r, _ = _gates(tmp_path, T=20, spike_frames=(9,))
    assert not r.fly[1, 8] and not r.fly[1, 9]
    assert r.fly[1, 11]
    assert r.quant["step_units"][1, 9] > 3.0


def test_identity_gate_rejects_a_sex_head_bout(tmp_path):
    r, _ = _gates(tmp_path, T=20, identity="sex")
    assert not r.frame.any() and r.reasons["identity"].all()


def test_reprojection_gate_fires_when_the_2d_head_disagrees(tmp_path):
    from jarvis_jax.data.pseudo_gates import GateThresholds, reprojection_gate
    from pseudo_fixtures import cam_mats
    d, npz, cm = make_bout(tmp_path, T=8)
    from jarvis_jax.data.pseudo_gates import load_bout_arrays
    a = load_bout_arrays(d, cameras=CAMS)
    ok, med = reprojection_gate(a.kp2d, a.kp3d, cm, a.conf, GateThresholds())
    assert ok.all() and float(np.nanmax(med)) < 1e-3
    bad = a.kp2d.copy(); bad[1, 3] += 10.0                # 10 px in every view
    ok2, med2 = reprojection_gate(bad, a.kp3d, cm, a.conf, GateThresholds())
    assert not ok2[1, 3] and ok2[0, 3]


def test_containment_gate_fires_when_the_pose_leaves_its_own_mask(tmp_path):
    from jarvis_jax.data.pseudo_gates import GateThresholds, containment_gate, load_bout_arrays
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, T=6)
    a = load_bout_arrays(d, cameras=CAMS)
    store = BoutMaskStore(npz, CAMS)
    ok, frac = containment_gate(a.kp3d, store, cm, GateThresholds())
    assert ok.all() and frac.min() >= 0.9
    moved = a.kp3d.copy(); moved[0, 2] += np.array([25.0, 0.0, 0.0])   # off its own mask
    ok2, frac2 = containment_gate(moved, store, cm, GateThresholds())
    assert not ok2[0, 2] and frac2[0, 2] < 0.9


def test_decorrelate_protects_partners(tmp_path):
    from jarvis_jax.data.pseudo_gates import decorrelate
    admit = np.ones(40, bool)
    keep = decorrelate(admit, 16)
    assert np.flatnonzero(keep).tolist() == [0, 16, 32]
    assert np.diff(np.flatnonzero(keep)).min() >= 16


def test_partner_masks_need_the_partner_frame_to_pass_too(tmp_path):
    from jarvis_jax.data.pseudo_gates import partner_masks
    admit = np.ones(20, bool); admit[5] = False
    p = partner_masks(admit, (1, 4, 16))
    assert not p[1][4] and p[1][6] is np.True_ or p[1][6]
    assert not p[4][1] and p[16][3] is np.False_ or not p[16][3]
```

- [ ] **Step 3: Run to verify they fail** — `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu pytest tests/test_pseudo_gates.py -q` -> `ModuleNotFoundError: jarvis_jax.data.pseudo_gates`.

- [ ] **Step 4: Implement `pseudo_gates.py`**

```python
"""Admission gates for P3b pseudo-labels (spec 2026-09-05 §3.2). Pure numpy
over ONE bout's on-disk arrays: no model, no GPU, so the whole gate stack is
unit-testable. Thresholds live in `GateThresholds` and nowhere else -- the
extractor, the gallery and the notes all read them from the same object, and
the resolved values are written into the export manifest's `gates` block."""
from __future__ import annotations

import dataclasses
import json
import os
from typing import NamedTuple

import numpy as np

MM_PER_UNIT = 0.1


@dataclasses.dataclass(frozen=True)
class GateThresholds:
    exist_min: float = 0.8            # spec §3.2, per present fly
    max_step_units: float = 3.0       # 0.3 mm per keypoint per frame
    reproj_px: float = 3.0            # per-view median over keypoints
    reproj_min_views: int = 5
    contain_frac: float = 0.9
    contain_min_cams: int = 5
    contain_dilate_px: float = 6.0
    decorrelation: int = 16
    deltas: tuple = (1, 4, 16)
    conf_min: float = 0.5             # per-view visibility below this is not scored


class BoutArrays(NamedTuple):
    kp3d: np.ndarray        # (F,T,K,3) MODEL order (what lift_mvq wrote)
    kp2d: np.ndarray        # (F,T,C,K,2) full-frame px, canonical camera order
    conf: np.ndarray        # (F,T,C,K) per-view visibility
    kp_names: list
    cameras: list
    meta: dict


class GateResult(NamedTuple):
    frame: np.ndarray                  # (T,) anchor frames after decorrelation
    fly: np.ndarray                    # (F,T) per-fly admission (False = absent/rejected)
    reasons: dict                      # gate name -> bool array, True = THIS gate rejected
    quant: dict                        # gate name -> the raw quantity, for histograms
    partners: dict                     # delta -> (T,) bool, both frames admitted


def load_bout_arrays(bout_dir, *, cameras, n_flies=2):
    """Read `fly{0..n}/kp{2,3}d.npz` + `mvq_meta.json`, checking the keypoint
    axis by NAME across flies and the camera axis by NAME against `cameras`."""
    kp3d, kp2d, conf, names = [], [], [], None
    for f in range(n_flies):
        z3 = np.load(os.path.join(bout_dir, f"fly{f}", "kp3d.npz"), allow_pickle=True)
        z2 = np.load(os.path.join(bout_dir, f"fly{f}", "kp2d.npz"), allow_pickle=True)
        n3 = [str(s) for s in z3["kp_names"]]
        if names is None:
            names = n3
        elif n3 != names:
            raise ValueError(f"{bout_dir} fly{f}: kp_names differ between flies")
        cam = [str(c) for c in z2["cameras"]]
        if cam != [str(c) for c in cameras]:
            raise ValueError(f"{bout_dir} fly{f}: kp2d camera order {cam} != canonical {list(cameras)}")
        kp3d.append(z3["kp3d"]); kp2d.append(z2["kp2d"]); conf.append(z2["conf"])
    meta = json.load(open(os.path.join(bout_dir, "mvq_meta.json")))
    return BoutArrays(np.stack(kp3d), np.stack(kp2d), np.stack(conf),
                      names, [str(c) for c in cameras], meta)


def existence_gate(meta, thr):
    e = np.asarray(meta["per_frame"]["exist"], np.float32)          # (F,T), -1 = no instance
    return e >= thr.exist_min, e


def identity_gate(meta, thr):
    """(T,) True where the frame is REJECTED: not mask identity, collapsed, or
    any containment drop. `identity_source` is per frame in `mvq_meta.json`."""
    pf = meta["per_frame"]; T = int(meta["n_frames"])
    src = np.array([s == "mask" for s in pf["identity_source"]], bool)
    collapsed = np.asarray(pf["collapsed"], bool)
    rep = (meta.get("containment_report") or {}).get("per_frame") or {}
    drops = np.asarray(rep.get("n_dropped", np.zeros((2, T)))).sum(0) > 0
    return ~(src & ~collapsed & ~drops)


def step_gate(kp3d, thr):
    """(F,T) admitted, and (F,T) the max per-keypoint step in units. A frame is
    rejected if EITHER neighbour step exceeds the threshold (a spike is bad in
    both frames of the pair it appears in)."""
    d = np.linalg.norm(np.diff(kp3d, axis=1), axis=-1)             # (F,T-1,K)
    with np.errstate(invalid="ignore"):
        s = np.nanmax(d, axis=-1)
    prev = np.concatenate([np.zeros((kp3d.shape[0], 1)), s], 1)
    nxt = np.concatenate([s, np.zeros((kp3d.shape[0], 1))], 1)
    worst = np.fmax(prev, nxt)
    return worst <= thr.max_step_units, worst


def reprojection_gate(kp2d, kp3d, cam_mats, conf, thr):
    """(F,T) admitted and (F,T,C) per-view median |2D head - reprojected 3D| px."""
    from jarvis_jax.tracking.lift_mvq import project_points
    F, T, C, K, _ = kp2d.shape
    med = np.full((F, T, C), np.nan, np.float64)
    for f in range(F):
        for t in range(T):
            uv = project_points(cam_mats, kp3d[f, t])               # (C,K,2)
            d = np.linalg.norm(kp2d[f, t] - uv, axis=-1)            # (C,K)
            d = np.where(conf[f, t] >= thr.conf_min, d, np.nan)
            with np.errstate(invalid="ignore"):
                med[f, t] = np.nanmedian(d, axis=-1)
    ok = (np.nan_to_num(med, nan=np.inf) <= thr.reproj_px).sum(-1) >= thr.reproj_min_views
    return ok, med


def containment_gate(kp3d, store, cam_mats, thr):
    """(F,T) admitted and (F,T) the best `contain_min_cams`-th containment
    fraction: per camera, the share of the fly's reprojected keypoints inside
    its OWN mask dilated by `contain_dilate_px`."""
    from jarvis_jax.tracking.lift_mvq import _disk_offsets, _inside_mask_dilated, project_points
    dy, dx = _disk_offsets(thr.contain_dilate_px)
    F, T = kp3d.shape[:2]; C = len(store.cameras)
    frac = np.zeros((F, T), np.float64)
    for f in range(F):
        for t in range(T):
            uv = project_points(cam_mats, kp3d[f, t])
            ok = np.isfinite(uv).all(-1)
            per_cam = []
            for c in range(C):
                if not store.valid_at(f, t)[c] or not ok[c].any():
                    continue
                inside = _inside_mask_dilated(store.mask_at(f, c, t), uv[c], ok[c], dy, dx)
                per_cam.append(inside[ok[c]].mean())
            per_cam = sorted(per_cam, reverse=True)
            frac[f, t] = per_cam[thr.contain_min_cams - 1] if len(per_cam) >= thr.contain_min_cams else 0.0
    return frac >= thr.contain_frac, frac


def decorrelate(admit, spacing, protect=None):
    """Greedy first-fit thinning: keep an admitted frame only if it is at least
    `spacing` frames after the last kept one. `protect` (T,) marks frames that
    are T=2 partners of a kept anchor and therefore exempt (spec §3.2)."""
    keep = np.zeros_like(admit, bool)
    last = -10 ** 9
    for t in np.flatnonzero(admit):
        if t - last >= spacing or (protect is not None and protect[t]):
            keep[t] = True; last = t
    return keep


def partner_masks(admit, deltas):
    """{delta: (T,) bool} -- True where frame t AND frame t+delta both pass."""
    out = {}
    for d in deltas:
        m = np.zeros_like(admit, bool)
        m[: len(admit) - d] = admit[: len(admit) - d] & admit[d:]
        out[int(d)] = m
    return out


def admit_bout(arrays, store, cam_mats, thr, *, use_identity=True):
    ex_ok, ex_q = existence_gate(arrays.meta, thr)
    st_ok, st_q = step_gate(arrays.kp3d, thr)
    rp_ok, rp_q = reprojection_gate(arrays.kp2d, arrays.kp3d, cam_mats, arrays.conf, thr)
    finite = np.isfinite(arrays.kp3d).all((-1, -2))
    if store is not None:
        ct_ok, ct_q = containment_gate(arrays.kp3d, store, cam_mats, thr)
    else:                                    # single-fly pass: no masks (spec §3.4)
        ct_ok = np.ones_like(ex_ok); ct_q = np.ones_like(ex_q)
    id_bad = identity_gate(arrays.meta, thr) if use_identity else np.zeros(ex_ok.shape[1], bool)
    fly = ex_ok & st_ok & rp_ok & ct_ok & finite & ~id_bad[None, :]
    frame_any = fly.any(0) & ~id_bad
    partners = partner_masks(frame_any, thr.deltas)
    protect = np.zeros_like(frame_any)
    for d, m in partners.items():
        protect[d:] |= m[: len(protect) - d]
    frame = decorrelate(frame_any, thr.decorrelation, protect=None)   # anchors only
    reasons = {"exist": ~ex_ok, "step": ~st_ok, "reproj": ~rp_ok, "contain": ~ct_ok,
               "nonfinite": ~finite, "identity": id_bad}
    quant = {"exist": ex_q, "step_units": st_q, "reproj_px": rp_q, "contain_frac": ct_q}
    return GateResult(frame, fly, reasons, quant, partners)
```

- [ ] **Step 5: Run** — `JAX_PLATFORMS=cpu pytest tests/test_pseudo_gates.py -q` -> 8 passed. Then a REAL bout, printing the admitted fraction (this is the sanity check that the thresholds are not vacuous or empty):

```bash
cd third_party/jarvis_jax && JAX_PLATFORMS=cpu python - <<'PY'
import numpy as np
from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
from jarvis_jax.tracking.lift_mvq import BoutMaskStore
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
B = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/pose_mvq_p3b/bouts/bout_00004"
rt = ReprojectionTool("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04/calibration")
cams = list(rt.cameras.keys())
a = load_bout_arrays(B, cameras=cams)
s = BoutMaskStore("/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/sam3_masks/bout_00004/sam3_masks.npz", cams)
r = admit_bout(a, s, np.asarray(rt.camera_matrices, np.float32), GateThresholds())
print("frames", a.kp3d.shape[1], "per-fly admitted", r.fly.mean(1), "anchors", int(r.frame.sum()))
print({k: float(v.mean()) for k, v in r.reasons.items()})
PY
```

Record the numbers in `docs/benchmark/2026-09-mvq/v2-pseudolabel-notes.md` (section "Gate calibration, bout 4"). If ANY gate rejects > 90 % of frames on this bout, stop and report: a gate that admits nothing is a bug, not a strict threshold.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/pseudo_gates.py third_party/jarvis_jax/tests/pseudo_fixtures.py third_party/jarvis_jax/tests/test_pseudo_gates.py
git add -f docs/benchmark/2026-09-mvq/v2-pseudolabel-notes.md
git commit -m "feat(mvq-v2): per-bout pseudo-label admission gates (spec 2026-09-05 §3.2)"
```

---

### Task 2: the v12-format writer and the extractor CLI

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/data/pseudo_export.py`
- Create: `scripts/pseudo_labels/__init__.py`, `scripts/pseudo_labels/extract_p3b_pseudolabels.py`
- Create: `third_party/jarvis_jax/tests/test_pseudo_export.py`, `third_party/jarvis_jax/tests/test_extract_pseudolabels.py`

**Interfaces:**
- Produces:
  - `PseudoRecord(recording, frame, host_fly, kp3d (F,K,3), kp2d (F,C,K,2), vis (F,C,K), sex (F,), stratum: dict, gates: dict, partners: dict[int,int], role: "anchor"|"partner"|"negative", center3D=None)`
  - `write_pseudo_export(out_root, records, *, export_names, cameras, recordings, checkpoint, gates, weight=0.3, frame_reader, mask_reader=None, split="train") -> dict` — writes `annotations/instances_train.json`, an intentionally EMPTY `annotations/instances_val.json`, `annotations/keypoint_names.json`, `images/<rec>/<cam>/Frame_<n>.jpg`, `masks/<rec>/<cam>/Frame_<n>.npz`, `calibrations/<group>/Cam*.yaml` (symlinks), `manifest.json`.
  - `to_export_order(arr, written_names, export_names)` — `detector_to_model_perm(written_names, export_names)` applied on the keypoint axis.
- Consumes: Task 1's `GateThresholds`/`admit_bout`; `jarvis_jax.predict.synced_reader` for frames.
- Consumed by: Tasks 3-6 and Plan B Task 1 (`V12WindowDataset(pseudo_root, "train", T=2, ...)`).

**On-disk contract** (everything the loader and Plan B depend on):

```jsonc
// manifest.json
{"version": "red_data_3d_v12_pseudo_p3b_20260905",
 "source": "pseudo",                       // NEW: the whole root is pseudo
 "checkpoint": "/…/mvq_t1_b16_p3b_contact_20260905/final",
 "gates": {"exist_min": 0.8, "max_step_units": 3.0, "reproj_px": 3.0, "reproj_min_views": 5,
           "contain_frac": 0.9, "contain_min_cams": 5, "contain_dilate_px": 6.0,
           "decorrelation": 16, "deltas": [1, 4, 16]},
 "weight": 0.3,                            // loss multiplier (spec §3.2)
 "calib_groups": ["2025_10_20_13_20_04", …],
 "recordings": {"2025_10_20_13_20_04": {
     "calib_group": "2025_10_20_13_20_04",  // one group per recording, its own calibration
     "n_framesets": 1234, "n_flies": 2, "behavior": "courtship", "sex": "mixed",
     "sex_source": "mvq_p3b_mask_identity", "fly_sex": {"fly0": "female", "fly1": "male"},
     "has_masks": true, "split": "train", "source": "pseudo", "weight": 0.3}}}
```

```jsonc
// annotations/instances_train.json  -- framesets carry the NEW keys
"framesets": {"2025_10_20_13_20_04/Frame_105207/fly1": {
   "recording": "2025_10_20_13_20_04", "fly_id": 1, "subset": "pseudo_p3b",
   "frames": [img ids per camera], "ann_ids": [ann ids per camera],
   "source": "pseudo", "weight": 0.3, "role": "anchor",
   "bout": 4, "stratum": {"host_sex": "male", "contact": true, "wall": false},
   "partners": {"1": 105208, "4": 105211, "16": 105223},   // absent delta = pair dropped
   "gates": {"exist": 0.93, "step_units": 0.7, "reproj_px": 1.2, "contain_frac": 0.97}}}
```

- [ ] **Step 1: Failing tests** (`tests/test_pseudo_export.py`)

```python
import json
import numpy as np
from pseudo_fixtures import CAMS, KP_NAMES, make_bout


def _records(a, r, rec="rec0"):
    from jarvis_jax.data.pseudo_export import PseudoRecord
    out = []
    for t in np.flatnonzero(r.frame):
        out.append(PseudoRecord(recording=rec, frame=1000 + int(t), host_fly=1,
                                kp3d=a.kp3d[:, t], kp2d=a.kp2d[:, t], vis=a.conf[:, t] >= 0.5,
                                sex=np.array([0, 1], np.int8),
                                stratum={"host_sex": "male", "contact": False, "wall": False},
                                gates={"exist": 0.95}, partners={1: 1001 + int(t)}, role="anchor"))
    return out


def test_export_round_trips_through_the_loader(tmp_path):
    from jarvis_jax.data.pseudo_export import write_pseudo_export
    from jarvis_jax.data.pseudo_gates import GateThresholds, admit_bout, load_bout_arrays
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.tracking.lift_mvq import BoutMaskStore
    d, npz, cm = make_bout(tmp_path, T=40)
    a = load_bout_arrays(d, cameras=CAMS)
    r = admit_bout(a, BoutMaskStore(npz, CAMS), cm, GateThresholds())
    out = str(tmp_path / "pseudo")
    frames = lambda rec, f: np.zeros((len(CAMS), 80, 120, 3), np.uint8)   # black frames are fine here
    export_names = list(reversed(KP_NAMES))                                # a DIFFERENT order on purpose
    write_pseudo_export(out, _records(a, r), export_names=export_names, cameras=CAMS,
                        recordings={"rec0": {"calib_dir": str(tmp_path / "calib"), "fly_sex": {"fly0": "female", "fly1": "male"}}},
                        checkpoint="/fake/final", gates=GateThresholds(), frame_reader=frames)
    man = json.load(open(f"{out}/manifest.json"))
    assert man["source"] == "pseudo" and man["weight"] == 0.3
    coco = json.load(open(f"{out}/annotations/instances_train.json"))
    assert coco["keypoint_names"] == export_names
    fs = next(iter(coco["framesets"].values()))
    assert fs["role"] == "anchor" and fs["source"] == "pseudo" and "1" in fs["partners"]
    ds = V12WindowDataset(out, "train", T=1, train=False)
    assert len(ds) == len(coco["framesets"])


def test_keypoints_are_written_in_EXPORT_order_not_model_order(tmp_path):
    """The CLAUDE.md trap: the campaign npz is in model order, the export's
    keypoint axis is `annotations/keypoint_names.json`. A by-index write would
    put Abd_tip's pixels under Antenna_Base and every metric would still look
    fine."""
    from jarvis_jax.data.pseudo_export import to_export_order
    arr = np.arange(len(KP_NAMES))[None, :, None].astype(np.float32)      # (1,K,1)
    export_names = list(reversed(KP_NAMES))
    got = to_export_order(arr, KP_NAMES, export_names)
    assert [KP_NAMES[int(v)] for v in got[0, :, 0]] == export_names


def test_dlt_of_the_written_2d_reproduces_the_written_3d(tmp_path):
    """The loader triangulates the 2D labels (`V12WindowDataset._dlt`); it never
    reads a 3D field. Writing the REPROJECTION of the gated 3D makes that
    round trip exact -- the gate only guarantees the 2D HEAD is within 3 px."""
    from jarvis_jax.data.pseudo_export import reprojected_labels
    from pseudo_fixtures import cam_mats, project
    cm = cam_mats()
    X = np.random.default_rng(0).normal(size=(6, 3)) * 5
    uv, vis = reprojected_labels(X, cm, np.ones((3, 6), bool), (80, 120))
    np.testing.assert_allclose(uv, project(cm, X), atol=1e-6)
    assert vis.shape == (3, 6)
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_pseudo_export.py -q` -> ModuleNotFoundError.

- [ ] **Step 3: Implement `pseudo_export.py`** — key bodies (the rest is bookkeeping):

```python
def to_export_order(arr, written_names, export_names):
    """`arr[..., k, ...]` in `written_names` order -> `export_names` order, BY NAME."""
    from jarvis_jax.tracking.predict_2d import detector_to_model_perm
    perm = detector_to_model_perm(list(written_names), list(export_names))
    return np.take(arr, perm, axis=-2 if arr.shape[-1] in (2, 3) else -1)


def reprojected_labels(kp3d, cam_mats, vis, hw):
    """(C,K,2) label pixels = the PROJECTION of the pseudo 3D (see the test
    above) and (C,K) visibility = vis & finite & inside the frame."""
    from jarvis_jax.tracking.lift_mvq import project_points
    uv = project_points(cam_mats, kp3d)
    H, W = hw
    inside = np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] <= W - 1) \
        & (uv[..., 1] >= 0) & (uv[..., 1] <= H - 1)
    return np.nan_to_num(uv, nan=0.0).astype(np.float32), (vis & inside)
```

`write_pseudo_export` loop, per record and per camera: append an `images` entry (`file_name = f"{rec}/{cam}/Frame_{frame}.jpg"`, real `width`/`height`), an `annotations` entry per labelled fly (`keypoints` = flattened `(u, v, int(vis))` in EXPORT order, `num_keypoints`, `bbox` from the visible points, `sex`, `fly_id`, `subset: "pseudo_p3b"`, `src_ann_id = id`), and a frameset per fly with the schema above. Decode each needed frame ONCE (`frame_reader(rec, frame) -> (C,H,W,3) uint8`, backed by `jarvis_jax.predict.synced_reader`) and write the JPEG at quality 95; when `mask_reader` is given write `masks/<rec>/<cam>/Frame_<n>.npz` with `ann_ids`/`matched`/`masks` exactly as `v5_3d._load_mask` reads them (the copy-paste augmentation in Plan B needs a real `prompt_mask`). Symlink `calibrations/<group>/` to the recording's `calibration/Cam*.yaml`. Negatives (`role == "negative"`, `host_fly is None`) get a frameset keyed `<rec>/Frame_<n>/neg0` with `fly_id: -1`, `negative: true`, `center3D: [x, y, z]` and one all-zero annotation per camera (`num_keypoints: 0`).

- [ ] **Step 4: The extractor CLI** `scripts/pseudo_labels/extract_p3b_pseudolabels.py`

Docstring EXPECTATION: *"a pseudo export whose gate histograms show every quantity well inside its threshold (exist mass above 0.9, step p99 below 3 units, reproj median below 1.5 px, containment above 0.95) and whose sampling report hits 50/50 host sex, >= 25 % contact, >= 25 % apart, all 11 recordings, no bout above 2 %. A histogram piled against a threshold means the gate, not the data, is choosing the set."*

```python
DEFAULT_ROOTS = "/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/*/*/pose_mvq_p3b"
TARGET = 10000

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help=f"run roots (default glob {DEFAULT_ROOTS})")
    ap.add_argument("--out", required=True)                 # …/red_data_3d_v12_pseudo_p3b_20260905
    ap.add_argument("--export-names-from", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--target", type=int, default=TARGET)
    ap.add_argument("--weight", type=float, default=0.3)
    ap.add_argument("--per-bout-frac", type=float, default=0.02)
    ap.add_argument("--contact-units", type=float, default=15.0)   # 1.5 mm (spec §3.2)
    ap.add_argument("--wall-height-units", type=float, default=30.0)
    ap.add_argument("--no-identity-gate", action="store_true")     # single-fly (Task 4)
    ap.add_argument("--num-animals", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gates-json", default=None, help="override GateThresholds fields")
    a = ap.parse_args()
```

Body, in order: (1) resolve run roots and their `sam3_masks/bout_*/sam3_masks.npz`; (2) per bout, `load_bout_arrays` + `admit_bout`, accumulating `quant` into histograms; (3) build the candidate pool — one row per admitted anchor with `stratum = {host_sex, contact: sep3d < 15 units, wall: height_above_floor > 30 units}` where the floor comes from `coarse_track.fit_floor(centroids_of_all_admitted_frames_of_this_recording)`; (4) **stratified draw**:

```python
rng = np.random.default_rng(a.seed)
cap = int(a.per_bout_frac * a.target)                       # 2 % => 200 frames per bout
quota = {"female": a.target // 2, "male": a.target - a.target // 2}
picked = []
for sex, n_sex in quota.items():
    pool = [r for r in rows if r.stratum["host_sex"] == sex]
    n_contact = max(int(0.25 * n_sex), 0); n_apart = max(int(0.25 * n_sex), 0)
    cells = [("contact", n_contact), ("apart", n_apart), ("free", n_sex - n_contact - n_apart)]
    for cell, n in cells:
        cand = [r for r in pool if cell == "free" or (r.stratum["contact"] == (cell == "contact"))]
        # round-robin over recordings so every one of the 11 appears; wall frames
        # are NOT boosted or suppressed -- they enter at their share of `cand`
        picked += round_robin_draw(cand, n, rng, key=lambda r: r.recording, cap_by=lambda r: r.bout, cap=cap)
```

`round_robin_draw` cycles recordings, drawing uniformly inside each and refusing a bout already at `cap`; it raises if a quota cannot be met and reports the shortfall per cell (do NOT silently under-fill). (5) add the partner rows for every picked anchor (`role="partner"`, exempt from the cap and from decorrelation); (6) `write_pseudo_export(...)`; (7) write `gate_histograms.json` (per gate: 50 bins, edges + counts, overall and per recording, plus the rejected fraction per gate and per recording) and `sampling_report.json` (realised composition vs targets, per-bout counts, per-recording counts, shortfalls) into the export root AND into `figures/2026-09-mvq/v2_pseudo/`.

- [ ] **Step 5: Test the CLI end to end on the fake root** (`tests/test_extract_pseudolabels.py`): build two fake bouts (one contact `sep_units=10`, one apart `sep_units=40`) under a fake run root, run `main()` with `--target 8 --per-bout-frac 0.5`, then assert: `manifest.json` has `source == "pseudo"`; the realised host-sex split is 4/4; both bouts contributed; `sampling_report.json`'s `contact_frac >= 0.25`; every frameset's `partners` keys are a subset of `{"1","4","16"}`; and `V12WindowDataset(out, "train", T=1, train=False)` builds a sample whose `kp3d_local + center3D` matches the gated `kp3d` for the same frame within 1e-3 units (the DLT round trip, in EXPORT keypoint order looked up BY NAME).

- [ ] **Step 6: Run for real** (CPU, ~1-2 h; 11 recordings x ~15 bouts, then ~70 k JPEG writes):

```bash
OMP_NUM_THREADS=8 python scripts/pseudo_labels/extract_p3b_pseudolabels.py \
  --out /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_p3b_20260905 \
  --target 10000 2>&1 | tee figures/2026-09-mvq/v2_pseudo/extract.log
```

Read `sampling_report.json` and confirm each spec §3.2 target line by line; paste the table into the notes.

- [ ] **Step 7: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/pseudo_export.py third_party/jarvis_jax/tests/test_pseudo_export.py \
        third_party/jarvis_jax/tests/test_extract_pseudolabels.py scripts/pseudo_labels/
git add -f docs/benchmark/2026-09-mvq/v2-pseudolabel-notes.md
git commit -m "feat(mvq-v2): v12-format pseudo-label export + stratified extractor (spec 2026-09-05 §3.2)"
```

---

### Task 3: the human review gallery

**Files:**
- Create: `scripts/pseudo_labels/pseudolabel_gallery.py`
- Create: `third_party/jarvis_jax/tests/test_pseudolabel_gallery.py`

**Interfaces:**
- Produces: `figures/2026-09-mvq/v2_pseudo/gallery/page_<nn>.png` (10 framesets per page, 2 columns = overhead `Cam2012630` + side `Cam2012855`, keypoints coloured by `viz.core.colors.keypoint_groups`, host in `PALETTE["fly0"]`/`PALETTE["fly1"]` by sex, mask outline in `PALETTE["mask"]`), `gallery/review.csv`, and under `--apply-review` a rewritten `annotations/instances_train.json` + `manifest.json["review"]` + `review_summary.json`.
- `review.csv` schema (one row per gallery frameset, `verdict` blank for the human to fill):
  `frameset,recording,bout,frame,host_fly,host_sex,contact,wall,page,cell,exist,step_units,reproj_px,contain_frac,verdict,note`
  with `verdict in {"", "accept", "reject"}`.
- Consumed by: spec §3.3's gate — training may not start until `review_summary.json` exists with `overall_reject_frac <= 0.03`.

- [ ] **Step 1: Tests** — on a 12-frameset fake export: (a) `main(["--export", root, "--n", "6", "--out", g])` writes one page PNG plus a `review.csv` with exactly 6 rows, header as above and every `verdict` empty; (b) sampling is stratified — with 3 contact and 3 apart requested the CSV has 3 of each; (c) `--apply-review` with 1 of 6 rejected (16.7 % > 3 %) exits non-zero and writes nothing; (d) with 0 rejected in a 40-row CSV it rewrites `instances_train.json` unchanged and sets `manifest["review"] = {"file": …, "n": 40, "reject_frac": 0.0, "dropped_strata": []}`; (e) with 1 rejected of 40 (2.5 %) inside the `contact/female` cell (whose own rate is 1/10 = 10 % > 3 %) the whole `contact/female` stratum is dropped from `instances_train.json`, `review_summary.json` names it, and every remaining frameset key still resolves.

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_pseudolabel_gallery.py -q` -> file missing.

- [ ] **Step 3: Implement.** Docstring EXPECTATION: *"every panel shows ONE fly's keypoints on that fly's own body, inside its own grey mask outline, in both cameras; the head (red) is at the antennae, the abdomen (magenta) at the tail, and leg chains do not cross between the two animals. A panel where the coloured skeleton straddles both bodies, or sits on the arena floor, is a reject — that is exactly the pseudo-label failure the 0.3 weight cannot fix."* Stratified draw of `--n 300` over `{host_sex} x {contact, apart} x {wall, floor}` proportional to the export's own composition; render with `cv2` on the export's own JPEGs, label each panel `f"{rec} b{bout} f{frame} fly{host} {sex} sep={sep:.1f}u"` and the camera NAME. `--apply-review`: read the CSV, compute overall and per-cell reject fractions, and follow spec §3.3 exactly (overall > 3 % -> exit 2 with the per-gate breakdown of the rejected rows so the offending gate can be tightened; else drop rejected framesets, then drop entire cells whose own rate > 3 %, then rewrite).

- [ ] **Step 4: Generate the real gallery, read it back**

```bash
python scripts/pseudo_labels/pseudolabel_gallery.py \
  --export /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_p3b_20260905 \
  --n 300 --out figures/2026-09-mvq/v2_pseudo/gallery
```

Open at least 5 pages with the Read tool (include one contact/female page and one wall page), and write what you saw against the expectation into the notes. **Then hand `review.csv` to Elliott** (spec §6 item 2) — do not fill verdicts yourself. When it comes back: `--apply-review figures/2026-09-mvq/v2_pseudo/gallery/review.csv`, commit the reviewed CSV and `review_summary.json` beside the notes.

- [ ] **Step 5: Commit**

```bash
git add scripts/pseudo_labels/pseudolabel_gallery.py third_party/jarvis_jax/tests/test_pseudolabel_gallery.py
git add -f docs/benchmark/2026-09-mvq/v2-pseudolabel-notes.md docs/benchmark/2026-09-mvq/v2-review/review.csv
git commit -m "feat(mvq-v2): pseudo-label review gallery and --apply-review gate (spec 2026-09-05 §3.3)"
```

---

### Task 4: single-fly pseudo-labels (free_running Session11, Clip Session6)

**Precondition (spec §3.4):** the P3b single-fly field check must be recorded as PASS in `docs/benchmark/2026-09-mvq/p3b-notes.md`. Step 1 below performs it if it is not there. If it FAILS (a phantom second fly, unstable keypoints), STOP: single-fly data enters v2 only through Task 5's negatives, and Plan B's acceptance drops its single-fly agreement row to "not evaluated".

**Files:**
- Create: `scripts/pseudo_labels/singlefly_p3b_pass.py`
- Create: `third_party/jarvis_jax/tests/test_singlefly_pass.py`
- Modify: `docs/benchmark/2026-09-mvq/p3b-notes.md` (field-check verdict)

**Interfaces:**
- Produces: P3b-SHAPED bout dirs `<out>/bouts/bout_<idx:05d>/fly0/{kp2d,kp3d}.npz` + `mvq_meta.json` with `per_frame = {exist, collapsed (all 0), slot_used, identity_source: ["single_typed"]*T}`, `containment: false`, `n_frames`, `frame_start`, `cameras`, `keypoint_names_written` — so Task 2's extractor runs on them unchanged with `--num-animals 1 --no-identity-gate`.
- Consumes: `MVQRunner(run_dir, calib_dir=…, cameras=…, identity="sex", containment=False)`, `MVQRunner.windows/infer/read_typed/to_pipeline`, `coarse_centres.lift_peaks_to_centres` (CenterDetect) for the per-frame centre.
- CLI: `--session-dir`, `--calib-dir`, `--run`, `--centerdetect`, `--bout start:end` (repeatable), `--num-animals 1`, `--out`, `--link-cameras` (Clip).

- [ ] **Step 1: Field check first.** Run the existing mask-free coarse pass at `--num-animals 1` over 2,000 frames of `free_running/Session11/2026_03_03_13_25_13` and of `Clip/Session6/2025_10_12_15_06_46`, then read the contact sheet:

```bash
python scripts/pseudo_labels/singlefly_p3b_pass.py --link-cameras \
  --session-dir /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/Clip/Session6/2025_10_12_15_06_46 \
  --out OutFiles/v2_singlefly_check/clip --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3b_contact_20260905/final \
  --centerdetect /gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004 \
  --bout 0:921 --num-animals 1
python scripts/viz/contact_sheet.py --tracks OutFiles/v2_singlefly_check/clip/bouts/bout_00000 \
  --out figures/2026-09-mvq/v2_pseudo/singlefly_check.png
```

Verdict lines to record: fraction of frames with EXACTLY one slot above 0.5 (target >= 0.99), the second-highest slot's mean existence (target < 0.2), and the median per-keypoint step in units. `--link-cameras` builds `<out>/_cams/<Cam>.mp4` symlinks from Clip's `Cam*_frames_<a>_<b>.mp4` names, because the readers resolve videos by `<Cam>.mp4`.

- [ ] **Step 2: Tests** — `FakeRunner` in the style of `tests/test_lift_masked_bout.py` (real `read_typed`/`to_pipeline`, arithmetic `infer`): assert the written `mvq_meta.json` has `identity_resolved == "single_typed"`, `collapsed` all zero, exactly one fly dir, `kp_names` equal to `configs/anatomy/v1.yaml` `model.KP_NAMES`, and the EyeL-EyeR invariant preserved (`_check_eye_invariant`); assert `--link-cameras` creates the seven `<Cam>.mp4` symlinks and refuses (ValueError) when two videos map to the same camera name.

- [ ] **Step 3: Implement**, then **Step 4: run for real** over 2 free_running recordings (2 bouts each, 4,000 frames) and Clip Session6 (921 frames) on a GPU, and **Step 5: extract**:

```bash
python scripts/pseudo_labels/extract_p3b_pseudolabels.py --runs OutFiles/v2_singlefly/*/ \
  --num-animals 1 --no-identity-gate --target 2000 \
  --out /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_singlefly_20260905
```

- [ ] **Step 6: Commit** (`scripts/pseudo_labels/singlefly_p3b_pass.py`, the test, and the notes with the field-check verdict table).

---

### Task 5: empty-window negatives

**Files:**
- Create: `scripts/pseudo_labels/extract_empty_windows.py`
- Create: `third_party/jarvis_jax/tests/test_empty_windows.py`

**Interfaces:**
- Produces: `PseudoRecord(role="negative", host_fly=None, center3D=(3,))` rows appended to the export (or written as their own root with `--out`), giving framesets keyed `<rec>/Frame_<n>/neg<k>` with `fly_id: -1`, `negative: true`, `center3D`, and one all-zero annotation per camera. Consumed by Plan B Task 1 (`fly_valid` all False, `unlabelled_sex = -1`, existence target 0 for every slot) and Plan B Task 3's negative test.
- Consumes: `coarse_tracks.npz` (`X3d (F,N,3)`, `exist (F,N)`, `coarse_frame (N,)`, `trackable`), `coarse_track.fit_floor`, `MVQRunner.cam_mats`, `coarse_centres._project_batch`.

- [ ] **Step 1: Tests** — with a synthetic `coarse_tracks.npz` (2 flies on a line, 200 coarse frames): (a) every emitted centre is `>= 60 units` (6 mm) from BOTH tracked centroids at its own frame and inside the frame in `>= 5` cameras when projected with `_project_batch`; (b) no centre is emitted for a frame where either fly's centroid is NaN AND `--require-tracked` is set; (c) with `--edge-frac 0.25`, a quarter of the rows are arena-edge windows (centre within `--edge-units` of the convex hull of the recording's tracked centroids) and each carries `stratum={"kind": "edge"}`; (d) the emitted records survive `write_pseudo_export` and `V12WindowDataset` yields them with `fly_valid.sum() == 0` once Plan B Task 1 lands — until then the test asserts the JSON contract only (`fly_id == -1`, `negative is True`, `len(center3D) == 3`).

- [ ] **Step 2-3: Implement.** Sampling: for each recording with a coarse pass, draw `--n-per-recording` frames uniformly over the recording, and per frame rejection-sample a centre uniformly inside the tracked-centroid bounding box inflated by 20 %, keeping it only if it is `>= 60 units` from every tracked centroid of that frame, `>= 6 units` above the fitted floor (a window buried in the substrate teaches nothing), and projects inside `>= 5` cameras. Arena-edge windows (`--edge-frac 0.25`, the gate-bout-1 failure mode) are drawn from the CenterDetect peaks of the same recording that lifted to a centre with `exist < 0.2` — the false peaks themselves, which is what the existence head must learn to refuse. Total default `--n 2000`.

- [ ] **Step 4: Run** over the 11 campaign recordings + the two single-fly sessions; read `figures/2026-09-mvq/v2_pseudo/negatives_check.png` (a 12-panel contact sheet of drawn empty crops, expectation: *"every panel shows bare substrate or an arena fixture, and NO fly — a panel with a fly in it is a mis-sampled negative that would teach the existence head to suppress a real animal"*) back with the Read tool before committing.

- [ ] **Step 5: Commit** (`scripts/pseudo_labels/extract_empty_windows.py`, test, notes).

---

### Task 6: loader reads the pseudo export (source / weight exposed)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py`
- Modify: `third_party/jarvis_jax/tests/test_v12_windows.py`

**Interfaces:**
- Produces: `ds.source(i) -> "real"|"pseudo"` (frameset `source`, else manifest `source`, else `"real"`); `ds.weight(i) -> float` (frameset `weight`, else manifest `weight`, else `1.0`); `ds.role(i) -> "anchor"|"partner"|"negative"`; sample key `sample_weight` (float32) added to `WINDOW_KEYS`.
- Consumed by: Plan B Task 1 (mixing real and pseudo roots), Plan B Task 3 (`batch["sample_weight"]` multiplies every per-sample loss term).

- [ ] **Step 1: Failing tests** (append to `tests/test_v12_windows.py`)

```python
def test_source_and_weight_default_to_real_on_the_human_export(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
    ds = V12WindowDataset(make_v12_root(tmp_path), "train", T=1, train=False)
    assert "sample_weight" in WINDOW_KEYS
    assert ds.source(0) == "real" and ds.weight(0) == 1.0 and ds.role(0) == "anchor"
    assert float(ds[0]["sample_weight"]) == 1.0


def test_pseudo_manifest_and_frameset_fields_override(tmp_path):
    import json, os
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    man = json.load(open(os.path.join(root, "manifest.json")))
    man["source"] = "pseudo"; man["weight"] = 0.3
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"))
    p = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(p))
    k = sorted(coco["framesets"])[0]
    coco["framesets"][k].update({"source": "pseudo", "weight": 0.1, "role": "partner"})
    json.dump(coco, open(p, "w"))
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = [j for j, w in enumerate(ds.windows) if f"{w[0]}/Frame_{w[2]}/fly{w[1]}" == k][0]
    assert ds.source(i) == "pseudo" and ds.weight(i) == 0.1 and ds.role(i) == "partner"
    assert float(ds[i]["sample_weight"]) == 0.1
    other = [j for j in range(len(ds)) if j != i][0]
    assert ds.weight(other) == 0.3            # manifest default for the rest of the root
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py -q -k "source or weight"` -> AttributeError.

- [ ] **Step 3: Implement** — in `__init__` keep the whole manifest dict (`self.manifest_root = json.load(...)`, `self.manifest = manifest_root["recordings"]`) and the frameset dicts already held in `self._fs`; add

```python
    def _fs_field(self, i, key, default):
        rec, fly, f0 = self.windows[i]
        v = (self._fs.get((rec, f0, fly)) or {}).get(key)
        if v is None:
            v = self.manifest.get(rec, {}).get(key, self.manifest_root.get(key, default))
        return v

    def source(self, i):
        return str(self._fs_field(i, "source", "real"))

    def weight(self, i):
        return float(self._fs_field(i, "weight", 1.0))

    def role(self, i):
        return str(self._fs_field(i, "role", "anchor"))
```

and `"sample_weight": np.float32(self.weight(i))` in `_build`'s returned dict, plus `"sample_weight"` in `WINDOW_KEYS`.

- [ ] **Step 4: Run the loader suite AND a real read of the pseudo export**

```bash
JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py tests/test_mv_copy_paste.py tests/test_train_mvq_smoke.py -q
JAX_PLATFORMS=cpu python -c "
from jarvis_jax.data.v12_windows import V12WindowDataset
ds = V12WindowDataset('/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_p3b_20260905', 'train', T=1, train=False)
s = ds[0]; print(len(ds), ds.source(0), ds.weight(0), s['crops'].shape, s['fly_valid'], ds.camera_names(0)[:2])"
```

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/v12_windows.py third_party/jarvis_jax/tests/test_v12_windows.py
git commit -m "feat(mvq-v2): loader exposes per-window source/weight/role for pseudo exports"
```

---

## Self-review (done while writing)

- Spec coverage: §3.2 -> Tasks 1-2; §3.3 -> Task 3; §3.4 -> Task 4; §3.5 -> Task 5; §6 items 1-3 -> Tasks 2-5; the loader hook §4 needs -> Task 6.
- Names crossing tasks: `GateThresholds`/`admit_bout`/`GateResult` (Task 1) are what Tasks 2, 4, 5 import; `PseudoRecord`/`write_pseudo_export`/`to_export_order` (Task 2) are what Tasks 3-5 write through; `ds.source/weight/role` and `sample_weight` (Task 6) are what Plan B Tasks 1 and 3 consume; the `role`/`partners`/`gates` frameset keys (Task 2) are what Plan B Task 1's Delta pairing reads.
- Ambiguities resolved in the text, each flagged where it is decided: the reprojection gate is per-view median with a `>= 5`-camera quorum (spec says "per-view … median over keypoints" without a quorum rule); labels are written as the REPROJECTION of the gated 3D so the loader's DLT round trip is exact; one calibration group per recording; the 2D-visibility flag comes from `kp2d.npz`'s `conf >= 0.5`; wall frames are defined by height above a per-recording fitted floor and enter at their pool proportion.
- Judgement calls left to the implementer, each flagged in its task: the exact `containment_report["per_frame"]` key names on real bouts (Task 1 Step 5 prints them), the JPEG quality/size budget (~10 GB, Task 2 Step 6), and whether Clip Session6 needs its own calibration group letter or its recording id (Task 4).
