# Coarse-to-Fine 3D Keypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce clean wing and leg kinematics for both flies on Session0 `2025_10_20_13_20_04` bout 28 by rebuilding the labeled 3D dataset without leakage and replacing the quantized 48³ single-stage lifter with a coarse-to-fine volumetric model plus continuous refinement.

**Architecture:** Rebuild `red_data_3d_v5` with the split as metadata rather than directory layout, recovering 264 silently-dropped framesets. Retrain ViTPose with sharp targets (σ 7.0 → 2.0). Add a stage-2 per-joint refinement volume at 4× resolution and a stage-3 gated continuous DLT refine on top of the existing stage-1 V2VNet. Fan 8 training arms across 8 GPUs so every design choice is attributed.

**Tech Stack:** JAX / Flax NNX, Orbax checkpointing, Hydra configs, NumPy/PIL, pytest, SAM3 (PyTorch) for masks, MuJoCo (`MUJOCO_GL=egl`) for IK renders.

**Spec:** `docs/specs/2026-08-29-coarse-to-fine-3d-design.md` (commit `5579ea5`)

## Global Constraints

- **Environment:** `micromamba activate 3d_tracking`; `unset LD_LIBRARY_PATH` before JAX; SAM3 needs `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` and `LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib`.
- **Node:** `g3102`, 8× L40S (46 GB), 32 CPUs, 256 GB cgroup, ~2d14h walltime. **Run directly — never `sbatch` from a GPU node, never run heavy work on a login node.**
- **JAX concurrency cap:** a prior 8-way run on a 128 GB cgroup died with `CUDA_ERROR_UNKNOWN`; working cap was ~4. Ramp 4 → 8 and set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`. Verify 4-way before launching 8.
- **Package root:** `third_party/jarvis_jax` (`PKG`). Repo root is `/mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset`.
- **Dataset root:** `V5=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5`
- **Target bout:** Session0 `2025_10_20_13_20_04`, bout 28, frames 446306–448312 (2007 frames), 2 flies, 7 cameras.
- **Units:** 1 world unit = 1 voxel at `grid_spacing: 1` ≈ 0.17 mm. Reprojection in px, positions in mm, angles in deg, time in frames.
- **Figures:** all under `figures/2026-08-29-c2f-3d/`; text artifacts under `docs/benchmark/2026-08-29-c2f-3d/`. **Never commit PNGs or videos.** Save the generating `.json`/`.npz` beside each PNG.
- **Every figure must be viewed with the Read tool and reported against its stated expectation before any claim is made about it.**
- **Naming:** real keypoint names (`T1L_FeTi`), camera names (`Cam2012630`), fly identities (`fly0`/`fly1`, male = fly1 after canonicalization). Never bare indices. Reuse `viz/core/colors.py` (`PALETTE`, `keypoint_groups`, `leg_chains`); white/cyan = observed/detector, green = fit, orange = fly1, grey = mask/mesh.
- **Tests:** `cd $PKG && python -m pytest tests/<file> -v` for package tests; `cd <repo> && python -m pytest tests/<file> -v` for repo tests.
- **Commit style:** end messages with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.

## Reference Constants (measured, do not re-derive)

| quantity | value |
|---|---|
| calibration groups | **bout 28's group is labelled `B`** (6 labeled recordings / 974 framesets); `A` is the big one (18 / 2532); `C` is 2 / 30. Labels are assigned by DESCENDING recording count (`group_calibrations`), so the letter is positional — never hardcode `A` to mean bout 28's group; compute it with `calib_fingerprint` on the Session0 calibration. Verified 2026-08-29. |
| union | 26 recordings, 3,536 framesets, 3,800 fly-samples |
| recovered second flies | 264 framesets / 1,673 annotations; 244 in group A |
| two-fly recordings | `2026_04_08_14_59_45` (214), `2026_04_07_11_33_33` (30), `2026_06_11_13_58_43` (17), `2026_06_11_13_58_45` (3) |
| confirmed female | 199 framesets in 5 recordings |
| recordings needing masks | 9: `2026_02_09_22_26_25`, `2026_02_13_13_44_49`, `2026_06_09_15_00_41`, `2026_06_09_15_21_14`, `2026_06_09_15_38_35`, `2026_06_09_15_46_55`, `2026_06_10_15_05_02`, `2026_07_30_13_28_99`, `2026_08_26_16_05_15` (737 framesets, ~5,160 images) |
| voxel scale | 1 voxel = 2.7–3.2 heatmap px; tarsal `T1L_TaT3→T1L_TaTip` = 1.59 voxels |

## File Structure

**Create:**
- `$PKG/jarvis_jax/data/calib_groups.py` — content-hash calibrations into groups
- `$PKG/jarvis_jax/data/build_v5.py` — dataset builder (manifest, symlinks, annotation merge)
- `$PKG/jarvis_jax/data/split_v5.py` — split generation and leakage audit
- `$PKG/jarvis_jax/data/v5_3d.py` — per-fly frameset loader (replaces `v3_3d.py`)
- `$PKG/jarvis_jax/hybridnet/refine.py` — stage-2 per-joint refinement
- `$PKG/scripts/build_v5_dataset.py` — Hydra entrypoint for the build
- `$PKG/scripts/train_arms.py` — 8-arm launcher
- `<repo>/scripts/viz/contact_sheet.py` — per-recording sexing contact sheets (committed)
- `<repo>/scripts/viz/voxel_resolution.py` — per-joint error vs segment length in voxels (committed)

**Modify:**
- `$PKG/jarvis_jax/hybridnet/reproject.py` — add `rotation` and `spacing`
- `$PKG/jarvis_jax/hybridnet/model.py:44` — generalize `soft_argmax_3d` past `assert grid_spacing == 1`
- `$PKG/jarvis_jax/data/transforms.py:47` — `sigma` default 7.0 → 2.0
- `$PKG/jarvis_jax/tracking/triangulate.py` — add stage-3 `refine_from_seed`
- `$PKG/configs/paths/hyak.yaml` — add `v5_root`
- `$PKG/configs/model/hybridnet.yaml` — add `refine` block

**Test:**
- `$PKG/tests/test_calib_groups.py`, `test_build_v5.py`, `test_split_v5.py`, `test_v5_3d.py`, `test_reproject_rotation.py`, `test_refine.py`, `test_triangulate_refine.py`

---

## Task 1: Baseline capture on bout 28 (Phase 0)

No new code. Produces the "before" every later comparison needs. Do this first — it is the only artifact that cannot be regenerated after the configs change.

**Files:**
- Create: `figures/2026-08-29-c2f-3d/phase0-baseline/` (outputs, not committed)

**Interfaces:**
- Consumes: nothing.
- Produces: `OutFiles/.../bouts/bout_00028/fly{0,1}/{kp2d.npz,kp3d.npz,outputs.h5}` — the baseline arrays every later task compares against; `figures/2026-08-29-c2f-3d/phase0-baseline/*.png`.

- [ ] **Step 1: Record the expectation before running**

Write to `figures/2026-08-29-c2f-3d/phase0-baseline/EXPECTATION.md`:

```markdown
# Phase 0 baseline expectation

Running bout 28 through the CURRENT pipeline (ViTPose v4_8gpu_20260808 -> robust
DLT -> STAC IK -> polish). This is the "before", not a result.

Expected from prior measurements:
- male (fly1) reprojection ~7 px, female (fly0) ~42 px, ~87 px when flies are close
- female silhouette IoU ~0
- leg-angle traces visibly jagged; tarsal-tip 3D traces show ~0.17 mm staircase
  quantization if the resolution hypothesis holds

If the female reprojection comes out near the male's, the benchmark baseline does
not describe this bout and the acceptance thresholds in Task 17 must be re-derived
from THIS run rather than from docs/benchmark/2026-08-07-baseline-scorecard.json.
```

- [ ] **Step 2: Run bout 28 end-to-end**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
python scripts/run_bout.py paths=hyak recording=session0 bout_ids=28 2>&1 \
  | tee figures/2026-08-29-c2f-3d/phase0-baseline/run.log
```

Expected: stages A–E complete for `fly0` and `fly1`; `outputs.h5` and per-camera
overlay videos written under the run root.

- [ ] **Step 3: Archive the baseline arrays**

```bash
OUT=$(python - <<'PY'
import hydra, os
from omegaconf import OmegaConf
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(os.path.normpath(str(p))), replace=True)
with hydra.initialize(config_path="configs", version_base=None):
    cfg = hydra.compose("config", overrides=["paths=hyak","recording=session0"])
print(cfg.outputs.out)
PY
)
DST=figures/2026-08-29-c2f-3d/phase0-baseline
mkdir -p $DST/arrays
cp -r "$OUT"/bouts/bout_00028 $DST/arrays/
ls -R $DST/arrays | head -20
```

Expected: `kp2d.npz`, `kp3d.npz`, `outputs.h5` present for both flies.

- [ ] **Step 4: Render the baseline figures**

**Controller ruling R1:** `python -m viz` is **argparse, not hydra**, and
`compare_stac_fits.py` requires `--xml`/`--anatomy` and has **no `--bout`**.
Verified real flags:

```bash
DST=figures/2026-08-29-c2f-3d/phase0-baseline
for FLY in 0 1; do
  python -m viz overlay --run "$OUT" --bout 28 --fly $FLY --out $DST
  python -m viz legskel --run "$OUT" --bout 28 --fly $FLY --out $DST
done
```

`legskel` is the leg-joint-chain view (detector 2D vs fitted 3D) and is the
most directly relevant figure for this goal. For `compare_stac_fits.py`, read
its argparse block and pass `--xml`/`--anatomy` from `configs/anatomy/` and the
`body_model_dir` in `configs/paths/hyak.yaml`. **Record the exact working
command in the report** so later tasks reuse it rather than rediscovering it.

- [ ] **Step 5: Read the figures and write what they show**

Open each PNG with the Read tool. Append to `EXPECTATION.md` a `## Observed`
section stating, per fly, the reprojection error, whether tarsal traces show
staircase quantization, and whether observed matches the expectation above.
**A figure that was generated but not viewed is not evidence.**

- [ ] **Step 6: Commit the text artifacts only**

```bash
mkdir -p docs/benchmark/2026-08-29-c2f-3d
cp figures/2026-08-29-c2f-3d/phase0-baseline/EXPECTATION.md \
   docs/benchmark/2026-08-29-c2f-3d/phase0-baseline-notes.md
git add docs/benchmark/2026-08-29-c2f-3d/phase0-baseline-notes.md
git commit -m "docs(3d): bout 28 phase-0 baseline, observed vs expected

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Calibration grouping

**Files:**
- Create: `$PKG/jarvis_jax/data/calib_groups.py`
- Test: `$PKG/tests/test_calib_groups.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `calib_fingerprint(calib_dir: str) -> str` (8-char hex);
  `group_calibrations(dirs: dict[str, str]) -> dict[str, str]` mapping
  recording name → group label `"A"`, `"B"`, `"C"`, … assigned by descending
  member count, ties broken by first recording name.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_calib_groups.py
import os
import pytest
from jarvis_jax.data.calib_groups import calib_fingerprint, group_calibrations

CAMS = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
        "Cam2012857", "Cam2012861", "Cam2012862"]

def _write_calib(tmp_path, name, first_coeff, *, pretty=False):
    d = tmp_path / name
    d.mkdir(parents=True)
    for c in CAMS:
        data = [first_coeff, 0.0074869, -0.031773, -2.828,
                0.0093308, -8.0788, -0.17912, 462.78, 0.0, 0.0, 0.0, 1.0]
        if pretty:
            body = ",\n       ".join(f"{v}" for v in data)
        else:
            body = ", ".join(f"{v:.16g}" for v in data)
        (d / f"{c}.yaml").write_text(
            "%YAML:1.0\n---\nimage_width: 1936\nimage_height: 448\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            f"   data: [ {body} ]\nscale: 10\n")
    return str(d)

def test_fingerprint_ignores_formatting(tmp_path):
    """Formatting-only differences must NOT create a new calibration group.
    Real case: 20_04_female_climbing is numerically identical to Session0 but
    written with fewer decimal places."""
    a = _write_calib(tmp_path, "terse", 8.1001)
    b = _write_calib(tmp_path, "pretty", 8.1001, pretty=True)
    assert calib_fingerprint(a) == calib_fingerprint(b)

def test_fingerprint_separates_real_differences(tmp_path):
    a = _write_calib(tmp_path, "grpA", 8.1001)
    b = _write_calib(tmp_path, "grpB", 8.1333)
    assert calib_fingerprint(a) != calib_fingerprint(b)

def test_groups_labelled_by_descending_size(tmp_path):
    dirs = {
        "rec_b1": _write_calib(tmp_path, "b1", 8.1333),
        "rec_b2": _write_calib(tmp_path, "b2", 8.1333),
        "rec_b3": _write_calib(tmp_path, "b3", 8.1333),
        "rec_a1": _write_calib(tmp_path, "a1", 8.1001),
        "rec_a2": _write_calib(tmp_path, "a2", 8.1001),
        "rec_c1": _write_calib(tmp_path, "c1", 8.0898),
    }
    g = group_calibrations(dirs)
    assert g["rec_b1"] == g["rec_b2"] == g["rec_b3"] == "A"   # biggest group
    assert g["rec_a1"] == g["rec_a2"] == "B"
    assert g["rec_c1"] == "C"

def test_missing_camera_raises(tmp_path):
    d = _write_calib(tmp_path, "short", 8.1001)
    os.remove(os.path.join(d, "Cam2012862.yaml"))
    with pytest.raises(ValueError, match="expected 7 cameras"):
        calib_fingerprint(d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_calib_groups.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.data.calib_groups'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/data/calib_groups.py
"""Group recordings by the CONTENT of their DLT calibration.

The V2VNet's voxel grid is axis-aligned to the world frame (see
hybridnet/reproject.py: `grid = base_grid + center3D`), so a recalibration that
rotates the world frame invalidates the model's learned orientation prior.
Calibration identity is therefore a first-class property of a recording, not
an incidental file path.

Fingerprints are computed from ROUNDED NUMERIC coefficients, never file bytes:
20_04_female_climbing's calibration is numerically identical to Session0's but
written with fewer decimal places, and a byte hash wrongly splits them.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from collections import Counter

CAM_GLOB = "Cam*.yaml"
NUM_CAMERAS = 7
_ROUND = 6


def _projection_matrix(path: str) -> list[float]:
    text = open(path).read()
    m = re.search(r"data:\s*\[(.*?)\]", text, re.S)
    if m is None:
        raise ValueError(f"no projectionMatrix data block in {path}")
    vals = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if len(vals) != 12:
        raise ValueError(f"expected 12 projection coefficients in {path}, got {len(vals)}")
    return [round(v, _ROUND) for v in vals]


def calib_fingerprint(calib_dir: str) -> str:
    """8-char hex fingerprint of a calibration directory's 7 camera matrices."""
    files = sorted(glob.glob(os.path.join(calib_dir, CAM_GLOB)))
    if len(files) != NUM_CAMERAS:
        raise ValueError(
            f"expected {NUM_CAMERAS} cameras in {calib_dir}, found {len(files)}")
    coeffs: list[float] = []
    for f in files:
        coeffs.extend(_projection_matrix(f))
    return hashlib.md5(repr(coeffs).encode()).hexdigest()[:8]


def group_calibrations(dirs: dict[str, str]) -> dict[str, str]:
    """Map recording name -> group label ('A', 'B', ...), largest group = 'A'."""
    fp = {rec: calib_fingerprint(d) for rec, d in dirs.items()}
    counts = Counter(fp.values())
    ordered = sorted(counts, key=lambda h: (-counts[h],
                                            min(r for r in fp if fp[r] == h)))
    label = {h: chr(ord("A") + i) for i, h in enumerate(ordered)}
    return {rec: label[h] for rec, h in fp.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_calib_groups.py -v`
Expected: 4 passed

- [ ] **Step 5: Verify against the real data**

```bash
cd $PKG && python - <<'PY'
import json, os
from jarvis_jax.data.calib_groups import group_calibrations
G = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
V = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/calib_params"
S = ("/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/"
     "2025_10_20_13_20_04/calibration")
dirs = {"__bout28__": S}
for s in sorted(os.listdir(G)):
    cp = os.path.join(G, s, "calib_params")
    if os.path.isdir(cp):
        for r in os.listdir(cp):
            dirs[r] = os.path.join(cp, r)
for r in os.listdir(V):
    dirs.setdefault(r, os.path.join(V, r))
g = group_calibrations(dirs)
from collections import Counter
print("group sizes:", Counter(g.values()))
print("bout 28 group:", g["__bout28__"])
print("2026_04_08 group:", g["2026_04_08_14_59_45"])
PY
```

Expected: exactly 3 groups; `bout 28 group` and `2026_04_08 group` are the SAME
label. If they differ, stop — the spec's Group-A membership is wrong.

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/data/calib_groups.py $PKG/tests/test_calib_groups.py
git commit -m "feat(data): content-hash calibrations into groups

Fingerprints rounded numeric coefficients, not file bytes: femclimb's
calibration is numerically identical to Session0's but formatted differently,
and a byte hash wrongly splits them.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Dataset builder — manifest, symlinks, deduplicated calibrations

**Files:**
- Create: `$PKG/jarvis_jax/data/build_v5.py`
- Test: `$PKG/tests/test_build_v5.py`

**Interfaces:**
- Consumes: `calib_groups.group_calibrations`.
- Produces:
  - `discover_sources(general_model_root: str, v3_root: str) -> dict[str, SourceRec]`
  - `SourceRec` dataclass: `recording: str`, `subset: str | None`, `ann_paths: list[str]`, `calib_dir: str`, `image_root: str`, `mask_root: str | None`
  - `build_manifest(sources, out_root) -> dict` writing `manifest.json` with per-recording `{subset, calib_group, n_framesets, n_flies, sex, behavior, has_masks, split}`; `sex` is `"unknown"` until Task 6 fills it.
  - `link_media(sources, out_root, *, copy: bool = False) -> None` writing `images/<recording>/<Cam>/Frame_*.jpg` and `masks/<recording>/<Cam>/Frame_*.npz`.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_build_v5.py
import json
import os
from jarvis_jax.data.build_v5 import SourceRec, build_manifest, link_media

CAMS = ["Cam2012630", "Cam2012631"]

def _fake_source(tmp_path, rec, n_frames=3, first=8.1001, masks=True):
    img_root = tmp_path / "src" / rec / "images"
    for c in CAMS:
        (img_root / c).mkdir(parents=True)
        for i in range(n_frames):
            (img_root / c / f"Frame_{i:06d}.jpg").write_bytes(b"\xff\xd8fake")
    calib = tmp_path / "src" / rec / "calib"
    calib.mkdir(parents=True)
    for c in ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
              "Cam2012857", "Cam2012861", "Cam2012862"]:
        data = ", ".join(str(v) for v in [first, 0.0074869, -0.031773, -2.828,
                                          0.0093308, -8.0788, -0.17912, 462.78,
                                          0.0, 0.0, 0.0, 1.0])
        (calib / f"{c}.yaml").write_text(
            f"%YAML:1.0\n---\nprojectionMatrix: !!opencv-matrix\n   data: [ {data} ]\n")
    mask_root = None
    if masks:
        mask_root = tmp_path / "src" / rec / "masks"
        for c in CAMS:
            (mask_root / c).mkdir(parents=True)
            for i in range(n_frames):
                (mask_root / c / f"Frame_{i:06d}.npz").write_bytes(b"npz")
    return SourceRec(recording=rec, subset=f"sub_{rec}", ann_paths=[],
                     calib_dir=str(calib), image_root=str(img_root),
                     mask_root=str(mask_root) if mask_root else None)

def test_manifest_records_calib_group_and_mask_availability(tmp_path):
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a", first=8.1001, masks=True),
            "rec_b": _fake_source(tmp_path, "rec_b", first=8.1333, masks=False)}
    out = tmp_path / "v5"
    man = build_manifest(srcs, str(out))
    assert man["recordings"]["rec_a"]["has_masks"] is True
    assert man["recordings"]["rec_b"]["has_masks"] is False
    assert man["recordings"]["rec_a"]["calib_group"] != man["recordings"]["rec_b"]["calib_group"]
    assert man["recordings"]["rec_a"]["sex"] == "unknown"
    assert json.load(open(out / "manifest.json")) == man

def test_calibrations_are_deduplicated_not_copied_per_recording(tmp_path):
    srcs = {f"rec_{i}": _fake_source(tmp_path, f"rec_{i}", first=8.1001)
            for i in range(3)}
    out = tmp_path / "v5"
    build_manifest(srcs, str(out))
    groups = sorted(os.listdir(out / "calibrations"))
    assert groups == ["A"], f"3 identical calibrations must dedup to 1 dir, got {groups}"
    assert len(os.listdir(out / "calibrations" / "A")) == 7

def test_link_media_creates_symlinks_with_no_split_dirs(tmp_path):
    srcs = {"rec_a": _fake_source(tmp_path, "rec_a")}
    out = tmp_path / "v5"
    build_manifest(srcs, str(out))
    link_media(srcs, str(out))
    p = out / "images" / "rec_a" / "Cam2012630" / "Frame_000000.jpg"
    assert p.is_symlink(), "images must be symlinked, not copied"
    assert (out / "masks" / "rec_a" / "Cam2012630" / "Frame_000000.npz").exists()
    # The whole point: no train/ or val/ directory anywhere in the media tree.
    for root, dirs, _ in os.walk(out / "images"):
        assert "train" not in dirs and "val" not in dirs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_build_v5.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.data.build_v5'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/data/build_v5.py
"""Build red_data_3d_v5: one correctly-organized root for 3D training.

Two organizational defects in the old layout drive this module:

1. The SPLIT WAS THE DIRECTORY LAYOUT. general_model/<subset>/ held train/ and
   val/ dirs over the SAME recording, so the split was frame-level and leaked
   (~50% of val framesets had a train frameset within +/-3 frames), and
   regenerating it meant moving image files. Here the split is METADATA
   (split.json, Task 5) and the media tree has no train/ or val/ dir at all.
2. Calibrations were duplicated per subset (21 copies of 3 distinct
   calibrations), hiding the fact that calibration identity is a real
   experimental variable.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field

from jarvis_jax.data.calib_groups import CAM_GLOB, group_calibrations

import glob


@dataclass
class SourceRec:
    recording: str
    subset: str | None
    ann_paths: list[str]
    calib_dir: str
    image_root: str
    mask_root: str | None = None
    n_framesets: int = 0
    n_flies: int = 0
    behavior: str = "unknown"
    sex: str = "unknown"
    extra: dict = field(default_factory=dict)


def discover_sources(general_model_root: str, v3_root: str) -> dict[str, SourceRec]:
    """Enumerate every labeled recording across BOTH source trees.

    general_model contributes 21 recordings; red_data_unified_V3 contributes 5
    more that general_model lacks (2026_03_22, 2026_04_07, 2026_04_08, and the
    two 2026_06_11 recordings). The union is 26 -- V3's 3D training only ever
    used 17.
    """
    srcs: dict[str, SourceRec] = {}
    for subset in sorted(os.listdir(general_model_root)):
        cp = os.path.join(general_model_root, subset, "calib_params")
        if not os.path.isdir(cp):
            continue
        for rec in sorted(os.listdir(cp)):
            anns = sorted(glob.glob(os.path.join(
                general_model_root, subset, "annotations", "instances_*.json")))
            srcs[rec] = SourceRec(
                recording=rec, subset=subset, ann_paths=anns,
                calib_dir=os.path.join(cp, rec),
                image_root=os.path.join(general_model_root, subset),
                mask_root=None)
    v3_calib = os.path.join(v3_root, "calib_params")
    v3_anns = sorted(glob.glob(os.path.join(v3_root, "annotations", "instances_*.json")))
    for rec in sorted(os.listdir(v3_calib)):
        if rec in srcs:
            # Already covered by general_model, but V3 is where its masks live.
            srcs[rec].mask_root = os.path.join(v3_root, "sam3_masks")
            continue
        srcs[rec] = SourceRec(
            recording=rec, subset=None, ann_paths=v3_anns,
            calib_dir=os.path.join(v3_calib, rec),
            image_root=v3_root,
            mask_root=os.path.join(v3_root, "sam3_masks"))
    return srcs


def build_manifest(sources: dict[str, SourceRec], out_root: str) -> dict:
    """Write manifest.json and the deduplicated calibrations/ tree."""
    os.makedirs(out_root, exist_ok=True)
    groups = group_calibrations({r: s.calib_dir for r, s in sources.items()})

    calib_out = os.path.join(out_root, "calibrations")
    for rec, grp in groups.items():
        dst = os.path.join(calib_out, grp)
        if os.path.isdir(dst):
            continue
        os.makedirs(dst, exist_ok=True)
        for f in sorted(glob.glob(os.path.join(sources[rec].calib_dir, CAM_GLOB))):
            shutil.copy2(f, os.path.join(dst, os.path.basename(f)))

    man = {
        "version": "red_data_3d_v5",
        "calib_groups": sorted(set(groups.values())),
        "recordings": {
            rec: {
                "subset": s.subset,
                "calib_group": groups[rec],
                "n_framesets": s.n_framesets,
                "n_flies": s.n_flies,
                "behavior": s.behavior,
                "sex": s.sex,                      # filled by the Task 6 sexing pass
                "has_masks": s.mask_root is not None,
                "split": None,                     # filled by Task 5
            }
            for rec, s in sorted(sources.items())
        },
    }
    with open(os.path.join(out_root, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    return man


def link_media(sources: dict[str, SourceRec], out_root: str, *,
               copy: bool = False) -> None:
    """Materialize images/ and masks/ as ONE flat per-recording tree.

    Symlinks by default: the sources are stable on gscratch and copying costs
    4.4 GB for no benefit. `copy=True` if the tree must be self-contained.
    """
    for rec, s in sources.items():
        for kind, src_root in (("images", s.image_root), ("masks", s.mask_root)):
            if src_root is None:
                continue
            for split in ("train", "val", ""):
                base = os.path.join(src_root, split, rec) if split else os.path.join(src_root, rec)
                if not os.path.isdir(base):
                    continue
                for cam in sorted(os.listdir(base)):
                    src_cam = os.path.join(base, cam)
                    if not os.path.isdir(src_cam):
                        continue
                    dst_cam = os.path.join(out_root, kind, rec, cam)
                    os.makedirs(dst_cam, exist_ok=True)
                    for fn in sorted(os.listdir(src_cam)):
                        dst = os.path.join(dst_cam, fn)
                        if os.path.lexists(dst):
                            continue
                        if copy:
                            shutil.copy2(os.path.join(src_cam, fn), dst)
                        else:
                            os.symlink(os.path.join(src_cam, fn), dst)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_build_v5.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add $PKG/jarvis_jax/data/build_v5.py $PKG/tests/test_build_v5.py
git commit -m "feat(data): v5 builder — manifest, deduped calibrations, flat media tree

The split stops being the directory layout: images/ and masks/ carry no
train/ or val/ dir, so regenerating a split moves zero files.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Annotation merge with per-fly identity — recover the 264 dropped framesets

This is the task that fixes the silent data loss. `v3_3d.py:122-128` keeps only
the first annotation per image; 1,673 annotations across 264 framesets are
discarded, 244 of them in bout 28's own calibration group.

**Files:**
- Modify: `$PKG/jarvis_jax/data/build_v5.py`
- Test: `$PKG/tests/test_build_v5.py`

**Interfaces:**
- Consumes: `SourceRec`.
- Produces: `merge_annotations(sources, out_root) -> dict` writing
  `annotations/instances.json` with schema:
  - `images[]`: `{id, file_name: "<recording>/<Cam>/Frame_*.jpg", width, height, recording}`
  - `annotations[]`: `{id, image_id, bbox, keypoints, num_keypoints, sex, behavior, fly_id}` where `fly_id` is the 0-based index of this fly WITHIN its frameset, assigned consistently across cameras by source ordering
  - `framesets{}`: `"<recording>/Frame_<N>/fly<k>"` → `{recording, fly_id, frames: [image_id × 7], ann_ids: [ann_id | None × 7]}` — **`ann_ids` may contain `None`** for a camera where this fly's identity could not be resolved (ruling R15); at least `MIN_CAMS` = 3 entries are non-`None`
  - `categories`: `[{"id": 1, "name": "fly", "num_keypoints": 50}]`
  - `keypoint_names`, `skeleton` carried through unchanged

- [ ] **Step 1: Write the failing test**

```python
# append to $PKG/tests/test_build_v5.py
import json
from jarvis_jax.data.build_v5 import merge_annotations, SourceRec

CAMS7 = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
         "Cam2012857", "Cam2012861", "Cam2012862"]

def _coco(tmp_path, name, rec, n_flies):
    """One frameset over 7 cameras with n_flies annotations per image."""
    images, anns = [], []
    aid = 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"{rec}/{cam}/Frame_000100.jpg"})
        for k in range(n_flies):
            anns.append({"id": aid, "image_id": ci, "bbox": [k * 100.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    blob = {"keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
            "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
            "images": images, "annotations": anns,
            "framesets": {f"{rec}/Frame_000100": {"datasetName": rec,
                                                  "frames": list(range(7))}},
            "calibrations": {rec: {}}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(blob))
    return str(p)

def test_two_fly_frameset_yields_two_samples_not_one(tmp_path):
    """The v3_3d 'keep first annotation' rule silently dropped the second fly.
    A 7-camera frameset with 2 annotations per image must produce TWO framesets."""
    p = _coco(tmp_path, "two", "rec_two", n_flies=2)
    srcs = {"rec_two": SourceRec("rec_two", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    keys = sorted(merged["framesets"])
    assert keys == ["rec_two/Frame_000100/fly0", "rec_two/Frame_000100/fly1"]
    for k in keys:
        assert len(merged["framesets"][k]["frames"]) == 7
        assert len(merged["framesets"][k]["ann_ids"]) == 7
    # no annotation may appear in two framesets
    used = [a for k in keys for a in merged["framesets"][k]["ann_ids"]]
    assert len(used) == len(set(used)) == 14

def test_single_fly_frameset_unchanged(tmp_path):
    p = _coco(tmp_path, "one", "rec_one", n_flies=1)
    srcs = {"rec_one": SourceRec("rec_one", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert sorted(merged["framesets"]) == ["rec_one/Frame_000100/fly0"]

def test_fly_id_is_consistent_across_cameras(tmp_path):
    p = _coco(tmp_path, "two", "rec_two", n_flies=2)
    srcs = {"rec_two": SourceRec("rec_two", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    by_id = {a["id"]: a for a in merged["annotations"]}
    for k, fs in merged["framesets"].items():
        fly_ids = {by_id[a]["fly_id"] for a in fs["ann_ids"]}
        assert len(fly_ids) == 1, f"{k} mixes fly_ids {fly_ids} across cameras"

def test_camera_with_fewer_annotations_is_marked_absent_not_guessed(tmp_path):
    """Ruling R15. When one camera sees fewer flies than the rest, its lone
    annotation may belong to EITHER fly — verified real case:
    2026_04_08_14_59_45/Frame_149677, where Cam2012631's single annotation is
    the RIGHT fly while positional index 0 would file it as the left one.
    The merge must record that camera ABSENT for every fly, never guess."""
    p = tmp_path / "uneq.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_u/{cam}/Frame_000100.jpg"})
        # Cam2012631 (index 1) sees only ONE fly; every other camera sees two.
        n = 1 if ci == 1 else 2
        for k in range(n):
            anns.append({"id": aid, "image_id": ci,
                         "bbox": [500.0 if n == 1 else k * 500.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_u/Frame_000100": {"datasetName": "rec_u",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_u": {}}}))
    srcs = {"rec_u": SourceRec("rec_u", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    # BOTH flies survive (6 cameras each, above MIN_CAMS=3) ...
    assert sorted(merged["framesets"]) == ["rec_u/Frame_000100/fly0",
                                           "rec_u/Frame_000100/fly1"]
    for key in merged["framesets"]:
        ids = merged["framesets"][key]["ann_ids"]
        assert len(ids) == 7
        # ... and the disagreeing camera is ABSENT, not guessed at.
        assert ids[1] is None, f"{key} guessed an identity for Cam2012631"
        assert sum(a is not None for a in ids) == 6


def test_fly_dropped_when_too_few_cameras_resolve_it(tmp_path):
    """Below MIN_CAMS=3 resolvable cameras a fly cannot be triangulated, so it
    must be dropped rather than emitted with mostly-None slots."""
    p = tmp_path / "sparse.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_s/{cam}/Frame_000100.jpg"})
        n = 2 if ci < 2 else 1          # only 2 cameras resolve two flies
        for k in range(n):
            anns.append({"id": aid, "image_id": ci, "bbox": [k * 500.0, 0.0, 50.0, 50.0],
                         "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship"})
            aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_s/Frame_000100": {"datasetName": "rec_s",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_s": {}}}))
    srcs = {"rec_s": SourceRec("rec_s", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert merged["framesets"] == {}, "2 resolvable cameras is below MIN_CAMS"


def test_single_fly_frameset_survives_a_camera_with_no_annotation(tmp_path):
    """The bug ruling R15 also fixes: under the old `min` rule a single camera
    with zero annotations discarded the WHOLE frameset, costing 91 framesets
    from 2026_01_13_18_47_45 and every frameset of wall_frames."""
    p = tmp_path / "gap.json"
    images, anns, aid = [], [], 0
    for ci, cam in enumerate(CAMS7):
        images.append({"id": ci, "width": 1936, "height": 448,
                       "file_name": f"rec_g/{cam}/Frame_000100.jpg"})
        if ci == 3:
            continue                      # this camera saw nothing
        anns.append({"id": aid, "image_id": ci, "bbox": [10.0, 0.0, 50.0, 50.0],
                     "keypoints": [1.0, 2.0, 2] * 50, "num_keypoints": 50,
                     "sex": "female", "behavior": "general"})
        aid += 1
    p.write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "Rat", "num_keypoints": 50}],
        "images": images, "annotations": anns,
        "framesets": {"rec_g/Frame_000100": {"datasetName": "rec_g",
                                             "frames": list(range(7))}},
        "calibrations": {"rec_g": {}}}))
    srcs = {"rec_g": SourceRec("rec_g", None, [str(p)], "", "")}
    out = tmp_path / "v5"; out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert list(merged["framesets"]) == ["rec_g/Frame_000100/fly0"]
    ids = merged["framesets"]["rec_g/Frame_000100/fly0"]["ann_ids"]
    assert ids[3] is None and sum(a is not None for a in ids) == 6


def test_category_renamed_from_rat(tmp_path):
    p = _coco(tmp_path, "one", "rec_one", n_flies=1)
    srcs = {"rec_one": SourceRec("rec_one", None, [p], "", "")}
    out = tmp_path / "v5"
    out.mkdir()
    merged = merge_annotations(srcs, str(out))
    assert merged["categories"][0]["name"] == "fly"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_build_v5.py -k merge -v`
Expected: FAIL — `ImportError: cannot import name 'merge_annotations'`

- [ ] **Step 3: Write the implementation**

```python
# append to $PKG/jarvis_jax/data/build_v5.py
import re
from collections import defaultdict

_FRAME_RE = re.compile(r"Frame_(\d+)")
# Minimum cameras a fly must appear in to be usable. 3 matches the robust
# triangulation path, which needs >=3 views for its consensus check.
MIN_CAMS = 3


def merge_annotations(sources: dict[str, SourceRec], out_root: str) -> dict:
    """Merge every source COCO file into one instances.json with per-fly identity.

    CONTROLLER RULING R15 -- how a fly's identity is resolved ACROSS cameras.
    Real courtship data frequently has one camera drop an annotation (occlusion),
    so per-camera annotation counts DISAGREE within a frameset. Three rules were
    considered and two are wrong:

      * `min(counts)` (the original draft) requires all 7 cameras to agree before
        counting a fly at all. It silently discarded 316 framesets with unequal
        counts, and dropped 8 recordings' worth of SINGLE-fly data too wherever
        some camera had zero annotations -- 91 framesets from 2026_01_13_18_47_45
        and 73 from 2026_03_22_12_07_40 among them. `2026_07_30_13_28_99`
        (wall_frames) collapsed to ZERO framesets.
      * `max(counts)` with positional indexing reaches the right TOTAL but creates
        CHIMERAS. Verified counterexample: in
        `2026_04_08_14_59_45/Frame_149677`, Cam2012631 carries a single annotation
        at bbox x=507 while the other six cameras carry two, at x~30-60 (left fly)
        and x~546-575 (right fly). That lone annotation is the RIGHT fly, so
        `cam_anns[0]` would file it as fly0 -- a fly whose views come from two
        different animals, which every downstream metric would report as fine.

    The rule used here: a camera whose annotation count EQUALS the frameset max
    contributes `cam_anns[k]` -- position-based identity is verified safe in that
    regime (two-fly framesets triangulate at 0.39 px, identical to single-fly
    controls). A camera that disagrees on the count cannot be assigned by
    position, so it is recorded as ABSENT (`None`) for every fly in that
    frameset. A fly is emitted only if at least MIN_CAMS cameras remain.

    Consumers must therefore treat `ann_ids` as possibly containing `None`.

    THE FIX: v3_3d.py mapped image_id -> FIRST annotation, commented "shouldn't
    happen for single-fly". It does happen -- on 2026_04_08 (214 framesets),
    2026_04_07 (30), and the two 2026_06_11 recordings (20) -- silently dropping
    264 complete framesets / 1,673 annotations. A chimera hypothesis (fly A in
    one camera mixed with fly B in another) was TESTED AND REFUTED: two-fly
    framesets triangulate at 0.39 px, identical to single-fly controls, so
    annotation order IS consistent across cameras. Ordering by annotation id
    within an image therefore assigns fly_id consistently.
    """
    out_images: list[dict] = []
    out_anns: list[dict] = []
    framesets: dict[str, dict] = {}
    kp_names: list[str] = []
    skeleton: list = []
    next_img, next_ann = 0, 0

    for rec, s in sorted(sources.items()):
        seen_paths: set[str] = set()
        img_key_to_id: dict[tuple[str, str], int] = {}
        anns_by_img: dict[int, list[dict]] = defaultdict(list)

        for ann_path in s.ann_paths:
            with open(ann_path) as f:
                blob = json.load(f)
            kp_names = kp_names or blob.get("keypoint_names", [])
            skeleton = skeleton or blob.get("skeleton", [])
            src_img = {i["id"]: i for i in blob["images"]}
            src_anns = defaultdict(list)
            for a in blob["annotations"]:
                src_anns[a["image_id"]].append(a)

            for key, fsv in blob.get("framesets", {}).items():
                if fsv.get("datasetName") != rec:
                    continue
                frame_no = _FRAME_RE.search(key)
                if frame_no is None:
                    continue
                frame = frame_no.group(1)
                # Deduplicate across the source's train/ and val/ files.
                if (rec, frame) in seen_paths:
                    continue
                seen_paths.add((rec, frame))

                per_cam: list[list[dict]] = []
                cam_img_ids: list[int] = []
                for iid in fsv["frames"]:
                    info = src_img.get(iid)
                    if info is None:
                        break
                    parts = info["file_name"].split("/")
                    cam = parts[-2]
                    ikey = (cam, frame)
                    if ikey not in img_key_to_id:
                        img_key_to_id[ikey] = next_img
                        out_images.append({
                            "id": next_img, "width": info["width"],
                            "height": info["height"], "recording": rec,
                            "file_name": f"{rec}/{cam}/Frame_{frame}.jpg"})
                        next_img += 1
                    cam_img_ids.append(img_key_to_id[ikey])
                    # Deterministic order => consistent fly_id across cameras.
                    per_cam.append(sorted(src_anns.get(iid, []), key=lambda a: a["id"]))

                if len(cam_img_ids) != len(fsv["frames"]) or not per_cam:
                    continue
                # CONTROLLER RULING R15 -- see the block comment above for why
                # this is NOT `min` (silently drops data) and NOT `max`
                # (creates chimeras).
                n_flies = max(len(v) for v in per_cam)
                if n_flies == 0:
                    continue

                for k in range(n_flies):
                    ann_ids = []
                    for img_id, cam_anns in zip(cam_img_ids, per_cam):
                        if len(cam_anns) != n_flies:
                            # This camera disagrees on how many flies it sees,
                            # so positional index k is NOT safe here. Record the
                            # camera as ABSENT for this fly rather than guessing.
                            ann_ids.append(None)
                            continue
                        a = cam_anns[k]
                        out_anns.append({
                            "id": next_ann, "image_id": img_id,
                            "bbox": a["bbox"], "keypoints": a["keypoints"],
                            "num_keypoints": a.get("num_keypoints", 50),
                            "sex": a.get("sex", "unknown"),
                            "behavior": a.get("behavior", "unknown"),
                            "fly_id": k,
                            "src_ann_id": a["id"],   # masks are keyed by this
                        })
                        ann_ids.append(next_ann)
                        next_ann += 1
                    if sum(a is not None for a in ann_ids) < MIN_CAMS:
                        continue   # too few views to triangulate; drop the fly
                    framesets[f"{rec}/Frame_{frame}/fly{k}"] = {
                        "recording": rec, "fly_id": k,
                        "frames": cam_img_ids, "ann_ids": ann_ids}

        s.n_framesets = len({k for k in framesets if k.startswith(f"{rec}/")})
        s.n_flies = sum(1 for k in framesets if k.startswith(f"{rec}/"))

    merged = {
        "keypoint_names": kp_names, "skeleton": skeleton,
        "categories": [{"id": 1, "name": "fly", "num_keypoints": 50}],
        "images": out_images, "annotations": out_anns, "framesets": framesets,
    }
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, "instances.json"), "w") as f:
        json.dump(merged, f)
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_build_v5.py -v`
Expected: 10 passed

- [ ] **Step 5: Verify the recovery count against the real data**

```bash
cd $PKG && python - <<'PY'
from jarvis_jax.data.build_v5 import discover_sources, merge_annotations
import collections, tempfile
G = "/gscratch/portia/eabe/data/Johnson_lab/red_data/general_model"
V = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3"
srcs = discover_sources(G, V)
with tempfile.TemporaryDirectory() as td:
    m = merge_annotations(srcs, td)
per_rec = collections.Counter(v["recording"] for v in m["framesets"].values())
two = {r: sum(1 for k, v in m["framesets"].items()
              if v["recording"] == r and v["fly_id"] == 1)
       for r in per_rec}
two = {r: n for r, n in two.items() if n}
print("recordings:", len(per_rec))
print("total fly-samples:", len(m["framesets"]))
print("second flies by recording:", two, "total", sum(two.values()))
PY
```

**CONTROLLER RULING R14 -- the earlier "exactly 264" gate was WRONG** and is
replaced. That number came from a probe counting framesets where at least ONE
camera has a second annotation; the algorithm counts something different. Both
numbers are real, they answer different questions. Re-measured under the R15
rule:

Expected: **26 recordings**; second flies **260 total** —
`2026_04_08_14_59_45` **211**, `2026_04_07_11_33_33` **30**,
`2026_06_11_13_58_43` **16**, `2026_06_11_13_58_45` **3**.

Also assert the regression R15 fixes: **`2026_07_30_13_28_99` (wall_frames) must
yield MORE THAN ZERO framesets.** Under the old `min` rule it produced none,
because some camera has zero annotations in every one of its framesets.

If the second-fly total is not 260, stop and reconcile — do NOT adjust the
algorithm to reach the number.

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/data/build_v5.py $PKG/tests/test_build_v5.py
git commit -m "fix(data): recover the 264 framesets v3_3d silently dropped

image_id -> FIRST annotation discarded 1,673 annotations across 264
two-fly framesets, 244 of them in bout 28's own calibration group. The
dropped fly triangulates at 0.39 px -- identical to the kept fly.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Leakage-free split generation

**Files:**
- Create: `$PKG/jarvis_jax/data/split_v5.py`
- Test: `$PKG/tests/test_split_v5.py`

**Interfaces:**
- Consumes: `merge_annotations` output; `manifest.json`.
- Produces:
  - `make_split(merged, manifest, *, val_recordings, female_val_frac=0.10, guard=50, seed=0) -> dict` mapping frameset key → `"train"` or `"val"`, written to `annotations/split.json`
  - `audit_split(merged, split, *, guard=50) -> dict` returning `{"cross_recording_leaks": int, "min_guard_distance": dict[str, int], "val_frac": float}`
  - `write_derived(merged, split, out_root) -> None` writing `instances_train.json` / `instances_val.json`

Policy: recordings whose manifest `sex == "female"` keep **every frameset in
training** except a contiguous tail block of `female_val_frac`, separated from
train by at least `guard` frames. All other recordings are held out whole.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_split_v5.py
import pytest
from jarvis_jax.data.split_v5 import make_split, audit_split

def _merged(recs):
    """recs: {recording: n_frames}. One fly per frameset, frames spaced by 1."""
    fs = {}
    for rec, n in recs.items():
        for i in range(n):
            fs[f"{rec}/Frame_{i:06d}/fly0"] = {"recording": rec, "fly_id": 0,
                                               "frames": [], "ann_ids": []}
    return {"framesets": fs}

def _manifest(recs, female=()):
    return {"recordings": {r: {"sex": "female" if r in female else "male"}
                           for r in recs}}

def test_whole_recording_holdout_has_zero_overlap():
    m = _merged({"rec_a": 100, "rec_b": 100})
    split = make_split(m, _manifest(["rec_a", "rec_b"]), val_recordings=["rec_b"])
    assert {k.split("/")[0] for k, v in split.items() if v == "val"} == {"rec_b"}
    assert {k.split("/")[0] for k, v in split.items() if v == "train"} == {"rec_a"}
    assert audit_split(m, split)["cross_recording_leaks"] == 0

def test_female_recording_stays_in_training_with_guarded_tail_val():
    """Female data is the binding constraint (199 framesets, 7%). A female
    recording must NOT be held out whole -- only a guarded tail block."""
    m = _merged({"fem": 200})
    split = make_split(m, _manifest(["fem"], female=["fem"]),
                       val_recordings=[], female_val_frac=0.10, guard=50)
    train = [k for k, v in split.items() if v == "train"]
    val = [k for k, v in split.items() if v == "val"]
    assert len(val) == 20 and len(train) > 0
    # val must be the TAIL, contiguous
    val_frames = sorted(int(k.split("Frame_")[1].split("/")[0]) for k in val)
    assert val_frames == list(range(180, 200))

def test_guard_band_separates_female_train_from_val():
    m = _merged({"fem": 200})
    split = make_split(m, _manifest(["fem"], female=["fem"]),
                       female_val_frac=0.10, guard=50, val_recordings=[])
    a = audit_split(m, split, guard=50)
    assert a["min_guard_distance"]["fem"] >= 50, a

def test_audit_detects_a_leaky_split():
    """Regression guard for the exact defect in the old data: adjacent frames
    split across train and val."""
    m = _merged({"rec_a": 100})
    bad = {k: ("val" if i % 10 == 0 else "train")
           for i, k in enumerate(sorted(m["framesets"]))}
    a = audit_split(m, bad, guard=50)
    assert a["min_guard_distance"]["rec_a"] < 50

def test_empty_val_is_rejected():
    m = _merged({"rec_a": 10})
    with pytest.raises(ValueError, match="empty val"):
        make_split(m, _manifest(["rec_a"]), val_recordings=[], female_val_frac=0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_split_v5.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.data.split_v5'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/data/split_v5.py
"""Split red_data_3d_v5 without leakage.

THE DEFECT THIS REPLACES: general_model split frame-level WITHIN each recording,
so ~50% of val framesets had a train frameset within +/-3 frames (courtship_V3
86%, S8_male_R_amp 100%). Every 3D val number produced before 2026-08-29 --
MPJPE 1.083, 0.708, and the laplacian ablation that set cached3d.yaml -- scored
near-duplicate frames.

POLICY. Male/general recordings are data-rich, so they are held out WHOLE: zero
leakage, and the val number means "generalizes to an unseen fly and session".
Female recordings are the binding constraint (199 framesets, 7% of the data),
so every one stays in training and val is a contiguous tail block separated by a
guard band. That measures "new pose, same fly and session" and WILL read
optimistic against bout 28 -- an accepted, documented limit. The real female
test is bout 28 under GT-free metrics plus rendered overlays.
"""
from __future__ import annotations

import json
import os
import re

_FRAME_RE = re.compile(r"Frame_(\d+)")
# Minimum cameras a fly must appear in to be usable. 3 matches the robust
# triangulation path, which needs >=3 views for its consensus check.
MIN_CAMS = 3


def _frame_no(key: str) -> int:
    return int(_FRAME_RE.search(key).group(1))


def _by_recording(merged: dict) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for k, v in merged["framesets"].items():
        out.setdefault(v["recording"], []).append(k)
    for rec in out:
        out[rec].sort(key=_frame_no)
    return out


def make_split(merged: dict, manifest: dict, *, val_recordings=(),
               female_val_frac: float = 0.10, guard: int = 50,
               seed: int = 0) -> dict:
    groups = _by_recording(merged)
    recs_meta = manifest.get("recordings", {})
    split: dict[str, str] = {}
    val_recordings = set(val_recordings)

    for rec, keys in groups.items():
        is_female = recs_meta.get(rec, {}).get("sex") == "female"
        if rec in val_recordings and not is_female:
            for k in keys:
                split[k] = "val"
            continue
        if not is_female:
            for k in keys:
                split[k] = "train"
            continue

        # Female: contiguous tail block for val, guard band dropped entirely.
        n_val = int(round(len(keys) * female_val_frac))
        if n_val == 0:
            for k in keys:
                split[k] = "train"
            continue
        val_keys = keys[-n_val:]
        val_start = _frame_no(val_keys[0])
        for k in keys[:-n_val]:
            if val_start - _frame_no(k) <= guard:
                continue          # guard band: belongs to neither split
            split[k] = "train"
        for k in val_keys:
            split[k] = "val"

    if not any(v == "val" for v in split.values()):
        raise ValueError("empty val split — every frameset landed in train")
    return split


def audit_split(merged: dict, split: dict, *, guard: int = 50) -> dict:
    """Prove the split does not leak. This is the test the old data would fail."""
    groups = _by_recording(merged)
    leaks = 0
    min_dist: dict[str, int] = {}
    for rec, keys in groups.items():
        tr = [_frame_no(k) for k in keys if split.get(k) == "train"]
        va = [_frame_no(k) for k in keys if split.get(k) == "val"]
        if not tr or not va:
            continue
        # Same recording in BOTH splits: only legal for the female guard-band case.
        leaks += 0
        tr_s = sorted(tr)
        d = min(min(abs(v - t) for t in tr_s) for v in va)
        min_dist[rec] = d
    for rec, keys in groups.items():
        tr = any(split.get(k) == "train" for k in keys)
        va = any(split.get(k) == "val" for k in keys)
        if tr and va and min_dist.get(rec, 0) < guard:
            leaks += 1
    n = sum(1 for v in split.values() if v in ("train", "val"))
    return {"cross_recording_leaks": leaks, "min_guard_distance": min_dist,
            "val_frac": sum(1 for v in split.values() if v == "val") / max(n, 1)}


def write_derived(merged: dict, split: dict, out_root: str) -> None:
    """Emit instances_{train,val}.json DERIVED from instances.json + split.json."""
    ann_dir = os.path.join(out_root, "annotations")
    os.makedirs(ann_dir, exist_ok=True)
    with open(os.path.join(ann_dir, "split.json"), "w") as f:
        json.dump(split, f, indent=2)

    by_id = {a["id"]: a for a in merged["annotations"]}
    img_by_id = {i["id"]: i for i in merged["images"]}
    for name in ("train", "val"):
        keys = [k for k, v in split.items() if v == name]
        fs = {k: merged["framesets"][k] for k in keys}
        ann_ids = {a for v in fs.values() for a in v["ann_ids"]}
        img_ids = {i for v in fs.values() for i in v["frames"]}
        blob = {
            "keypoint_names": merged["keypoint_names"],
            "skeleton": merged["skeleton"],
            "categories": merged["categories"],
            "images": [img_by_id[i] for i in sorted(img_ids)],
            "annotations": [by_id[a] for a in sorted(ann_ids)],
            "framesets": fs,
        }
        with open(os.path.join(ann_dir, f"instances_{name}.json"), "w") as f:
            json.dump(blob, f)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_split_v5.py -v`
Expected: 5 passed

- [ ] **Step 5: Prove the audit detects a genuinely leaky split**

**Controller ruling R11:** an earlier draft pointed this at
`red_data_unified_V3/annotations`, which is already RECORDING-level split
(13 train / 4 val, zero overlap) and comes back CLEAN — it does not exercise
the audit at all. The genuinely leaky source is
`general_model/<subset>/annotations/instances_{train,val}.json`, where each
subset splits the SAME recording frame-level. Point the audit there and expect
**20 of 21 recordings to fail a 50-frame guard, worst case 1 frame apart**,
reproducing courtship_V3 at 86% and S8_male_R_amp at 100% on the ±3-frame check.
If it comes back clean, the audit is not measuring what it claims.

```bash
cd $PKG && python - <<'PY'
"""EXPECTATION: audit_split must report a sub-guard distance for the old
frame-level split. If it reports a clean split, the audit is not measuring
what we think and Task 5 is not done."""
import json, re
from jarvis_jax.data.split_v5 import audit_split
V = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_unified_V3/annotations"
fs, split = {}, {}
for name in ("train", "val"):
    d = json.load(open(f"{V}/instances_{name}.json"))
    img = {i["id"]: i["file_name"] for i in d["images"]}
    for k, v in d["framesets"].items():
        fs[k] = {"recording": v["datasetName"]}
        split[k] = name
a = audit_split({"framesets": fs}, split, guard=50)
bad = {r: v for r, v in a["min_guard_distance"].items() if v < 50}
print("recordings failing the 50-frame guard:", len(bad), "of",
      len(a["min_guard_distance"]))
print("worst:", sorted(bad.items(), key=lambda kv: kv[1])[:6])
PY
```

Expected: most recordings fail with distances of 1–3 frames. This is the
regression the new builder prevents.

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/data/split_v5.py $PKG/tests/test_split_v5.py
git commit -m "feat(data): leakage-free split with female guard bands

Whole-recording holdout for male/general; female recordings all stay in
training with a guarded tail block, because female data is 7% of the set
and is the binding constraint on the goal.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Contact sheets and the manual sexing pass

54% of framesets are `sex: unknown`, including the three big Group-A courtship
recordings (677 framesets in bout 28's exact calibration). Sex is a
per-recording property, so this is ~30 judgements against 30 images.

**Files:**
- Create: `<repo>/scripts/viz/contact_sheet.py` (committed — a regenerable figure)
- Modify: `$PKG/jarvis_jax/data/build_v5.py` (add `apply_sex_labels`)
- Test: `<repo>/tests/test_contact_sheet.py`

**Interfaces:**
- Consumes: `manifest.json`, `annotations/instances.json`, `images/`.
- Produces:
  - `build_contact_sheet(v5_root, recording, *, n_frames=6, cams=("Cam2012630","Cam2012855"), out_dir) -> str` returning the PNG path
  - `apply_sex_labels(v5_root: str, labels: dict[str, str]) -> dict` writing `sex` into `manifest.json`; accepts `"male"`, `"female"`, `"unknown"`; for two-fly recordings the key is `"<recording>/fly<k>"`

- [ ] **Step 1: Write the failing test**

```python
# <repo>/tests/test_contact_sheet.py
import json
import os
import numpy as np
import pytest
from PIL import Image

from scripts.viz.contact_sheet import build_contact_sheet
from jarvis_jax.data.build_v5 import apply_sex_labels


def _tiny_v5(tmp_path):
    root = tmp_path / "v5"
    (root / "images" / "rec_a" / "Cam2012630").mkdir(parents=True)
    (root / "images" / "rec_a" / "Cam2012855").mkdir(parents=True)
    for cam in ("Cam2012630", "Cam2012855"):
        for i in range(3):
            Image.fromarray(np.zeros((448, 1936, 3), np.uint8)).save(
                root / "images" / "rec_a" / cam / f"Frame_{i:06d}.jpg")
    imgs, anns, fs = [], [], {}
    iid = 0
    for i in range(3):
        ids = []
        for cam in ("Cam2012630", "Cam2012855"):
            imgs.append({"id": iid, "width": 1936, "height": 448, "recording": "rec_a",
                         "file_name": f"rec_a/{cam}/Frame_{i:06d}.jpg"})
            anns.append({"id": iid, "image_id": iid, "bbox": [10.0, 10.0, 60.0, 60.0],
                         "keypoints": [20.0, 20.0, 2] * 50, "num_keypoints": 50,
                         "sex": "unknown", "behavior": "courtship", "fly_id": 0})
            ids.append(iid)
            iid += 1
        fs[f"rec_a/Frame_{i:06d}/fly0"] = {"recording": "rec_a", "fly_id": 0,
                                           "frames": ids, "ann_ids": ids}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances.json").write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": imgs, "annotations": anns, "framesets": fs}))
    (root / "manifest.json").write_text(json.dumps(
        {"recordings": {"rec_a": {"sex": "unknown"}}}))
    return root


def test_contact_sheet_written_and_non_blank(tmp_path):
    root = _tiny_v5(tmp_path)
    out = build_contact_sheet(str(root), "rec_a", n_frames=2,
                              cams=("Cam2012630", "Cam2012855"),
                              out_dir=str(tmp_path / "sheets"))
    assert os.path.exists(out)
    arr = np.asarray(Image.open(out).convert("RGB"))
    assert arr.max() > 0, "sheet is entirely black — keypoints were not drawn"


def test_apply_sex_labels_updates_manifest(tmp_path):
    root = _tiny_v5(tmp_path)
    man = apply_sex_labels(str(root), {"rec_a": "female"})
    assert man["recordings"]["rec_a"]["sex"] == "female"
    assert json.load(open(root / "manifest.json"))["recordings"]["rec_a"]["sex"] == "female"


def test_apply_sex_labels_rejects_bad_value(tmp_path):
    root = _tiny_v5(tmp_path)
    with pytest.raises(ValueError, match="sex must be"):
        apply_sex_labels(str(root), {"rec_a": "F"})


def test_apply_sex_labels_rejects_unknown_recording(tmp_path):
    root = _tiny_v5(tmp_path)
    with pytest.raises(KeyError, match="not in manifest"):
        apply_sex_labels(str(root), {"nope": "male"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo> && python -m pytest tests/test_contact_sheet.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.viz.contact_sheet'`

- [ ] **Step 3: Write the implementation**

```python
# <repo>/scripts/viz/contact_sheet.py
"""Per-recording contact sheets for the manual sexing pass.

WHY THIS EXISTS: 54% of red_data framesets carry sex: "unknown", including the
three big Group-A courtship recordings (677 framesets in bout 28's own
calibration). Sex is a per-RECORDING property, so ~30 judgements settle it --
far cheaper than a GUI.

WHAT TO LOOK FOR: the male is smaller with a darker, blunter abdomen tip and
carries sex combs on the T1 tarsi; the female is larger with a pointed
ovipositor. NOTE the sexing gotcha recorded in ab-hybridnet-vs-dlt-ik: mask
AREA is backwards as a size cue during courtship, because the male extends a
wing during song and so has the LARGER silhouette. Judge by body shape, not
extent.

EXPECTATION for a correct sheet: every crop shows one fly filling most of the
frame with keypoints landing on eyes, thorax, abdomen and legs. Keypoints
scattered into empty space mean the annotation-to-image mapping is wrong and
the sheet must not be used for sexing.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from viz.core.colors import PALETTE, keypoint_groups


def _crop_box(bbox, w, h, pad=1.4, size=256):
    x, y, bw, bh = bbox
    cx, cy = x + bw / 2, y + bh / 2
    half = max(bw, bh) * pad / 2
    x0 = int(np.clip(cx - half, 0, max(w - 1, 0)))
    y0 = int(np.clip(cy - half, 0, max(h - 1, 0)))
    x1 = int(np.clip(cx + half, 1, w))
    y1 = int(np.clip(cy + half, 1, h))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def build_contact_sheet(v5_root: str, recording: str, *, n_frames: int = 6,
                        cams=("Cam2012630", "Cam2012855"),
                        out_dir: str, cell: int = 256, fly_id: int = 0) -> str:
    with open(os.path.join(v5_root, "annotations", "instances.json")) as f:
        blob = json.load(f)
    img_by_id = {i["id"]: i for i in blob["images"]}
    ann_by_id = {a["id"]: a for a in blob["annotations"]}
    keys = sorted(k for k, v in blob["framesets"].items()
                  if v["recording"] == recording and v["fly_id"] == fly_id)
    if not keys:
        raise ValueError(f"no framesets for {recording} fly{fly_id}")
    pick = keys[:: max(1, len(keys) // max(n_frames, 1))][:n_frames]

    groups = keypoint_groups(blob["keypoint_names"])
    sheet = Image.new("RGB", (cell * len(pick), cell * len(cams)), (12, 12, 14))
    draw = ImageDraw.Draw(sheet)

    for col, key in enumerate(pick):
        fs = blob["framesets"][key]
        for row, cam in enumerate(cams):
            hit = next(((i, a) for i, a in zip(fs["frames"], fs["ann_ids"])
                        if img_by_id[i]["file_name"].split("/")[-2] == cam), None)
            if hit is None:
                continue
            img_id, ann_id = hit
            info, ann = img_by_id[img_id], ann_by_id[ann_id]
            path = os.path.join(v5_root, "images", info["file_name"])
            if not os.path.exists(path):
                continue
            with Image.open(path) as pil:
                im = pil.convert("RGB")
            x0, y0, x1, y1 = _crop_box(ann["bbox"], info["width"], info["height"])
            crop = im.crop((x0, y0, x1, y1)).resize((cell, cell))
            kp = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
            cd = ImageDraw.Draw(crop)
            sx, sy = cell / (x1 - x0), cell / (y1 - y0)
            for j, (px, py, v) in enumerate(kp):
                if v <= 0:
                    continue
                cx, cy = (px - x0) * sx, (py - y0) * sy
                if not (0 <= cx < cell and 0 <= cy < cell):
                    continue
                colour = PALETTE.get(groups.get(j, "other"), (255, 255, 255))
                cd.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=tuple(colour))
            sheet.paste(crop, (col * cell, row * cell))
            draw.text((col * cell + 4, row * cell + 4), f"{cam}", fill=(230, 230, 230))
        draw.text((col * cell + 4, cell * len(cams) - 14),
                  key.split("/")[1], fill=(180, 180, 180))

    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{recording}_fly{fly_id}.png")
    sheet.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v5-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--recordings", nargs="*", default=None)
    ap.add_argument("--n-frames", type=int, default=6)
    args = ap.parse_args()
    man = json.load(open(os.path.join(args.v5_root, "manifest.json")))
    recs = args.recordings or sorted(man["recordings"])
    with open(os.path.join(args.v5_root, "annotations", "instances.json")) as f:
        blob = json.load(f)
    for rec in recs:
        flies = sorted({v["fly_id"] for v in blob["framesets"].values()
                        if v["recording"] == rec})
        for k in flies:
            print(build_contact_sheet(args.v5_root, rec, n_frames=args.n_frames,
                                      out_dir=args.out_dir, fly_id=k))


if __name__ == "__main__":
    main()
```

```python
# append to $PKG/jarvis_jax/data/build_v5.py
_VALID_SEX = {"male", "female", "unknown"}


def apply_sex_labels(v5_root: str, labels: dict[str, str]) -> dict:
    """Write manually-determined sex into manifest.json.

    Keys are recording names, or '<recording>/fly<k>' for the four two-fly
    recordings where the two slots differ.
    """
    path = os.path.join(v5_root, "manifest.json")
    with open(path) as f:
        man = json.load(f)
    for key, sex in labels.items():
        if sex not in _VALID_SEX:
            raise ValueError(f"sex must be one of {sorted(_VALID_SEX)}, got {sex!r}")
        rec = key.split("/")[0]
        if rec not in man["recordings"]:
            raise KeyError(f"{rec!r} not in manifest")
        if "/" in key:
            man["recordings"][rec].setdefault("fly_sex", {})[key.split("/")[1]] = sex
        else:
            man["recordings"][rec]["sex"] = sex
    with open(path, "w") as f:
        json.dump(man, f, indent=2)
    return man
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo> && python -m pytest tests/test_contact_sheet.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit the tooling**

```bash
git add scripts/viz/contact_sheet.py tests/test_contact_sheet.py \
        third_party/jarvis_jax/jarvis_jax/data/build_v5.py
git commit -m "feat(viz): per-recording contact sheets for the manual sexing pass

54% of framesets are sex:unknown, including 677 in bout 28's calibration
group. Sex is per-recording, so ~30 judgements settle it.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: STOP — hand the sheets to the user**

This step is **not automatable**. After Task 7 has run the full build, generate
the sheets and present them:

```bash
python scripts/viz/contact_sheet.py \
  --v5-root /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5 \
  --out-dir figures/2026-08-29-c2f-3d/contact-sheets
```

Read several sheets with the Read tool first and confirm keypoints land on
anatomy (eyes on eyes, legs on legs). If they do not, the annotation-to-image
mapping is wrong — fix that before asking anyone to sex anything. Then ask the
user for a `recording -> sex` mapping and apply it with `apply_sex_labels`.
Do not guess sex from the images yourself.

**Controller ruling R2 — REGENERATE THE SPLIT AFTERWARDS.** `make_split` reads
`manifest["recordings"][rec]["sex"]`, but Task 7 builds before this sexing pass,
so on the first build no recording is female and the female guard-band path
never fires. After `apply_sex_labels`, re-run the (idempotent) build so
`split.json` and the derived `instances_{train,val}.json` reflect known sex:

```bash
cd $PKG && python scripts/build_v5_dataset.py paths=hyak
python - <<'PY'
import json
V5 = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5"
man = json.load(open(f"{V5}/manifest.json"))["recordings"]
fem = [r for r, v in man.items() if v.get("sex") == "female"]
split = json.load(open(f"{V5}/annotations/split.json"))
val_fem = [k for k, v in split.items() if v == "val" and k.split("/")[0] in fem]
print("female recordings:", len(fem), "-> female val framesets:", len(val_fem))
PY
```

Expected: a non-zero female val count. **Zero means the guard-band path still
did not fire and the female val metric does not exist** — the whole reason this
step is here.

---

## Task 7: Build entrypoint and SAM3 masks for the 9 new recordings (Phase 1b)

**Files:**
- Create: `$PKG/scripts/build_v5_dataset.py`
- Modify: `$PKG/configs/paths/hyak.yaml`

**Interfaces:**
- Consumes: `discover_sources`, `build_manifest`, `link_media`, `merge_annotations`, `make_split`, `audit_split`, `write_derived`.
- Produces: a populated `$V5` tree and `build_report.json`.

- [ ] **Step 1: Add the v5 path**

```yaml
# append to $PKG/configs/paths/hyak.yaml
v5_root:        /gscratch/portia/${paths.user}/data/Johnson_lab/red_data/red_data_3d_v5
general_model:  /gscratch/portia/${paths.user}/data/Johnson_lab/red_data/general_model
v3_root:        /gscratch/portia/${paths.user}/data/Johnson_lab/red_data/red_data_unified_V3
```

- [ ] **Step 2: Write the build entrypoint**

```python
# $PKG/scripts/build_v5_dataset.py
"""Build red_data_3d_v5 end to end.

    python scripts/build_v5_dataset.py paths=hyak

Idempotent: re-running relinks nothing that already exists and rewrites the
annotation/split JSON from scratch.
"""
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)
if _PKG not in sys.path:
    sys.path.insert(0, _PKG)

import hydra

from jarvis_jax.hydra_utils import CONFIG_DIR, register_resolvers
from jarvis_jax.data.build_v5 import (
    discover_sources, build_manifest, link_media, merge_annotations)
from jarvis_jax.data.split_v5 import make_split, audit_split, write_derived

register_resolvers()

# Held out WHOLE. Chosen to resemble bout 28: a Group-A courtship recording
# plus one from each other calibration group so per-group val is reportable.
VAL_RECORDINGS = [
    "2026_03_18_15_31_22",   # group A, courtship  (137 framesets)
    "2026_06_15_12_12_33",   # group B, courtship male
    "2026_05_27_11_57_05",   # group C, courtship male
]


@hydra.main(config_path=CONFIG_DIR, config_name="config", version_base=None)
def main(cfg):
    out = cfg.paths.v5_root
    os.makedirs(out, exist_ok=True)

    srcs = discover_sources(cfg.paths.general_model, cfg.paths.v3_root)
    print(f"discovered {len(srcs)} recordings")

    merged = merge_annotations(srcs, out)
    print(f"merged {len(merged['framesets'])} fly-samples, "
          f"{len(merged['annotations'])} annotations")

    man = build_manifest(srcs, out)
    link_media(srcs, out)

    split = make_split(merged, man, val_recordings=VAL_RECORDINGS)
    audit = audit_split(merged, split)
    print("split audit:", json.dumps(audit, indent=2)[:400])
    if audit["cross_recording_leaks"]:
        raise SystemExit(f"REFUSING to write a leaky split: {audit}")
    write_derived(merged, split, out)

    for k, v in split.items():
        man["recordings"][merged["framesets"][k]["recording"]]["split"] = (
            "mixed" if man["recordings"][merged["framesets"][k]["recording"]]
            .get("split") not in (None, v) else v)
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=2)
    with open(os.path.join(out, "build_report.json"), "w") as f:
        json.dump({"n_recordings": len(srcs),
                   "n_fly_samples": len(merged["framesets"]),
                   "n_annotations": len(merged["annotations"]),
                   "split_audit": audit,
                   "val_recordings": VAL_RECORDINGS}, f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the build**

```bash
cd $PKG && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
python scripts/build_v5_dataset.py paths=hyak 2>&1 | tail -25
```

Expected: `discovered 26 recordings`; ~3,800 fly-samples; `cross_recording_leaks: 0`.
If leaks > 0 the script refuses to write — fix the split before continuing.

- [ ] **Step 4: Generate masks for the 9 recordings without them**

These 9 were never in 3D training and have no SAM3 masks; the detector is
4-channel (RGB + mask) so they are unusable until this runs. `20_04_female_climbing`
and `wall_frames` are the reason we want them — female climbing and wall poses
are exactly the OOD cases bout 28's female needs.

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset/third_party/JARVIS-HybridNet
micromamba activate 3d_tracking
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia/cu13/lib"
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3
V5=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5
python tools/sam3_label_masks.py \
  --data-root "$V5" --out-root "$V5/masks_new" \
  --splits train val --text-prompt insect --confidence 0.5 \
  --qc-thresh 0.60 --resolution 1008 2>&1 | tail -20
```

Expected: ~5,160 images processed, `anns_matched / anns_total` above 0.95 in
`$V5/masks_new/mask_report.json`. A much lower match rate means the prompt or
QC threshold does not transfer to these recordings — inspect before proceeding.

- [ ] **Step 5: Merge the new masks into the flat tree and re-link**

```bash
V5=/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5
python - <<'PY'
import os, shutil
V5 = os.environ.get("V5", "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5")
src = os.path.join(V5, "masks_new")
moved = 0
for root, _, files in os.walk(src):
    for fn in files:
        if not fn.endswith(".npz"):
            continue
        rel = os.path.relpath(os.path.join(root, fn), src).split(os.sep)
        rec, cam = rel[-3], rel[-2]
        dst = os.path.join(V5, "masks", rec, cam)
        os.makedirs(dst, exist_ok=True)
        d = os.path.join(dst, fn)
        if not os.path.lexists(d):
            shutil.move(os.path.join(root, fn), d)
            moved += 1
print("masks merged:", moved)
PY
```

Expected: ~5,160 mask files merged.

- [ ] **Step 6: Read the mask overlays for the two OOD recordings**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
python scripts/viz/contact_sheet.py --v5-root $V5 \
  --recordings 2026_08_26_16_05_15 2026_07_30_13_28_99 \
  --out-dir figures/2026-08-29-c2f-3d/mask-check
```

**Expectation:** each crop shows one fly with keypoints on anatomy. Open both
PNGs with the Read tool and state what they show. If keypoints scatter into
empty space, the mask/annotation matching failed for these recordings and they
must be excluded rather than trained on.

- [ ] **Step 7: Commit**

```bash
git add $PKG/scripts/build_v5_dataset.py $PKG/configs/paths/hyak.yaml
git commit -m "feat(data): v5 build entrypoint; masks for the 9 unused recordings

26 recordings, ~3,800 fly-samples, split refuses to write if it leaks.
Adds both amputee sets, all 5 headless, wall_frames and female climbing —
never used in 3D training before.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: The v5 frameset loader

**Files:**
- Create: `$PKG/jarvis_jax/data/v5_3d.py`
- Test: `$PKG/tests/test_v5_3d.py`

**Interfaces:**
- Consumes: `$V5/annotations/instances_{train,val}.json`, `$V5/images`, `$V5/masks`, `$V5/calibrations/<group>`. **`ann_ids` may contain `None`** (ruling R15) — such cameras contribute a black crop and invisible keypoints.
- Produces: `V5FramesetDataset(root, split, *, recordings=None, calib_groups=None, sex=None)` with `__len__`, `__getitem__` returning the SAME dict schema as `V3FramesetDataset` (`crops4`, `centerHM`, `center3D`, `cameraMatrices`, `kp3d`, `vis`) plus `fly_id` and `calib_group`; and `frameset_batches(ds, batch_size, *, shuffle, seed, drop_last, female_weight=1.0)`.

The schema match is deliberate: `precompute_repro_cache.py` and
`train_3d_cached.py` consume it unchanged.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_v5_3d.py
import json
import numpy as np
import pytest
from PIL import Image

from jarvis_jax.data.v5_3d import V5FramesetDataset, frameset_batches

CAMS7 = ["Cam2012630", "Cam2012631", "Cam2012853", "Cam2012855",
         "Cam2012857", "Cam2012861", "Cam2012862"]


def _v5(tmp_path, n_flies=1, n_frames=2):
    root = tmp_path / "v5"
    (root / "calibrations" / "A").mkdir(parents=True)
    for ci, c in enumerate(CAMS7):
        data = ", ".join(str(v) for v in
                         [8.1 + ci * 0.01, 0.007, -0.03, -2.8,
                          0.009, -8.07, -0.179, 462.7, 0.0, 0.0, 0.0, 1.0])
        (root / "calibrations" / "A" / f"{c}.yaml").write_text(
            "%YAML:1.0\n---\nimage_width: 1936\nimage_height: 448\n"
            "projectionMatrix: !!opencv-matrix\n   rows: 3\n   cols: 4\n   dt: d\n"
            f"   data: [ {data} ]\nscale: 10\n")
    imgs, anns, fs = [], [], {}
    iid = aid = 0
    for f in range(n_frames):
        per_fly = {k: [] for k in range(n_flies)}
        ids = []
        for cam in CAMS7:
            d = root / "images" / "rec_a" / cam
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.zeros((448, 1936, 3), np.uint8)).save(
                d / f"Frame_{f:06d}.jpg")
            imgs.append({"id": iid, "width": 1936, "height": 448, "recording": "rec_a",
                         "file_name": f"rec_a/{cam}/Frame_{f:06d}.jpg"})
            ids.append(iid)
            for k in range(n_flies):
                anns.append({"id": aid, "image_id": iid,
                             "bbox": [800.0 + k * 200, 100.0, 80.0, 80.0],
                             "keypoints": [900.0 + k * 200, 150.0, 2] * 50,
                             "num_keypoints": 50, "sex": "female", "fly_id": k,
                             "behavior": "courtship", "src_ann_id": aid})
                per_fly[k].append(aid)
                aid += 1
            iid += 1
        for k in range(n_flies):
            fs[f"rec_a/Frame_{f:06d}/fly{k}"] = {
                "recording": "rec_a", "fly_id": k,
                "frames": ids, "ann_ids": per_fly[k]}
    (root / "annotations").mkdir()
    (root / "annotations" / "instances_train.json").write_text(json.dumps({
        "keypoint_names": [f"kp{i}" for i in range(50)], "skeleton": [],
        "categories": [{"id": 1, "name": "fly"}],
        "images": imgs, "annotations": anns, "framesets": fs}))
    (root / "manifest.json").write_text(json.dumps({"recordings": {
        "rec_a": {"calib_group": "A", "sex": "female"}}}))
    return root


def test_two_fly_recording_yields_two_samples(tmp_path):
    root = _v5(tmp_path, n_flies=2, n_frames=2)
    ds = V5FramesetDataset(str(root), "train")
    assert len(ds) == 4          # 2 frames x 2 flies
    assert {ds[i]["fly_id"] for i in range(len(ds))} == {0, 1}


def test_sample_schema_matches_v3_loader(tmp_path):
    root = _v5(tmp_path)
    s = V5FramesetDataset(str(root), "train")[0]
    assert s["crops4"].shape == (7, 448, 448, 4) and s["crops4"].dtype == np.uint8
    assert s["centerHM"].shape == (7, 2)
    assert s["center3D"].shape == (3,)
    assert s["cameraMatrices"].shape == (7, 4, 3)
    assert s["kp3d"].shape == (50, 3)
    assert s["vis"].shape == (50,) and s["vis"].dtype == bool


def test_calib_group_filter(tmp_path):
    root = _v5(tmp_path)
    assert len(V5FramesetDataset(str(root), "train", calib_groups=["A"])) > 0
    assert len(V5FramesetDataset(str(root), "train", calib_groups=["B"])) == 0


def test_female_weight_oversamples_female_framesets(tmp_path):
    root = _v5(tmp_path, n_frames=4)
    ds = V5FramesetDataset(str(root), "train")
    n = sum(len(b["kp3d"]) for b in frameset_batches(
        ds, 2, shuffle=True, seed=0, drop_last=False, female_weight=3.0))
    assert n > len(ds), "female_weight>1 must repeat female framesets"


def test_missing_mask_is_zeros_not_a_crash(tmp_path):
    """The 9 newly-added recordings may have gaps; a missing mask must degrade
    to an empty 4th channel, never raise."""
    root = _v5(tmp_path)
    s = V5FramesetDataset(str(root), "train")[0]
    assert s["crops4"][..., 3].max() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_v5_3d.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.data.v5_3d'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/data/v5_3d.py
"""Per-fly frameset dataset for red_data_3d_v5.

Differs from data/v3_3d.py in exactly three ways, all deliberate:
  1. A frameset is keyed by (recording, frame, FLY) rather than (recording,
     frame), so the 264 two-fly framesets contribute both flies instead of the
     first one only.
  2. Media paths are flat -- images/<rec>/<cam>/Frame_*.jpg with no train/ or
     val/ component -- because the split is metadata here.
  3. Calibration is looked up by GROUP, not per-recording, so the three
     distinct rig calibrations are explicit and filterable.

The __getitem__ dict schema is IDENTICAL to V3FramesetDataset so that
scripts/precompute_repro_cache.py and train/train_3d_cached.py consume it
unchanged.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from jarvis_jax.data.transforms import crop_origin
from jarvis_jax.geometry.center3d import quantize_center3d
from jarvis_jax.geometry.reprojection_tool import ReprojectionTool

_CROP = 448
_GRID_SPACING = 1


def _load_mask(v5_root, file_name, src_ann_id, w, h):
    """Mask for ONE annotation. Keyed by src_ann_id, which is how
    tools/sam3_label_masks.py stores them (ann_ids array parallel to masks)."""
    rec, cam, fn = file_name.split("/")
    path = os.path.join(v5_root, "masks", rec, cam, fn.replace(".jpg", ".npz"))
    if not os.path.exists(path):
        return np.zeros((h, w), np.uint8)
    with np.load(path) as z:
        ids = z["ann_ids"]
        hit = np.nonzero(ids == src_ann_id)[0]
        if hit.size == 0 or not z["matched"][hit[0]]:
            return np.zeros((h, w), np.uint8)
        return z["masks"][hit[0]].astype(np.uint8)


class V5FramesetDataset:
    def __init__(self, root: str, split: str, *, recordings=None,
                 calib_groups=None, sex=None):
        self.root = root
        self.split = split
        with open(os.path.join(root, "annotations", f"instances_{split}.json")) as f:
            coco = json.load(f)
        with open(os.path.join(root, "manifest.json")) as f:
            self.manifest = json.load(f)["recordings"]

        self.keypoint_names = coco.get("keypoint_names", [])
        self.skeleton = coco.get("skeleton", [])
        self._id2img = {i["id"]: i for i in coco["images"]}
        self._id2ann = {a["id"]: a for a in coco["annotations"]}

        self._tools: dict[str, ReprojectionTool] = {}
        self.keys: list[str] = []
        self._fs: list[dict] = []
        for key, fsv in sorted(coco["framesets"].items()):
            rec = fsv["recording"]
            meta = self.manifest.get(rec, {})
            grp = meta.get("calib_group")
            if recordings is not None and rec not in recordings:
                continue
            if calib_groups is not None and grp not in calib_groups:
                continue
            if sex is not None and meta.get("sex") != sex:
                continue
            if grp not in self._tools:
                d = os.path.join(root, "calibrations", str(grp))
                if not os.path.isdir(d):
                    continue
                self._tools[grp] = ReprojectionTool(d)
            self.keys.append(key)
            self._fs.append(fsv)

    def __len__(self) -> int:
        return len(self._fs)

    def is_female(self, idx: int) -> bool:
        return self.manifest.get(self._fs[idx]["recording"], {}).get("sex") == "female"

    def __getitem__(self, idx: int) -> dict:
        fsv = self._fs[idx]
        rec = fsv["recording"]
        grp = self.manifest[rec]["calib_group"]
        rt = self._tools[grp]
        n_cam = rt.num_cameras

        crops4 = np.zeros((n_cam, _CROP, _CROP, 4), np.uint8)
        centerHM = np.zeros((n_cam, 2), np.float32)
        kp2d = np.zeros((n_cam, 50, 3), np.float32)

        for c, (img_id, ann_id) in enumerate(
                zip(fsv["frames"][:n_cam], fsv["ann_ids"][:n_cam])):
            if ann_id is None:
                # Ruling R15: this camera could not resolve which fly is which,
                # so it contributes nothing. Leave the crop black and the
                # keypoints invisible (v=0) -- triangulation already ignores
                # non-visible views, and a black crop is honest about absence.
                continue
            info, ann = self._id2img[img_id], self._id2ann[ann_id]
            w, h = info["width"], info["height"]
            kp2d[c] = np.asarray(ann["keypoints"], np.float32).reshape(-1, 3)
            path = os.path.join(self.root, "images", info["file_name"])
            with Image.open(path) as pil:
                img = np.asarray(pil.convert("RGB"), np.uint8)
            mask = _load_mask(self.root, info["file_name"],
                              ann.get("src_ann_id", ann_id), w, h)
            bbox = np.asarray(ann["bbox"], np.float32)
            x0, y0 = crop_origin(bbox, w, h, _CROP)
            crops4[c, ..., :3] = img[y0:y0 + _CROP, x0:x0 + _CROP]
            crops4[c, ..., 3] = mask[y0:y0 + _CROP, x0:x0 + _CROP]
            centerHM[c] = [x0 + _CROP / 2, y0 + _CROP / 2]

        kp3d = np.zeros((50, 3), np.float32)
        vis = np.zeros(50, bool)
        for j in range(50):
            pts = np.zeros((n_cam, 2), np.float64)
            use = []
            for c in range(n_cam):
                x, y, v = kp2d[c, j]
                if v > 0:
                    pts[c] = [x, y]
                    use.append(c)
            if len(use) >= 2:
                kp3d[j] = rt.reconstruct_point(pts, cams_to_use=use).astype(np.float32)
                vis[j] = True

        return {
            "crops4": crops4,
            "centerHM": centerHM,
            "center3D": quantize_center3d(kp3d[vis], _GRID_SPACING),
            "cameraMatrices": rt.camera_matrices,
            "kp3d": kp3d,
            "vis": vis,
            "fly_id": np.int32(fsv["fly_id"]),
            "calib_group": grp,
        }


def frameset_batches(ds: V5FramesetDataset, batch_size: int, *,
                     shuffle: bool = True, seed: int = 0,
                     drop_last: bool = True, female_weight: float = 1.0):
    """Batch iterator. `female_weight` > 1 repeats female framesets, because
    female data is 7% of the set and is the binding constraint on the goal."""
    idx = []
    for i in range(len(ds)):
        reps = int(round(female_weight)) if ds.is_female(i) else 1
        idx.extend([i] * max(reps, 1))
    idx = np.asarray(idx)
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    stop = (len(idx) // batch_size) * batch_size if drop_last else len(idx)
    for s in range(0, stop, batch_size):
        samples = [ds[int(i)] for i in idx[s:s + batch_size]]
        yield {k: np.stack([smp[k] for smp in samples], 0)
               for k in samples[0] if k != "calib_group"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_v5_3d.py -v`
Expected: 5 passed

- [ ] **Step 5: Sanity-check against the real build**

```bash
cd $PKG && python - <<'PY'
from jarvis_jax.data.v5_3d import V5FramesetDataset
V5 = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5"
for split in ("train", "val"):
    ds = V5FramesetDataset(V5, split)
    print(split, len(ds), "samples;",
          sum(ds.is_female(i) for i in range(len(ds))), "female")
s = V5FramesetDataset(V5, "train")[0]
print({k: (v.shape if hasattr(v, "shape") else v) for k, v in s.items()})
PY
```

Expected: train + val ≈ 3,800; the mask channel is non-zero for at least some
samples (if `crops4[...,3].max() == 0` everywhere, mask lookup is broken).

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/data/v5_3d.py $PKG/tests/test_v5_3d.py
git commit -m "feat(data): per-fly v5 frameset loader with calib-group filtering

Same __getitem__ schema as V3FramesetDataset so the cache builder and
cached trainer consume it unchanged.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Sharpen the 2D targets and retrain ViTPose (Phase 2)

**Controller ruling R11 — the contamination argument for this retrain is
WITHDRAWN.** An earlier framing said the shipped detector had seen frames we
would hold out, so any 3D val number was contaminated through the front-end.
That is FALSE. Measured directly: `red_data_unified_V4` (what
`v4_8gpu_20260808` trained on) is split by RECORDING with **zero** train/val
overlap — 11 train recordings vs 9 val, none shared. Same for
`red_data_unified_V3`, which the 3D trainer read (13 vs 4, zero shared). The
frame-level leak is real but confined to `general_model/<subset>/annotations/`,
which neither trainer reads.

**What still justifies this retrain — and it is sufficient on its own — is the
sigma measurement in Step 3:** the target Gaussian's σ of 7.0 heatmap px is
~1.5× longer than the entire 4.6 px distal tarsal segment it must resolve. Do
not claim a contamination benefit anywhere in this task's commit message or
report.

**Files:**
- Modify: `$PKG/jarvis_jax/data/transforms.py:47`, `$PKG/jarvis_jax/data/v3.py:79`
- Modify: `$PKG/configs/train/vit2d.yaml`
- Test: `$PKG/tests/test_transforms_sigma.py`

**Interfaces:**
- Consumes: `$V5`.
- Produces: a detector checkpoint at `${paths.vit_runs_root}/v5_sigma2/final`, and `train.target_sigma` as a config-visible knob.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_transforms_sigma.py
import numpy as np
from jarvis_jax.data.transforms import gaussian_heatmaps


def _concentration(hm, radius=7):
    """Fraction of heatmap mass within `radius` px of the peak — the same
    statistic the 2D wobble diagnostic reported as 0.352 for the shipped model."""
    j, i = np.unravel_index(np.argmax(hm), hm.shape)
    yy, xx = np.mgrid[:hm.shape[0], :hm.shape[1]]
    near = ((yy - j) ** 2 + (xx - i) ** 2) <= radius ** 2
    return float(hm[near].sum() / max(hm.sum(), 1e-9))


def test_sigma_two_is_far_more_concentrated_than_seven():
    """sigma=7 px is wider than an entire tarsal segment (~4.6 heatmap px),
    so adjacent tarsal targets overlap almost completely."""
    xy = np.array([[112.0, 112.0]], np.float32)
    vis = np.array([1], np.int32)
    wide = np.asarray(gaussian_heatmaps(xy, vis, heatmap_size=224, sigma=7.0))[0]
    tight = np.asarray(gaussian_heatmaps(xy, vis, heatmap_size=224, sigma=2.0))[0]
    assert _concentration(wide) < 0.55
    assert _concentration(tight) > 0.95


def test_adjacent_tarsal_targets_separate_at_sigma_two():
    """Two keypoints 4.6 px apart — the measured tarsal segment length."""
    xy = np.array([[112.0, 112.0], [116.6, 112.0]], np.float32)
    vis = np.array([1, 1], np.int32)
    for sigma, expect_separated in ((7.0, False), (2.0, True)):
        hm = np.asarray(gaussian_heatmaps(xy, vis, heatmap_size=224, sigma=sigma))
        mid = hm[0][112, 114]           # midpoint between the two peaks
        peak = hm[0][112, 112]
        separated = mid < 0.5 * peak
        assert separated == expect_separated, (sigma, mid, peak)


def test_default_sigma_is_two():
    import inspect
    assert inspect.signature(gaussian_heatmaps).parameters["sigma"].default == 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_transforms_sigma.py -v`
Expected: FAIL — `test_default_sigma_is_two` fails (default is 7.0); the
separation test fails at sigma=2.0 if the implementation is otherwise correct.

- [ ] **Step 3: Change the default and thread the knob**

```python
# $PKG/jarvis_jax/data/transforms.py:47 — change the signature only
def gaussian_heatmaps(hm_xy, vis, heatmap_size=224, sigma=2.0):
    """Render Gaussian keypoint targets.

    sigma DEFAULT CHANGED 7.0 -> 2.0 (2026-08-29). Measured: 1 voxel is 2.7-3.2
    heatmap px and the distal tarsal segment T1L_TaT3->T1L_TaTip is 1.59 voxels
    ~ 4.6 heatmap px, so a sigma of 7 px is ~1.5x LONGER than the whole segment
    and adjacent tarsal targets overlap almost completely. That is the direct
    cause of the 0.352 peak concentration measured on the shipped detector.
    sigma/heatmap_size = 3.1% matches the COCO convention, but COCO was tuned
    for humans whose limb segments span a large fraction of the frame; against
    the fly's ~37 heatmap-px body length, sigma=7 is ~19% -- about 5x too wide.
    """
```

```python
# $PKG/jarvis_jax/data/v3.py:79 — same default so the 2D loader agrees
    def __init__(self, root, split, *, crop=448, heatmap_size=224, sigma=2.0,
```

```yaml
# append to $PKG/configs/train/vit2d.yaml
# Gaussian target sigma in heatmap px. 2.0 from the 2026-08-29 resolution
# analysis (see data/transforms.py). 7.0 reproduces the pre-2026-08-29 detector.
target_sigma: 2.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_transforms_sigma.py -v`
Expected: 3 passed

- [ ] **Step 5: Verify nothing else silently depended on sigma=7**

Run: `cd $PKG && python -m pytest tests/ -k "heatmap or decode or transforms or mpjpe" -v`
Expected: all pass. Any failure asserting a sigma-7 numeric constant must be
updated to pass `sigma=7.0` explicitly, not by reverting the default.

- [ ] **Step 6: Commit the config change**

```bash
git add $PKG/jarvis_jax/data/transforms.py $PKG/jarvis_jax/data/v3.py \
        $PKG/configs/train/vit2d.yaml $PKG/tests/test_transforms_sigma.py
git commit -m "fix(2d): target sigma 7.0 -> 2.0 — the detector was trained to blur

sigma=7 px exceeds the 4.6 px tarsal segment it must resolve, so adjacent
tarsal targets overlapped almost completely. Direct cause of the measured
0.352 heatmap peak concentration.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Retrain the detector on all 8 GPUs**

This is the critical path — one 8-GPU data-parallel run, not a fan-out. All 32
CPUs feed one process's JPEG decode.

```bash
cd $PKG && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
nvidia-smi -L
python -u -m jarvis_jax.train.train run_id=v5_sigma2 train=vit2d model=vitpose \
  paths=hyak paths.data_root=$V5 \
  train.target_sigma=2.0 train.total_steps=30000 train.num_workers=8 \
  2>&1 | tee /gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5_sigma2.log
```

Expected: val 2D MPJPE printed at each `eval_every`; run auto-resumes from its
own `ckpt/` if preempted.

- [ ] **Step 8: Measure the heatmap concentration and read the overlays**

**Expectation:** mean peak concentration rises from 0.352 toward >0.7, and 2D
overlays on female wall/occlusion frames put keypoints on anatomy across ≥3
cameras. If concentration barely moves, the sigma change did not reach the
training target and Step 3 is incomplete.

```bash
cd $PKG
python scripts/diagnose_2d_wobble.py paths=hyak \
  detector.ckpt=${paths.vit_runs_root}/v5_sigma2/final \
  +out=../../figures/2026-08-29-c2f-3d/phase2-detector
```

Open the concentration plot and the female overlays with the Read tool and
write what they show into
`docs/benchmark/2026-08-29-c2f-3d/phase2-detector-notes.md`, including the
before/after concentration numbers. Commit that notes file only.

---

## Task 10: Rotation and spacing in the reprojection layer

**Files:**
- Modify: `$PKG/jarvis_jax/hybridnet/reproject.py`
- Test: `$PKG/tests/test_reproject_rotation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `_build_half_grid(grid_size, grid_spacing, *, rotation=None)` and
  `reproject_heatmaps(..., grid_size=48, grid_spacing=1.0, heatmap_size=226, rotation=None)`.
  `rotation` is `(3,3)` or `(B,3,3)` float32 applied to the grid basis.
  Defaults reproduce current behaviour byte-for-byte.

  **Controller ruling R3:** an earlier draft also added `center_offset`. It is
  DROPPED — Task 12 recenters by passing each joint's coarse estimate directly
  as `center3D`, so the parameter would be dead on arrival. Do not add it, and
  do not write its test.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_reproject_rotation.py
import jax.numpy as jnp
import numpy as np
import pytest

from jarvis_jax.hybridnet.reproject import reproject_heatmaps, _build_half_grid


def _inputs(seed=0, B=2, C=7, J=3, hm=64):
    rng = np.random.default_rng(seed)
    return dict(
        heatmaps=jnp.asarray(rng.random((B, C, J, hm, hm), np.float32)),
        center3D=jnp.asarray(rng.normal(0, 5, (B, 3)).astype(np.float32)),
        centerHM=jnp.asarray(rng.uniform(100, 900, (B, C, 2)).astype(np.float32)),
        camera_matrices=jnp.asarray(rng.normal(0, 1, (B, C, 4, 3)).astype(np.float32)),
    )


def test_identity_rotation_is_byte_identical_to_default():
    """Guards the refactor: the shipped path must not move."""
    a = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64)
    b = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64,
                           rotation=jnp.eye(3, dtype=jnp.float32))
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_rotation_changes_the_volume():
    theta = np.pi / 3
    R = jnp.asarray(np.array([[np.cos(theta), -np.sin(theta), 0],
                              [np.sin(theta), np.cos(theta), 0],
                              [0, 0, 1]], np.float32))
    a = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64)
    b = reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64, rotation=R)
    assert not np.allclose(np.asarray(a), np.asarray(b))


def test_grid_spacing_scales_the_sampled_extent():
    """A finer spacing must span a proportionally smaller world extent —
    this is what makes stage-2 refinement 4x higher resolution."""
    g1 = np.asarray(_build_half_grid(48, 1.0))
    g4 = np.asarray(_build_half_grid(48, 0.25))
    assert np.isclose(g1.max() / g4.max(), 4.0, rtol=1e-5)


def test_rotation_is_orthogonality_checked():
    bad = jnp.asarray(np.diag([2.0, 1.0, 1.0]).astype(np.float32))
    with pytest.raises(ValueError, match="orthogonal"):
        reproject_heatmaps(**_inputs(), grid_size=16, heatmap_size=64, rotation=bad)


```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_reproject_rotation.py -v`
Expected: FAIL — `TypeError: reproject_heatmaps() got an unexpected keyword argument 'rotation'`

- [ ] **Step 3: Write the implementation**

Modify `_build_half_grid` and `_reproject_single` / `reproject_heatmaps`:

```python
# $PKG/jarvis_jax/hybridnet/reproject.py

def _build_half_grid(grid_size: int, grid_spacing: float,
                     rotation: jnp.ndarray | None = None) -> jnp.ndarray:
    """Base (G/2)^3 grid in world coordinates, optionally rotated.

    `grid_spacing` is now a FLOAT. At the shipped 1.0 the grid spans 48 world
    units with 1 voxel = 1 unit ~ 0.17 mm; the distal tarsal segment is 1.59
    voxels, which is why stage 2 uses 0.25.

    `rotation` (3,3) rotates the grid BASIS. The V2VNet is otherwise
    world-frame-locked (it learns "dorsal is roughly +Z" for whichever
    calibration dominates training), so rotating the basis during training --
    with the labels rotated by the same R -- both removes that dependence and
    multiplies an otherwise small (3,800-sample) dataset.
    """
    half_g = grid_size // 2
    half_half_g = half_g // 2
    idx = jnp.arange(half_g, dtype=jnp.float32)
    ig, jg, kg = jnp.meshgrid(idx, idx, idx, indexing="ij")
    grid = jnp.stack([ig - half_half_g, jg - half_half_g, kg - half_half_g], axis=-1)
    grid = grid * (grid_spacing * 2)
    if rotation is not None:
        grid = jnp.einsum("...i,ij->...j", grid, rotation.T,
                          precision=lax.Precision.HIGHEST)
    return grid
```

In `_reproject_single`, accept `rotation`, pass it to `_build_half_grid`,
and change:

```python
    base_grid = _build_half_grid(grid_size, grid_spacing, rotation)
    grid = base_grid + center3D
```

In `reproject_heatmaps`, add the parameter, validate, and vmap:

```python
def reproject_heatmaps(heatmaps, center3D, centerHM, camera_matrices, *,
                       grid_size: int = 48, grid_spacing: float = 1.0,
                       heatmap_size: int = 226, rotation=None):
    if rotation is not None:
        R = jnp.asarray(rotation, jnp.float32)
        R2 = R if R.ndim == 2 else R[0]
        if not bool(jnp.allclose(R2 @ R2.T, jnp.eye(3, dtype=jnp.float32), atol=1e-4)):
            raise ValueError("rotation must be orthogonal (R @ R.T == I)")
    ...
```

vmap over the batch with `in_axes` of `None` for a shared `(3,3)` rotation and
`0` for a per-sample `(B,3,3)`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_reproject_rotation.py -v`
Expected: 4 passed

- [ ] **Step 5: Verify the existing reprojection parity tests still pass**

Run: `cd $PKG && python -m pytest tests/ -k "reproject or repro" -v`
Expected: all pass, including the PyTorch-parity test. A parity failure means
the default path moved — the refactor is wrong.

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/hybridnet/reproject.py $PKG/tests/test_reproject_rotation.py
git commit -m "feat(3d): rotation + float grid spacing in reprojection

Defaults are byte-identical to the shipped path. Enables stage-2 refinement
at spacing 0.25 and rotation augmentation of the world-locked grid.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 11: Generalize `soft_argmax_3d` past `grid_spacing == 1`

`hybridnet/model.py:44` currently carries `assert grid_spacing == 1` with a
message naming exactly what to generalize. Stage 2 needs spacing 0.25.

**Files:**
- Modify: `$PKG/jarvis_jax/hybridnet/model.py:44-80`
- Test: `$PKG/tests/test_soft_argmax_spacing.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `soft_argmax_3d(vol, *, grid_spacing: float = 1.0, roi_cube: float = 48.0, sharpen: float = 1.0) -> tuple[points (B,J,3), conf (B,J)]`, world offset `idx * grid_spacing * 2 - roi_cube / 2`.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_soft_argmax_spacing.py
import jax.numpy as jnp
import numpy as np

from jarvis_jax.hybridnet.model import soft_argmax_3d


def _one_hot(G, idx):
    v = np.zeros((1, 1, G, G, G), np.float32)
    v[0, 0, idx[0], idx[1], idx[2]] = 1.0
    return jnp.asarray(v)


def test_spacing_one_unchanged():
    G = 8
    pts, _ = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=1.0, roi_cube=8.0)
    expect = np.array([5, 2, 6], np.float32) * 1.0 * 2 - 8.0 / 2
    np.testing.assert_allclose(np.asarray(pts)[0, 0], expect, atol=1e-4)


def test_quarter_spacing_gives_quarter_extent():
    """The whole point of stage 2: same voxel index, 1/4 the world offset."""
    G = 8
    pts, _ = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=0.25, roi_cube=2.0)
    expect = np.array([5, 2, 6], np.float32) * 0.25 * 2 - 2.0 / 2
    np.testing.assert_allclose(np.asarray(pts)[0, 0], expect, atol=1e-4)


def test_subvoxel_interpolation_on_a_two_voxel_blob():
    """A peak split evenly between neighbours must land BETWEEN them —
    this is the sub-voxel precision stage 2 exists to exploit."""
    G = 8
    v = np.zeros((1, 1, G, G, G), np.float32)
    v[0, 0, 4, 4, 4] = 1.0
    v[0, 0, 5, 4, 4] = 1.0
    pts, _ = soft_argmax_3d(jnp.asarray(v), grid_spacing=0.25, roi_cube=2.0)
    assert 4.4 < (np.asarray(pts)[0, 0, 0] + 1.0) / 0.5 < 4.6


def test_conf_unaffected_by_spacing():
    G = 8
    _, c1 = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=1.0, roi_cube=8.0)
    _, c2 = soft_argmax_3d(_one_hot(G, (5, 2, 6)), grid_spacing=0.25, roi_cube=2.0)
    np.testing.assert_allclose(np.asarray(c1), np.asarray(c2), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_soft_argmax_spacing.py -v`
Expected: FAIL — `AssertionError: soft_argmax_3d world offset assumes grid_spacing==1`

- [ ] **Step 3: Write the implementation**

Delete the assert and generalize the world-coordinate mapping:

```python
# $PKG/jarvis_jax/hybridnet/model.py
def soft_argmax_3d(vol, *, grid_spacing: float = 1.0, roi_cube: float = 48.0,
                   sharpen: float = 1.0):
    """...
    World coordinate: ``idx * grid_spacing * 2 - roi_cube / 2``.

    GENERALIZED 2026-08-29: grid_spacing was asserted == 1. Stage-2 refinement
    samples at 0.25 so that 1 voxel ~ 0.8 heatmap px instead of 2.7-3.2, which
    is the whole point — at spacing 1 the distal tarsal segment spans only 1.59
    voxels and the kinematics are quantization-limited at ~0.17 mm.
    """
    # (assert removed)
    ...
    offset = roi_cube / 2.0
    coords = idx * grid_spacing * 2.0 - offset
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_soft_argmax_spacing.py -v`
Expected: 4 passed

- [ ] **Step 5: Verify the stage-1 path is unmoved**

Run: `cd $PKG && python -m pytest tests/ -k "soft_argmax or hybridnet or model_parity" -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/hybridnet/model.py $PKG/tests/test_soft_argmax_spacing.py
git commit -m "feat(3d): soft_argmax_3d supports fractional grid_spacing

Removes assert grid_spacing == 1 so stage-2 refinement can sample at 0.25.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 12: Stage-2 per-joint refinement

**Files:**
- Create: `$PKG/jarvis_jax/hybridnet/refine.py`
- Modify: `$PKG/configs/model/hybridnet.yaml`
- Test: `$PKG/tests/test_refine.py`

**Interfaces:**
- Consumes: `reproject_heatmaps` (with `grid_spacing`), `soft_argmax_3d` (fractional spacing), `V2VNet`.
- Produces:
  - `refine_volumes(heatmaps, coarse_kp3d, centerHM, camera_matrices, *, cube=24, spacing=0.25, heatmap_size=224) -> (B, J, cube, cube, cube)` — one small volume per joint, centred on that joint's coarse estimate
  - `class RefineNet(nnx.Module)`: `__init__(self, *, channels: int = 32, rngs)`, `__call__(vols (B*J, cube, cube, cube, 1), use_running_average: bool) -> (B*J, cube, cube, cube, 1)` — weights SHARED across joints (joint folded into batch)
  - `refine_keypoints(net, heatmaps, coarse_kp3d, centerHM, camera_matrices, *, cube=24, spacing=0.25, sharpen=3.0) -> (B, J, 3)`

Sizing rationale: `cube=24, spacing=0.25` spans ±3 world units around the
coarse estimate — adequate against a ~1-unit stage-1 error — and costs
50 × 24³ = 691k voxels against stage 1's 50 × 48³ = 5.53M, i.e. **8× cheaper
than stage 1**, not equal.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_refine.py
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from jarvis_jax.hybridnet.refine import (
    RefineNet, refine_volumes, refine_keypoints)

B, C, J, HM = 2, 7, 5, 64


def _inputs(seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.random((B, C, J, HM, HM), np.float32)),
            jnp.asarray(rng.normal(0, 3, (B, J, 3)).astype(np.float32)),
            jnp.asarray(rng.uniform(200, 800, (B, C, 2)).astype(np.float32)),
            jnp.asarray(rng.normal(0, 1, (B, C, 4, 3)).astype(np.float32)))


def test_refine_volume_shape_is_per_joint_and_small():
    hm, kp, chm, cm = _inputs()
    v = refine_volumes(hm, kp, chm, cm, cube=8, spacing=0.25, heatmap_size=HM)
    assert v.shape == (B, J, 8, 8, 8)


def test_refine_volume_is_much_cheaper_than_stage1():
    """50 joints x 24^3 must be ~8x SMALLER than one 50 x 48^3 stage-1 volume."""
    assert (50 * 24 ** 3) * 8 == 50 * 48 ** 3


def test_each_joint_volume_is_centred_on_its_own_coarse_estimate():
    """Two joints far apart must produce different volumes — a single shared
    volume would silently destroy the per-joint refinement."""
    hm, kp, chm, cm = _inputs()
    kp = kp.at[0, 0].set(jnp.array([0.0, 0.0, 0.0]))
    kp = kp.at[0, 1].set(jnp.array([20.0, 20.0, 20.0]))
    v = refine_volumes(hm, kp, chm, cm, cube=8, spacing=0.25, heatmap_size=HM)
    assert not np.allclose(np.asarray(v)[0, 0], np.asarray(v)[0, 1])


def test_refinenet_shares_weights_across_joints():
    net = RefineNet(channels=8, rngs=nnx.Rngs(0))
    x = jnp.zeros((B * J, 8, 8, 8, 1), jnp.float32)
    y = net(x, use_running_average=True)
    assert y.shape == (B * J, 8, 8, 8, 1)
    n_params = sum(int(np.prod(p.shape)) for p in
                   jax.tree_util.tree_leaves(nnx.state(net, nnx.Param)))
    assert n_params < 200_000, "RefineNet must stay small; it runs 50x per sample"


def test_refined_keypoints_stay_within_the_capture_window():
    """A refinement may move a joint by at most +/- cube/2 * spacing * 2."""
    hm, kp, chm, cm = _inputs()
    net = RefineNet(channels=8, rngs=nnx.Rngs(0))
    out = refine_keypoints(net, hm, kp, chm, cm, cube=8, spacing=0.25,
                           heatmap_size=HM)
    assert out.shape == (B, J, 3)
    limit = 8 / 2 * 0.25 * 2
    assert np.all(np.abs(np.asarray(out - kp)) <= limit + 1e-3)


import jax  # noqa: E402  (used by the param-count assertion above)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_refine.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.hybridnet.refine'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/hybridnet/refine.py
"""Stage-2 per-joint refinement volumes.

WHY. Stage 1's 48^3 grid at spacing 1 puts 1 voxel at ~0.17 mm and 2.7-3.2
heatmap px. The distal tarsal segment T1L_TaT3->T1L_TaTip is 1.59 voxels, so
leg kinematics are quantization-limited and the 3D stage samples the 2D
heatmaps ~3x below their own resolution. Stage 2 re-samples a SMALL volume
around each joint's coarse estimate at spacing 0.25, where 1 voxel ~ 0.8
heatmap px -- matched to the 2D detail that already exists.

COST. cube=24 at spacing 0.25 spans +/-3 world units, adequate against a
~1-unit stage-1 error, and 50 joints x 24^3 = 691k voxels is 8x CHEAPER than
one 50 x 48^3 stage-1 volume (5.53M). Weights are shared across joints by
folding the joint axis into the batch.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from jarvis_jax.hybridnet.model import soft_argmax_3d
from jarvis_jax.hybridnet.reproject import reproject_heatmaps


def refine_volumes(heatmaps, coarse_kp3d, centerHM, camera_matrices, *,
                   cube: int = 24, spacing: float = 0.25,
                   heatmap_size: int = 224):
    """(B,J,cube,cube,cube) — one volume per joint, centred on its own estimate."""
    B, C, J = heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2]

    def one_joint(j):
        # Reproject ONLY joint j, with the grid recentred on that joint.
        hm_j = heatmaps[:, :, j: j + 1]                       # (B,C,1,hm,hm)
        centre = coarse_kp3d[:, j]                            # (B,3)
        vol = reproject_heatmaps(
            hm_j, centre, centerHM, camera_matrices,
            grid_size=cube, grid_spacing=spacing, heatmap_size=heatmap_size)
        return vol[:, 0]                                      # (B,cube,cube,cube)

    vols = jax.vmap(one_joint, out_axes=1)(jnp.arange(J))
    return vols                                               # (B,J,cube,...)


class RefineNet(nnx.Module):
    """Small shared 3D CNN over per-joint refinement volumes.

    Deliberately tiny: it runs 50 times per sample, and its job is local peak
    sharpening, not global reasoning — stage 1 already did the localization and
    cross-view disambiguation.
    """

    def __init__(self, *, channels: int = 32, rngs: nnx.Rngs):
        k = dict(kernel_size=(3, 3, 3), strides=(1, 1, 1), padding="SAME", rngs=rngs)
        self.c1 = nnx.Conv(1, channels, **k)
        self.n1 = nnx.BatchNorm(channels, rngs=rngs)
        self.c2 = nnx.Conv(channels, channels, **k)
        self.n2 = nnx.BatchNorm(channels, rngs=rngs)
        self.out = nnx.Conv(channels, 1, kernel_size=(1, 1, 1), strides=(1, 1, 1),
                            padding="VALID",
                            kernel_init=nnx.initializers.normal(0.001),
                            bias_init=nnx.initializers.zeros, rngs=rngs)

    def __call__(self, x, use_running_average: bool = False):
        x = jax.nn.relu(self.n1(self.c1(x), use_running_average=use_running_average))
        x = jax.nn.relu(self.n2(self.c2(x), use_running_average=use_running_average))
        return self.out(x)


def refine_keypoints(net: RefineNet, heatmaps, coarse_kp3d, centerHM,
                     camera_matrices, *, cube: int = 24, spacing: float = 0.25,
                     heatmap_size: int = 224, sharpen: float = 3.0,
                     use_running_average: bool = False):
    """Coarse (B,J,3) -> refined (B,J,3). Offsets are RELATIVE to the coarse
    estimate, so a refinement can never move a joint outside its window."""
    B, J = coarse_kp3d.shape[0], coarse_kp3d.shape[1]
    vols = refine_volumes(heatmaps, coarse_kp3d, centerHM, camera_matrices,
                          cube=cube, spacing=spacing, heatmap_size=heatmap_size)
    flat = vols.reshape(B * J, cube, cube, cube, 1)
    logits = net(flat, use_running_average=use_running_average)
    logits = logits.reshape(B * J, 1, cube, cube, cube)
    roi = cube * spacing * 2.0
    delta, _ = soft_argmax_3d(logits, grid_spacing=spacing, roi_cube=roi,
                              sharpen=sharpen)
    return coarse_kp3d + delta.reshape(B, J, 3)
```

```yaml
# append to $PKG/configs/model/hybridnet.yaml
# Stage-2 per-joint refinement. cube*spacing*2 = +/-3 world units of capture,
# against a ~1-unit stage-1 error. 1 voxel ~ 0.8 heatmap px at spacing 0.25.
refine:
  enabled: true
  cube: 24
  spacing: 0.25
  channels: 32
  sharpen: 3.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_refine.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add $PKG/jarvis_jax/hybridnet/refine.py $PKG/configs/model/hybridnet.yaml \
        $PKG/tests/test_refine.py
git commit -m "feat(3d): stage-2 per-joint refinement at spacing 0.25

Re-samples a 24^3 volume around each joint's coarse estimate, where 1 voxel
is ~0.8 heatmap px instead of 2.7-3.2. 8x cheaper than the stage-1 volume;
weights shared across joints.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 13: Stage-3 gated continuous refine

**Files:**
- Modify: `$PKG/jarvis_jax/tracking/triangulate.py`
- Test: `$PKG/tests/test_triangulate_refine.py`

**Interfaces:**
- Consumes: existing `_reproject_px`, `_point_residuals`, `TRIGGER_FACTOR`, `MIN_CONSENSUS_VIEWS`.
- Produces: `refine_from_seed(seed_kp3d, kp2d, conf, cam_mats, *, gate_px=15.0, conf_thresh=0.3, min_views=2) -> (kp3d (T,K,3), conf3d (T,K), n_views (T,K))`. Views whose 2D lies further than `gate_px` from the seed's reprojection are dropped; survivors are triangulated continuously. Where fewer than `min_views` survive, the seed is returned unchanged.

This is post-hoc inference applied to **every** training arm, so it costs no arm.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_triangulate_refine.py
import numpy as np
import pytest

from jarvis_jax.tracking.triangulate import refine_from_seed


def _rig(n_cam=7, seed=0):
    rng = np.random.default_rng(seed)
    cams = []
    for i in range(n_cam):
        th = 2 * np.pi * i / n_cam
        R = np.array([[np.cos(th), -np.sin(th), 0],
                      [np.sin(th), np.cos(th), 0], [0, 0, 1]])
        t = np.array([0.0, 0.0, 200.0])
        P = np.hstack([8.0 * R, (8.0 * t)[:, None]])
        cams.append(P.T.astype(np.float32))          # (4,3), p_h @ M convention
    return np.stack(cams)


def _project(cam_mats, X):
    h = np.append(np.asarray(X, np.float64), 1.0)
    out = []
    for M in cam_mats:
        p = h @ M
        out.append([p[0] / p[2], p[1] / p[2]])
    return np.asarray(out)


def test_clean_views_recover_the_true_point():
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X)[None, :, None, :]                 # (1,C,1,2)
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    seed = (X + 0.4)[None, None, :].astype(np.float32)       # (1,1,3)
    out, _, nv = refine_from_seed(seed, xy.astype(np.float32), conf, cams,
                                  gate_px=15.0)
    np.testing.assert_allclose(out[0, 0], X, atol=1e-3)
    assert nv[0, 0] == cams.shape[0]


def test_a_swapped_view_is_gated_out():
    """The failure the whole gate exists for: one camera's tarsal tip jumps to
    the wrong leg at conf 0.4-0.8 and drags the DLT."""
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X)
    xy[3] += 250.0                                            # gross swap
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    conf[0, 3, 0] = 0.6
    seed = (X + 0.4)[None, None, :].astype(np.float32)
    out, _, nv = refine_from_seed(seed, xy[None, :, None, :].astype(np.float32),
                                  conf, cams, gate_px=15.0)
    np.testing.assert_allclose(out[0, 0], X, atol=1e-2)
    assert nv[0, 0] == cams.shape[0] - 1


def test_seed_is_returned_when_too_few_views_survive():
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X) + 500.0                            # every view gated out
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    seed = np.array([[[9.0, 9.0, 9.0]]], np.float32)
    out, _, nv = refine_from_seed(seed, xy[None, :, None, :].astype(np.float32),
                                  conf, cams, gate_px=5.0, min_views=2)
    np.testing.assert_allclose(out[0, 0], seed[0, 0], atol=1e-6)
    assert nv[0, 0] < 2


def test_output_is_continuous_not_quantized():
    """Stage 3 exists to escape the ~0.17 mm voxel staircase — two seeds a
    tenth of a voxel apart must give distinguishable, non-identical output."""
    cams = _rig()
    outs = []
    for d in (0.0, 0.1):
        X = np.array([1.0 + d, 2.0, 3.0])
        xy = _project(cams, X)[None, :, None, :].astype(np.float32)
        conf = np.ones((1, cams.shape[0], 1), np.float32)
        seed = np.zeros((1, 1, 3), np.float32)
        out, _, _ = refine_from_seed(seed, xy, conf, cams, gate_px=1e6)
        outs.append(out[0, 0, 0])
    assert 0.05 < abs(outs[1] - outs[0]) < 0.15


def test_low_conf_views_are_dropped():
    cams = _rig()
    X = np.array([1.0, 2.0, 3.0])
    xy = _project(cams, X)[None, :, None, :].astype(np.float32)
    conf = np.ones((1, cams.shape[0], 1), np.float32)
    conf[0, :3, 0] = 0.05
    seed = X[None, None, :].astype(np.float32)
    _, _, nv = refine_from_seed(seed, xy, conf, cams, gate_px=15.0,
                                conf_thresh=0.3)
    assert nv[0, 0] == cams.shape[0] - 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_triangulate_refine.py -v`
Expected: FAIL — `ImportError: cannot import name 'refine_from_seed'`

- [ ] **Step 3: Write the implementation**

```python
# append to $PKG/jarvis_jax/tracking/triangulate.py
def refine_from_seed(seed_kp3d, kp2d, conf, cam_mats, *, gate_px: float = 15.0,
                     conf_thresh: float = 0.3, min_views: int = 2):
    """Stage 3: continuous sub-voxel 3D, gated by a volumetric seed.

    The two lifters fail orthogonally. Plain DLT is continuous and sub-pixel
    but has no outlier rejection, so a single swapped view drags the solve
    (measured: 834 jitter spikes on a 921-frame clip). The volumetric net fuses
    every view and is robust (156 spikes) but quantizes at ~0.17 mm, which is
    larger than the tarsal segments that ARE the leg kinematics.

    This takes the robustness from the volume and the precision from the DLT:
    the seed says where the joint is and therefore which views are lying; the
    surviving views are triangulated continuously.

    Args:
        seed_kp3d: (T,K,3) stage-1/2 estimate.
        kp2d:      (T,C,K,2) detector 2D, must be FINITE everywhere.
        conf:      (T,C,K).
        cam_mats:  (C,4,3).
        gate_px:   a view is dropped if its 2D is further than this from the
                   seed's reprojection.
        min_views: below this, the seed is returned unchanged rather than
                   producing a confident wrong point.

    Returns:
        (kp3d (T,K,3), conf3d (T,K), n_views (T,K))
    """
    seed_kp3d = np.asarray(seed_kp3d, np.float64)
    kp2d = np.asarray(kp2d, np.float64)
    conf = np.asarray(conf, np.float64)
    T, C, K = conf.shape

    out = seed_kp3d.copy()
    out_conf = np.zeros((T, K), np.float32)
    n_used = np.zeros((T, K), np.int32)

    for t in range(T):
        for k in range(K):
            X0 = seed_kp3d[t, k]
            if not np.all(np.isfinite(X0)):
                continue
            proj = _reproject_px(cam_mats, X0)              # (C,2)
            d = np.linalg.norm(kp2d[t, :, k, :] - proj, axis=-1)
            keep = (conf[t, :, k] >= conf_thresh) & (d <= gate_px)
            idx = np.nonzero(keep)[0]
            n_used[t, k] = idx.size
            if idx.size < min_views:
                continue
            A = []
            for c in idx:
                M = cam_mats[c]                              # (4,3)
                x, y = kp2d[t, c, k]
                A.append(x * M[:, 2] - M[:, 0])
                A.append(y * M[:, 2] - M[:, 1])
            _, _, Vt = np.linalg.svd(np.stack(A))
            h = Vt[-1]
            if abs(h[3]) < 1e-12:
                continue
            out[t, k] = h[:3] / h[3]
            out_conf[t, k] = float(np.mean(conf[t, idx, k]))
    return out.astype(np.float32), out_conf, n_used
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_triangulate_refine.py -v`
Expected: 5 passed

- [ ] **Step 5: Verify the existing triangulation tests still pass**

Run: `cd $PKG && python -m pytest tests/test_courtship_triangulate.py -v`
Expected: all pass — `refine_from_seed` is additive and must not touch
`triangulate_keypoints`.

- [ ] **Step 6: Commit**

```bash
git add $PKG/jarvis_jax/tracking/triangulate.py $PKG/tests/test_triangulate_refine.py
git commit -m "feat(3d): stage-3 continuous refine gated by the volumetric seed

Robustness from the volume, sub-voxel precision from the DLT. Post-hoc, so
it applies to every training arm and costs none.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 14: Rotation augmentation

**Files:**
- Create: `$PKG/jarvis_jax/data/rot_augment.py`
- Test: `$PKG/tests/test_rot_augment.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `gravity_axis(cam_mats) -> np.ndarray (3,)` — the world axis most nearly shared as "up" by the rig; cached per calibration group
  - `sample_rotation(rng, *, yaw_range=(0, 2*np.pi), tilt_deg=30.0, axis=(0,0,1)) -> np.ndarray (3,3)`
  - `augment_sample(sample, R) -> dict` — rotates `kp3d` and `center3D` about the grid centre by `R`, returning the sample plus `rotation`

Bounded by default (full yaw about gravity, ±30° tilt) rather than full SO(3):
the world frame carries a real physical prior — flies are usually upright on
the arena floor — so unbounded rotation destroys free information. Expected to
help the female on walls and cost a little on the easy male; arms A1/A2 and
A3/A4 measure exactly that.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_rot_augment.py
import numpy as np
import pytest

from jarvis_jax.data.rot_augment import sample_rotation, augment_sample


def test_sampled_rotations_are_orthogonal_with_unit_determinant():
    rng = np.random.default_rng(0)
    for _ in range(20):
        R = sample_rotation(rng)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_tilt_is_bounded():
    """Full SO(3) would destroy the gravity prior ('legs point down'), which is
    real free information for a fly on the arena floor."""
    rng = np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    for _ in range(200):
        R = sample_rotation(rng, tilt_deg=30.0, axis=up)
        cos = float(np.clip((R @ up) @ up, -1, 1))
        assert np.degrees(np.arccos(cos)) <= 30.0 + 1e-4


def test_zero_tilt_keeps_the_gravity_axis_fixed():
    rng = np.random.default_rng(0)
    up = np.array([0.0, 0.0, 1.0])
    R = sample_rotation(rng, tilt_deg=0.0, axis=up)
    np.testing.assert_allclose(R @ up, up, atol=1e-6)


def test_augment_rotates_labels_about_the_grid_centre():
    """The label must move with the grid, or the net learns a wrong mapping."""
    kp = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], np.float32)
    c = np.array([0.0, 0.0, 0.0], np.float32)
    th = np.pi / 2
    R = np.array([[np.cos(th), -np.sin(th), 0],
                  [np.sin(th), np.cos(th), 0], [0, 0, 1]], np.float32)
    out = augment_sample({"kp3d": kp, "center3D": c, "vis": np.ones(2, bool)}, R)
    np.testing.assert_allclose(out["kp3d"][0], [0.0, 1.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(out["kp3d"][1], [-2.0, 0.0, 0.0], atol=1e-5)


def test_augment_preserves_pairwise_distances():
    """A rotation is rigid — bone lengths must not change."""
    rng = np.random.default_rng(1)
    kp = rng.normal(0, 5, (50, 3)).astype(np.float32)
    c = np.zeros(3, np.float32)
    R = sample_rotation(rng)
    out = augment_sample({"kp3d": kp, "center3D": c, "vis": np.ones(50, bool)}, R)
    d0 = np.linalg.norm(kp[1:] - kp[:-1], axis=-1)
    d1 = np.linalg.norm(out["kp3d"][1:] - out["kp3d"][:-1], axis=-1)
    np.testing.assert_allclose(d0, d1, atol=1e-4)


def test_invisible_keypoints_are_left_alone():
    kp = np.zeros((3, 3), np.float32)
    vis = np.array([True, False, True])
    rng = np.random.default_rng(0)
    out = augment_sample({"kp3d": kp, "center3D": np.zeros(3, np.float32),
                          "vis": vis}, sample_rotation(rng))
    np.testing.assert_array_equal(out["kp3d"][1], np.zeros(3, np.float32))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_rot_augment.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.data.rot_augment'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/data/rot_augment.py
"""Bounded rotation augmentation of the world-locked voxel grid.

WHY. hybridnet/reproject.py builds the grid axis-aligned to the WORLD frame
(`grid = base_grid + center3D`) and mean-pools the cameras, so the V2VNet's
only geometric dependency is world-frame ORIENTATION -- it learns "dorsal is
roughly +Z" for whichever calibration dominates training. There are three
distinct rig calibrations in the labeled data. Rotating the grid basis during
training, with the labels rotated by the same R, removes that dependence.

At 3,800 fly-samples the bigger win is simply that this is a free data
multiplier for a 3D conv net.

WHY BOUNDED, NOT FULL SO(3). The world frame carries a real physical prior:
flies are usually upright on the arena floor, so "legs point down in world -Z"
is genuine free information. Full SO(3) throws it away. Expect this to HELP the
female on walls (where the upright prior fails anyway) and cost a little on the
easy male -- which is the trade we want, and which arms A1/A2 and A3/A4 measure
rather than assume.
"""
from __future__ import annotations

import numpy as np


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def gravity_axis(cam_mats: np.ndarray) -> np.ndarray:
    """World axis most nearly shared as 'up' across the rig.

    The arena floor is level and every camera looks roughly inward, so the
    world axis with the most consistent sign across camera image-y directions
    is the gravity axis. Falls back to +Z, which is correct for this rig.
    """
    M = np.asarray(cam_mats, np.float64)
    if M.ndim != 3:
        return np.array([0.0, 0.0, 1.0])
    ydir = M[:, :3, 1]                                   # (C,3)
    ydir = ydir / np.maximum(np.linalg.norm(ydir, axis=1, keepdims=True), 1e-12)
    u, s, vt = np.linalg.svd(ydir - ydir.mean(0, keepdims=True))
    axis = vt[0] if s[0] > 0 else np.array([0.0, 0.0, 1.0])
    if axis[2] < 0:
        axis = -axis
    return axis / max(np.linalg.norm(axis), 1e-12)


def sample_rotation(rng: np.random.Generator, *, yaw_range=(0.0, 2 * np.pi),
                    tilt_deg: float = 30.0, axis=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Full yaw about `axis`, plus a bounded tilt away from it."""
    axis = np.asarray(axis, np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    yaw = rng.uniform(*yaw_range)
    R = _rodrigues(axis, yaw)
    if tilt_deg > 0:
        # A random axis perpendicular to `axis`, tilted by <= tilt_deg.
        perp = np.cross(axis, rng.normal(size=3))
        n = np.linalg.norm(perp)
        if n > 1e-9:
            tilt = np.radians(rng.uniform(-tilt_deg, tilt_deg))
            R = _rodrigues(perp / n, tilt) @ R
    return R.astype(np.float32)


def augment_sample(sample: dict, R: np.ndarray) -> dict:
    """Rotate kp3d about center3D by R. Invisible keypoints are left untouched."""
    out = dict(sample)
    R = np.asarray(R, np.float32)
    c = np.asarray(sample["center3D"], np.float32)
    kp = np.asarray(sample["kp3d"], np.float32).copy()
    vis = np.asarray(sample["vis"], bool)
    kp[vis] = ((kp[vis] - c) @ R.T) + c
    out["kp3d"] = kp
    out["rotation"] = R
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_rot_augment.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add $PKG/jarvis_jax/data/rot_augment.py $PKG/tests/test_rot_augment.py
git commit -m "feat(data): bounded rotation augmentation of the world-locked grid

Full yaw about gravity + /-30 deg tilt. Bounded on purpose: the world frame
carries a real gravity prior that full SO(3) would destroy.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 15: Build the caches and run the 8 arms

**Files:**
- Create: `$PKG/scripts/train_arms.py`
- Modify: `$PKG/scripts/precompute_repro_cache.py` (accept `V5FramesetDataset`)

**Interfaces:**
- Consumes: `V5FramesetDataset`, `RefineNet`, `rot_augment`, the `v5_sigma2` detector.
- Produces: 8 run dirs under `${paths.runs_root}/v5_<arm>/` each with `final/` and a `val_mpjpe.json` carrying overall and **per-calibration-group, per-sex** breakdowns.

| arm | config |
|---|---|
| `A1_base` | stage-1 48³ sp1, no aug |
| `A2_base_aug` | + rotation aug |
| `A3_c2f` | coarse-to-fine, no aug |
| `A4_c2f_aug` | coarse-to-fine + aug (**expected winner**) |
| `A5_hires` | single-stage 96³ sp0.5 |
| `A6_c2f_aug_norecover` | A4 minus the 264 second flies |
| `A7_c2f_aug_femwt` | A4 + `female_weight=3.0` |
| `A8_c2f_aug_seed2` | A4, `seed=1` — the noise floor |

- [ ] **Step 1: Point the cache builder at v5**

In `$PKG/scripts/precompute_repro_cache.py`, select the dataset class on a new
`cache.dataset_version` key (`"v5"` default, `"v3"` preserved):

```python
if str(cfg.cache.get("dataset_version", "v5")) == "v5":
    from jarvis_jax.data.v5_3d import V5FramesetDataset as FramesetDataset
    from jarvis_jax.data.v5_3d import frameset_batches
else:
    from jarvis_jax.data.v3_3d import V3FramesetDataset as FramesetDataset
    from jarvis_jax.data.v3_3d import frameset_batches
```

- [ ] **Step 2: Build the train and val caches**

```bash
cd $PKG && micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
VIT=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5_sigma2/final
CACHE=/gscratch/portia/eabe/data/Johnson_lab/jax_repro_cache/v5_sigma2
for SPLIT in train val; do
  python -u scripts/precompute_repro_cache.py paths=hyak cache=default \
    cache.front_end=vitpose cache.dataset_version=v5 \
    paths.data_root=$V5 paths.vitpose_ckpt="$VIT" \
    paths.cache_dir="$CACHE" cache.split=$SPLIT
done
```

Expected: `train` ≈ 3,400 and `val` ≈ 400 volumes. Cache is idempotent.

- [ ] **Step 3: Write the arm launcher**

```python
# $PKG/scripts/train_arms.py
"""Launch the 8 training arms, one per GPU.

Wall-clock cost of 8 arms equals 1 arm: features are pre-cached, V2VNet is
small, and the prior single-arm runs took ~8 h on one GPU. That is what buys
attribution under a tight budget.

RAMP 4 -> 8. A prior 8-way JAX run on a 128 GB cgroup died with
CUDA_ERROR_UNKNOWN; the working cap was ~4. This node has 256 GB (32 GB per
process) and cached features keep host RAM low, but --max-parallel defaults to
4 and must be raised only after a 4-way run is seen to be healthy.
"""
import argparse
import os
import subprocess
import sys

ARMS = {
    "A1_base":              ["train.rot_augment=false", "model.refine.enabled=false"],
    "A2_base_aug":          ["train.rot_augment=true",  "model.refine.enabled=false"],
    "A3_c2f":               ["train.rot_augment=false", "model.refine.enabled=true"],
    "A4_c2f_aug":           ["train.rot_augment=true",  "model.refine.enabled=true"],
    "A5_hires":             ["train.rot_augment=true",  "model.refine.enabled=false",
                             "model.roi_cube=96", "model.grid_spacing=0.5",
                             "train.batch_size=4"],
    "A6_c2f_aug_norecover": ["train.rot_augment=true",  "model.refine.enabled=true",
                             "train.exclude_second_fly=true"],
    "A7_c2f_aug_femwt":     ["train.rot_augment=true",  "model.refine.enabled=true",
                             "train.female_weight=3.0"],
    "A8_c2f_aug_seed2":     ["train.rot_augment=true",  "model.refine.enabled=true",
                             "train.seed=1"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--arms", nargs="*", default=sorted(ARMS))
    ap.add_argument("--max-parallel", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    running = []
    for gpu, arm in enumerate(args.arms):
        cmd = [sys.executable, "-u", "-m", "jarvis_jax.train.train_3d_cached",
               f"run_id=v5_{arm}", "train=cached3d", "paths=hyak",
               f"paths.cache_dir={args.cache_dir}",
               f"train.total_steps={args.steps}",
               "train.sharpen=3", "train.laplacian_weight=0.0", *ARMS[arm]]
        env = dict(os.environ,
                   CUDA_VISIBLE_DEVICES=str(gpu % 8),
                   XLA_PYTHON_CLIENT_MEM_FRACTION="0.9")
        print("LAUNCH", arm, "on GPU", gpu % 8, " ".join(cmd))
        if args.dry_run:
            continue
        log = open(f"/gscratch/portia/eabe/data/Johnson_lab/"
                   f"jax_cached3d_runs/v5_{arm}.log", "a")
        running.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=log))
        while len([p for p in running if p.poll() is None]) >= args.max_parallel:
            running[0].wait()
            running = [p for p in running if p.poll() is None]
    for p in running:
        p.wait()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dry-run, then launch 4, then 8**

```bash
cd $PKG
python scripts/train_arms.py --cache-dir $CACHE --dry-run
# health check at 4-way first (the documented CUDA_ERROR_UNKNOWN cap)
python scripts/train_arms.py --cache-dir $CACHE --max-parallel 4 \
  --arms A1_base A2_base_aug A3_c2f A4_c2f_aug &
sleep 600 && nvidia-smi --query-gpu=index,memory.used --format=csv
```

Expected: 4 processes each holding GPU memory, no `CUDA_ERROR_UNKNOWN` in any
log. Only then launch the remaining 4 arms.

- [ ] **Step 4b: Emit the arrays Task 16 consumes (controller ruling R4)**

Task 16 reads `val_pred.npz`, `val_gt.npz` and `keypoint_names.json`, which no
other task creates. Add to the trainer's eval path: after the final val pass,
write `<run_dir>/val_pred.npz` with key `kp3d` of shape `(N, 50, 3)`; and write
`$V5/annotations/val_gt.npz` (same key and shape, ground truth) plus
`$V5/annotations/keypoint_names.json` (the `keypoint_names` list taken from
`instances_val.json`) once. Verify:

```bash
python - <<'PY'
import numpy as np, json, glob
for f in sorted(glob.glob("/gscratch/portia/eabe/data/Johnson_lab/"
                          "jax_cached3d_runs/v5_A*/val_pred.npz")):
    print(f, np.load(f)["kp3d"].shape)
V5 = "/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v5"
print("gt", np.load(f"{V5}/annotations/val_gt.npz")["kp3d"].shape,
      "names", len(json.load(open(f"{V5}/annotations/keypoint_names.json"))))
PY
```

Expected: every arm's `val_pred.npz` and the single `val_gt.npz` share
`(N, 50, 3)`, and `keypoint_names.json` has 50 entries.

- [ ] **Step 5: Collect the scorecard**

```bash
cd $PKG && python - <<'PY'
import json, glob, os
rows = []
for d in sorted(glob.glob("/gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/v5_A*")):
    p = os.path.join(d, "val_mpjpe.json")
    if os.path.exists(p):
        rows.append((os.path.basename(d), json.load(open(p))))
# Ruling R10: group letters are positional (descending recording count), so
# bout 28's group is 'B', not 'A'. Print EVERY group rather than hardcoding a
# letter — a hardcoded 'A' silently reports the wrong cohort.
for name, r in rows:
    groups = r.get('by_calib_group', {})
    gtxt = "  ".join(f"{g}={groups[g]:.3f}" for g in sorted(groups))
    print(f"{name:24s} overall {r.get('overall'):.3f}  [{gtxt}]  "
          f"female {r.get('by_sex', {}).get('female', float('nan')):.3f}")
print("\nbout 28's calibration group is 'B' (verified 2026-08-29) — read that "
      "column, not 'A'.")
PY
```

**Read A8 against A4 first.** A8 is the same config with a different seed, so
`|A4 − A8|` is the noise floor; treat any arm-to-arm difference smaller than
that as no difference at all. Write the table to
`docs/benchmark/2026-08-29-c2f-3d/phase3-arms.md` and commit that file.

- [ ] **Step 6: Commit**

```bash
git add $PKG/scripts/train_arms.py $PKG/scripts/precompute_repro_cache.py \
        docs/benchmark/2026-08-29-c2f-3d/phase3-arms.md
git commit -m "feat(3d): 8-arm training fan-out with per-group/per-sex val

Same wall clock as one arm, so every design choice is attributed. A8 is a
seed replica of A4 and defines the noise floor.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 16: The falsifiable resolution figure

**Files:**
- Create: `<repo>/scripts/viz/voxel_resolution.py` (committed)
- Test: `<repo>/tests/test_voxel_resolution.py`

**Interfaces:**
- Consumes: `$V5` val split, an arm's predictions.
- Produces: `segment_lengths_voxels(kp3d, keypoint_names, grid_spacing=1.0) -> dict[str, float]` and `plot_error_vs_segment(pred, gt, keypoint_names, *, out_png, grid_spacing=1.0) -> dict`.

**This is the gate that can falsify the whole premise.** If per-joint 3D error
does NOT concentrate on the sub-2-voxel segments, the resolution hypothesis is
wrong and the spec says so.

- [ ] **Step 1: Write the failing test**

```python
# <repo>/tests/test_voxel_resolution.py
import numpy as np
from scripts.viz.voxel_resolution import segment_lengths_voxels

NAMES = ["Scutellum", "Abd_tip", "T1L_TaT3", "T1L_TaTip"]


def test_segment_lengths_in_voxels_scale_with_spacing():
    kp = np.zeros((4, 4, 3), np.float32)
    kp[:, 1] = [12.8, 0, 0]      # Scutellum -> Abd_tip = 12.8 world units
    kp[:, 3] = [1.59, 0, 0]      # tarsal segment = 1.59 world units
    kp[:, 2] = [0, 0, 0]
    a = segment_lengths_voxels(kp, NAMES, grid_spacing=1.0)
    b = segment_lengths_voxels(kp, NAMES, grid_spacing=0.25)
    assert np.isclose(a["Scutellum->Abd_tip"], 12.8, atol=0.1)
    assert np.isclose(a["T1L_TaT3->T1L_TaTip"], 1.59, atol=0.1)
    # finer spacing => MORE voxels per segment (the point of stage 2)
    assert np.isclose(b["T1L_TaT3->T1L_TaTip"] / a["T1L_TaT3->T1L_TaTip"], 4.0,
                      rtol=1e-3)


def test_nan_keypoints_are_ignored():
    kp = np.full((2, 4, 3), np.nan, np.float32)
    kp[0, 0] = [0, 0, 0]
    kp[0, 1] = [10, 0, 0]
    out = segment_lengths_voxels(kp, NAMES, grid_spacing=1.0)
    assert np.isclose(out["Scutellum->Abd_tip"], 10.0, atol=1e-3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd <repo> && python -m pytest tests/test_voxel_resolution.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.viz.voxel_resolution'`

- [ ] **Step 3: Write the implementation**

```python
# <repo>/scripts/viz/voxel_resolution.py
"""Per-joint 3D error against segment length in voxels — the falsifiable gate.

STATED EXPECTATION. If the resolution hypothesis is right, per-joint error at
stage 1 (48^3, spacing 1) concentrates on the SHORT segments: the tarsal links
T1L_TiTa->T1L_TaT1, T1L_TaT1->T1L_TaT3 and T1L_TaT3->T1L_TaTip are 1.6-2.2
voxels and should show the largest error relative to their own length, while
the body (12.8 voxels) and wing (17.5 voxels) segments should be comparatively
clean. Stage 2 at spacing 0.25 should collapse that dependence.

IF ERROR IS FLAT ACROSS SEGMENT LENGTHS, THE HYPOTHESIS IS WRONG and the
remaining error lives upstream (2D) or downstream (IK). Report that outcome
rather than reaching for it.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SEGMENTS = [
    ("Scutellum", "Abd_tip"), ("WingL_base", "WingL_V12"),
    ("WingL_V12", "WingL_V13"), ("T1L_FeTi", "T1L_TiTa"),
    ("T1L_TiTa", "T1L_TaT1"), ("T1L_TaT1", "T1L_TaT3"),
    ("T1L_TaT3", "T1L_TaTip"), ("T3L_TiTa", "T3L_TaT1"), ("EyeL", "EyeR"),
]


def segment_lengths_voxels(kp3d, keypoint_names, grid_spacing: float = 1.0) -> dict:
    """Median segment length expressed in VOXELS at the given grid spacing."""
    kp3d = np.asarray(kp3d, np.float64)
    out = {}
    for a, b in SEGMENTS:
        if a not in keypoint_names or b not in keypoint_names:
            continue
        ia, ib = keypoint_names.index(a), keypoint_names.index(b)
        d = np.linalg.norm(kp3d[:, ia] - kp3d[:, ib], axis=-1)
        med = np.nanmedian(d)
        if np.isfinite(med):
            out[f"{a}->{b}"] = float(med / grid_spacing)
    return out


def plot_error_vs_segment(pred, gt, keypoint_names, *, out_png,
                          grid_spacing: float = 1.0, label: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lengths = segment_lengths_voxels(gt, keypoint_names, grid_spacing)
    err = np.nanmedian(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1), 0)
    xs, ys, names = [], [], []
    for seg, L in lengths.items():
        a, b = seg.split("->")
        e = float(np.nanmean([err[keypoint_names.index(a)],
                              err[keypoint_names.index(b)]]))
        xs.append(L); ys.append(e / max(L, 1e-6)); names.append(seg)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.scatter(xs, ys, s=48, c="#00c2c7")
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.axvline(2.0, ls="--", c="#888",
               label="2 voxels — below this, quantization dominates")
    ax.set_xlabel("segment length (voxels at grid_spacing=%.2f)" % grid_spacing)
    ax.set_ylabel("median 3D error / segment length")
    ax.set_title(f"Per-joint error vs segment resolution {label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    stats = {"lengths_voxels": lengths,
             "err_over_length": dict(zip(names, ys)),
             "short_segments_mean": float(np.mean([y for x, y in zip(xs, ys) if x < 2.0]))
             if any(x < 2.0 for x in xs) else None,
             "long_segments_mean": float(np.mean([y for x, y in zip(xs, ys) if x >= 4.0]))
             if any(x >= 4.0 for x in xs) else None}
    with open(out_png.replace(".png", ".json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-npz", required=True)
    ap.add_argument("--gt-npz", required=True)
    ap.add_argument("--names-json", required=True)
    ap.add_argument("--out-png", required=True)
    ap.add_argument("--grid-spacing", type=float, default=1.0)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    names = json.load(open(a.names_json))
    stats = plot_error_vs_segment(np.load(a.pred_npz)["kp3d"],
                                  np.load(a.gt_npz)["kp3d"], names,
                                  out_png=a.out_png, grid_spacing=a.grid_spacing,
                                  label=a.label)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd <repo> && python -m pytest tests/test_voxel_resolution.py -v`
Expected: 2 passed

- [ ] **Step 5: Generate and READ the figure for A1 and A4**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
for ARM in A1_base A4_c2f_aug; do
  python scripts/viz/voxel_resolution.py \
    --pred-npz /gscratch/portia/eabe/data/Johnson_lab/jax_cached3d_runs/v5_$ARM/val_pred.npz \
    --gt-npz $V5/annotations/val_gt.npz \
    --names-json $V5/annotations/keypoint_names.json \
    --out-png figures/2026-08-29-c2f-3d/phase3-resolution/$ARM.png \
    --label "$ARM"
done
```

Open both PNGs with the Read tool. **Report whether `short_segments_mean`
exceeds `long_segments_mean` for A1 (hypothesis holds) and whether A4 narrows
the gap.** If A1 shows no dependence on segment length, say so plainly — the
resolution hypothesis is refuted and Task 17's interpretation changes.

- [ ] **Step 6: Commit**

```bash
git add scripts/viz/voxel_resolution.py tests/test_voxel_resolution.py
git commit -m "feat(viz): falsifiable per-joint error vs segment-resolution gate

If error does not concentrate on the sub-2-voxel tarsal segments, the
resolution hypothesis is wrong and the figure will say so.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 17: Bout 28 end-to-end and acceptance (Phase 4)

**Files:**
- Modify: `<repo>/configs/detector/vitpose_v3.yaml` (point at the new checkpoint, behind a variant)
- Create: `docs/benchmark/2026-08-29-c2f-3d/acceptance.md` (committed)

**Interfaces:**
- Consumes: the winning arm, `refine_from_seed`, Task 1's baseline arrays.
- Produces: the acceptance figures and a committed notes file.

- [ ] **Step 1: Write the expectation BEFORE running**

Create `figures/2026-08-29-c2f-3d/phase4-acceptance/EXPECTATION.md`:

```markdown
# Phase 4 acceptance expectation

Bout 28, both flies, new stack (v5_sigma2 detector -> winning arm -> stage-3
refine -> STAC IK) against the Task-1 baseline.

**Controller rulings R5/R6 — this expectation was REWRITTEN from Task 1's
measured baseline.** An earlier draft predicted the new stack would remove a
~0.17 mm voxel staircase "visible in the baseline". That was wrong: the
baseline pipeline has NO volumetric lifter at all (`run_bout.py` contains zero
v2vnet/hybridnet references; Stage B is plain DLT), so it is continuous by
construction and Task 1 correctly found no staircase. Its absence neither
confirms nor falsifies the resolution hypothesis — that test is Task 16
(arm A1's volumetric output against GT), not this one.

Task 1 measured what bout 28's failure ACTUALLY is:

| metric | fly0 (female) | fly1 (male) |
|---|---|---|
| reproj median | 10.26 px | 6.02 px |
| reproj p99 / max | 62.74 / **90.26** px | 7.13 / 7.56 px |
| **frames with NaN reproj** | **25.2%** | 0.0% |
| worst window | offsets ~1484-1501 (proximity event) | — |

The female is NOT broadly bad on this bout (10.26 px typical, only 1.7x the
male — the frozen cohort's 42.3 px median does not describe it). The failure is
LOCALIZED: a proximity event where the fitted mesh detaches and floats in empty
space, plus a quarter of frames where DLT cannot triangulate at all.

**So the win must come from coverage and occlusion robustness, not precision.**
Stage 1 (volumetric fusion) and stage 3 (continuous refine with seed fallback)
carry it: fusion always yields an answer where DLT NaNs out, and stage 3 keeps
DLT-grade precision where views are trustworthy. Stage 2 addresses resolution,
which this bout says is not its binding constraint — judge it on Task 16, not here.

Expected if the approach works:
- fly0 NaN-frame rate falls well below 25.2%, and the mesh stays ON the animal
  through the offset ~1484-1501 proximity window
- fly0 max/p99 reprojection falls from 90.26/62.74 px; fly1 does NOT regress
  from 6.02 px median (it is already near the detector noise floor)
- typical-frame reprojection does not regress: fly0 stays at or below 10.26 px
- leg-angle traces smooth WITHIN the proximity window specifically (fly0 whole-
  bout p99 step was 30.3 deg, max 71.3 deg, all inside that window) — without
  added lag, since a smoother trace that also lags means a filter crept in
- bone-length CV does NOT collapse to ~0: rigid bones enforced for free are a
  known false signal here, not a win

Expected if it does not work:
- the NaN rate and the proximity failure persist, meaning the problem is
  upstream (2D detection under occlusion) or downstream (IK). The prior
  measurement that STAC IK reprojects 30-60 px off the DLT 3D it consumes makes
  IK the first suspect. Report that outcome rather than reaching past it.

Do NOT reuse the frozen 13-bout scorecard's cohort medians as thresholds for
this bout — Task 1 showed they describe its worst moments but not its typical
state. Thresholds come from the table above.
```

- [ ] **Step 2: Run bout 28 through the new stack**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
micromamba activate 3d_tracking && unset LD_LIBRARY_PATH
export MUJOCO_GL=egl XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
# Ruling R8 (verified in Task 1): `+bout_ids=28` FAILS — the key is already
# defined, so use `bout_ids=28` with no `+`. Likewise the hydra config name is
# `pipeline`, not `config`.
python scripts/run_bout.py paths=hyak recording=session0 bout_ids=28 \
  detector.ckpt=/gscratch/portia/eabe/data/Johnson_lab/jax_vitpose_runs/v5_sigma2/final \
  outputs.out=OutFiles/c2f_3d_bout28 2>&1 \
  | tee figures/2026-08-29-c2f-3d/phase4-acceptance/run.log
```

Expected: stages A–E complete for both flies.

- [ ] **Step 3: Produce the three acceptance figures**

```bash
DST=figures/2026-08-29-c2f-3d/phase4-acceptance
BASE=figures/2026-08-29-c2f-3d/phase0-baseline/arrays/bout_00028
# 1. wing + leg angle traces, both flies, baseline vs new.
#    compare_stac_fits.py takes --xml/--anatomy and has NO --bout (ruling R1);
#    use the exact invocation Task 1's report recorded.
python scripts/viz/compare_stac_fits.py \
  --fit baseline=$BASE --fit c2f=OutFiles/c2f_3d_bout28 \
  --xml <from Task 1 report> --anatomy <from Task 1 report> --out $DST
# 2. per-camera overlays + leg chains, both flies, incl. occlusion and wall frames
for FLY in 0 1; do
  python -m viz overlay --run OutFiles/c2f_3d_bout28 --bout 28 --fly $FLY \
    --compare $BASE --out $DST
  python -m viz legskel --run OutFiles/c2f_3d_bout28 --bout 28 --fly $FLY \
    --compare $BASE --out $DST
done
# 3. benchmark regression check
python -m scripts.benchmark.run_variant collect --variant c2f_3d --out $DST
```

- [ ] **Step 4: READ every figure and write the observations**

Open each PNG with the Read tool. For videos, extract a few frames and read
those. Append an `## Observed` section to `EXPECTATION.md` reporting, per fly
and in real units:
- reprojection error (px) baseline vs new, for `Cam2012630`, `Cam2012855` and
  one more camera, on a mid-bout frame, an occlusion frame and a wall frame
- whether tarsal-tip traces lost the staircase
- whether leg-angle traces smoothed without lag
- the benchmark scorecard delta, with any cohort that regressed

State disagreements with the expectation explicitly. **A figure that was
generated but never viewed is not evidence, and "the plot confirms it" without
having looked is a false claim about verification.**

- [ ] **Step 5: Write and commit the acceptance notes**

```bash
cp figures/2026-08-29-c2f-3d/phase4-acceptance/EXPECTATION.md \
   docs/benchmark/2026-08-29-c2f-3d/acceptance.md
git add docs/benchmark/2026-08-29-c2f-3d/acceptance.md
git commit -m "docs(3d): bout 28 acceptance — observed vs expected, both flies

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 6: Report honestly, and do not promote on your own authority**

Summarize for the user: which arm won, by how much against the A4/A8 noise
floor, whether the female improved, and whether the IK absorbed the gain.
Promoting the new lifter into the production pipeline is a **separate
decision** (spec §2 non-goals) — present the evidence and let the user make it.

---

---

## Task 18: Pairwise male/female classifier (Phase 5)

The existing wing-song sexing got **24 of 160 hand-reviewed bouts wrong (15%)**.
This task builds a classifier that decides, for two flies **in the same
frameset**, which is the male — the shape the courtship problem actually has.

**Files:**
- Create: `$PKG/jarvis_jax/tracking/sex_pairwise.py`
- Create: `<repo>/scripts/viz/sex_pair_sheet.py` (committed — the labelling aid)
- Test: `$PKG/tests/test_sex_pairwise.py`

**Interfaces:**
- Consumes: `$V5/annotations/instances.json` (needs Task 4's recovered second flies), `V5FramesetDataset` for 3D keypoints.
- Produces:
  - `pair_features(kp3d_a, kp3d_b, keypoint_names) -> np.ndarray` — **antisymmetric**: `pair_features(b, a) == -pair_features(a, b)` exactly.
  - `fit_pairwise(X, y, groups) -> dict` — logistic regression, **no intercept**, group-aware CV.
  - `predict_male_slot(model, kp3d_a, kp3d_b, names) -> tuple[int, float]` — 0 or 1, plus probability.
  - `clip_index(merged) -> dict[str, list[str]]` — frameset keys grouped into contiguous clips.

### Why pairwise, and why antisymmetric

Sex is **perfectly confounded with recording** in the absolute labels: all 4
male-labelled recordings are 100% male, all 4 female ones 100% female. A
per-fly classifier trained on those can score perfectly by learning lighting or
one individual's quirks. The two-fly recordings remove that entirely — both
animals appear in the **same frame, same camera, same instant**, so the only
difference is the animal.

Antisymmetry is enforced by construction, not hoped for: every feature is
`f(a) - f(b)`, and the model has **no intercept**, so
`P(a is male) = 1 - P(b is male)` identically. A classifier that could call
both flies male is not a classifier of a courtship pair.

### Labelling: 29 decisions, not 264

Measured: the 264 two-fly framesets fall into **29 contiguous clips**
(`2026_04_08_14_59_45` 7 clips / 214 framesets, `2026_04_07_11_33_33` 9 / 30,
`2026_06_11_13_58_43` 10 / 17, `2026_06_11_13_58_45` 3 / 3).

Annotation order is **not** spatial (`ann0 < ann1` in only 53% of framesets) but
**is** a stable track identity within a clip: on 113 near-adjacent frameset
pairs, slot 0 stayed with the nearer fly **100%** of the time. So one
"which slot is the male" decision per clip labels every frameset in it.

- [ ] **Step 1: Write the failing test**

```python
# $PKG/tests/test_sex_pairwise.py
import numpy as np
import pytest

from jarvis_jax.tracking.sex_pairwise import (
    pair_features, fit_pairwise, predict_male_slot, clip_index)

NAMES = ["Scutellum", "Abd_A4", "Abd_tip", "WingL_base", "WingL_V12",
         "EyeL", "EyeR", "T1L_FeTi", "T1L_TiTa"]


def _fly(scale=1.0, abd=1.0):
    """Synthetic fly: `scale` sets overall size, `abd` the abdomen extension."""
    kp = np.zeros((len(NAMES), 3), np.float32)
    kp[NAMES.index("Scutellum")] = [0, 0, 0]
    kp[NAMES.index("Abd_A4")] = [0, -6 * scale, 0]
    kp[NAMES.index("Abd_tip")] = [0, -13 * scale * abd, 0]
    kp[NAMES.index("WingL_base")] = [1 * scale, -1 * scale, 0]
    kp[NAMES.index("WingL_V12")] = [3 * scale, -18 * scale, 0]
    kp[NAMES.index("EyeL")] = [-2 * scale, 4 * scale, 0]
    kp[NAMES.index("EyeR")] = [2 * scale, 4 * scale, 0]
    kp[NAMES.index("T1L_FeTi")] = [-3 * scale, 1 * scale, 0]
    kp[NAMES.index("T1L_TiTa")] = [-5 * scale, -2 * scale, 0]
    return kp


def test_pair_features_are_exactly_antisymmetric():
    """The property the whole design rests on: swapping the pair must negate
    the features, so P(a male) == 1 - P(b male) by construction."""
    a, b = _fly(1.0), _fly(1.25, abd=1.1)
    fab = pair_features(a, b, NAMES)
    fba = pair_features(b, a, NAMES)
    np.testing.assert_allclose(fab, -fba, atol=1e-6)


def test_identical_flies_give_zero_features():
    a = _fly(1.0)
    np.testing.assert_allclose(pair_features(a, a.copy(), NAMES), 0.0, atol=1e-6)


def test_features_are_not_all_scale_normalised_away():
    """Size IS the signal (females are larger). A feature vector that is
    invariant to overall scale has thrown away the main cue."""
    small, big = _fly(1.0), _fly(1.3)
    f = pair_features(small, big, NAMES)
    assert np.abs(f).max() > 1e-3


def test_prediction_is_swap_consistent():
    rng = np.random.default_rng(0)
    X, y, g = [], [], []
    for i in range(40):
        male = _fly(1.0 + rng.normal(0, .02), abd=1.0)
        female = _fly(1.25 + rng.normal(0, .02), abd=1.15)
        # alternate which slot holds the male so the label is not slot-correlated
        if i % 2:
            X.append(pair_features(male, female, NAMES)); y.append(0)
        else:
            X.append(pair_features(female, male, NAMES)); y.append(1)
        g.append(f"clip{i // 8}")
    m = fit_pairwise(np.array(X), np.array(y), np.array(g))
    male, female = _fly(1.0), _fly(1.25, abd=1.15)
    s0, p0 = predict_male_slot(m, male, female, NAMES)
    s1, p1 = predict_male_slot(m, female, male, NAMES)
    assert s0 == 0 and s1 == 1, (s0, s1)
    assert abs(p0 - p1) < 1e-6, "swap must give the mirrored probability"


def test_model_has_no_intercept():
    """An intercept would break antisymmetry: the model could prefer slot 0."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 4)); y = (X[:, 0] > 0).astype(int)
    g = np.array([f"c{i//5}" for i in range(30)])
    m = fit_pairwise(X, y, g)
    assert float(np.abs(m["intercept"])) == 0.0


def test_cv_groups_by_clip_never_by_frameset():
    """Adjacent framesets in a clip are near-duplicates. Grouping CV by
    frameset would leak exactly the way the dataset split used to."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 3)); y = (X[:, 0] > 0).astype(int)
    g = np.array([f"clip{i//10}" for i in range(40)])
    m = fit_pairwise(X, y, g)
    assert set(m["cv_groups"]) == {"clip0", "clip1", "clip2", "clip3"}
    assert len(m["fold_accuracy"]) == 4


def test_clip_index_splits_on_frame_gaps():
    merged = {"framesets": {}}
    for f in [10, 11, 12, 900, 901]:
        merged["framesets"][f"rec_a/Frame_{f:06d}/fly0"] = {
            "recording": "rec_a", "fly_id": 0, "frames": [], "ann_ids": []}
    clips = clip_index(merged, gap=100)
    assert len(clips) == 2
    assert sorted(len(v) for v in clips.values()) == [2, 3]


def test_nan_keypoints_do_not_poison_features():
    a, b = _fly(1.0), _fly(1.25)
    b[NAMES.index("WingL_V12")] = np.nan
    f = pair_features(a, b, NAMES)
    assert np.all(np.isfinite(f)), "NaN keypoints must degrade, not propagate"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd $PKG && python -m pytest tests/test_sex_pairwise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jarvis_jax.tracking.sex_pairwise'`

- [ ] **Step 3: Write the implementation**

```python
# $PKG/jarvis_jax/tracking/sex_pairwise.py
"""Which of two flies in one frameset is the male?

WHY PAIRWISE. In the absolute labels, sex is perfectly confounded with
recording: all 4 male-labelled recordings are 100% male, all 4 female ones 100%
female. A per-fly classifier can score perfectly by learning lighting or one
individual's quirks, and nothing in an in-recording validation would show it.
The two-fly courtship recordings remove the confound completely -- both animals
appear in the same frame, same camera, same instant.

WHY ANTISYMMETRIC. Every feature is f(a) - f(b) and the model carries NO
intercept, so P(a is male) == 1 - P(b is male) identically. A model that could
call both flies in a courtship pair male is not modelling the problem.

CUES. Females are larger with a longer, more pointed abdomen; males are smaller
with a blunter, darker tip. Overall SIZE is a real cue here, so features are
deliberately NOT scale-normalised. Note the recorded gotcha: mask AREA is
backwards during courtship because the male extends a wing during song, so his
silhouette is LARGER -- that is why these features come from 3-D keypoint
geometry, not from silhouette extent.
"""
from __future__ import annotations

import re

import numpy as np

_FRAME_RE = re.compile(r"Frame_(\d+)")

# (name_a, name_b) segment lengths used as scalar descriptors.
_SEGMENTS = [
    ("Scutellum", "Abd_tip"),     # body length
    ("Scutellum", "Abd_A4"),      # thorax->mid-abdomen
    ("Abd_A4", "Abd_tip"),        # abdomen taper section
    ("WingL_base", "WingL_V12"),  # wing length
    ("EyeL", "EyeR"),             # head width
    ("T1L_FeTi", "T1L_TiTa"),     # tibia
]


def _scalars(kp3d: np.ndarray, names: list[str]) -> np.ndarray:
    out = []
    for a, b in _SEGMENTS:
        if a in names and b in names:
            d = kp3d[names.index(a)] - kp3d[names.index(b)]
            v = float(np.linalg.norm(d))
            out.append(v if np.isfinite(v) else 0.0)
        else:
            out.append(0.0)
    finite = kp3d[np.isfinite(kp3d).all(axis=-1)]
    extent = (float(np.linalg.norm(finite.max(0) - finite.min(0)))
              if finite.shape[0] >= 2 else 0.0)
    out.append(extent)
    body = out[0]
    # abdomen fraction of body: shape cue that survives a size difference
    out.append(out[2] / body if body > 1e-9 else 0.0)
    return np.asarray(out, np.float64)


def pair_features(kp3d_a, kp3d_b, keypoint_names) -> np.ndarray:
    """Antisymmetric descriptor of the ORDERED pair (a, b)."""
    a = _scalars(np.asarray(kp3d_a, np.float64), list(keypoint_names))
    b = _scalars(np.asarray(kp3d_b, np.float64), list(keypoint_names))
    f = a - b
    return np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)


def fit_pairwise(X, y, groups) -> dict:
    """Logistic regression, NO intercept, leave-one-group-out CV.

    `groups` MUST be clip ids, never frameset ids: adjacent framesets inside a
    clip are near-duplicates, and grouping by frameset would leak exactly the
    way this project's dataset split used to.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut

    X = np.asarray(X, np.float64)
    y = np.asarray(y, int)
    groups = np.asarray(groups)

    accs, uniq = [], sorted(set(groups.tolist()))
    logo = LeaveOneGroupOut()
    for tr, te in logo.split(X, y, groups):
        if len(set(y[tr].tolist())) < 2:
            continue
        m = LogisticRegression(fit_intercept=False, max_iter=2000)
        m.fit(X[tr], y[tr])
        accs.append(float(m.score(X[te], y[te])))

    final = LogisticRegression(fit_intercept=False, max_iter=2000)
    final.fit(X, y)
    return {"coef": final.coef_[0].copy(), "intercept": 0.0,
            "fold_accuracy": accs, "cv_groups": uniq,
            "cv_mean": float(np.mean(accs)) if accs else float("nan"),
            "_model": final}


def predict_male_slot(model, kp3d_a, kp3d_b, keypoint_names):
    """-> (slot, probability). slot 0 means `kp3d_a` is the male."""
    f = pair_features(kp3d_a, kp3d_b, keypoint_names)
    z = float(np.dot(model["coef"], f))       # no intercept => exactly antisymmetric
    p_b_male = 1.0 / (1.0 + np.exp(-z))
    return (1, p_b_male) if p_b_male >= 0.5 else (0, 1.0 - p_b_male)


def clip_index(merged: dict, *, gap: int = 100) -> dict[str, list[str]]:
    """Group frameset keys into contiguous clips (frame gaps > `gap` split).

    Annotation order is NOT spatial (ann0 < ann1 in only 53% of framesets) but
    IS a stable track identity within a clip -- measured: on 113 near-adjacent
    frameset pairs, slot 0 stayed with the nearer fly 100% of the time. So one
    'which slot is the male' decision labels a whole clip.
    """
    per_rec: dict[str, list[tuple[int, str]]] = {}
    for key, v in merged["framesets"].items():
        m = _FRAME_RE.search(key)
        if m is None:
            continue
        per_rec.setdefault(v["recording"], []).append((int(m.group(1)), key))
    clips: dict[str, list[str]] = {}
    for rec, items in per_rec.items():
        items.sort()
        idx, cur, prev = 0, [], None
        for fr, key in items:
            if prev is not None and fr - prev > gap:
                clips[f"{rec}#clip{idx:03d}"] = cur
                idx += 1
                cur = []
            cur.append(key)
            prev = fr
        if cur:
            clips[f"{rec}#clip{idx:03d}"] = cur
    return clips
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd $PKG && python -m pytest tests/test_sex_pairwise.py -v`
Expected: 8 passed

- [ ] **Step 5: Build the 29 clip sheets and STOP for labelling**

Write `<repo>/scripts/viz/sex_pair_sheet.py`, reusing `scripts/viz/contact_sheet.py`
(Task 6) for crop extraction. One PNG per clip: both flies side by side, slot 0
left and slot 1 right, several frames across the clip, two cameras, keypoints
overlaid, filename `<recording>#clip<NNN>.png`.

**Read at least 5 sheets with the Read tool and confirm the two panels really
show different animals** — if slot 0 and slot 1 look like the same fly, the
clip's track identity is broken and that clip must be excluded, not labelled.

Then hand them to the user for 29 `clip -> male_slot` decisions. **Do not guess
sex yourself.** Write the answers to `$V5/annotations/sex_pairs.json` as
`{"<recording>#clip<NNN>": 0|1}`.

- [ ] **Step 6: Train, and report the HONEST number**

```bash
cd $PKG && python -m jarvis_jax.tracking.sex_pairwise \
  --v5-root $V5 --labels $V5/annotations/sex_pairs.json \
  --out ../../figures/2026-08-29-c2f-3d/phase5-sex/
```

Report BOTH numbers, and lead with the second:
1. **leave-one-CLIP-out** accuracy — optimistic; clips within a recording share
   the same two individuals.
2. **leave-one-RECORDING-out** accuracy — the honest one. Only 4 recordings, so
   this is 4 folds and will be noisy; report per-fold, not just the mean.

**Independent check:** run the trained model on the 8 sex-labelled single-fly
recordings by pairing a male frameset against a female frameset (it never saw
any of them). Report accuracy. If leave-one-recording-out is near chance while
leave-one-clip-out is high, the model learned the individuals, not the sex —
**say so plainly rather than reporting the flattering number.**

**Figure gate:** per-feature male-vs-female distributions, **coloured by
recording**. Stated expectation: if the separation is real, the male and female
clusters stay separated *within* every recording's own colour; if the
separation only appears between recording colours, the model is reading
individuals or lighting, and the pairwise design has not saved us.

- [ ] **Step 7: Commit**

```bash
git add $PKG/jarvis_jax/tracking/sex_pairwise.py $PKG/tests/test_sex_pairwise.py \
        scripts/viz/sex_pair_sheet.py
git commit -m "feat(sexing): antisymmetric pairwise male/female classifier

Wing-song sexing got 24 of 160 hand-reviewed bouts wrong (15%). This decides
which of two flies in ONE frameset is the male, so lighting, session and
individual are controlled by construction. No intercept => P(a male) is
exactly 1 - P(b male). CV groups by clip, never by frameset.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Task 19: Diagnose the bout-28 identity/coverage failure (Phase 5)

**Diagnosis only. No fix, no pipeline source changes.** The fix task is written
only after this answers what actually breaks.

**Files:**
- Create: `<repo>/scripts/qc/diagnose_mask_dropout.py` (committed — regenerable diagnostic)
- Outputs: `figures/2026-08-29-c2f-3d/phase5-dropout/`, notes committed to `docs/benchmark/2026-08-29-c2f-3d/dropout-diagnosis.md`

**Interfaces:**
- Consumes: `sam3_masks/bout_00028/sam3_masks.npz`, the archived `kp2d.npz` / `kp3d.npz` / `qc_perframe.npz` from Task 1, `configs/detector/vitpose_v3.yaml`.
- Produces: `dropout_report(mask_npz, kp2d, kp3d, *, split_frame) -> dict` with per-camera per-stage survival counts, and a per-stage attrition figure.

### START HERE: two accounts conflict, and resolving that is job one

Task 1's fix round concluded the failure was *"a SAM3 mask/identity failure —
the mask drifts onto empty background in several cameras."* **Direct
measurement of `sam3_masks.npz` contradicts that:**

| measurement | value | implication |
|---|---|---|
| fly0 centroid step, frames 1495–1510 | **0.1–3.2 px**, every camera | no drift, no jump |
| fly0↔fly1 centroid separation | ~250–300 px **before and after** 1502 | no merge, no identity swap |
| fly0 `valid`% after 1502 | 853:100 855:100 630:100 861:75 857:69 862:39 **631:17** | fly0 lost in 4 of 7 cameras |
| fly1 `valid`% after 1502 | **100 on all seven** | male entirely unaffected |
| `in_frame` vs `valid` | **identical arrays** | flagged out-of-frame, not mis-segmented |

So the masks are smooth and correctly separated; fly0 is simply marked
**not in frame** in up to 4 cameras from ~1502, while **three cameras keep 100%
valid masks** — and the 3D is still NaN for all 505 frames.

**Leading hypothesis for the contradiction:** the Task 1 overlays were rendered
at frames 1550/1700/1900 without honouring `valid`, so cameras with
`valid=False` displayed stale or garbage mask content, which reads visually as
"drifted onto background". Check that first — if true, the recorded root cause
is wrong and must be corrected before anyone builds a fix on it.

- [ ] **Step 1: Settle the contradiction**

Re-render fly0 mask overlays at frames 1550, 1700, 1900 across all 7 cameras,
**this time drawing only where `valid[fly, cam, frame]` is True** and annotating
each panel with its `valid`/`in_frame` flag. Read every panel with the Read
tool. State plainly which account holds: masks genuinely drifted onto
background, or masks are absent-and-flagged while the fly is out of view.

- [ ] **Step 2: Follow one frame through every gate**

For a representative dead frame (e.g. offset 1600) and a live one (e.g. 1400),
tabulate for fly0 what survives each stage, per camera:

| stage | gate | source |
|---|---|---|
| mask present | `valid[0, cam, f]` | `sam3_masks.npz` |
| keypoints emitted | any finite in `kp2d` | Task 1's archived `kp2d.npz` |
| per-keypoint conf | `conf_thresh: 0.3` | `configs/detector/vitpose_v3.yaml` |
| per-view median conf | `view_conf_thresh: 0.6` | same |
| consensus | `reproj_resid_px: 10.0`, needs ≥3 views | `tracking/triangulate.py` |
| 3D emitted | finite in `kp3d.npz` | Task 1's archive |

**The question this answers:** three cameras hold 100% valid masks through the
dead zone, which is enough to triangulate. So which gate removes them? If
`view_conf_thresh: 0.6` is dropping views whose masks are fine, the fix is a
threshold/coverage problem, not an identity problem — a completely different
task from the one implied by "identity failure".

- [ ] **Step 3: Test the "most cameras are right" assumption**

`configs/sam3/default.yaml` ships `repair_outliers` (per-camera leave-one-out
residual > 40 px → box-prompt re-segment) and `repair_missing` (a camera that
never picked up a fly that walked in; SAM3VideoTracker propagates forward-only
from frame 0). Neither fired here. Determine which of these is true:
(a) they fired and were rejected by their accept thresholds
(`repair_missing_accept_resid: 25.0`, `repair_resid_thresh: 40.0`);
(b) they never triggered because the fly is flagged out-of-frame rather than
mis-segmented; or (c) they are structurally blind because they assume most
cameras are right, and here 4 of 7 fail together.
Cite the code path and the numbers, not a guess.

- [ ] **Step 4: Establish whether the fly is really out of view**

`repair_missing` exists precisely because the tracker propagates forward-only
and misses a fly that enters a camera mid-bout. Reproject fly1's 3D position
and the last good fly0 3D position into each camera at frames 1550/1700/1900
and check whether those pixel locations lie inside the image bounds. **If fly0
is predicted inside the frame in cameras reporting `in_frame=0`, this is a
recoverable tracker failure, not a genuine exit** — and that distinction
decides the whole fix.

- [ ] **Step 5: Write the attrition figure and READ it**

One figure: x = frame (0–2006), stacked count of fly0 cameras surviving each
gate in Step 2, with the 1502 boundary marked. **Stated expectation:** if the
cause is coverage, the mask-present curve drops to ~3 at 1502 and the
keypoint/conf curves drop to 0 — locating the loss downstream of the masks. If
the mask curve itself drops to 0, the masks are the whole story. Read the PNG
and report which shape it has.

- [ ] **Step 6: Commit the diagnostic and the notes**

```bash
git add scripts/qc/diagnose_mask_dropout.py \
        docs/benchmark/2026-08-29-c2f-3d/dropout-diagnosis.md
git commit -m "diag(masks): locate the bout-28 fly0 dropout in the stage chain

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 7: Recommend the fix task, do not write it**

Close the notes with a recommendation naming the single gate or component that
loses the female, and what a fix would change. If the evidence says the cause is
a confidence/coverage gate rather than identity, **say so** — Task 20 will be
scoped from this, and scoping it as an identity fix when it is a coverage
problem would waste the whole task.

---

## Task 20: Graceful degradation of a partial 3D signal (Phase 5)

Scoped **from Task 19's evidence**, not from the assumption that preceded it.
Task 19 refuted the identity/mask-drift framing outright; do not reintroduce it.

**Files:**
- Modify: `<repo>/scripts/run_bout.py` (`finite_frame_mask`, `NAN_SOLVE_MIN_SEG` path)
- Modify: `<repo>/configs/pipeline.yaml` (new gate settings, defaults preserving current behaviour)
- Possibly: `third_party/jarvis_jax/jarvis_jax/tracking/stac.py` (`ik_only_bout`)
- Test: `<repo>/tests/test_partial_frame_solve.py`

**Interfaces:**
- Produces: `finite_frame_mask(kp3d, *, min_keypoints: int | None = None)` — `None` preserves today's all-or-nothing behaviour exactly; an int accepts a frame with at least that many finite keypoints.
- Produces: a marker-validity mask threaded to the IK so absent markers contribute nothing to the residual.

### What Task 19 established (numbers, not guesses)

On bout 28 fly0, frames 1502–2006 read as 100% NaN. Three mechanisms compound:

| mechanism | measured contribution |
|---|---|
| `masks.min_views: 4` on the `kp_mask_agree_fly_lengths: 3.0`-adjusted count | blanks **338/505 (66.9%)** of frames outright |
| `view_conf_thresh: 0.6` vs real median confidence **0.44–0.59** | only **20/505 (4.0%)** frames retain all 50 keypoints |
| `finite_frame_mask` + `NAN_SOLVE_MIN_SEG=30` (after `NAN_SOLVE_MAX_GAP=10`) | needs ≥30 *consecutive fully-complete* frames; the only usable segment in the whole bout is `[0, 1502)` |

Underneath all three, **raw DLT still recovers a partial signal in 18.2% of the tail frames** — and the female is genuinely leaving 4 of 7 camera views, with coverage bottoming out at 2–3 cameras for a sustained run.

### Non-goals — each rejected on evidence

- **Do NOT chase an identity or segmentation bug.** Masks are smooth, correctly separated, and `in_frame` agrees with `valid` bit-for-bit.
- **Do NOT globally lower `masks.min_views`.** The config's own comment records a measurement that this makes bones flex 20–50%. If you touch it at all it must be per-frame and gated, never a corpus-wide default change.
- **Do NOT aim for zero loss.** Where only 2 cameras genuinely see her, full-body DLT is ill-conditioned and the frames are not recoverable. The target is **less total loss**, and the report must state honestly how much remains.

- [ ] **Step 1: Determine whether the IK can accept partial markers — this decides everything after it**

`finite_frame_mask` (`scripts/run_bout.py:567`) requires **every** keypoint finite. The fix depends on whether the solver can ignore absent markers.

Read `stac-mjx/stac_mjx/stac_core.py` and `jarvis_jax/tracking/stac.py::ik_only_bout`. Establish, with the code path cited:
- Does the residual support a **per-marker weight or mask**? (`q_reg_weights` exists for qpos regularization — that is NOT the same thing.)
- What does the solver currently do with a NaN marker — propagate NaN through the whole solve, or ignore it?

Write the answer in your report before writing any fix. **If per-marker masking already exists, the fix is small: relax the frame gate and pass the mask. If it does not, STOP and report** — adding it means changing the `stac-mjx` submodule, which is a separate decision I need to make (that submodule already carries uncommitted local changes).

- [ ] **Step 2: Write the failing test**

```python
# <repo>/tests/test_partial_frame_solve.py
import numpy as np
import pytest

from scripts.run_bout import finite_frame_mask, contiguous_segments, NAN_SOLVE_MIN_SEG


def _kp(T=100, K=50, missing=()):
    kp = np.random.default_rng(0).normal(size=(T, K, 3)).astype(np.float32)
    for t, joints in missing:
        kp[t, list(joints)] = np.nan
    return kp


def test_default_is_byte_identical_to_all_or_nothing():
    """The shipped behaviour must not move unless explicitly opted in."""
    kp = _kp(missing=[(5, [0]), (6, [1, 2])])
    np.testing.assert_array_equal(finite_frame_mask(kp),
                                  finite_frame_mask(kp, min_keypoints=None))
    assert not finite_frame_mask(kp)[5]


def test_min_keypoints_accepts_a_partial_frame():
    """A frame missing 3 of 50 keypoints is usable; today it is discarded."""
    kp = _kp(missing=[(5, [0, 1, 2])])
    assert finite_frame_mask(kp, min_keypoints=40)[5]
    assert not finite_frame_mask(kp, min_keypoints=50)[5]


def test_min_keypoints_still_rejects_a_mostly_empty_frame():
    kp = _kp(missing=[(5, range(45))])
    assert not finite_frame_mask(kp, min_keypoints=40)[5]


def test_partial_frames_can_form_a_solvable_segment():
    """The bout-28 failure in miniature: scattered complete frames never reach
    NAN_SOLVE_MIN_SEG, but the same frames counted partially do."""
    kp = _kp(T=100)
    for t in range(100):
        if t % 3:                       # 2 of every 3 frames lose 3 keypoints
            kp[t, [0, 1, 2]] = np.nan
    assert contiguous_segments(finite_frame_mask(kp), NAN_SOLVE_MIN_SEG) == []
    segs = contiguous_segments(finite_frame_mask(kp, min_keypoints=40),
                               NAN_SOLVE_MIN_SEG)
    assert segs and (segs[0][1] - segs[0][0]) >= NAN_SOLVE_MIN_SEG


def test_marker_mask_matches_the_accepted_frames():
    """Every keypoint the solver is told to ignore must be exactly the NaN ones —
    an off-by-one here silently drops real observations."""
    from scripts.run_bout import marker_validity_mask
    kp = _kp(missing=[(5, [0, 1, 2]), (7, [9])])
    m = marker_validity_mask(kp)
    assert m.shape == kp.shape[:2]
    assert not m[5, 0] and not m[5, 1] and not m[5, 2] and m[5, 3]
    assert not m[7, 9] and m[7, 8]
    np.testing.assert_array_equal(m, np.isfinite(kp).all(axis=-1))
```

- [ ] **Step 3: Run it and see it fail**

Run: `cd <repo> && python -m pytest tests/test_partial_frame_solve.py -v`
Expected: `TypeError: finite_frame_mask() got an unexpected keyword argument 'min_keypoints'`, and `ImportError` for `marker_validity_mask`.

- [ ] **Step 4: Implement**

Add `min_keypoints` to `finite_frame_mask` (default `None` = unchanged), add
`marker_validity_mask`, thread the mask through `ik_only_bout` to the solver
per Step 1's finding, and add config keys under `postprocessing` (or wherever
`NAN_SOLVE_*` is configured) whose **defaults reproduce today's behaviour
exactly**. A run that does not opt in must be byte-identical.

- [ ] **Step 5: Measure on bout 28 and bout 3 — read the figures**

**Stated expectation:** with `min_keypoints` around 40, bout 28 fly0's usable
segment extends past frame 1502 and the NaN-frame rate falls from 25.2%. It
will **not** reach zero — Task 19 showed coverage genuinely bottoms out at 2–3
cameras — and a fitted pose in those frames must be treated with suspicion, not
celebrated. Bout 3 (fly0 0% NaN, median 9.52 px, p99 132.1 px) is the control:
**it must not regress**, since it has no coverage problem to fix.

If the recovered tail frames show the mesh detached from the animal, the
partial solve is fabricating kinematics from too few markers and `min_keypoints`
is too low. Report that rather than the improved NaN rate alone — a lower NaN
rate with a worse fit is a regression wearing a better number.

- [ ] **Step 6: Frozen-benchmark gate**

Run the 13-bout benchmark (`scripts/benchmark/run_variant.py`) with the feature
ON and compare against the committed baseline scorecard. **No cohort may
regress.** Task 19 was explicit that a `min_views`/STAC-criterion change must be
validated corpus-wide rather than assumed safe from one bout — this is that gate.

- [ ] **Step 7: Commit**

```bash
git add scripts/run_bout.py configs/pipeline.yaml tests/test_partial_frame_solve.py \
        docs/benchmark/2026-08-29-c2f-3d/partial-solve-results.md
git commit -m "feat(ik): accept partial keypoint frames instead of all-or-nothing

Task 19 measured that bout 28 fly0's 505-frame NaN block is a coverage-gate
cascade, not an identity failure: raw DLT recovers a partial signal in 18.2%
of those frames, but finite_frame_mask requires all 50 keypoints and
NAN_SOLVE_MIN_SEG needs 30 consecutive complete frames, so the only solvable
segment in the bout is [0, 1502).

Defaults preserve the existing behaviour exactly; the relaxation is opt-in.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

## Self-Review

**Spec coverage.** §1.1 resolution → Tasks 11, 12, 16. §1.2 sigma → Task 9.
§1.3 leaky split → Tasks 5, 7. §1.4 dropped framesets → Tasks 4, 8. §1.5
calibration → Tasks 2, 14. Phase 0 → Task 1. Phase 1 → Tasks 3–7. Phase 1b →
Task 7. Phase 2 → Task 9. Phase 3 → Tasks 10–15. Phase 4 → Task 17. §4 code
placement → all `Files:` blocks. §5 testing: every listed test is present —
split leakage (Task 5), 264 recovery (Task 4), reproject parity (Task 10),
rotation equivariance (Task 14), stage-2 capture window (Task 12), calibration
grouping (Task 2). §6 risks: 8-way ramp (Task 15 Step 4), A5 memory
(`train.batch_size=4`), IK absorption (Task 17), walltime (Task 1 runs first).

**Type consistency.** `SourceRec` fields are set in Task 3 and consumed in
Tasks 4 and 7. `merge_annotations` emits `src_ann_id`, which `v5_3d._load_mask`
reads in Task 8. `make_split`/`audit_split`/`write_derived` signatures match
their Task 7 call sites. `reproject_heatmaps(..., grid_spacing: float,
rotation)` from Task 10 is called by `refine_volumes` in Task 12.
`soft_argmax_3d(..., grid_spacing: float, roi_cube: float)` from Task 11 is
called by `refine_keypoints` in Task 12. `refine_from_seed` returns a 3-tuple in
both Task 13's tests and its Task 17 use.

**Known gaps, deliberately left to execution.** Task 15 assumes
`train_3d_cached` grows `rot_augment`, `female_weight`, `exclude_second_fly`
and per-group/per-sex val reporting; those are small additions to an existing
trainer and belong with the arm that first needs them. Task 16 Step 5 assumes
each arm writes `val_pred.npz` — add that to the trainer's eval path when
implementing Task 15.
