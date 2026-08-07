# Benchmark Suite + Scale A/B (Tracks 0–1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frozen benchmark + metric suite (Track 0) that gates every tracking upgrade, then run the Track 1 scale/calibration 2×2 on it.

**Architecture:** A `scripts/benchmark/` package in the parent repo: bout selector → input freezer → per-bout metrics → cohort scorecard (with gap-to-free-running ratios and partner-proximity splits) → render grid → variant runner that re-runs only STAC-and-later stages of `scripts/run_bout.py` against frozen inputs by pre-seeding a variant run_root (stage skipping is `stage_done()` = output-file-exists). Track 1 adds a `scaling.scale_keypoints: trunk|all` mode to the pipeline and runs 4 variants.

**Tech Stack:** Python 3.12, numpy, h5py, mujoco, OmegaConf/PyYAML (all already in the `3d_tracking` env), pytest 9. `jarvis_jax` is pip-installed editable (importable from parent tests).

**Spec:** `docs/specs/2026-08-07-tracking-upgrade-program-design.md` (Tracks 0 and 1 only)

## Global Constraints

- Benchmark experiments never mutate shared session outputs; every variant writes under its own run_root.
- Scorecards are committed artifacts (JSON + render grid) so history is auditable.
- Cohorts (exact strings): `free_running`, `courtship_male`, `courtship_female`. Courtship metrics also report a `gap_ratio` = courtship_value / free_running_value, and split into `apart` / `close` by partner proximity.
- Courtship male/female cohort assignment comes from the bout's `sex.json` (`male_fly` = current dir index) — never from the fly index alone.
- Data roots (read-only sources): courtship `/gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session*/<rec>/pose/`, free-running (legacy flat layout) `/gscratch/portia/eabe/data/Johnson_lab/free_running/Session11_*_bouts/`.
- Benchmark data root (all frozen inputs + variant outputs): `/gscratch/portia/eabe/data/Johnson_lab/processed/benchmark/`.
- Per-bout file schemas (verified): `kp2d.npz` = `kp2d (T,C,K,2)` + `conf (T,C,K)`; `kp3d.npz` = `kp3d (T,K,3)` + `conf3d (T,K)`; `kp3d_filt.npz` same keys (float64 kp3d); `outputs.h5` = `qpos (T,93)`, `kp3d_mm (T,50,3)`, `kp_names (50,)`, `mesh_mm`, `root_se3`, `scale`; `stac_ik.h5` has `names_qpos (93,)`; `qc.json` = `{per_camera_reproj_px:{median,n}, loo_reproj_px:{median,n_frames}, silhouette_iou:{hard_median,soft_median,n_frames}, n_frames}`; `qc_perframe.npz` = `soft_iou, hard_iou, reproj_px, n_cams` each `(T,)`.
- Pipeline re-run mechanics (verified in `scripts/run_bout.py`): stages skip via `stage_done(path)` (file exists, size>0). Pre-seed a variant run_root with `bouts/bout_<idx:05d>/fly<f>/{kp2d.npz,kp3d.npz,kp3d_filt.npz}` to freeze stages A/B/B2; do NOT pre-seed `scale.json`/`segment_scales.json`/`offsets.h5` when the treatment is scale/calibration (they must recompute under the variant config). Invocation: `python scripts/run_bout.py paths=hyak recording=<cfg> recording.session_dir=<dir> outputs.out=<variant_run_root> +bout_ids=<idx> <overrides>`.
- Track 1's 2×2 override strings (exact): baseline `{}`; all+norm `scaling.scale_keypoints=all scaling.estimator=norm_ratio`; segcal-off `anatomy.model.segment_calibration=false`; both combined.
- GPU work (variant re-runs) goes through sbatch on ckpt-g2 — never on a login node. Metric computation is CPU-light and may run anywhere.
- Commit style: `feat(benchmark): ...` / `feat(pipeline): ...` / `test(...): ...`.
- Python: stdlib + numpy/h5py/mujoco/omegaconf/yaml only in `scripts/benchmark/`; `jarvis_jax` imported only inside functions that need it (keeps unit tests import-light).

## File Structure

- Create `scripts/benchmark/__init__.py` — empty.
- Create `scripts/benchmark/manifest.py` — manifest schema load/validate + path resolution (both layouts).
- Create `scripts/benchmark/select_bouts.py` — candidate scanner/ranker (CLI).
- Create `scripts/benchmark/freeze_inputs.py` — copy frozen inputs + checksums (CLI).
- Create `scripts/benchmark/metrics.py` — per-bout-fly metric computation (pure functions + loaders).
- Create `scripts/benchmark/scorecard.py` — cohort aggregation, gap ratios, proximity split, JSON + markdown.
- Create `scripts/benchmark/render_grid.py` — fixed-frame MuJoCo render grid (CLI).
- Create `scripts/benchmark/run_variant.py` — variant run_root builder + command emitter + post-run collection (CLI).
- Create `configs/benchmark/bouts.yaml` — the curated manifest (Task 7 fills the real bouts; Task 1 defines the format).
- Modify `scripts/run_bout.py:510-522` — scale-keypoint mode resolution (Track 1).
- Modify `configs/pipeline.yaml:48-52` — add `scaling.scale_keypoints: trunk`.
- Tests: `tests/test_benchmark_manifest.py`, `tests/test_benchmark_metrics.py`, `tests/test_benchmark_scorecard.py`, `tests/test_benchmark_freeze.py`, `tests/test_benchmark_runner.py`, `tests/test_scale_mode.py`.

---

### Task 1: Manifest schema + path resolution

**Files:**
- Create: `scripts/benchmark/__init__.py` (empty), `scripts/benchmark/manifest.py`
- Create: `configs/benchmark/bouts.yaml` (format example, placeholder bouts replaced in Task 7)
- Test: `tests/test_benchmark_manifest.py`

**Interfaces:**
- Produces (all later tasks consume):
  - `load_manifest(path: Path) -> dict` — validated manifest.
  - `bout_fly_dir(entry: dict, root: Path | None = None) -> Path` — source bout-fly dir for an entry (handles `layout: canonical|flat`); with `root` given, the same relative layout under a variant/frozen root: `<root>/<entry['run_key']>/bouts/bout_<idx:05d>/fly<f>`.
  - `entries(manifest) -> list[dict]` — flat entry list with defaults applied.
  - Manifest YAML format:

```yaml
# configs/benchmark/bouts.yaml
benchmark_root: /gscratch/portia/eabe/data/Johnson_lab/processed/benchmark
proximity_threshold_bl: 2.0        # body-lengths; apart vs close split
bouts:
  - run_key: courtship_S1_2026_04_02_16_03_48   # unique per source run_root
    layout: canonical                            # pose/bouts/... under source_root
    source_root: /gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session1/2026_04_02_16_03_48/pose
    recording_cfg: session1                      # hydra recording group
    session_dir: /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session1/2026_04_02_16_03_48
    assay: courtship
    bout: 1
    flies: [0, 1]                                # cohort resolved via sex.json
    tags: [close_interaction, wing_extension]
    render_frames: [10, 200, 400, 700]
  - run_key: freerun_S11_2026_03_03_13_37_35
    layout: flat                                 # bouts/ directly under source_root
    source_root: /gscratch/portia/eabe/data/Johnson_lab/free_running/Session11_2026_03_03_13_37_35_bouts
    recording_cfg: free_running_session11
    session_dir: /gscratch/portia/eabe/data/Johnson_lab/Video_recordings/free_running/Session11/2026_03_03_13_37_35
    assay: free_running
    bout: 1
    flies: [0]
    tags: [walking]
    render_frames: [10, 200, 400]
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_manifest.py`:

```python
"""Tests for scripts/benchmark/manifest.py (benchmark manifest + path resolution)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from scripts.benchmark.manifest import bout_fly_dir, entries, load_manifest

MANIFEST = {
    "benchmark_root": "/tmp/bench",
    "proximity_threshold_bl": 2.0,
    "bouts": [
        {"run_key": "courtship_S1_recA", "layout": "canonical",
         "source_root": "/data/processed/courtship/Session1/recA/pose",
         "recording_cfg": "session1", "session_dir": "/data/video/recA",
         "assay": "courtship", "bout": 3, "flies": [0, 1],
         "tags": ["close"], "render_frames": [5]},
        {"run_key": "freerun_S11_recB", "layout": "flat",
         "source_root": "/data/free_running/Session11_recB_bouts",
         "recording_cfg": "free_running_session11", "session_dir": "/data/video/recB",
         "assay": "free_running", "bout": 1, "flies": [0],
         "tags": [], "render_frames": [5, 9]},
    ],
}


def write_manifest(tmp_path, data=MANIFEST) -> Path:
    p = tmp_path / "bouts.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_and_entries(tmp_path):
    m = load_manifest(write_manifest(tmp_path))
    es = entries(m)
    assert len(es) == 2
    assert es[0]["run_key"] == "courtship_S1_recA"
    assert m["proximity_threshold_bl"] == 2.0


def test_bout_fly_dir_canonical_source():
    e = MANIFEST["bouts"][0]
    assert bout_fly_dir({**e, "fly": 0}) == Path(
        "/data/processed/courtship/Session1/recA/pose/bouts/bout_00003/fly0")


def test_bout_fly_dir_flat_source():
    e = MANIFEST["bouts"][1]
    assert bout_fly_dir({**e, "fly": 0}) == Path(
        "/data/free_running/Session11_recB_bouts/bouts/bout_00001/fly0")


def test_bout_fly_dir_under_root():
    e = MANIFEST["bouts"][0]
    assert bout_fly_dir({**e, "fly": 1}, root=Path("/tmp/var1")) == Path(
        "/tmp/var1/courtship_S1_recA/bouts/bout_00003/fly1")


def test_load_manifest_rejects_duplicate_run_key_bout(tmp_path):
    bad = dict(MANIFEST, bouts=[MANIFEST["bouts"][0], MANIFEST["bouts"][0]])
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(write_manifest(tmp_path, bad))


def test_load_manifest_rejects_bad_layout(tmp_path):
    bad = dict(MANIFEST, bouts=[{**MANIFEST["bouts"][0], "layout": "weird"}])
    with pytest.raises(ValueError, match="layout"):
        load_manifest(write_manifest(tmp_path, bad))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_manifest.py -v`
Expected: FAIL at import — `ModuleNotFoundError: No module named 'scripts.benchmark'`

- [ ] **Step 3: Implement**

`scripts/benchmark/__init__.py`: empty file.

Create `scripts/benchmark/manifest.py`:

```python
"""Benchmark manifest: which bouts form the frozen benchmark, and where they live.

Two source layouts exist today (verified on disk):
  canonical: <source_root>/bouts/bout_XXXXX/flyN   (courtship pose/ dirs)
  flat:      <source_root>/bouts/bout_XXXXX/flyN   (legacy free-running *_bouts dirs)
Both place bouts under <source_root>/bouts; `layout` is kept explicit in the
manifest anyway because future sources may differ and the freezer/runner treat
the two assays differently (sex.json, num_animals).
"""
from __future__ import annotations

from pathlib import Path

import yaml

VALID_LAYOUTS = ("canonical", "flat")
REQUIRED_KEYS = ("run_key", "layout", "source_root", "recording_cfg",
                 "session_dir", "assay", "bout", "flies")


def load_manifest(path: Path) -> dict:
    manifest = yaml.safe_load(Path(path).read_text())
    seen = set()
    for e in manifest["bouts"]:
        missing = [k for k in REQUIRED_KEYS if k not in e]
        if missing:
            raise ValueError(f"manifest entry missing keys {missing}: {e}")
        if e["layout"] not in VALID_LAYOUTS:
            raise ValueError(f"bad layout {e['layout']!r} (want {VALID_LAYOUTS})")
        key = (e["run_key"], e["bout"])
        if key in seen:
            raise ValueError(f"duplicate manifest entry: {key}")
        seen.add(key)
        e.setdefault("tags", [])
        e.setdefault("render_frames", [])
    return manifest


def entries(manifest: dict) -> list[dict]:
    return list(manifest["bouts"])


def bout_fly_dir(entry: dict, root: Path | None = None) -> Path:
    """Bout-fly dir for entry['fly'] — source layout, or mirrored under `root`."""
    rel = Path("bouts") / f"bout_{int(entry['bout']):05d}" / f"fly{int(entry['fly'])}"
    if root is not None:
        return Path(root) / entry["run_key"] / rel
    return Path(entry["source_root"]) / rel
```

`configs/benchmark/bouts.yaml`: the example from the Interfaces block above, verbatim, with a top comment `# Curated in Task 7 — entries below are format examples only.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_manifest.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/ configs/benchmark/ tests/test_benchmark_manifest.py
git commit -m "feat(benchmark): manifest schema + source/variant path resolution"
```

---

### Task 2: Per-bout metrics

**Files:**
- Create: `scripts/benchmark/metrics.py`
- Test: `tests/test_benchmark_metrics.py`

**Interfaces:**
- Consumes: file schemas from Global Constraints.
- Produces:
  - `LEG_QPOS_PREFIXES = ('coxa_', 'trochanter_', 'femur_', 'tibia_', 'tarsus')`
  - `kp_group(name: str) -> str` — `'leg'|'wing'|'trunk'|'other'`.
  - `jitter_series(qpos: np.ndarray, names_qpos: list[str]) -> np.ndarray` — `(T-2,)` per-frame median |second difference| over leg qpos columns (rad/frame²).
  - `joint_bounds(mj_model) -> tuple[np.ndarray, np.ndarray]` — `(nq,)` lb/ub, ±inf where unlimited/free/ball.
  - `joint_limit_violation_rate(qpos, lb, ub, tol=1e-4) -> float`.
  - `reproj_series_by_group(kp3d_mm, kp2d, conf, cam_mats, kp_names, conf_thr=0.3) -> dict[str, np.ndarray]` — per-frame median reprojection px per group; `cam_mats` is `(C,4,3)` as exposed by `jarvis_jax.geometry.reprojection_tool.ReprojectionTool.camera_matrices`.
  - `proximity_bl(kp3d_self, kp3d_partner) -> np.ndarray` — `(T,)` inter-fly centroid distance in units of the median per-fly keypoint spread ("body lengths", unit-free).
  - `compute_bout_metrics(bout_dir: Path, partner_dir: Path | None, calib_dir: Path | None) -> dict` — loads the files, returns `{"series": {...}, "scalars": {...}, "n_frames": int}`; IoU series/medians come from `qc_perframe.npz`/`qc.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_metrics.py`:

```python
"""Tests for scripts/benchmark/metrics.py (pure metric functions, synthetic data)."""
from __future__ import annotations

import numpy as np
import pytest

from scripts.benchmark.metrics import (
    jitter_series, joint_bounds, joint_limit_violation_rate, kp_group,
    proximity_bl, reproj_series_by_group,
)


def test_kp_group():
    assert kp_group("T1L_FeTi") == "leg"
    assert kp_group("T3R_TaT5") == "leg"
    assert kp_group("WingL_V12") == "wing"
    assert kp_group("WingL_base") == "trunk"
    assert kp_group("Scutellum") == "trunk"
    assert kp_group("Abd_tip") == "trunk"
    assert kp_group("Head") == "other"


def test_jitter_series_flags_noisy_legs():
    T = 100
    names = ["free"] * 7 + ["coxa_T1_left", "wing_yaw_left"]
    smooth = np.zeros((T, 9))
    smooth[:, 7] = np.linspace(0, 1, T)          # smooth leg ramp
    noisy = smooth.copy()
    rng = np.random.default_rng(0)
    noisy[:, 7] += rng.normal(0, 0.05, T)         # jittery leg
    noisy[:, 8] += rng.normal(0, 0.5, T)          # noisy WING: must not count
    js, jn = jitter_series(smooth, names), jitter_series(noisy, names)
    assert js.shape == (T - 2,)
    assert np.median(jn) > 10 * max(np.median(js), 1e-12)


def test_joint_limits_synthetic_model():
    import mujoco
    xml = """<mujoco><worldbody><body><joint name="free" type="free"/>
      <body><joint name="hinge_lim" type="hinge" range="-0.5 0.5" limited="true"/>
        <geom size="0.01"/></body><geom size="0.01"/></body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    lb, ub = joint_bounds(m)
    assert lb.shape == (m.nq,)
    assert np.isinf(lb[:7]).all()                 # free joint unbounded
    assert lb[7] == pytest.approx(-0.5) and ub[7] == pytest.approx(0.5)
    qpos = np.zeros((10, m.nq))
    qpos[5:, 7] = 0.6                             # 5 of 10 frames beyond limit
    assert joint_limit_violation_rate(qpos, lb, ub) == pytest.approx(0.5)


def test_reproj_series_by_group_zero_for_perfect_projection():
    rng = np.random.default_rng(1)
    T, C, K = 4, 3, 4
    kp_names = ["T1L_FeTi", "T2R_Tro", "WingL_V12", "Scutellum"]
    kp3d = rng.normal(0, 5, (T, K, 3))
    # camera matrices (C,4,3): uv_h = [X 1] @ P
    cams = []
    for c in range(C):
        A = np.eye(3, 4)                          # simple projective rows
        A[2, 3] = 10.0 + c                        # keep depth positive
        cams.append(A.T)                          # (4,3)
    cam_mats = np.stack(cams)
    Xh = np.concatenate([kp3d, np.ones((T, K, 1))], axis=-1)      # (T,K,4)
    uvh = np.einsum("tkf,cfe->tcke", Xh, cam_mats)                # (T,C,K,3)
    kp2d = uvh[..., :2] / uvh[..., 2:3]
    conf = np.ones((T, C, K))
    out = reproj_series_by_group(kp3d, kp2d, conf, cam_mats, kp_names)
    assert set(out) == {"leg", "wing", "trunk"}
    for series in out.values():
        assert series.shape == (T,)
        assert np.nanmax(series) < 1e-6


def test_reproj_low_conf_ignored():
    T, C, K = 2, 2, 2
    kp_names = ["T1L_FeTi", "T1R_FeTi"]
    kp3d = np.zeros((T, K, 3))
    cam_mats = np.stack([np.eye(3, 4).T + [[0], [0], [0], [1e-9]] for _ in range(C)])
    cam_mats[..., 2] = np.array([0, 0, 0, 1.0])   # depth row -> w=1
    kp2d = np.full((T, C, K, 2), 100.0)           # wildly wrong 2D
    conf = np.zeros((T, C, K))                    # ...but zero confidence
    out = reproj_series_by_group(kp3d, kp2d, conf, cam_mats, kp_names)
    assert np.isnan(out["leg"]).all()             # nothing confident -> NaN


def test_proximity_bl_scales_with_distance():
    T, K = 5, 10
    rng = np.random.default_rng(2)
    body = rng.normal(0, 1.0, (T, K, 3))          # spread ~ its own size
    near = body + np.array([1.0, 0, 0])
    far = body + np.array([50.0, 0, 0])
    assert np.median(proximity_bl(body, far)) > 10 * np.median(proximity_bl(body, near))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_metrics.py -v`
Expected: FAIL at import — `cannot import name` from `scripts.benchmark.metrics`.

- [ ] **Step 3: Implement**

Create `scripts/benchmark/metrics.py`:

```python
"""Per-bout tracking-quality metrics for the benchmark suite.

All heavy inputs are the pipeline's existing per-bout artifacts (see plan
Global Constraints for schemas). Functions here are pure numpy where possible;
mujoco/jarvis_jax are imported lazily inside the functions that need them.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LEG_QPOS_PREFIXES = ('coxa_', 'trochanter_', 'femur_', 'tibia_', 'tarsus')
TRUNK_KEYPOINTS = {'Scutellum', 'WingL_base', 'WingR_base', 'Abd_A4', 'Abd_tip',
                   'Abd_A1', 'Abd_A2', 'Abd_A3'}


def kp_group(name: str) -> str:
    if name in TRUNK_KEYPOINTS:
        return 'trunk'
    if name[:2] in ('T1', 'T2', 'T3'):
        return 'leg'
    if name.startswith('Wing'):
        return 'wing'
    return 'other'


# --- temporal stability -----------------------------------------------------

def jitter_series(qpos: np.ndarray, names_qpos: list[str]) -> np.ndarray:
    """(T-2,) per-frame median |second difference| over LEG qpos columns.

    Rad/frame^2 acceleration proxy: high-frequency wobble scores high, smooth
    fast motion scores low. Sampling-rate free (both assays share the rig fps).
    """
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names_qpos]
    cols = [i for i, n in enumerate(names) if n.startswith(LEG_QPOS_PREFIXES)]
    if not cols:
        raise ValueError("no leg qpos columns found")
    d2 = np.diff(np.asarray(qpos, dtype=np.float64)[:, cols], n=2, axis=0)
    return np.nanmedian(np.abs(d2), axis=1)


# --- joint limits -----------------------------------------------------------

def joint_bounds(mj_model) -> tuple[np.ndarray, np.ndarray]:
    """(nq,) lower/upper qpos bounds; +/-inf for free/ball/unlimited joints."""
    import mujoco
    lb = np.full(mj_model.nq, -np.inf)
    ub = np.full(mj_model.nq, np.inf)
    for j in range(mj_model.njnt):
        jtype = mj_model.jnt_type[j]
        adr = int(mj_model.jnt_qposadr[j])
        if jtype in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            continue
        lo, hi = mj_model.jnt_range[j]
        if lo == 0.0 and hi == 0.0:          # MuJoCo "unlimited" sentinel
            continue
        lb[adr], ub[adr] = float(lo), float(hi)
    return lb, ub


def joint_limit_violation_rate(qpos: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                               tol: float = 1e-4) -> float:
    """Fraction of (frame, bounded-dof) samples outside [lb-tol, ub+tol]."""
    qpos = np.asarray(qpos, dtype=np.float64)
    bounded = np.isfinite(lb) | np.isfinite(ub)
    if not bounded.any():
        return 0.0
    q = qpos[:, bounded]
    viol = (q < (lb[bounded] - tol)) | (q > (ub[bounded] + tol))
    return float(viol.mean())


# --- reprojection -----------------------------------------------------------

def _project(cam_mats: np.ndarray, X: np.ndarray) -> np.ndarray:
    """cam_mats (C,4,3) [uv_h = [X 1] @ P], X (T,K,3) -> (T,C,K,2) pixels."""
    Xh = np.concatenate([X, np.ones((*X.shape[:-1], 1))], axis=-1)   # (T,K,4)
    uvh = np.einsum('tkf,cfe->tcke', Xh, cam_mats)                    # (T,C,K,3)
    return uvh[..., :2] / np.where(np.abs(uvh[..., 2:3]) < 1e-12, np.nan,
                                   uvh[..., 2:3])


def reproj_series_by_group(kp3d_mm: np.ndarray, kp2d: np.ndarray,
                           conf: np.ndarray, cam_mats: np.ndarray,
                           kp_names: list[str], conf_thr: float = 0.3
                           ) -> dict[str, np.ndarray]:
    """Per-frame median reprojection error (px) per keypoint group.

    Projects the FITTED 3D sites into every camera and compares against the
    observed 2D detections, masking low-confidence detections. Frames where a
    group has no confident observation are NaN.
    """
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in kp_names]
    err = np.linalg.norm(_project(cam_mats, kp3d_mm) - kp2d, axis=-1)  # (T,C,K)
    err = np.where(conf >= conf_thr, err, np.nan)
    out: dict[str, np.ndarray] = {}
    for grp in ('leg', 'wing', 'trunk'):
        cols = [i for i, n in enumerate(names) if kp_group(n) == grp]
        if not cols:
            continue
        with np.errstate(all='ignore'):
            out[grp] = np.nanmedian(err[:, :, cols].reshape(err.shape[0], -1),
                                    axis=1)
    return out


# --- multi-animal proximity ---------------------------------------------------

def proximity_bl(kp3d_self: np.ndarray, kp3d_partner: np.ndarray) -> np.ndarray:
    """(T,) inter-fly centroid distance in unit-free "body lengths".

    Body length proxy = median per-frame RMS keypoint spread of the self fly,
    so the number is invariant to the raw triangulation units.
    """
    a = np.asarray(kp3d_self, dtype=np.float64)
    b = np.asarray(kp3d_partner, dtype=np.float64)
    with np.errstate(all='ignore'):
        ca = np.nanmedian(a, axis=1)
        cb = np.nanmedian(b, axis=1)
        spread = np.sqrt(np.nansum((a - ca[:, None, :]) ** 2, axis=(1, 2))
                         / max(a.shape[1], 1))
        body = float(np.nanmedian(spread))
    return np.linalg.norm(ca - cb, axis=-1) / max(body, 1e-12)


# --- bout-level driver --------------------------------------------------------

def compute_bout_metrics(bout_dir: Path, partner_dir: Path | None = None,
                         calib_dir: Path | None = None) -> dict:
    """Compute all metrics for one bout-fly dir from its on-disk artifacts."""
    import h5py
    bout_dir = Path(bout_dir)
    with h5py.File(bout_dir / 'outputs.h5', 'r') as f:
        qpos = f['qpos'][:]
        kp3d_mm = f['kp3d_mm'][:]
        kp_names = [n.decode() for n in f['kp_names'][:]]
    with h5py.File(bout_dir / 'stac_ik.h5', 'r') as f:
        names_qpos = [n.decode() for n in f['names_qpos'][:]]
    qc = json.loads((bout_dir / 'qc.json').read_text())
    pf = np.load(bout_dir / 'qc_perframe.npz')

    series = {
        'jitter': jitter_series(qpos, names_qpos),
        'soft_iou': pf['soft_iou'],
        'hard_iou': pf['hard_iou'],
        'reproj_px': pf['reproj_px'],
    }
    scalars = {
        'reproj_px_median': qc['per_camera_reproj_px']['median'],
        'loo_px_median': qc['loo_reproj_px']['median'],
        'soft_iou_median': qc['silhouette_iou']['soft_median'],
        'hard_iou_median': qc['silhouette_iou']['hard_median'],
    }

    if calib_dir is not None:
        from jarvis_jax.geometry.reprojection_tool import ReprojectionTool
        rt = ReprojectionTool(str(calib_dir))
        kp2d_f = np.load(bout_dir / 'kp2d.npz')
        by_group = reproj_series_by_group(
            kp3d_mm, kp2d_f['kp2d'], kp2d_f['conf'],
            np.asarray(rt.camera_matrices), kp_names)
        for grp, s in by_group.items():
            series[f'reproj_px_{grp}'] = s
            scalars[f'reproj_px_{grp}_median'] = float(np.nanmedian(s))

    kp3d_src = bout_dir / ('kp3d_filt.npz'
                           if (bout_dir / 'kp3d_filt.npz').exists() else 'kp3d.npz')
    if partner_dir is not None:
        p_src = Path(partner_dir) / ('kp3d_filt.npz'
                                     if (Path(partner_dir) / 'kp3d_filt.npz').exists()
                                     else 'kp3d.npz')
        series['proximity_bl'] = proximity_bl(
            np.load(kp3d_src)['kp3d'], np.load(p_src)['kp3d'])

    scalars['jitter_median'] = float(np.nanmedian(series['jitter']))
    return {'series': {k: np.asarray(v) for k, v in series.items()},
            'scalars': scalars, 'n_frames': int(qpos.shape[0])}
```

Note on `joint_limit_violation_rate` in `compute_bout_metrics`: it needs the MuJoCo model, which is config-dependent — the scorecard runner (Task 4/6) passes it in by loading the model once (`mujoco.MjModel.from_xml_path`) and calling `joint_bounds` + `joint_limit_violation_rate(qpos, lb, ub)` per bout; add these two lines there rather than loading the model per bout here.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_metrics.py -v` — all PASS. (If the perfect-projection test fails on the einsum convention, verify against `ReprojectionTool.reproject_point` on one real bout and fix `_project`, not the test's math.)

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/metrics.py tests/test_benchmark_metrics.py
git commit -m "feat(benchmark): per-bout metrics (jitter, joint limits, grouped reproj, proximity)"
```

---

### Task 3: Scorecard (cohorts, gap ratios, proximity split)

**Files:**
- Create: `scripts/benchmark/scorecard.py`
- Test: `tests/test_benchmark_scorecard.py`

**Interfaces:**
- Consumes: Task 2's `compute_bout_metrics` result shape; Task 1's manifest.
- Produces:
  - `cohort_of(entry: dict, fly: int, male_fly: int | None) -> str` — `free_running` for that assay; else `courtship_male` if `fly == male_fly` else `courtship_female`.
  - `split_series(series: np.ndarray, proximity: np.ndarray | None, threshold: float) -> dict` — `{"all": median, "apart": median|None, "close": median|None}` (proximity series is `(T,)`, metric series may be `(T,)` or `(T-2,)` — align by trimming proximity to the series length from the front).
  - `build_scorecard(per_bout: list[dict]) -> dict` — input items: `{"cohort", "run_key", "bout", "fly", "tags", "scalars", "splits": {metric: {all/apart/close}}}`; output: `{"cohorts": {...}, "gap_ratios": {...}, "n_bouts": {...}}` where cohort aggregates are medians over bouts and `gap_ratios[courtship_x][metric] = cohort_median / free_running_median`.
  - `scorecard_markdown(scorecard: dict) -> str` — one table per cohort + a gap-ratio table.
  - `write_scorecard(path: Path, scorecard: dict)` — JSON (+ `.md` sibling).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_scorecard.py`:

```python
"""Tests for scripts/benchmark/scorecard.py (aggregation, gaps, splits)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts.benchmark.scorecard import (
    build_scorecard, cohort_of, scorecard_markdown, split_series, write_scorecard,
)


def test_cohort_of():
    fr = {"assay": "free_running"}
    ct = {"assay": "courtship"}
    assert cohort_of(fr, 0, None) == "free_running"
    assert cohort_of(ct, 1, 1) == "courtship_male"
    assert cohort_of(ct, 0, 1) == "courtship_female"


def test_cohort_of_courtship_requires_sex():
    with pytest.raises(ValueError, match="sex"):
        cohort_of({"assay": "courtship"}, 0, None)


def test_split_series_thresholds():
    s = np.arange(10, dtype=float)          # metric series
    prox = np.array([0.5] * 5 + [5.0] * 5)  # first half close, second apart
    out = split_series(s, prox, threshold=2.0)
    assert out["close"] == pytest.approx(np.median(s[:5]))
    assert out["apart"] == pytest.approx(np.median(s[5:]))
    assert out["all"] == pytest.approx(np.median(s))


def test_split_series_alignment_and_no_proximity():
    s = np.arange(8, dtype=float)           # e.g. jitter is (T-2,)
    prox = np.ones(10) * 5.0
    out = split_series(s, prox, threshold=2.0)
    assert out["apart"] == pytest.approx(np.median(s))
    out2 = split_series(s, None, threshold=2.0)
    assert out2["apart"] is None and out2["close"] is None


def _bout(cohort, reproj, jitter):
    return {"cohort": cohort, "run_key": "r", "bout": 1, "fly": 0, "tags": [],
            "scalars": {"reproj_px_median": reproj, "jitter_median": jitter},
            "splits": {"reproj_px": {"all": reproj, "apart": reproj, "close": None}}}


def test_build_scorecard_gap_ratios():
    per_bout = [_bout("free_running", 4.0, 0.001),
                _bout("free_running", 6.0, 0.001),
                _bout("courtship_male", 10.0, 0.002),
                _bout("courtship_female", 20.0, 0.004)]
    sc = build_scorecard(per_bout)
    assert sc["cohorts"]["free_running"]["reproj_px_median"] == pytest.approx(5.0)
    assert sc["gap_ratios"]["courtship_male"]["reproj_px_median"] == pytest.approx(2.0)
    assert sc["gap_ratios"]["courtship_female"]["jitter_median"] == pytest.approx(4.0)
    assert sc["n_bouts"]["free_running"] == 2


def test_write_and_markdown(tmp_path):
    sc = build_scorecard([_bout("free_running", 4.0, 0.001),
                          _bout("courtship_male", 8.0, 0.002)])
    p = tmp_path / "scorecard.json"
    write_scorecard(p, sc)
    assert json.loads(p.read_text())["cohorts"]
    md = scorecard_markdown(sc)
    assert "free_running" in md and "gap" in md.lower()
    assert (tmp_path / "scorecard.md").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_scorecard.py -v` — FAIL at import.

- [ ] **Step 3: Implement**

Create `scripts/benchmark/scorecard.py`:

```python
"""Cohort scorecard: aggregate per-bout metrics, gap-to-free-running ratios,
and the apart/close partner-proximity split (spec: Track 0)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

COHORTS = ("free_running", "courtship_male", "courtship_female")


def cohort_of(entry: dict, fly: int, male_fly: int | None) -> str:
    if entry["assay"] == "free_running":
        return "free_running"
    if male_fly is None:
        raise ValueError(f"courtship entry {entry.get('run_key')} needs sex.json "
                         f"male_fly to resolve the cohort")
    return "courtship_male" if fly == male_fly else "courtship_female"


def split_series(series: np.ndarray, proximity: np.ndarray | None,
                 threshold: float) -> dict:
    series = np.asarray(series, dtype=np.float64)
    out = {"all": float(np.nanmedian(series)) if series.size else None,
           "apart": None, "close": None}
    if proximity is None or not series.size:
        return out
    prox = np.asarray(proximity, dtype=np.float64)
    prox = prox[prox.shape[0] - series.shape[0]:]      # align (T-2,) vs (T,)
    for name, mask in (("apart", prox >= threshold), ("close", prox < threshold)):
        if mask.any():
            with np.errstate(all="ignore"):
                v = np.nanmedian(series[mask])
            out[name] = None if np.isnan(v) else float(v)
    return out


def build_scorecard(per_bout: list[dict]) -> dict:
    cohorts: dict[str, dict] = {}
    n_bouts: dict[str, int] = {}
    for c in COHORTS:
        items = [b for b in per_bout if b["cohort"] == c]
        if not items:
            continue
        n_bouts[c] = len(items)
        keys = sorted({k for b in items for k in b["scalars"]})
        agg = {k: float(np.nanmedian([b["scalars"][k] for b in items
                                      if k in b["scalars"]])) for k in keys}
        split_keys = sorted({k for b in items for k in b.get("splits", {})})
        for sk in split_keys:
            for part in ("apart", "close"):
                vals = [b["splits"][sk][part] for b in items
                        if b.get("splits", {}).get(sk, {}).get(part) is not None]
                if vals:
                    agg[f"{sk}_{part}"] = float(np.nanmedian(vals))
        cohorts[c] = agg
    gap_ratios: dict[str, dict] = {}
    base = cohorts.get("free_running", {})
    for c in ("courtship_male", "courtship_female"):
        if c not in cohorts:
            continue
        gap_ratios[c] = {k: cohorts[c][k] / base[k]
                         for k in cohorts[c] if base.get(k) not in (None, 0)}
    return {"cohorts": cohorts, "gap_ratios": gap_ratios, "n_bouts": n_bouts,
            "per_bout": per_bout}


def scorecard_markdown(scorecard: dict) -> str:
    lines = ["# Benchmark scorecard", ""]
    for c, agg in scorecard["cohorts"].items():
        lines += [f"## {c} (n={scorecard['n_bouts'][c]})", "",
                  "| metric | median |", "|---|---|"]
        lines += [f"| {k} | {v:.4g} |" for k, v in sorted(agg.items())]
        lines.append("")
    if scorecard["gap_ratios"]:
        lines += ["## Gap ratios (courtship / free_running, 1.0 = parity)", "",
                  "| metric | " + " | ".join(scorecard["gap_ratios"]) + " |",
                  "|---|" + "---|" * len(scorecard["gap_ratios"])]
        keys = sorted({k for g in scorecard["gap_ratios"].values() for k in g})
        for k in keys:
            row = [f"{scorecard['gap_ratios'][c].get(k, float('nan')):.3g}"
                   for c in scorecard["gap_ratios"]]
            lines.append(f"| {k} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_scorecard(path: Path, scorecard: dict) -> None:
    path = Path(path)
    slim = dict(scorecard)
    slim["per_bout"] = [{k: v for k, v in b.items() if k != "series"}
                        for b in scorecard.get("per_bout", [])]
    path.write_text(json.dumps(_jsonable(slim), indent=2, sort_keys=True))
    path.with_suffix(".md").write_text(scorecard_markdown(scorecard))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_scorecard.py tests/test_benchmark_metrics.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/scorecard.py tests/test_benchmark_scorecard.py
git commit -m "feat(benchmark): cohort scorecard with gap ratios + proximity split"
```

---

### Task 4: Input freezer + candidate selector

**Files:**
- Create: `scripts/benchmark/freeze_inputs.py`, `scripts/benchmark/select_bouts.py`
- Test: `tests/test_benchmark_freeze.py`

**Interfaces:**
- Consumes: Task 1 `load_manifest/entries/bout_fly_dir`.
- Produces:
  - `freeze(manifest: dict, dest_root: Path) -> dict` — copies, per entry×fly, `kp2d.npz`, `kp3d.npz`, `kp3d_filt.npz` (if present) and the bout-level `sex.json` (courtship) from source into `<dest_root>/frozen/<run_key>/bouts/bout_XXXXX/flyN/`; returns `{relpath: sha256}` and writes it to `<dest_root>/frozen/checksums.json`. Raises `FileNotFoundError` naming the missing artifact.
  - `verify(dest_root: Path) -> list[str]` — re-hash, return mismatched relpaths (empty = intact).
  - `select_bouts.main(argv)` — CLI: `--courtship-root`, `--freerun-root`, `--top N`; scans sources, reads each bout's `qc.json` + (courtship) computes median partner proximity from kp3d, prints a ranked candidate table (CSV to stdout: run_key, bout, assay, n_frames, reproj_px, soft_iou, proximity_median, suggestion). Selection itself stays human (Task 7).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_freeze.py`:

```python
"""Tests for scripts/benchmark/freeze_inputs.py (synthetic source trees)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.benchmark.freeze_inputs import freeze, verify


def make_source_bout(root: Path, bout: int, flies=(0,), with_filt=True,
                     sex_json=None) -> Path:
    bdir = root / "bouts" / f"bout_{bout:05d}"
    for f in flies:
        d = bdir / f"fly{f}"
        d.mkdir(parents=True)
        np.savez(d / "kp2d.npz", kp2d=np.zeros((4, 2, 3, 2)), conf=np.ones((4, 2, 3)))
        np.savez(d / "kp3d.npz", kp3d=np.zeros((4, 3, 3)), conf3d=np.ones((4, 3)))
        if with_filt:
            np.savez(d / "kp3d_filt.npz", kp3d=np.zeros((4, 3, 3)),
                     conf3d=np.ones((4, 3)))
    if sex_json is not None:
        (bdir / "sex.json").write_text(json.dumps(sex_json))
    return bdir


def manifest_for(src: Path, run_key="courtship_r", assay="courtship",
                 flies=(0, 1), bout=1):
    return {"benchmark_root": "", "proximity_threshold_bl": 2.0, "bouts": [{
        "run_key": run_key, "layout": "canonical", "source_root": str(src),
        "recording_cfg": "session1", "session_dir": "/nope", "assay": assay,
        "bout": bout, "flies": list(flies), "tags": [], "render_frames": []}]}


def test_freeze_copies_inputs_and_checksums(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0, 1), sex_json={"male_fly": 1})
    sums = freeze(manifest_for(src), dest)
    base = dest / "frozen" / "courtship_r" / "bouts" / "bout_00001"
    for f in (0, 1):
        for name in ("kp2d.npz", "kp3d.npz", "kp3d_filt.npz"):
            assert (base / f"fly{f}" / name).exists()
    assert (base / "sex.json").exists()
    assert (dest / "frozen" / "checksums.json").exists()
    assert any(k.endswith("kp2d.npz") for k in sums)
    assert verify(dest) == []


def test_freeze_missing_artifact_raises(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    bdir = make_source_bout(src, 1, flies=(0,))
    (bdir / "fly0" / "kp3d.npz").unlink()
    with pytest.raises(FileNotFoundError, match="kp3d.npz"):
        freeze(manifest_for(src, assay="free_running", flies=(0,)), dest)


def test_freeze_optional_filt_and_sex(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 2, flies=(0,), with_filt=False)   # no filt, no sex.json
    freeze(manifest_for(src, run_key="fr", assay="free_running",
                        flies=(0,), bout=2), dest)
    base = dest / "frozen" / "fr" / "bouts" / "bout_00002"
    assert not (base / "fly0" / "kp3d_filt.npz").exists()


def test_verify_detects_tamper(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0,))
    freeze(manifest_for(src, run_key="fr", assay="free_running", flies=(0,)), dest)
    victim = dest / "frozen" / "fr" / "bouts" / "bout_00001" / "fly0" / "kp3d.npz"
    victim.write_bytes(b"corrupt")
    bad = verify(dest)
    assert len(bad) == 1 and bad[0].endswith("kp3d.npz")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_freeze.py -v` — FAIL at import.

- [ ] **Step 3: Implement**

Create `scripts/benchmark/freeze_inputs.py`:

```python
"""Freeze benchmark 2D/3D inputs so every variant re-runs from identical data."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from scripts.benchmark.manifest import bout_fly_dir, entries

REQUIRED = ("kp2d.npz", "kp3d.npz")
OPTIONAL = ("kp3d_filt.npz",)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def freeze(manifest: dict, dest_root: Path) -> dict:
    dest_root = Path(dest_root)
    frozen = dest_root / "frozen"
    sums: dict[str, str] = {}

    def copy(src: Path, rel: Path):
        dst = frozen / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        sums[str(rel)] = _sha256(dst)

    for e in entries(manifest):
        for fly in e["flies"]:
            src_dir = bout_fly_dir({**e, "fly": fly})
            rel_dir = (Path(e["run_key"]) / "bouts"
                       / f"bout_{int(e['bout']):05d}" / f"fly{fly}")
            for name in REQUIRED:
                if not (src_dir / name).is_file():
                    raise FileNotFoundError(f"{e['run_key']} bout {e['bout']} "
                                            f"fly{fly}: missing {name}")
                copy(src_dir / name, rel_dir / name)
            for name in OPTIONAL:
                if (src_dir / name).is_file():
                    copy(src_dir / name, rel_dir / name)
        if e["assay"] == "courtship":
            sex = src_dir.parent / "sex.json"          # bout-level
            if sex.is_file():
                copy(sex, rel_dir.parent / "sex.json")
    frozen.mkdir(parents=True, exist_ok=True)
    (frozen / "checksums.json").write_text(json.dumps(sums, indent=2,
                                                      sort_keys=True))
    return sums


def verify(dest_root: Path) -> list[str]:
    frozen = Path(dest_root) / "frozen"
    sums = json.loads((frozen / "checksums.json").read_text())
    return [rel for rel, digest in sums.items()
            if not (frozen / rel).is_file() or _sha256(frozen / rel) != digest]
```

Create `scripts/benchmark/select_bouts.py`:

```python
"""Rank candidate benchmark bouts from existing pipeline outputs (human picks).

Usage:
  python -m scripts.benchmark.select_bouts \
      --courtship-root /gscratch/portia/eabe/data/Johnson_lab/processed/courtship \
      --freerun-root   /gscratch/portia/eabe/data/Johnson_lab/free_running \
      > candidates.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def scan_bout_fly(fly_dir: Path) -> dict | None:
    qc_path = fly_dir / "qc.json"
    if not qc_path.is_file():
        return None
    qc = json.loads(qc_path.read_text())
    return {"n_frames": qc.get("n_frames"),
            "reproj_px": qc.get("per_camera_reproj_px", {}).get("median"),
            "soft_iou": qc.get("silhouette_iou", {}).get("soft_median")}


def proximity_median(bout_dir: Path) -> float | None:
    from scripts.benchmark.metrics import proximity_bl
    srcs = []
    for f in (0, 1):
        d = bout_dir / f"fly{f}"
        p = d / ("kp3d_filt.npz" if (d / "kp3d_filt.npz").exists() else "kp3d.npz")
        if not p.exists():
            return None
        srcs.append(np.load(p)["kp3d"])
    return float(np.nanmedian(proximity_bl(srcs[0], srcs[1])))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--courtship-root", type=Path, default=None)
    ap.add_argument("--freerun-root", type=Path, default=None)
    args = ap.parse_args(argv)
    w = csv.writer(sys.stdout)
    w.writerow(["assay", "run", "bout", "fly", "n_frames", "reproj_px",
                "soft_iou", "proximity_median"])
    if args.courtship_root:
        for bout_dir in sorted(args.courtship_root.glob(
                "Session*/*/pose/bouts/bout_*")):
            prox = proximity_median(bout_dir)
            for f in (0, 1):
                row = scan_bout_fly(bout_dir / f"fly{f}")
                if row:
                    w.writerow(["courtship",
                                f"{bout_dir.parents[3].name}/{bout_dir.parents[2].name}",
                                bout_dir.name, f, row["n_frames"],
                                row["reproj_px"], row["soft_iou"], prox])
    if args.freerun_root:
        for bout_dir in sorted(args.freerun_root.glob("*_bouts/bouts/bout_*")):
            row = scan_bout_fly(bout_dir / "fly0")
            if row:
                w.writerow(["free_running", bout_dir.parents[1].name,
                            bout_dir.name, 0, row["n_frames"],
                            row["reproj_px"], row["soft_iou"], ""])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_freeze.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/freeze_inputs.py scripts/benchmark/select_bouts.py tests/test_benchmark_freeze.py
git commit -m "feat(benchmark): input freezer with checksums + candidate selector"
```

---

### Task 5: Variant runner

**Files:**
- Create: `scripts/benchmark/run_variant.py`
- Test: `tests/test_benchmark_runner.py`

**Interfaces:**
- Consumes: Tasks 1–4.
- Produces:
  - `build_variant_root(manifest, dest_root: Path, variant: str) -> Path` — creates `<dest_root>/variants/<variant>/<run_key>/bouts/...` populated with hardlinks (fallback: copies) of the frozen inputs. Never links `scale.json`/`segment_scales.json`/`offsets.h5` (recomputed per variant by design).
  - `variant_commands(manifest, variant_root: Path, overrides: list[str]) -> list[str]` — one `python scripts/run_bout.py ...` command per (run_key, bout): `python scripts/run_bout.py paths=hyak recording=<recording_cfg> recording.session_dir=<session_dir> outputs.out=<variant_root>/<run_key> +bout_ids=<bout> <overrides...>`.
  - `collect(manifest, root_for_outputs: Path | None, mjcf_path: str | None, out_json: Path) -> dict` — per entry×fly: resolve bout dir (source dirs when `root_for_outputs is None` — the baseline; else under the variant root), read `sex.json` for cohort, run `compute_bout_metrics` (+ joint-limit rate when `mjcf_path` given, loading the model once), build splits via `split_series`, then `build_scorecard` + `write_scorecard(out_json, ...)`.
  - CLI: `python -m scripts.benchmark.run_variant --manifest configs/benchmark/bouts.yaml {build|commands|collect|baseline} ...` with `--variant`, `--override` (repeatable), `--mjcf`, `--calib` optional per-run_key from manifest `session_dir` + `/calibration`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_runner.py`:

```python
"""Tests for scripts/benchmark/run_variant.py (no GPU: build + command emission)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.benchmark.freeze_inputs import freeze
from scripts.benchmark.run_variant import build_variant_root, variant_commands
from tests.test_benchmark_freeze import make_source_bout, manifest_for


def _frozen(tmp_path):
    src, dest = tmp_path / "src", tmp_path / "dest"
    make_source_bout(src, 1, flies=(0, 1), sex_json={"male_fly": 1})
    m = manifest_for(src)
    freeze(m, dest)
    return m, dest


def test_build_variant_root_links_inputs_not_shared_artifacts(tmp_path):
    m, dest = _frozen(tmp_path)
    # simulate stale shared artifacts in frozen tree root: must NOT propagate
    (dest / "frozen" / "courtship_r" / "scale.json").write_text("{}")
    vroot = build_variant_root(m, dest, "all_norm")
    assert vroot == dest / "variants" / "all_norm"
    fly0 = vroot / "courtship_r" / "bouts" / "bout_00001" / "fly0"
    assert (fly0 / "kp2d.npz").exists()
    assert (fly0 / "kp3d.npz").exists()
    assert not (vroot / "courtship_r" / "scale.json").exists()


def test_variant_commands_shape(tmp_path):
    m, dest = _frozen(tmp_path)
    vroot = build_variant_root(m, dest, "v")
    cmds = variant_commands(m, vroot,
                            ["scaling.scale_keypoints=all",
                             "scaling.estimator=norm_ratio"])
    assert len(cmds) == 1
    c = cmds[0]
    assert c.startswith("python scripts/run_bout.py ")
    assert "recording=session1" in c
    assert "recording.session_dir=/nope" in c
    assert f"outputs.out={vroot}/courtship_r" in c
    assert "+bout_ids=1" in c
    assert "scaling.scale_keypoints=all" in c and "scaling.estimator=norm_ratio" in c
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark_runner.py -v` — FAIL at import.

- [ ] **Step 3: Implement**

Create `scripts/benchmark/run_variant.py`:

```python
"""Build variant run_roots from frozen inputs, emit run commands, collect scorecards.

A variant re-runs scripts/run_bout.py per bout with stage-skipping doing the
freezing: kp2d/kp3d/kp3d_filt pre-exist in the variant root (hardlinked from
frozen/), so stages A/B/B2 skip; scale.json / segment_scales.json / offsets.h5
are deliberately ABSENT so they recompute under the variant's config (that is
the Track 1 treatment). Stage C (STAC) onward runs fresh.

GPU note: emitted commands must run on a compute node (sbatch ckpt-g2 or an
interactive GPU shell) -- never a login node.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
from pathlib import Path

import numpy as np

from scripts.benchmark.manifest import bout_fly_dir, entries, load_manifest
from scripts.benchmark.metrics import (
    compute_bout_metrics, joint_bounds, joint_limit_violation_rate)
from scripts.benchmark.scorecard import (
    build_scorecard, cohort_of, split_series, write_scorecard)

FROZEN_INPUTS = ("kp2d.npz", "kp3d.npz", "kp3d_filt.npz")


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        import shutil
        shutil.copy2(src, dst)


def build_variant_root(manifest: dict, dest_root: Path, variant: str) -> Path:
    dest_root = Path(dest_root)
    frozen = dest_root / "frozen"
    vroot = dest_root / "variants" / variant
    for e in entries(manifest):
        for fly in e["flies"]:
            rel = (Path(e["run_key"]) / "bouts"
                   / f"bout_{int(e['bout']):05d}" / f"fly{fly}")
            for name in FROZEN_INPUTS:
                src = frozen / rel / name
                if src.exists():
                    _link_or_copy(src, vroot / rel / name)
        sex = frozen / e["run_key"] / "bouts" / f"bout_{int(e['bout']):05d}" / "sex.json"
        if sex.exists():
            _link_or_copy(sex, vroot / e["run_key"] / "bouts"
                          / f"bout_{int(e['bout']):05d}" / "sex.json")
    return vroot


def variant_commands(manifest: dict, variant_root: Path,
                     overrides: list[str]) -> list[str]:
    cmds = []
    for e in entries(manifest):
        parts = ["python", "scripts/run_bout.py", "paths=hyak",
                 f"recording={e['recording_cfg']}",
                 f"recording.session_dir={e['session_dir']}",
                 f"outputs.out={Path(variant_root) / e['run_key']}",
                 f"+bout_ids={int(e['bout'])}", *overrides]
        cmds.append(" ".join(shlex.quote(p) if " " in p else p for p in parts))
    return cmds


def _male_fly(bout_dir_parent: Path) -> int | None:
    sex = bout_dir_parent / "sex.json"
    if sex.is_file():
        return int(json.loads(sex.read_text()).get("male_fly"))
    return None


def collect(manifest: dict, root_for_outputs: Path | None,
            mjcf_path: str | None, out_json: Path) -> dict:
    lb = ub = None
    if mjcf_path:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        lb, ub = joint_bounds(model)
    thr = float(manifest.get("proximity_threshold_bl", 2.0))
    per_bout = []
    for e in entries(manifest):
        for fly in e["flies"]:
            bd = bout_fly_dir({**e, "fly": fly}, root=root_for_outputs)
            partner = None
            if e["assay"] == "courtship" and len(e["flies"]) > 1:
                partner = bout_fly_dir({**e, "fly": 1 - fly},
                                       root=root_for_outputs)
            calib = Path(e["session_dir"]) / "calibration"
            res = compute_bout_metrics(bd, partner_dir=partner,
                                       calib_dir=calib if calib.is_dir() else None)
            if lb is not None:
                import h5py
                with h5py.File(bd / "outputs.h5", "r") as f:
                    res["scalars"]["jl_violation_rate"] = \
                        joint_limit_violation_rate(f["qpos"][:], lb, ub)
            prox = res["series"].get("proximity_bl")
            splits = {k: split_series(v, prox, thr)
                      for k, v in res["series"].items() if k != "proximity_bl"}
            per_bout.append({
                "cohort": cohort_of(e, fly, _male_fly(bd.parent)),
                "run_key": e["run_key"], "bout": e["bout"], "fly": fly,
                "tags": e.get("tags", []), "scalars": res["scalars"],
                "splits": splits})
    sc = build_scorecard(per_bout)
    write_scorecard(Path(out_json), sc)
    return sc


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["build", "commands", "collect", "baseline"])
    ap.add_argument("--manifest", type=Path,
                    default=Path("configs/benchmark/bouts.yaml"))
    ap.add_argument("--variant", type=str, default=None)
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--mjcf", type=str, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    manifest = load_manifest(args.manifest)
    dest_root = Path(manifest["benchmark_root"])
    if args.mode in ("build", "commands"):
        vroot = build_variant_root(manifest, dest_root, args.variant)
        if args.mode == "commands":
            for c in variant_commands(manifest, vroot, args.override):
                print(c)
    elif args.mode == "collect":
        vroot = dest_root / "variants" / args.variant
        collect(manifest, vroot, args.mjcf,
                args.out or vroot / "scorecard.json")
    else:  # baseline: metrics straight off the SOURCE dirs, no re-run
        collect(manifest, None, args.mjcf,
                args.out or dest_root / "baseline_scorecard.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark_runner.py tests/test_benchmark_freeze.py -v` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/run_variant.py tests/test_benchmark_runner.py
git commit -m "feat(benchmark): variant runner (build/commands/collect/baseline)"
```

---

### Task 6: Render grid

**Files:**
- Create: `scripts/benchmark/render_grid.py`
- Test: smoke assertions inside `tests/test_benchmark_runner.py` (append)

**Interfaces:**
- Consumes: manifest `render_frames`; `outputs.h5` qpos; the anatomy MJCF.
- Produces: `render_bout(qpos: np.ndarray, frames: list[int], mjcf_path: str, out_png: Path, size=(320, 320))` — offscreen MuJoCo render of each listed frame, horizontally tiled into one PNG (pattern proven in `scripts/viz/compare_stac_fits.py`: `d.qpos[:] = qpos[f]; mj_forward; Renderer.render()` with a fixed MjvCamera tracking the root). CLI `python -m scripts.benchmark.render_grid --manifest ... --root <variant_root|SOURCE> --mjcf <xml> --out-dir <dir>` writes `<run_key>_bout_<idx>_fly<f>.png` per entry.

- [ ] **Step 1: Write the failing test** (append to `tests/test_benchmark_runner.py`)

```python
def test_render_bout_smoke(tmp_path):
    import mujoco  # noqa: F401  (skip if unavailable)
    from scripts.benchmark.render_grid import render_bout
    xml = tmp_path / "m.xml"
    xml.write_text("""<mujoco><worldbody><body name="root">
      <joint type="free"/><geom size="0.02"/></body></worldbody></mujoco>""")
    qpos = np.zeros((10, 7)); qpos[:, 3] = 1.0     # identity quat
    out = tmp_path / "grid.png"
    render_bout(qpos, [0, 5, 9], str(xml), out, size=(64, 64))
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_benchmark_runner.py::test_render_bout_smoke -v` — FAIL at import.

- [ ] **Step 3: Implement**

Create `scripts/benchmark/render_grid.py`:

```python
"""Fixed-frame MuJoCo render grid for benchmark bouts (visual A/B).

Same frames every run -> renders are directly comparable across variants.
Render pattern follows scripts/viz/compare_stac_fits.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def render_bout(qpos: np.ndarray, frames: list[int], mjcf_path: str,
                out_png: Path, size: tuple[int, int] = (320, 320)) -> None:
    import imageio.v2 as imageio
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(mjcf_path))
    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.azimuth, cam.elevation, cam.distance = 90.0, -20.0, 1.0
    panels = []
    with mujoco.Renderer(m, height=size[1], width=size[0]) as r:
        for f in frames:
            f = int(np.clip(f, 0, qpos.shape[0] - 1))
            d.qpos[:] = qpos[f]
            mujoco.mj_forward(m, d)
            cam.lookat[:] = d.qpos[:3]
            r.update_scene(d, camera=cam)
            panels.append(r.render())
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out_png, np.concatenate(panels, axis=1))


def main(argv=None) -> None:
    import h5py
    from scripts.benchmark.manifest import bout_fly_dir, entries, load_manifest
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path,
                    default=Path("configs/benchmark/bouts.yaml"))
    ap.add_argument("--root", type=str, default="SOURCE",
                    help="variant root, or SOURCE for the original outputs")
    ap.add_argument("--mjcf", type=str, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    manifest = load_manifest(args.manifest)
    root = None if args.root == "SOURCE" else Path(args.root)
    for e in entries(manifest):
        for fly in e["flies"]:
            bd = bout_fly_dir({**e, "fly": fly}, root=root)
            if not (bd / "outputs.h5").exists():
                print(f"skip (no outputs.h5): {bd}")
                continue
            with h5py.File(bd / "outputs.h5", "r") as f:
                qpos = f["qpos"][:]
            frames = e.get("render_frames") or [0, qpos.shape[0] // 2,
                                                qpos.shape[0] - 1]
            render_bout(qpos, frames, args.mjcf,
                        args.out_dir / f"{e['run_key']}_bout_{e['bout']}_fly{fly}.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_benchmark_runner.py -v` — all PASS. (If `mujoco.Renderer` needs a GL backend on the node, set `MUJOCO_GL=osmesa` for the test run and note it in the report.)

- [ ] **Step 5: Commit**

```bash
git add scripts/benchmark/render_grid.py tests/test_benchmark_runner.py
git commit -m "feat(benchmark): fixed-frame render grid"
```

---

### Task 7: Curate the benchmark + baseline scorecard (execution)

**Files:**
- Modify: `configs/benchmark/bouts.yaml` (real entries)
- Create (committed artifacts): `docs/benchmark/2026-08-07-baseline-scorecard.json` + `.md`

No new code — run on real data. GPU not required (metrics only), but run from a compute node anyway if memory is heavy (courtship bouts are small; a login-node run of pure-numpy metrics on ~15 bouts is acceptable).

- [ ] **Step 1: Generate candidates**

```bash
python -m scripts.benchmark.select_bouts \
  --courtship-root /gscratch/portia/eabe/data/Johnson_lab/processed/courtship \
  --freerun-root /gscratch/portia/eabe/data/Johnson_lab/free_running > /tmp/candidates.csv
```

- [ ] **Step 2: Curate ~12–15 bouts WITH THE USER** — present the ranked table; pick per cohort: ≥4 free-running (varied locomotion), ≥4 courtship with low median proximity (apart-heavy), ≥4 with high-close fraction (interaction-heavy), including wing-extension tags (use the fly-ID GUI's videos for a quick look). Write the chosen entries into `configs/benchmark/bouts.yaml` (schema from Task 1), with `render_frames` picked as 4 evenly spaced frames plus the worst-`reproj_px` frame from `qc_perframe.npz`.

- [ ] **Step 3: Freeze + verify**

```bash
python - <<'EOF'
from pathlib import Path
from scripts.benchmark.manifest import load_manifest
from scripts.benchmark.freeze_inputs import freeze, verify
m = load_manifest(Path("configs/benchmark/bouts.yaml"))
freeze(m, Path(m["benchmark_root"]))
assert verify(Path(m["benchmark_root"])) == []
print("frozen + verified")
EOF
```

- [ ] **Step 4: Baseline scorecard + render grid off the SOURCE outputs**

```bash
python -m scripts.benchmark.run_variant baseline \
  --mjcf models/fruitfly_v1/fruitfly_v1_free.xml \
  --out docs/benchmark/2026-08-07-baseline-scorecard.json
MUJOCO_GL=osmesa python -m scripts.benchmark.render_grid --root SOURCE \
  --mjcf models/fruitfly_v1/fruitfly_v1_free.xml --out-dir docs/benchmark/renders-baseline
```

Sanity-check the scorecard against known facts: courtship gap ratios > 1, female worse than male, close worse than apart. Surprises get reported, not silently accepted.

- [ ] **Step 5: Commit manifest + baseline artifacts**

```bash
git add configs/benchmark/bouts.yaml docs/benchmark/
git commit -m "feat(benchmark): curated benchmark bouts + baseline scorecard"
```

---

### Task 8: Scale-keypoint mode (Track 1 plumbing)

**Files:**
- Modify: `scripts/run_bout.py:510-522` (the `scale.json` block)
- Modify: `configs/pipeline.yaml` `scaling:` block (~line 48)
- Test: `tests/test_scale_mode.py`

**Interfaces:**
- Consumes: existing `jarvis_jax.tracking.scale.compute_trunk_scale(kp3d, kp_names, model_xml, *, trunk_names, estimator, robust_stat, robust)` — already supports `estimator='norm_ratio'` and an arbitrary `trunk_names` list; no submodule change needed.
- Produces:
  - `resolve_scale_keypoints(cfg, kp_names: list[str]) -> list[str]` in `scripts/run_bout.py` — `'trunk'` → `list(cfg.scaling.trunk_keypoints)`, `'all'` → `list(kp_names)`, else `ValueError`.
  - `configs/pipeline.yaml` gains `scaling.scale_keypoints: trunk` (default unchanged behavior).
  - `scale.json` written by the pipeline gains `"scale_keypoints"` (mode) and keeps `"trunk_keypoints"` (now the actual names used) + `"estimator"`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scale_mode.py`:

```python
"""Tests for the scaling.scale_keypoints mode plumbing in scripts/run_bout.py."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from scripts.run_bout import resolve_scale_keypoints

KP = ["Scutellum", "WingL_base", "T1L_FeTi", "T1R_FeTi", "Abd_tip"]


def cfg_with(mode=None):
    scaling = {"trunk_keypoints": ["Scutellum", "WingL_base", "Abd_tip"],
               "estimator": "umeyama", "robust_stat": "median", "robust": "none"}
    if mode is not None:
        scaling["scale_keypoints"] = mode
    return OmegaConf.create({"scaling": scaling})


def test_default_is_trunk():
    assert resolve_scale_keypoints(cfg_with(), KP) == \
        ["Scutellum", "WingL_base", "Abd_tip"]


def test_trunk_explicit():
    assert resolve_scale_keypoints(cfg_with("trunk"), KP) == \
        ["Scutellum", "WingL_base", "Abd_tip"]


def test_all_uses_every_keypoint():
    assert resolve_scale_keypoints(cfg_with("all"), KP) == KP


def test_bad_mode_raises():
    with pytest.raises(ValueError, match="scale_keypoints"):
        resolve_scale_keypoints(cfg_with("some"), KP)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scale_mode.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_scale_keypoints'` (importing `scripts.run_bout` must not require a GPU: it only imports at module level — verify; if module import fails in the test env, report BLOCKED rather than restructuring imports).

- [ ] **Step 3: Implement**

In `scripts/run_bout.py`, add near the other helpers (above `process_bout_fly`):

```python
def resolve_scale_keypoints(cfg, kp_names):
    """scaling.scale_keypoints: 'trunk' -> configured trunk list; 'all' -> every
    keypoint (pair with estimator=norm_ratio -- all+umeyama provably collapses,
    see configs/preprocessing/v2_3.yaml)."""
    mode = str(cfg.scaling.get("scale_keypoints", "trunk"))
    if mode == "all":
        return list(kp_names)
    if mode == "trunk":
        return list(cfg.scaling.trunk_keypoints)
    raise ValueError(
        f"scaling.scale_keypoints must be 'trunk' or 'all', got {mode!r}")
```

Then modify the `scale.json` block (currently lines ~510–522) to use it:

```python
        scale_names = resolve_scale_keypoints(cfg, kp_names)
        _scale = compute_trunk_scale(
            kp3d, kp_names, cfg.silhouette.xml,
            trunk_names=scale_names,
            estimator=cfg.scaling.estimator,
            robust_stat=cfg.scaling.robust_stat,
            robust=cfg.scaling.robust)
        atomic_save_json(scale_path, {
            "scale": float(_scale),
            "scale_keypoints": str(cfg.scaling.get("scale_keypoints", "trunk")),
            "trunk_keypoints": list(scale_names),
            "estimator": str(cfg.scaling.estimator)})
```

(Keep every surrounding line untouched; only the `trunk_names=` argument and the saved dict change, plus the one helper call above them.)

In `configs/pipeline.yaml`, add one line to the existing `scaling:` block:

```yaml
scaling:
  scale_keypoints: trunk   # 'trunk' | 'all' (pair 'all' with estimator=norm_ratio)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_scale_mode.py -v` — all PASS. Also run `python -c "from omegaconf import OmegaConf; import yaml; yaml.safe_load(open('configs/pipeline.yaml'))"` to confirm the YAML stays valid.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_bout.py configs/pipeline.yaml tests/test_scale_mode.py
git commit -m "feat(pipeline): scaling.scale_keypoints mode (trunk|all) for courtship STAC"
```

---

### Task 9: Track 1 — run the 2×2 (execution)

**Files:** committed artifacts only: `docs/benchmark/2026-08-07-scale-ab/` (4 scorecards + render grids + `comparison.md`).

GPU required — all `run_bout.py` invocations go through sbatch on ckpt-g2 (or an interactive GPU node). The four variants (exact names and overrides):

| variant | overrides |
|---|---|
| `trunk_umeyama_segcal` | *(none — baseline config)* |
| `trunk_umeyama_nosegcal` | `anatomy.model.segment_calibration=false` |
| `all_norm_segcal` | `scaling.scale_keypoints=all scaling.estimator=norm_ratio` |
| `all_norm_nosegcal` | `scaling.scale_keypoints=all scaling.estimator=norm_ratio anatomy.model.segment_calibration=false` |

- [ ] **Step 1: Build variant roots + emit commands** (per variant; shown for one)

```bash
python -m scripts.benchmark.run_variant commands --variant all_norm_nosegcal \
  --override scaling.scale_keypoints=all --override scaling.estimator=norm_ratio \
  --override anatomy.model.segment_calibration=false > /tmp/cmds_all_norm_nosegcal.sh
```

- [ ] **Step 2: Run each variant's commands on a GPU node** (sbatch wrapper or interactive `salloc -p ckpt-g2 --gpus=1`; run the four variants' command files sequentially or as four array jobs). Verify every bout dir under each variant root gets `stac_ik.h5` → `outputs.h5` (`find <vroot> -name outputs.h5 | wc -l` equals entries×flies).

- [ ] **Step 3: Collect scorecards + render grids** (per variant)

```bash
python -m scripts.benchmark.run_variant collect --variant all_norm_nosegcal \
  --mjcf models/fruitfly_v1/fruitfly_v1_free.xml
MUJOCO_GL=osmesa python -m scripts.benchmark.render_grid \
  --root <benchmark_root>/variants/all_norm_nosegcal \
  --mjcf models/fruitfly_v1/fruitfly_v1_free.xml \
  --out-dir docs/benchmark/2026-08-07-scale-ab/renders-all_norm_nosegcal
```

- [ ] **Step 4: Write `comparison.md`** — a 4-row table (variant × headline metrics: reproj by group, soft IoU, jitter, JL-violation rate, per-cohort + gap ratios) plus each variant's `scale.json` value per run_key, and the render-grid images side by side. State the recommended default and WHY; flag if the winner differs between assays (the spec allows per-assay analysis but demands ONE default — surface the conflict to the user rather than deciding silently).

- [ ] **Step 5: Commit artifacts**

```bash
git add docs/benchmark/2026-08-07-scale-ab/
git commit -m "feat(benchmark): Track 1 scale/calibration 2x2 results + recommendation"
```

---

## Self-Review (completed)

- **Spec coverage:** cohorts + gap ratios (T3), proximity split + annotation (T2/T3), frozen inputs (T4), metric suite incl. jitter/joint-limits/grouped reproj/IoU (T2), render grid (T6), baseline (T7), variant mechanics honoring stage-skipping and recompute-the-treatment (T5), Track 1 port-free plumbing + 2×2 + decision gate (T8/T9). Committed-artifact requirement satisfied by T7/T9 committing scorecards + renders.
- **Placeholder scan:** clean — every code step carries full code; execution tasks carry exact commands.
- **Type consistency:** `bout_fly_dir(entry, root=None)` used identically in T4/T5/T6; `compute_bout_metrics` return shape consumed by `collect` matches T2's definition; `split_series`/`build_scorecard` shapes match between T3 and T5; `resolve_scale_keypoints(cfg, kp_names)` defined and called with the same signature in T8.
- **Known risk (stated, not hidden):** `_project`'s `(C,4,3)` einsum convention and `ReprojectionTool(str(calib_dir))` constructor signature are inferred from explorer reports; Task 2/5 implementers must verify both against one real bout and adjust the implementation (not the tests' math) if wrong.
