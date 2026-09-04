# MVQ P3a Implementation Plan — sex-typed slots, existence targets, copy-paste, warm start

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every mvq instance slot a fixed meaning (prompted / female / male / other), make existence and sex targets deterministic functions of the labels (with an ignore mask for the unlabelled fly in 26 % of training windows), add an affine-exact multi-view copy-paste augmentation, extend evaluation with per-slot existence, sex accuracy and mask containment, and launch a 10k-step warm-started fine-tune of the 30k run on ckpt-all.

**Architecture:** Label-driven slot assignment replaces cost-based matching in the loss (`train/matching.py` -> `assign_slots`). The loader emits per-fly sex and the unlabelled-fly code; a pure-numpy compositor pastes a donor fly from the same calibration group with per-camera shifts `M D + t_src - t_tgt`. The model gains one sex head and a fourth slot; a shape-tolerant warm start copies every matching leaf from the 30k run.

**Tech Stack:** JAX 0.11 / Flax NNX 0.12.x, optax, orbax 0.11, numpy, OpenCV (`cv2`, already a dependency), PIL, Hydra; pytest from `third_party/jarvis_jax/` (CPU, `JAX_PLATFORMS=cpu`).

**Spec:** `docs/specs/2026-09-04-mvq-p3a-identity-existence-design.md` — read it first; §3 slot table, §4 targets, §6 copy-paste, §7 eval, §8 run. The original spec `docs/specs/2026-09-03-mvq-dinov3-query-decoder-design.md` still governs everything the P3a spec does not mention.

## Global Constraints

- Package root: `third_party/jarvis_jax/jarvis_jax/`; tests in `third_party/jarvis_jax/tests/` (bare `from mvq_fixtures import ...`, no `tests/__init__.py`); run pytest from `third_party/jarvis_jax/` with `JAX_PLATFORMS=cpu`. Every CPU test must keep passing: the pre-P3a suite is 69 mvq tests (`pytest tests/test_dinov3.py tests/test_mvq_*.py tests/test_v12_windows.py tests/test_mv_augment.py tests/test_train_mvq_smoke.py -q`).
- Slot codes (spec §3): `SLOT_PROMPTED=0, SLOT_FEMALE=1, SLOT_MALE=2, SLOT_OTHER=3`. Sex codes: `SEX_FEMALE=0, SEX_MALE=1, SEX_UNKNOWN=-1`; `unlabelled_sex` adds `2` = present, sex unknown, and `-1` = nobody unlabelled. Define them ONCE in `train/matching.py` and import everywhere.
- `n_instances` is 4 everywhere from Task 4 on (config default, tiny test configs). `mvq_loss` raises if `out["xyz"].shape[1] != 4`.
- Cameras are AFFINE; `M (C,2,3)`, `t_local (T,C,2)` are crop-local: `uv_crop = M @ X_local + t_local`. World units 0.1 mm; `px_scale` ~8 px/unit. Never index keypoints or cameras by integer in figures — names from `ds.keypoint_names` / `ds.camera_names(i)`.
- Data root: `/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902`. The 30k run: `/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_local8_20260904/` (its `final/` appears when it reaches step 30000, ~11:15 on 2026-09-04; until then `ckpt/<step>` exists).
- Compute: this session sits on a GPU node (g3102) whose 8 GPUs are BUSY with the 30k run until it finishes — CPU tests run here directly; anything needing a GPU or heavy CPU goes through `scripts/slurm/submit_task.sh` (ckpt-all, constraint `h200|a100|l40s|l40|a40`). JAX GPU jobs need `module load cuda/12.9.1; export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6; unset LD_LIBRARY_PATH JAX_PLATFORMS; export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN=`.
- Figures under `figures/2026-09-mvq/<topic>/` (gitignored), the generating script committed under `scripts/viz/`, expectation in its docstring, PNG read back with the Read tool before any claim. Notes and small JSON go in `docs/benchmark/2026-09-mvq/` (force-add: the directory is gitignored).
- Commits: end messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01Npo4HiYC4t5M2xYjKUJsFP`; commit only the task's files (the working tree has unrelated uncommitted changes — never `git add -A`).

---

## File map

| file | responsibility |
|---|---|
| `jarvis_jax/train/matching.py` | slot/sex constants, `assign_slots`, `slot_ignore` (old `match`/`enumerate_assignments` deleted) |
| `jarvis_jax/train/losses_mvq.py` | uses `assign_slots`; existence with ignore; term 8 sex; `LossWeights.sex` |
| `jarvis_jax/models/mvq/decoder.py` | `Heads.sex`, `sex_logit` readout |
| `jarvis_jax/models/mvq/model.py` | `n_instances=4` default, `assemble` returns `sex_prob` |
| `jarvis_jax/models/mvq/checkpoint.py` | `mvq_run.json` looked up in the run dir first |
| `jarvis_jax/train/checkpoint.py` | `warm_start_partial` |
| `jarvis_jax/data/v12_windows.py` | `fly_sex`, `unlabelled_sex`, `fly_centroids`, donor index, copy-paste hook, `paste_window` |
| `jarvis_jax/data/mv_copy_paste.py` (new) | `CopyPasteParams`, body plane, offset sampling, view shifts, warp, composite (pure numpy/cv2) |
| `jarvis_jax/train/train_mvq.py` | `copy_paste_p`/`warm_start` config, run-dir json at start, evaluate: policy, per-slot exist, sex_acc, mask containment, contact_pair cohort |
| `configs/model/mvq.yaml`, `configs/train/mvq.yaml` | `n_instances: 4`, `loss.sex`, `copy_paste_*`, `warm_start` |
| `scripts/viz/mvq_sex_label_check.py`, `scripts/viz/mvq_copy_paste_check.py` (new, repo root) | pre-launch figure gates |
| `scripts/viz/mvq_overlay.py` | `contact_pair` case, slot index + sex prob in row labels |
| tests: `test_mvq_matching.py` (new), `test_mv_copy_paste.py` (new), `test_checkpoint_warm_start_partial.py` (new), updates to `test_mvq_losses.py`, `test_mvq_model.py`, `test_v12_windows.py`, `test_train_mvq_smoke.py`, `mvq_fixtures.py` | |

---

### Task 1: Label-driven slot assignment

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/matching.py` (replace whole file)
- Create: `third_party/jarvis_jax/tests/test_mvq_matching.py`
- Modify: `third_party/jarvis_jax/tests/test_mvq_losses.py` (delete the three `match`/`enumerate` tests at the top: `test_enumerate_assignments_is_injective_and_complete`, `test_match_agrees_with_scipy_hungarian`, `test_match_pins_fly0_to_instance0_and_ignores_invalid_flies`)

**Interfaces:**
- Produces: `assign_slots(fly_sex, fly_valid, prompt_on, dist, n_instances=4) -> (assign (B,F) int32, slot_target (B,I) bool)`, `slot_ignore(unlabelled_sex, n_instances=4) -> (B,I) bool`, constants `SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER, SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, SEX_PRESENT_UNKNOWN`. All JAX, jit-able, vmapped over B.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mvq_matching.py
import numpy as np
import jax, jax.numpy as jnp


def _run(fly_sex, fly_valid, prompt_on, dist):
    from jarvis_jax.train.matching import assign_slots
    a, t = assign_slots(jnp.asarray(fly_sex, jnp.int8), jnp.asarray(fly_valid), jnp.asarray(prompt_on),
                        jnp.asarray(dist, jnp.float32))
    return np.asarray(a).tolist(), np.asarray(t).tolist()


def test_single_female_unprompted_goes_to_female_slot():
    a, t = _run([[0, -1]], [[True, False]], [False], [[0.0, 0.0]])
    assert a == [[1, -1]] and t == [[False, True, False, False]]


def test_single_male_prompted_goes_to_slot0_and_male_slot_is_empty():
    a, t = _run([[1, -1]], [[True, False]], [True], [[0.0, 0.0]])
    assert a == [[0, -1]] and t == [[True, False, False, False]]


def test_mixed_pair_unprompted_and_prompted():
    a, t = _run([[0, 1]], [[True, True]], [False], [[0.0, 30.0]])
    assert a == [[1, 2]] and t == [[False, True, True, False]]
    a, t = _run([[0, 1]], [[True, True]], [True], [[0.0, 30.0]])
    assert a == [[0, 2]] and t == [[True, False, True, False]]


def test_same_sex_pair_nearer_fly_takes_typed_slot():
    # host is fly 0 but the OTHER female is nearer the ROI origin: order is host first, so the
    # host still takes slot 1 (host-first rule beats distance); the other goes to slot 3
    a, t = _run([[0, 0]], [[True, True]], [False], [[10.0, 2.0]])
    assert a == [[1, 3]] and t == [[False, True, False, True]]
    # prompted: host -> 0, the other female -> the (free) female slot 1
    a, _ = _run([[0, 0]], [[True, True]], [True], [[10.0, 2.0]])
    assert a == [[0, 1]]


def test_unknown_sex_goes_to_other_slot_and_invalid_is_minus_one():
    a, t = _run([[-1, 1]], [[True, False]], [False], [[0.0, 0.0]])
    assert a == [[3, -1]] and t == [[False, False, False, True]]


def test_batched_and_jittable():
    from jarvis_jax.train.matching import assign_slots
    f = jax.jit(assign_slots)
    a, t = f(jnp.asarray([[0, 1], [1, -1]], jnp.int8), jnp.asarray([[True, True], [True, False]]),
             jnp.asarray([False, True]), jnp.zeros((2, 2), jnp.float32))
    assert np.asarray(a).tolist() == [[1, 2], [0, -1]]
    assert np.asarray(t).shape == (2, 4)


def test_slot_ignore_codes():
    from jarvis_jax.train.matching import slot_ignore
    ig = np.asarray(slot_ignore(jnp.asarray([-1, 0, 1, 2], jnp.int8)))
    assert ig.tolist() == [[False, False, False, False],
                           [False, True, False, True],
                           [False, False, True, True],
                           [False, True, True, True]]
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu pytest tests/test_mvq_matching.py -q`
Expected: FAIL with `ImportError: cannot import name 'assign_slots'`.

- [ ] **Step 3: Replace `matching.py`**

```python
"""Label-driven instance-slot assignment (P3a spec §3-4). No prediction enters:
a slot's existence/sex target is a deterministic function of the labels and
the prompt flag, so the existence head never sees coin-flip targets."""
from __future__ import annotations

import jax
import jax.numpy as jnp

SLOT_PROMPTED, SLOT_FEMALE, SLOT_MALE, SLOT_OTHER = 0, 1, 2, 3
SEX_FEMALE, SEX_MALE, SEX_UNKNOWN = 0, 1, -1
SEX_PRESENT_UNKNOWN = 2          # only in `unlabelled_sex`: an unlabelled fly of unknown sex
N_SLOTS = 4


def _assign_one(sex, valid, on, dist, n_instances):
    """One sample. sex (F,) int8, valid (F,) bool, on () bool, dist (F,) float."""
    F = sex.shape[0]
    # host (fly 0) first, then the others by increasing distance from the ROI origin
    order = jnp.argsort(jnp.where(jnp.arange(F) == 0, -jnp.inf, dist))
    typed_all = jnp.where(sex == SEX_FEMALE, SLOT_FEMALE, jnp.where(sex == SEX_MALE, SLOT_MALE, SLOT_OTHER))

    def body(carry, f):
        taken, assign = carry
        typed = typed_all[f]
        slot = jnp.where((f == 0) & on, SLOT_PROMPTED, jnp.where(taken[typed], SLOT_OTHER, typed))
        slot = jnp.where(valid[f], slot, -1)
        taken = jnp.where(valid[f], taken.at[jnp.maximum(slot, 0)].set(True) | taken, taken)
        return (taken, assign.at[f].set(slot)), None

    (taken, assign), _ = jax.lax.scan(body, (jnp.zeros((n_instances,), bool), jnp.full((F,), -1, jnp.int32)), order)
    return assign, taken


def assign_slots(fly_sex, fly_valid, prompt_on, dist, n_instances=N_SLOTS):
    """fly_sex (B,F) int8 {0 F, 1 M, -1 unknown}; fly_valid (B,F); prompt_on (B,);
    dist (B,F) distance of each fly's labelled-3D centroid from the ROI origin.
    Returns assign (B,F) int32 slot per fly (-1 = invalid fly) and
    slot_target (B,I) bool = a fly was assigned to that slot. F <= 2 is assumed
    (slot 3 can hold one fly)."""
    return jax.vmap(lambda s, v, o, d: _assign_one(s, v, o, d, n_instances))(fly_sex, fly_valid, prompt_on, dist)


def slot_ignore(unlabelled_sex, n_instances=N_SLOTS):
    """(B,) int8 -> (B,I) bool: slots that get NO existence loss because an
    unlabelled fly present in the window could legitimately occupy them.
    -1: none; 0/1: that sex's slot and OTHER; 2: every slot but PROMPTED."""
    u = unlabelled_sex[:, None]
    s = jnp.arange(n_instances)[None, :]
    other = s == SLOT_OTHER
    return (((u == SEX_FEMALE) & ((s == SLOT_FEMALE) | other))
            | ((u == SEX_MALE) & ((s == SLOT_MALE) | other))
            | ((u == SEX_PRESENT_UNKNOWN) & (s != SLOT_PROMPTED)))
```

- [ ] **Step 4: Delete the three old tests at the top of `tests/test_mvq_losses.py`** (keep `_perfect_batch` and everything below; Task 3 updates those). Remove `import pytest` only if unused.

- [ ] **Step 5: Run**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mvq_matching.py -q`
Expected: 7 passed. (`tests/test_mvq_losses.py` now fails on the `match` import — expected until Task 3.)

- [ ] **Step 6: Commit**

```bash
git add jarvis_jax/train/matching.py tests/test_mvq_matching.py tests/test_mvq_losses.py
git commit -m "feat(mvq): label-driven slot assignment replaces cost-based matching (P3a §3)"
```

---

### Task 2: Loader emits `fly_sex`, `unlabelled_sex`, `fly_centroids`

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` (`WINDOW_KEYS`, `__init__`, new methods, `__getitem__` return)
- Modify: `third_party/jarvis_jax/tests/mvq_fixtures.py` (manifest `n_flies`)
- Modify: `third_party/jarvis_jax/tests/test_v12_windows.py`

**Interfaces:**
- Produces: sample keys `fly_sex (F,) int8`, `unlabelled_sex () int8`; `ds.fly_sex_code(rec, fly) -> int`, `ds.unlabelled_sex(i) -> int`, `ds.fly_centroids(i) -> (n_labelled_flies, 3) float32` (labelled-3D centroid per fly, world coords, frame 0, host first; no image decode).

- [ ] **Step 1: Fixture: add `n_flies` to the manifest** in `make_v12_root` — change the manifest dict to `{"calib_group": "A", "sex": "mixed", "behavior": "courtship", "n_flies": 2, "fly_sex": {"fly0": "female", "fly1": "male"}, "split": "train"}`.

- [ ] **Step 2: Write the failing tests** (append to `tests/test_v12_windows.py`)

```python
def test_fly_sex_and_unlabelled_sex_keys(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS
    from jarvis_jax.train.matching import SEX_FEMALE, SEX_MALE
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    assert "fly_sex" in WINDOW_KEYS and "unlabelled_sex" in WINDOW_KEYS
    both = ds.windows.index((REC, 0, 1))           # frame 1 has fly0 + fly1 labelled
    s = ds[both]
    assert s["fly_sex"].dtype == np.int8 and s["fly_sex"].tolist() == [SEX_FEMALE, SEX_MALE]
    assert s["unlabelled_sex"].dtype == np.int8 and int(s["unlabelled_sex"]) == -1
    only_host = ds.windows.index((REC, 0, 0))      # frame 0: fly1 present per manifest, not labelled
    s0 = ds[only_host]
    assert s0["fly_sex"].tolist() == [SEX_FEMALE, -1]
    assert int(s0["unlabelled_sex"]) == SEX_MALE
    assert ds.unlabelled_sex(only_host) == SEX_MALE and ds.unlabelled_sex(both) == -1
    # a fly1-host window in the two-fly frame: fly0 (female) is the OTHER labelled fly
    host1 = ds.windows.index((REC, 1, 1))
    assert ds[host1]["fly_sex"].tolist() == [SEX_MALE, SEX_FEMALE]


def test_fly_centroids_match_sample_labels(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = ds.windows.index((REC, 0, 1))
    c = ds.fly_centroids(i)
    s = ds[i]
    assert c.shape == (2, 3)
    for f in range(2):
        has = s["has3d"][f, 0]
        ref = (s["kp3d_local"][f, 0][has]).mean(0) + s["center3D"]
        np.testing.assert_allclose(c[f], ref, atol=1e-3)
```

- [ ] **Step 3: Run to verify they fail**

Run: `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py -q -k "fly_sex or centroids"`
Expected: FAIL (`fly_sex` not in `WINDOW_KEYS`).

- [ ] **Step 4: Implement**

In `v12_windows.py`:

```python
from jarvis_jax.train.matching import SEX_FEMALE, SEX_MALE, SEX_UNKNOWN, SEX_PRESENT_UNKNOWN

WINDOW_KEYS = ("crops", "cam_valid", "M", "t_local", "center3D", "kp3d_local", "has3d",
               "kp2d", "vis2d", "fly_valid", "px_scale", "is_female", "prompt_mask", "crop_origin",
               "fly_sex", "unlabelled_sex")

_SEX_CODE = {"female": SEX_FEMALE, "male": SEX_MALE}
```

Methods on `V12WindowDataset` (after `n_flies`):

```python
    def fly_sex_code(self, rec, fly):
        return _SEX_CODE.get(self._sex.get((rec, fly), "unknown"), SEX_UNKNOWN)

    def _window_flies(self, i):
        """Labelled fly ids in window i, host first, capped at max_flies (same rule as __getitem__)."""
        rec, host, f0 = self.windows[i]
        frames = [f0 + k for k in range(self.T)]
        others = sorted({k[2] for k in self._fs if k[0] == rec and k[1] in frames and k[2] != host})
        return rec, [host] + others[: self.max_flies - 1]

    def unlabelled_sex(self, i):
        """-1 if every animal the manifest says is in this recording is labelled in the
        window; else the sex code of the one unlabelled animal (SEX_PRESENT_UNKNOWN if
        the manifest does not name its sex)."""
        rec, flies = self._window_flies(i)
        meta = self.manifest.get(rec, {})
        n_present = int(meta.get("n_flies", len(flies)))
        if n_present <= len(flies):
            return SEX_UNKNOWN
        named = {int(k[3:]): v for k, v in (meta.get("fly_sex") or {}).items()}
        missing = [fid for fid in sorted(named) if fid not in flies]
        if not missing:
            return SEX_PRESENT_UNKNOWN
        return _SEX_CODE.get(named[missing[0]], SEX_PRESENT_UNKNOWN)

    def fly_centroids(self, i):
        """(n_labelled_flies, 3) world centroid of each fly's DLT-able labels at frame 0
        (host first). Labels only -- no JPEG decode -- so it is cheap enough for cohorts."""
        rec, flies = self._window_flies(i)
        rt = self._rt(rec); f0 = self.windows[i][2]
        out = np.zeros((len(flies), 3), np.float32)
        for fi, fly in enumerate(flies):
            fsv = self._fs.get((rec, f0, fly))
            if fsv is None:
                out[fi] = np.nan; continue
            kp, _ = self._labels_full(fsv, rt)
            X, has = self._dlt(kp, rt)
            out[fi] = X[has].mean(0) if has.any() else np.nan
        return out
```

In `__getitem__`, replace the `others = ...; flies = ...` two lines with `rec_, flies = self._window_flies(i)` (keep `rec`), and add to the returned dict:

```python
            "fly_sex": np.array([self.fly_sex_code(rec, fly) if fi < len(flies) else SEX_UNKNOWN
                                 for fi, fly in enumerate(flies + [None] * (F - len(flies)))], np.int8),
            "unlabelled_sex": np.int8(self.unlabelled_sex(i)),
```

(Write the list comprehension so a padded slot yields `SEX_UNKNOWN`; e.g. `[self.fly_sex_code(rec, flies[fi]) if fi < len(flies) else SEX_UNKNOWN for fi in range(F)]`.)

- [ ] **Step 5: Run the whole loader test file**

Run: `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py -q`
Expected: all pass (the old `test_sample_shapes_and_instances` asserts `set(s) == set(WINDOW_KEYS)`, which still holds).

- [ ] **Step 6: Commit**

```bash
git add jarvis_jax/data/v12_windows.py tests/mvq_fixtures.py tests/test_v12_windows.py
git commit -m "feat(mvq): loader emits fly_sex, unlabelled_sex and label-only fly centroids (P3a §4)"
```

---

### Task 3: Loss uses slot assignment; existence ignore; sex term

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/losses_mvq.py`
- Modify: `third_party/jarvis_jax/tests/test_mvq_losses.py`

**Interfaces:**
- Consumes: `assign_slots`, `slot_ignore`, batch keys `fly_sex (B,F) int8`, `unlabelled_sex (B,) int8`, `prompt_on (B,) bool`; `out["sex_logit"] (B,I)`.
- Produces: `LossWeights.sex: float = 0.5`; metrics add `sex`, `sex_acc`; `exist`/`exist_acc` are masked by ignore. `mvq_loss` raises `ValueError` unless `I == 4`.

- [ ] **Step 1: Update `_perfect_batch` in `tests/test_mvq_losses.py`** to I=4 and the new keys. Instance layout: fly 0 (female) -> slot 1, fly 1 (male) -> slot 2; slots 0 and 3 empty, unprompted.

```python
def _perfect_batch(B=2, I=4, T=1, C=3, K=5, seed=0):
    """Labels + an output that reproduces them exactly (fly 0 female -> slot 1, fly 1 male -> slot 2)."""
    from jarvis_jax.models.mvq.geometry import project_local
    rng = np.random.default_rng(seed)
    M = np.stack([np.array([[8.0, 0.1 * c, 0.0], [0.0, -8.0, 0.2 * c]]) for c in range(C)]).astype(np.float32)
    M = np.broadcast_to(M, (B, C, 2, 3)).copy()
    tl = np.broadcast_to(np.array([[224.0, 224.0]] * C, np.float32), (B, T, C, 2)).copy()
    X = rng.normal(size=(B, 2, T, K, 3)).astype(np.float32) * 5
    X[:, 1] += 40.0                                                                 # fly 1 sits away from the ROI origin
    kp2d = np.stack([np.stack([np.stack([np.asarray(project_local(jnp.asarray(X[b, f, t]), jnp.asarray(M[b]), jnp.asarray(tl[b, t])))
                                          for t in range(T)])
                                for f in range(2)])
                      for b in range(B)])
    kp2d = np.moveaxis(kp2d, 4, 3)
    batch = {"M": M, "t_local": tl, "kp3d_local": X, "has3d": np.ones((B, 2, T, K), bool),
             "kp2d": kp2d.astype(np.float32), "vis2d": np.ones((B, 2, T, C, K), bool),
             "fly_valid": np.ones((B, 2), bool), "px_scale": np.full((B,), 8.0, np.float32),
             "cam_valid": np.ones((B, T, C), bool), "prompt_on": np.zeros((B,), bool),
             "fly_sex": np.array([[0, 1]] * B, np.int8), "unlabelled_sex": np.full((B,), -1, np.int8)}
    xyz = np.full((B, I, T, K, 3), 50.0, np.float32); xyz[:, 1] = X[:, 0]; xyz[:, 2] = X[:, 1]
    uv = np.zeros((B, I, T, C, K, 2), np.float32); uv[:, 1] = kp2d[:, 0]; uv[:, 2] = kp2d[:, 1]
    big = np.full((B, I, T, K), 6.0, np.float32)
    out = {"xyz": xyz, "conf_logit": big, "exist_logit": np.array([[-6.0, 6.0, 6.0, -6.0]] * B, np.float32),
           "sex_logit": np.array([[0.0, 6.0, -6.0, 0.0]] * B, np.float32),
           "uv": uv, "vis_logit": np.full((B, I, T, C, K), 6.0, np.float32), "aux_pass1": None, "aux_layers": []}
    out["aux_pass1"] = {k: v for k, v in out.items() if k not in ("aux_pass1", "aux_layers")}
    j = lambda d: {k: (jnp.asarray(v) if not isinstance(v, (dict, list)) and v is not None else v) for k, v in d.items()}
    out = j(out); out["aux_pass1"] = j(out["aux_pass1"])
    return out, j(batch)
```

Then fix the existing tests that hard-code slot indices: in `test_masked_entries_do_not_contribute`, `test_repulsion_fires_only_near_other_fly_same_part`, `test_confidence_term_prefers_low_c_on_bad_points`, `test_loss_is_differentiable_and_jittable` replace every `out["xyz"].at[:, 0` / `out["uv"].at[:, 0` (fly 0's slot) with `[:, 1`. In `test_single_valid_fly_zero_repulsion_and_correct_exist_acc` the expected matched vector becomes `[[False, True, False, False]] * B`. In `test_aux_layers_deep_supervision_raises_total_and_stays_jittable` add `"sex_logit": out["sex_logit"]` to the aux layer dict is NOT needed (aux layers only use xyz) — leave it.

- [ ] **Step 2: Add the new tests**

```python
def test_loss_rejects_wrong_slot_count():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    out, batch = _perfect_batch(I=3)
    with pytest.raises(ValueError):
        mvq_loss(out, batch, LossWeights(), np.arange(5, dtype=np.int32))


def test_existence_ignores_unlabelled_fly_slots():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch()
    batch["fly_valid"] = batch["fly_valid"].at[:, 1].set(False)          # only the female is labelled...
    batch["unlabelled_sex"] = jnp.full((2,), 1, jnp.int8)                # ...but a male is present
    _, m_ref = mvq_loss(out, batch, LossWeights(), pk)
    out2 = dict(out); out2["exist_logit"] = out["exist_logit"].at[:, 2].set(6.0).at[:, 3].set(6.0)   # claim male + other
    _, m_ign = mvq_loss(out2, batch, LossWeights(), pk)
    assert abs(float(m_ign["exist"]) - float(m_ref["exist"])) < 1e-6      # ignored slots change nothing
    assert float(m_ign["exist_acc"]) == 1.0
    batch["unlabelled_sex"] = jnp.full((2,), -1, jnp.int8)               # nobody unlabelled: now they count
    _, m_cnt = mvq_loss(out2, batch, LossWeights(), pk)
    assert float(m_cnt["exist"]) > float(m_ign["exist"]) + 1.0 and float(m_cnt["exist_acc"]) < 1.0


def test_prompted_host_moves_to_slot0():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch()
    batch["prompt_on"] = jnp.ones((2,), bool)
    # move the female's perfect prediction from slot 1 to slot 0 and flip the exist/sex logits to match
    o = dict(out)
    o["xyz"] = out["xyz"].at[:, 0].set(out["xyz"][:, 1]).at[:, 1].set(50.0)
    o["uv"] = out["uv"].at[:, 0].set(out["uv"][:, 1]).at[:, 1].set(0.0)
    o["exist_logit"] = jnp.asarray([[6.0, -6.0, 6.0, -6.0]] * 2, jnp.float32)
    o["sex_logit"] = jnp.asarray([[6.0, 0.0, -6.0, 0.0]] * 2, jnp.float32)
    o["aux_pass1"] = {k: o[k] for k in out["aux_pass1"]}
    _, m = mvq_loss(o, batch, LossWeights(), pk)
    assert float(m["reproj"]) < 1e-3 and float(m["exist_acc"]) == 1.0 and float(m["sex_acc"]) == 1.0


def test_sex_term_only_on_assigned_known_sex_slots():
    from jarvis_jax.train.losses_mvq import mvq_loss, LossWeights
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch()
    _, m0 = mvq_loss(out, batch, LossWeights(), pk)
    assert float(m0["sex_acc"]) == 1.0 and float(m0["sex"]) < 1e-2
    o = dict(out); o["sex_logit"] = out["sex_logit"].at[:, 0].set(-9.0).at[:, 3].set(9.0)   # unassigned slots
    _, m1 = mvq_loss(o, batch, LossWeights(), pk)
    assert abs(float(m1["sex"]) - float(m0["sex"])) < 1e-6
    o2 = dict(out); o2["sex_logit"] = out["sex_logit"].at[:, 1].set(-6.0)                 # female slot says male
    _, m2 = mvq_loss(o2, batch, LossWeights(), pk)
    assert float(m2["sex"]) > 1.0 and float(m2["sex_acc"]) == 0.5
    batch["fly_sex"] = batch["fly_sex"].at[:, 1].set(-1)                                     # fly 1 sex unknown -> slot 3
    _, m3 = mvq_loss(out, batch, LossWeights(), pk)
    assert float(m3["sex_acc"]) == 1.0                                                        # only slot 1 scored
```

- [ ] **Step 3: Run to verify failures**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mvq_losses.py -q`
Expected: ImportError on `match` (whole file fails).

- [ ] **Step 4: Implement in `losses_mvq.py`**

Replace the import and the matching block:

```python
from jarvis_jax.train.matching import assign_slots, slot_ignore, SEX_FEMALE, N_SLOTS
```

`LossWeights` gains `sex: float = 0.5` (after `exist`).

In `mvq_loss`, replace everything from `# ---------------- matching on the final pass` through `assign, inst_matched = match(...)` with:

```python
    if I != N_SLOTS:
        raise ValueError(f"mvq_loss expects n_instances == {N_SLOTS} (slots prompted/female/male/other), got {I}")
    # ---------------- label-driven slot assignment (P3a §3): no prediction enters
    has_f = batch["has3d"].astype(jnp.float32)                                          # (B,F,T,K)
    cen = (batch["kp3d_local"] * has_f[..., None]).sum((2, 3)) / jnp.maximum(has_f.sum((2, 3)), 1.0)[..., None]
    dist = jnp.linalg.norm(cen, axis=-1)                                                 # (B,F) from the ROI origin
    assign, slot_target = assign_slots(batch["fly_sex"], fv, batch["prompt_on"], dist, I)
    ignore = slot_ignore(batch["unlabelled_sex"], I)                                     # (B,I)
    inst_matched = slot_target
```

Replace term 6 with:

```python
    # term 6 existence -- masked mean over slots NOT ignored (an unlabelled fly could occupy an ignored slot)
    tgt = inst_matched.astype(jnp.float32)
    bce_e = jnp.maximum(out["exist_logit"], 0) - out["exist_logit"] * tgt + jnp.log1p(jnp.exp(-jnp.abs(out["exist_logit"])))
    keep = ~ignore
    exist = _mmean(bce_e, keep)
    exist_acc = _mmean(((out["exist_logit"] > 0) == inst_matched).astype(jnp.float32), keep)
    # term 8 sex (P3a §4): BCE female=1 on assigned slots whose fly has a known sex
    oh = ((assign[:, :, None] == jnp.arange(I)[None, None, :]) & fv[:, :, None]).astype(jnp.int32)   # (B,F,I)
    slot_sex = (oh * (batch["fly_sex"].astype(jnp.int32) + 1)[:, :, None]).sum(1) - 1               # (B,I) -1 = none/unknown
    m_sex = slot_sex >= 0
    sex_t = (slot_sex == SEX_FEMALE).astype(jnp.float32)
    bce_s = jnp.maximum(out["sex_logit"], 0) - out["sex_logit"] * sex_t + jnp.log1p(jnp.exp(-jnp.abs(out["sex_logit"])))
    sex = _mmean(bce_s, m_sex)
    sex_acc = _mmean(((out["sex_logit"] > 0) == (slot_sex == SEX_FEMALE)).astype(jnp.float32), m_sex)
```

Add `+ w.sex * sex` to `total`, and `"sex": sex, "sex_acc": sex_acc` to `metrics`. Note `_mmean` needs `m` broadcastable to `x`: `keep`/`m_sex` are `(B,I)` like the logits — fine. (`slot_sex` for a fly with `fly_sex=-1` contributes `oh*0`, i.e. unknown — verify with the last assertion of the sex test.)

- [ ] **Step 5: Run**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mvq_losses.py tests/test_mvq_matching.py -q`
Expected: all pass. If `test_loss_near_zero_at_ground_truth_and_metrics`'s `total < 0.05` fails, check that `sex` at the fixture's logits is `log1p(exp(-6))*0.5 ≈ 0.0012` — it should pass; do not loosen the bound.

- [ ] **Step 6: Commit**

```bash
git add jarvis_jax/train/losses_mvq.py tests/test_mvq_losses.py
git commit -m "feat(mvq): existence with unlabelled-fly ignore, sex term, label-driven slots in the loss (P3a §4)"
```

---

### Task 4: Model — sex head, four slots, assemble, configs

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/models/mvq/decoder.py` (`Heads`, `_read3d`)
- Modify: `third_party/jarvis_jax/jarvis_jax/models/mvq/model.py` (`n_instances` default, `assemble`)
- Modify: `third_party/jarvis_jax/configs/model/mvq.yaml`, `third_party/jarvis_jax/configs/train/mvq.yaml`
- Modify: `third_party/jarvis_jax/tests/test_mvq_model.py`, and every test constructing `MVQConfig(... n_instances=2 ...)` (grep `n_instances=` under `tests/` — `test_mvq_model.py`, `test_train_mvq_smoke.py`, `test_mvq_attention.py` if present) -> `n_instances=4`.

**Interfaces:**
- Produces: `out["sex_logit"] (B,I)` from `MVQModel.__call__` (final pass; `aux_pass1` carries it too since `_read3d` is shared); `assemble(...) -> (kp3d, conf3d, kp2d, sex_prob)`.

- [ ] **Step 1: Tests** (in `tests/test_mvq_model.py`; adapt `test_output_shapes_and_aux` to assert `out["sex_logit"].shape == (B, 4)`; adapt `test_assemble_nan_policy` to unpack four values and assert `sex_prob.shape == (B, I)` and `0 <= sex_prob <= 1`.)

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mvq_model.py -q -k "shapes or assemble"`
Expected: FAIL (`KeyError: 'sex_logit'` / unpack error).

- [ ] **Step 3: Implement**

`decoder.py` `Heads.__init__`: add `self.sex = nnx.Linear(D, 1, rngs=rngs)` after `self.exist`. `_read3d`: add `"sex_logit": self.heads.sex(h4.mean(axis=(2, 3)))[..., 0]`.

`model.py`: `n_instances: int = 4`; `assemble` docstring gets the slot table (`0 prompted, 1 female, 2 male, 3 other -- P3a spec §3`) and ends with:

```python
    sex_prob = 1 / (1 + np.exp(-np.asarray(out["sex_logit"])))
    return kp3d.astype(np.float32), conf3d.astype(np.float32), kp2d.astype(np.float32), sex_prob.astype(np.float32)
```

Grep for other `assemble(` callers (`grep -rn "assemble(" third_party/jarvis_jax scripts`) and update each unpack.

`configs/model/mvq.yaml`: `n_instances: 4` with the comment `# P3a: prompted / female / male / other (fixed meaning, spec 2026-09-04 §3)`.
`configs/train/mvq.yaml`: under `loss:` add `sex: 0.5`.

Tiny test configs: `n_instances=4` everywhere (`sed -i 's/n_instances=2/n_instances=4/' tests/*.py`, then read the diff).

- [ ] **Step 4: Run the model + smoke tests**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mvq_model.py tests/test_train_mvq_smoke.py -q`
Expected: model tests pass; smoke tests pass (the trainer already threads `fly_sex`/`unlabelled_sex` through `WINDOW_KEYS`; `augment_window` passes unknown keys through untouched — verify by reading `_per_view_affine`/`_mirror`/`_camera_dropout` return `dict(b, ...)` style updates; if any of them rebuilds the batch from an explicit key list, add the two keys).

- [ ] **Step 5: Commit**

```bash
git add jarvis_jax/models/mvq/decoder.py jarvis_jax/models/mvq/model.py configs/model/mvq.yaml configs/train/mvq.yaml tests/
git commit -m "feat(mvq): sex head, four fixed slots, assemble returns sex_prob (P3a §5)"
```

---

### Task 5: Multi-view copy-paste compositor (pure numpy)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/data/mv_copy_paste.py`
- Create: `third_party/jarvis_jax/tests/test_mv_copy_paste.py`

**Interfaces:**
- Produces:
  - `CopyPasteParams(p=0.5, opposite_sex_p=0.7, contact_p=0.3, contact_sep=(8.0, 15.0), far_sep=(15.0, 60.0), max_tries=8, gain_clip=(0.7, 1.4))` (frozen dataclass)
  - `body_plane_axes(kp3d_local (K,3), has3d (K,)) -> (2,3)` orthonormal
  - `sample_offset(rng, axes (2,3), params) -> D (3,) float32`
  - `view_shifts(M (C,2,3), D (3,), t_local_src (C,2), t_local_tgt (C,2)) -> (C,2) float32`
  - `composite(tgt: dict, src: dict, D, params) -> dict | None` (T=1 samples as `V12WindowDataset.__getitem__` returns them; `None` = rejected)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mv_copy_paste.py
import numpy as np
import jax.numpy as jnp
from mvq_fixtures import make_v12_root, REC


def _two_samples(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    tgt = ds[ds.windows.index((REC, 0, 0))]        # single labelled fly (frame 0)
    src = ds[ds.windows.index((REC, 0, 2))]        # donor: host of frame 2
    return ds, tgt, src


def test_view_shifts_are_exact_affine_translations():
    from jarvis_jax.data.mv_copy_paste import view_shifts
    from jarvis_jax.models.mvq.geometry import project_local
    rng = np.random.default_rng(0)
    M = rng.normal(size=(3, 2, 3)).astype(np.float32); ts = rng.normal(size=(3, 2)).astype(np.float32)
    tt = rng.normal(size=(3, 2)).astype(np.float32); D = np.array([3.0, -2.0, 1.0], np.float32)
    X = rng.normal(size=(5, 3)).astype(np.float32)
    uv_src = np.asarray(project_local(jnp.asarray(X), jnp.asarray(M), jnp.asarray(ts)))          # (K,C,2)
    uv_tgt = np.asarray(project_local(jnp.asarray(X + D), jnp.asarray(M), jnp.asarray(tt)))
    s = view_shifts(M, D, ts, tt)
    np.testing.assert_allclose(uv_src + s[None], uv_tgt, atol=1e-4)


def test_body_plane_axes_are_orthonormal_and_span_the_spread():
    from jarvis_jax.data.mv_copy_paste import body_plane_axes
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(50, 3)) * np.array([10.0, 4.0, 0.5])
    ax = body_plane_axes(pts.astype(np.float32), np.ones(50, bool))
    np.testing.assert_allclose(ax @ ax.T, np.eye(2), atol=1e-5)
    assert abs(ax[:, 2]).max() < 0.2                       # thin axis (z) is the normal, not in-plane


def test_sample_offset_respects_separation_and_plane():
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, sample_offset
    ax = np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    rng = np.random.default_rng(1)
    for _ in range(50):
        D = sample_offset(rng, ax, CopyPasteParams())
        assert abs(D[2]) < 1e-6 and 8.0 <= np.linalg.norm(D) <= 60.0


def test_composite_labels_are_geometrically_exact_and_host_is_occluded(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.models.mvq.geometry import project_local
    ds, tgt, src = _two_samples(tmp_path)
    D = np.array([12.0, 0.0, 0.0], np.float32)                      # contact range: donor overlaps the host
    out = composite(tgt, src, D, CopyPasteParams())
    assert out is not None
    assert out["fly_valid"].tolist() == [True, True] and out["fly_sex"][1] == src["fly_sex"][0]
    assert int(out["unlabelled_sex"]) == -1
    # pasted 3D reprojects onto pasted 2D in every valid camera
    uv = np.asarray(project_local(jnp.asarray(out["kp3d_local"][1, 0]), jnp.asarray(out["M"]), jnp.asarray(out["t_local"][0])))
    has = out["has3d"][1, 0]
    for c in range(7):
        np.testing.assert_allclose(uv[has, c], out["kp2d"][1, 0, c][has], atol=1e-3)
    # the donor's pixels landed where its labels say: the crop is brighter under the pasted mask centre
    c = 0; k = np.where(out["vis2d"][1, 0, c])[0][0]
    u, v = np.round(out["kp2d"][1, 0, c, k]).astype(int)
    assert out["crops"][0, c, v, u].max() > tgt["crops"][0, c, v, u].max() or out["crops"][0, c].mean() != tgt["crops"][0, c].mean()
    # host keypoints under the donor mask are now invisible; the host prompt mask lost those pixels
    assert out["vis2d"][0].sum() <= tgt["vis2d"][0].sum()
    assert out["prompt_mask"].sum() <= tgt["prompt_mask"].sum()
    # host labels untouched
    np.testing.assert_array_equal(out["kp2d"][0], tgt["kp2d"][0]); np.testing.assert_array_equal(out["kp3d_local"][0], tgt["kp3d_local"][0])


def test_composite_rejects_when_donor_leaves_crop_or_cameras_missing(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    ds, tgt, src = _two_samples(tmp_path)
    assert composite(tgt, src, np.array([500.0, 0.0, 0.0], np.float32), CopyPasteParams()) is None
    src2 = dict(src); cv = src["cam_valid"].copy(); cv[0, 0] = False; src2["cam_valid"] = cv
    assert composite(tgt, src2, np.array([12.0, 0.0, 0.0], np.float32), CopyPasteParams()) is None
```

- [ ] **Step 2: Run to verify failure**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mv_copy_paste.py -q`
Expected: ImportError.

- [ ] **Step 3: Implement `mv_copy_paste.py`**

```python
"""Affine-exact multi-view copy-paste (P3a spec §6). A donor fly (the host of
another window of the SAME calibration group) is translated by a 3D offset D
in the target's ROI-local frame; under affine cameras that is the per-camera
2D translation `M_c D + t_local_src[c] - t_local_tgt[c]` in crop px, so pixels,
2D labels and 3D labels stay mutually consistent in every view. Pure numpy +
cv2; runs in the loader thread pool."""
from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from jarvis_jax.train.matching import SEX_UNKNOWN


@dataclasses.dataclass(frozen=True)
class CopyPasteParams:
    p: float = 0.5
    opposite_sex_p: float = 0.7
    contact_p: float = 0.3
    contact_sep: tuple = (8.0, 15.0)      # world units (0.1 mm): the stacked-pair regime
    far_sep: tuple = (15.0, 60.0)
    max_tries: int = 8
    gain_clip: tuple = (0.7, 1.4)


def body_plane_axes(kp3d_local, has3d):
    """(2,3) orthonormal axes spanning the two largest principal directions of
    the labelled points (the floor for a walking fly, the wall for a climber)."""
    pts = np.asarray(kp3d_local, np.float64)[np.asarray(has3d, bool)]
    if pts.shape[0] < 3:
        return np.array([[1.0, 0, 0], [0, 1.0, 0]], np.float32)
    pts = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    return vt[:2].astype(np.float32)


def sample_offset(rng, axes, params: CopyPasteParams):
    lo, hi = params.contact_sep if rng.uniform() < params.contact_p else params.far_sep
    sep = rng.uniform(lo, hi); ang = rng.uniform(0, 2 * np.pi)
    return (sep * (np.cos(ang) * axes[0] + np.sin(ang) * axes[1])).astype(np.float32)


def view_shifts(M, D, t_local_src, t_local_tgt):
    """(C,2) crop-px translation that moves the donor's crop content to where the
    donor would appear in the TARGET crop after a 3D offset D."""
    return (np.einsum("cij,j->ci", np.asarray(M, np.float64), np.asarray(D, np.float64))
            + np.asarray(t_local_src, np.float64) - np.asarray(t_local_tgt, np.float64)).astype(np.float32)


def _translate(img, shift, nearest=False):
    h, w = img.shape[:2]
    A = np.array([[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]], np.float64)
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    return cv2.warpAffine(img, A, (w, h), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def composite(tgt, src, D, params: CopyPasteParams):
    """Paste `src`'s host fly into `tgt` as fly 1. Both are T=1 samples from
    V12WindowDataset (numpy). Returns a NEW sample dict, or None when rejected:
    a target-valid camera the donor lacks, or a donor keypoint (visible in the
    donor) that would leave the crop in a target-valid camera."""
    crops = tgt["crops"]; T, C, H, W, _ = crops.shape
    assert T == 1, "copy-paste is T=1 only (P3a)"
    tv, sv = tgt["cam_valid"][0], src["cam_valid"][0]
    if np.any(tv & ~sv):
        return None
    shifts = view_shifts(tgt["M"], D, src["t_local"][0], tgt["t_local"][0])         # (C,2)
    kp2d_d = src["kp2d"][0, 0] + shifts[:, None, :]                                  # (C,K,2)
    vis_d = src["vis2d"][0, 0].copy()                                                # (C,K)
    inside = (kp2d_d >= 0).all(-1) & (kp2d_d[..., 0] <= W - 1) & (kp2d_d[..., 1] <= H - 1)
    if np.any(vis_d & ~inside & tv[:, None]):
        return None
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in tgt.items()}
    lo, hi = params.gain_clip
    for c in range(C):
        if not tv[c]:
            continue
        m = _translate(src["prompt_mask"][0, c].astype(np.uint8), shifts[c], nearest=True).astype(bool)
        if not m.any():
            continue
        donor = _translate(src["crops"][0, c], shifts[c])
        gain = float(np.clip((np.median(crops[0, c]) + 1.0) / (np.median(src["crops"][0, c]) + 1.0), lo, hi))
        donor = np.clip(donor.astype(np.float32) * gain, 0, 255)
        alpha = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)[..., None]          # 1-px feather
        out["crops"][0, c] = (crops[0, c] * (1 - alpha) + donor * alpha).astype(np.uint8)
        # donor on top: host keypoints under it are occluded; host prompt loses those pixels
        hk = np.round(tgt["kp2d"][0, 0, c]).astype(int)
        ok = (hk[:, 0] >= 0) & (hk[:, 0] < W) & (hk[:, 1] >= 0) & (hk[:, 1] < H)
        covered = np.zeros(hk.shape[0], bool); covered[ok] = m[hk[ok, 1], hk[ok, 0]]
        out["vis2d"][0, 0, c] &= ~covered
        out["prompt_mask"][0, c] &= ~m
    out["kp2d"][1, 0] = kp2d_d.astype(np.float32)
    out["vis2d"][1, 0] = vis_d & inside & tv[:, None]
    out["kp3d_local"][1, 0] = (src["kp3d_local"][0, 0] + D) * src["has3d"][0, 0][:, None]
    out["has3d"][1, 0] = src["has3d"][0, 0]
    out["fly_valid"] = np.array([True, True])
    out["fly_sex"] = np.array([tgt["fly_sex"][0], src["fly_sex"][0]], np.int8)
    out["unlabelled_sex"] = np.int8(SEX_UNKNOWN)
    return out
```

(The fixture's donor images are white 24-px blobs on black with a mask only where the fixture wrote one — check `make_v12_root` writes masks; if `prompt_mask` is all-zero in the fixture, extend the fixture to write a `masks/` entry the same way `_load_mask` reads it, following how `test_mv_augment.py`/`test_v12_windows.py` already obtain a non-empty prompt mask. If they don't, add to the fixture a per-annotation mask npz using the same loader path `_load_mask(root, file_name, src_ann_id, ann_id, w, h)` — read that function first to learn the file layout.)

- [ ] **Step 4: Run**

Run: `JAX_PLATFORMS=cpu pytest tests/test_mv_copy_paste.py -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add jarvis_jax/data/mv_copy_paste.py tests/test_mv_copy_paste.py tests/mvq_fixtures.py
git commit -m "feat(mvq): affine-exact multi-view copy-paste compositor (P3a §6)"
```

---

### Task 6: Loader hook — donors, `paste_window`, `copy_paste_p`

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py`
- Modify: `third_party/jarvis_jax/tests/test_v12_windows.py`

**Interfaces:**
- Produces: `V12WindowDataset(..., copy_paste: CopyPasteParams | None = None)`; `ds.paste_window(i, rng) -> (sample, info) | None` with `info = {"donor": j, "D": D, "sep": float, "contact": bool}`; `__getitem__` applies it with probability `params.p` when `train and T == 1 and n_flies(i) == 1 and unlabelled_sex(i) == -1`.

- [ ] **Step 1: Tests**

```python
def test_copy_paste_hook_is_deterministic_and_skips_unlabelled_present(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams
    root = make_v12_root(tmp_path, n_frames=3, two_fly_frame=1)
    # manifest says 2 flies; frames 0 and 2 have only fly0 labelled -> unlabelled_sex = male -> never pasted
    ds = V12WindowDataset(root, "train", T=1, train=True, copy_paste=CopyPasteParams(p=1.0))
    i = ds.windows.index((REC, 0, 0))
    assert ds[i]["fly_valid"].tolist() == [True, False]
    # rewrite the manifest to n_flies=1 so those windows become fully labelled donors/targets
    import json, os
    man = json.load(open(os.path.join(root, "manifest.json"))); man["recordings"][REC]["n_flies"] = 1
    man["recordings"][REC]["fly_sex"] = {"fly0": "female"}
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"))
    ds = V12WindowDataset(root, "train", T=1, train=True, copy_paste=CopyPasteParams(p=1.0, max_tries=20))
    s1 = ds[i]; s2 = ds[i]
    assert s1["fly_valid"].tolist() == [True, True]
    np.testing.assert_array_equal(s1["crops"], s2["crops"])            # same (seed, i, epoch) -> same paste
    ds.epoch = 1
    s3 = ds[i]
    assert not np.array_equal(s1["kp2d"][1], s3["kp2d"][1])           # a new epoch draws a new paste
    r = ds.paste_window(i, np.random.default_rng(0))
    assert r is not None and set(r[1]) == {"donor", "D", "sep", "contact"}
    val = V12WindowDataset(root, "val", T=1, train=False, copy_paste=CopyPasteParams(p=1.0))
    assert val[i]["fly_valid"].tolist() == [True, False]               # never in eval mode
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py -q -k copy_paste` -> TypeError (unexpected kwarg).

- [ ] **Step 3: Implement**

`__init__` gains `copy_paste=None`, stores `self.copy_paste = copy_paste`, and builds the donor index when set (train split only is guaranteed by the caller; still, exclude nothing else):

```python
        self._donors = {}
        if self.copy_paste is not None and self.T == 1:
            for i, (rec, fly, _) in enumerate(self.windows):
                self._donors.setdefault((self.manifest[rec]["calib_group"], self.fly_sex_code(rec, fly)), []).append(i)
```

Rename the current body of `__getitem__` to `_build(self, i)` (unchanged), and add:

```python
    def paste_window(self, i, rng):
        """Copy-paste per spec §6; None when no donor fits after max_tries."""
        from jarvis_jax.data.mv_copy_paste import body_plane_axes, composite, sample_offset
        p = self.copy_paste
        rec, host, _ = self.windows[i]
        grp = self.manifest[rec]["calib_group"]; host_sex = self.fly_sex_code(rec, host)
        tgt = self._build(i)
        axes = body_plane_axes(tgt["kp3d_local"][0, 0], tgt["has3d"][0, 0])
        for _ in range(p.max_tries):
            want = (1 - host_sex) if (host_sex in (0, 1) and rng.uniform() < p.opposite_sex_p) else host_sex
            pool = self._donors.get((grp, want)) or self._donors.get((grp, host_sex)) or []
            pool = [j for j in pool if j != i]
            if not pool:
                return None
            j = int(pool[rng.integers(len(pool))])
            D = sample_offset(rng, axes, p)
            out = composite(tgt, self._build(j), D, p)
            if out is not None:
                sep = float(np.linalg.norm(D))
                return out, {"donor": j, "D": D, "sep": sep, "contact": sep <= p.contact_sep[1]}
        return None

    def __getitem__(self, i):
        p = self.copy_paste
        if (p is not None and self.train and self.T == 1 and self.n_flies(i) == 1
                and self.unlabelled_sex(i) == SEX_UNKNOWN):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(i), int(self.epoch), 7]))
            if rng.uniform() < p.p:
                r = self.paste_window(i, rng)
                if r is not None:
                    return r[0]
        return self._build(i)
```

(`n_flies(i) == 1` means one LABELLED fly; a donor window with two labelled flies is fine as a donor because only its host mask is pasted.)

- [ ] **Step 4: Run** — `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py tests/test_mv_copy_paste.py -q` -> all pass.

- [ ] **Step 5: Commit**

```bash
git add jarvis_jax/data/v12_windows.py tests/test_v12_windows.py
git commit -m "feat(mvq): copy-paste hook in the window loader with per-group donor index (P3a §6)"
```

---

### Task 7: Trainer plumbing — config, run-dir json, warm start

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` (`MVQTrainConfig`, `run_training`)
- Modify: `third_party/jarvis_jax/jarvis_jax/train/checkpoint.py` (`warm_start_partial`)
- Modify: `third_party/jarvis_jax/jarvis_jax/models/mvq/checkpoint.py` (run-dir json)
- Modify: `third_party/jarvis_jax/configs/train/mvq.yaml`
- Create: `third_party/jarvis_jax/tests/test_checkpoint_warm_start_partial.py`
- Modify: `third_party/jarvis_jax/tests/test_train_mvq_smoke.py`

**Interfaces:**
- Produces: `MVQTrainConfig.copy_paste_p: float = 0.0`, `copy_paste_opposite_sex_p: float = 0.7`, `copy_paste_contact_p: float = 0.3`, `warm_start: str | None = None`; `warm_start_partial(model, src_dir) -> (model, skipped: list[str])`; `<run_dir>/mvq_run.json` written before step 0; `load_mvq_model(run_dir, step=...)` reads it from the run dir first.

- [ ] **Step 1: Tests**

```python
# tests/test_checkpoint_warm_start_partial.py
import numpy as np, jax, orbax.checkpoint as ocp
from flax import nnx


def _tiny(n_inst):
    from jarvis_jax.models.mvq import MVQConfig, MVQModel
    cfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=n_inst,
                    n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                    refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                    backbone_heads=4, remat=False)
    return MVQModel(cfg, rngs=nnx.Rngs(0))


def test_warm_start_partial_copies_matching_leaves_and_reports_the_rest(tmp_path):
    from jarvis_jax.train.checkpoint import warm_start_partial
    src = _tiny(3)
    # make the source distinguishable from a fresh init
    src.decoder.e_inst.value = src.decoder.e_inst.value + 1.0
    src.decoder.heads.exist.kernel.value = src.decoder.heads.exist.kernel.value + 2.0
    ck = ocp.StandardCheckpointer(); ck.save(str(tmp_path / "final"), nnx.split(src)[1]); ck.wait_until_finished()
    dst = _tiny(4)
    fresh_row3 = np.asarray(dst.decoder.e_inst.value[3]).copy()
    fresh_sex = np.asarray(dst.decoder.heads.sex.kernel.value).copy()
    dst, skipped = warm_start_partial(dst, str(tmp_path / "final"))
    np.testing.assert_allclose(np.asarray(dst.decoder.heads.exist.kernel.value), np.asarray(src.decoder.heads.exist.kernel.value))
    np.testing.assert_allclose(np.asarray(dst.decoder.e_inst.value[:3]), np.asarray(src.decoder.e_inst.value))
    np.testing.assert_allclose(np.asarray(dst.decoder.e_inst.value[3]), fresh_row3)
    np.testing.assert_allclose(np.asarray(dst.decoder.heads.sex.kernel.value), fresh_sex)
    assert sorted(skipped) == sorted(["decoder/e_inst (partial rows 0:3)", "decoder/heads/sex/bias", "decoder/heads/sex/kernel"])
```

Append to `tests/test_train_mvq_smoke.py`:

```python
def test_run_dir_json_written_at_start_and_loader_prefers_it(tmp_path):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.models.mvq.checkpoint import load_mvq_model
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    run = tmp_path / "run"
    run_training(root, out_dir=str(run / "final"), ckpt_dir=str(run / "ckpt"), mcfg=mcfg, tcfg=tcfg,
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    meta = json.load(open(run / "mvq_run.json"))
    assert meta["val"] is None and meta["model"]["n_instances"] == 4 and len(meta["keypoint_names"]) == 50
    # a mid-run checkpoint loads with NO final/ present
    import shutil; shutil.rmtree(run / "final")
    m, meta2 = load_mvq_model(str(run), step="latest")
    assert meta2["model"] == meta["model"]


def test_warm_start_config_seeds_step0_from_another_run(tmp_path):
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    base = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1,), smoke=True)
    a = tmp_path / "a"
    run_training(root, out_dir=str(a / "final"), ckpt_dir=str(a / "ckpt"), mcfg=mcfg, tcfg=base,
                 aug=MVAugParams(enabled=False), weights=LossWeights())
    b = tmp_path / "b"
    res = run_training(root, out_dir=str(b / "final"), ckpt_dir=str(b / "ckpt"), mcfg=mcfg,
                       tcfg=dataclasses.replace(base, warm_start=str(a / "final")),
                       aug=MVAugParams(enabled=False), weights=LossWeights())
    assert res["resumed_from"] == 0 and np.isfinite(res["final_loss"])
```

(add `import json, dataclasses` at the top of the smoke test file.)

- [ ] **Step 2: Run to verify failures** — both new test files fail (ImportError / TypeError).

- [ ] **Step 3: Implement**

`train/checkpoint.py`:

```python
def warm_start_partial(model, src_dir):
    """Shape-tolerant warm start (P3a spec §8): restore every leaf of `src_dir`
    (a `final/` written by StandardCheckpointer) whose path AND shape match
    `model`'s state; for `decoder/e_inst` with fewer source rows copy the
    leading rows; leave everything else at its fresh init. Returns the new
    model and the list of leaves NOT fully restored (human-readable paths)."""
    gdef, state = nnx.split(model)
    pure = nnx.to_pure_dict(state)
    repl = NamedSharding(Mesh(jax.devices(), axis_names=("data",)), P())
    ck = ocp.StandardCheckpointer()
    meta = ck.metadata(os.path.abspath(src_dir)).item_metadata.tree
    is_arr = lambda x: hasattr(x, "shape") and hasattr(x, "dtype")
    target = jax.tree_util.tree_map(
        lambda m: jax.ShapeDtypeStruct(tuple(m.shape), m.dtype, sharding=repl) if is_arr(m) else m, meta, is_leaf=is_arr)
    src = ck.restore(os.path.abspath(src_dir), target=target)
    src_flat = {jax.tree_util.keystr(k): v for k, v in jax.tree_util.tree_flatten_with_path(src)[0]}
    skipped = []

    def merge(path, v):
        key = jax.tree_util.keystr(path)
        name = key.replace("['", "/").replace("']", "").strip("/")
        s = src_flat.get(key)
        if s is None or not hasattr(v, "shape"):
            skipped.append(name); return v
        if tuple(s.shape) == tuple(v.shape):
            return jnp.asarray(s, v.dtype)
        if name.endswith("e_inst") and s.shape[1:] == v.shape[1:] and s.shape[0] < v.shape[0]:
            skipped.append(f"{name} (partial rows 0:{s.shape[0]})")
            return v.at[: s.shape[0]].set(jnp.asarray(s, v.dtype))
        skipped.append(name); return v

    new_pure = jax.tree_util.tree_map_with_path(merge, pure)
    nnx.replace_by_pure_dict(state, new_pure)
    return nnx.merge(gdef, state), skipped
```

(Verify `nnx.to_pure_dict` / `nnx.replace_by_pure_dict` exist in the installed flax: `python -c "from flax import nnx; print(nnx.to_pure_dict, nnx.replace_by_pure_dict)"`. If the keystr formats of the two flattened trees differ — e.g. the source dict has `['decoder']['e_inst']` while the pure state has the same — print both key sets once while developing and normalise. The test pins the exact skipped names, so normalise to `a/b/c` form as above. Add `import jax.numpy as jnp` to the module.)

`train_mvq.py`:
- `MVQTrainConfig`: add `copy_paste_p: float = 0.0`, `copy_paste_opposite_sex_p: float = 0.7`, `copy_paste_contact_p: float = 0.3`, `warm_start: str | None = None`.
- `run_training`: build train sets with `copy_paste=CopyPasteParams(p=tcfg.copy_paste_p, opposite_sex_p=tcfg.copy_paste_opposite_sex_p, contact_p=tcfg.copy_paste_contact_p) if tcfg.copy_paste_p > 0 else None`.
- Right after the cohort check (before the model build) write the run-dir json:

```python
    run_dir = os.path.dirname(os.path.abspath(out_dir))
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "mvq_run.json"), "w") as f:
        json.dump({"model": dataclasses.asdict(mcfg), "train": dataclasses.asdict(tcfg), "val": None,
                   "keypoint_names": names}, f, indent=1)
```

- After the pretrained load and BEFORE `make_optimizer`: `if tcfg.warm_start: model, skipped = warm_start_partial(model, tcfg.warm_start); print(f"[mvq] warm start from {tcfg.warm_start}; not restored: {skipped}", flush=True)`. (Import `warm_start_partial` from `jarvis_jax.train.checkpoint`.) Resume from the run's own `ckpt/` still happens afterwards and overrides it, per the docstring rule.

`models/mvq/checkpoint.py` `load_mvq_model`: when `step is not None`, look for `os.path.join(run_dir_or_final, "mvq_run.json")` first and fall back to `final/mvq_run.json`; update the docstring paragraph that says `final/mvq_run.json` is REQUIRED.

`configs/train/mvq.yaml`: add `copy_paste_p: 0.0`, `copy_paste_opposite_sex_p: 0.7`, `copy_paste_contact_p: 0.3`, `warm_start: null` (with one comment line each referencing P3a §6/§8).

`scripts/train_mvq.py`'s dataclass filter already forwards any key in `MVQTrainConfig.__dataclass_fields__`; no change.

- [ ] **Step 4: Run** — `JAX_PLATFORMS=cpu pytest tests/test_checkpoint_warm_start_partial.py tests/test_train_mvq_smoke.py -q` -> all pass.

- [ ] **Step 5: Commit**

```bash
git add jarvis_jax/train/train_mvq.py jarvis_jax/train/checkpoint.py jarvis_jax/models/mvq/checkpoint.py configs/train/mvq.yaml tests/test_checkpoint_warm_start_partial.py tests/test_train_mvq_smoke.py
git commit -m "feat(mvq): copy-paste/warm-start config, run-dir mvq_run.json at start, shape-tolerant warm start (P3a §5, §8)"
```

---

### Task 8: Evaluate — typed-slot policy, per-slot existence, sex_acc, mask containment, contact_pair

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` (`evaluate`, `_cohorts`)
- Modify: `third_party/jarvis_jax/tests/test_train_mvq_smoke.py`

**Interfaces:**
- Produces per mode in `evaluate`'s result: `exist_prec_slot{0..3}`, `exist_rec_slot{0..3}`, `sex_acc`, `mask_containment`, `cohort_contact_pair`, plus the existing keys. `policy_miss_frac` now counts unprompted windows where no typed slot (1-3) clears 0.5.

- [ ] **Step 1: Test** (append to the smoke file; extend the assertion set in `test_two_steps_cpu_and_eval` instead of a new run to keep CPU time down):

```python
        assert {"sex_acc", "mask_containment", "cohort_contact_pair"} <= set(v)
        assert {f"exist_prec_slot{i}" for i in range(4)} | {f"exist_rec_slot{i}" for i in range(4)} <= set(v)
        assert np.isnan(v["mask_containment"]) or 0.0 <= v["mask_containment"] <= 1.0
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_train_mvq_smoke.py -q -k two_steps` -> KeyError.

- [ ] **Step 3: Implement**

In `evaluate`, per batch after the forward: compute the label-driven targets on the host

```python
            from jarvis_jax.train.matching import assign_slots, slot_ignore, SLOT_OTHER
            has_f = b["has3d"].astype(np.float32)
            cen = (b["kp3d_local"] * has_f[..., None]).sum((2, 3)) / np.maximum(has_f.sum((2, 3)), 1.0)[..., None]
            dist = np.linalg.norm(cen, axis=-1)
            assign, slot_t = assign_slots(jnp.asarray(b["fly_sex"]), jnp.asarray(b["fly_valid"]), on, jnp.asarray(dist), I)
            assign, slot_t = np.asarray(assign), np.asarray(slot_t)
            ignore = np.asarray(slot_ignore(jnp.asarray(b["unlabelled_sex"]), I))
            sex_logit = np.asarray(out["sex_logit"])
```

(`I = xyz.shape[1]`.) Per real sample `bi`:

```python
                exist = exist_probs >= 0.5
                # POLICY (P3a §7): prompted -> slot 0 when this window has a usable mask, else the
                # unprompted rule; unprompted -> among TYPED slots 1..3 that exist, nearest centroid to the ROI origin
                use_prompt = prompted and bool(np.asarray(has_mask)[bi])
                if use_prompt:
                    inst_policy = 0
                else:
                    cand = np.array([s for s in range(1, I) if exist[s]])
                    inst_policy = (int(cand[np.argmin(np.linalg.norm(xyz[bi, cand].mean(axis=(1, 2)), axis=-1))])
                                   if cand.size else None)
                # per-slot existence bookkeeping (non-ignored slots), sex accuracy on assigned known-sex slots
                for s in range(I):
                    if ignore[bi, s]:
                        continue
                    slot_counts[mode][s] += np.array([exist[s] and slot_t[bi, s], exist[s] and not slot_t[bi, s],
                                                      (not exist[s]) and slot_t[bi, s]], int)     # TP, FP, FN
                for f in range(b["fly_valid"].shape[1]):
                    if b["fly_valid"][bi, f] and b["fly_sex"][bi, f] >= 0 and assign[bi, f] >= 0:
                        sex_hits[mode].append(int((sex_logit[bi, assign[bi, f]] > 0) == (b["fly_sex"][bi, f] == 0)))
                # mask containment of the policy instance's reprojection inside the HOST mask
                contain = np.nan
                if inst_policy is not None:
                    uv = np.asarray(project_local(jnp.asarray(xyz[bi, inst_policy, 0]), jnp.asarray(b["M"][bi]), jnp.asarray(b["t_local"][bi, 0])))  # (K,C,2)
                    pm = b["prompt_mask"][bi, 0]; hits = []
                    for c in range(pm.shape[0]):
                        if not b["cam_valid"][bi, 0, c] or not pm[c].any():
                            continue
                        vis = b["vis2d"][bi, 0, 0, c]
                        p = np.round(uv[vis, c]).astype(int)
                        okp = (p[:, 0] >= 0) & (p[:, 0] < pm.shape[2]) & (p[:, 1] >= 0) & (p[:, 1] < pm.shape[1])
                        inside = np.zeros(len(p), bool); inside[okp] = pm[c][p[okp, 1], p[okp, 0]]
                        hits.extend(inside.tolist())
                    contain = float(np.mean(hits)) if hits else np.nan
```

Initialise `slot_counts = {m: np.zeros((I_max, 3), int) ...}` lazily on the first batch (I known then), `sex_hits = {"prompted": [], "unprompted": []}`, and append `contain` as a new trailing field of the `per_sample` row (document the row layout comment). In `_finish`: `res[f"exist_prec_slot{s}"] = TP/max(TP+FP,1)`, `res[f"exist_rec_slot{s}"] = TP/max(TP+FN,1)` (nan when TP+FN == 0), `res["sex_acc"] = mean(sex_hits) or nan`, `res["mask_containment"] = nanmean of the per-sample field` (nan if all nan). Keep the old `exist_prec`/`exist_rec` keys as they are (they are logged historically) but base them on the same per-slot counts summed over slots 1..3.

`project_local` is imported at module top (`from jarvis_jax.models.mvq.geometry import project_local`) if not already.

`_cohorts(ds)`: add

```python
    def _contact(i):
        if ds.n_flies(i) < 2:
            return False
        c = ds.fly_centroids(i)
        return bool(np.isfinite(c).all() and np.linalg.norm(c[0] - c[1]) < 15.0)
    c["contact_pair"] = np.array([_contact(i) for i in range(n)])
```

- [ ] **Step 4: Run** — `JAX_PLATFORMS=cpu pytest tests/test_train_mvq_smoke.py -q` -> all pass. Then the whole mvq CPU suite: `JAX_PLATFORMS=cpu pytest tests/test_dinov3.py tests/test_mvq_*.py tests/test_v12_windows.py tests/test_mv_augment.py tests/test_mv_copy_paste.py tests/test_train_mvq_smoke.py tests/test_checkpoint_warm_start_partial.py -q` -> all pass (count it; expect ~80).

- [ ] **Step 5: Commit**

```bash
git add jarvis_jax/train/train_mvq.py tests/test_train_mvq_smoke.py
git commit -m "feat(mvq): typed-slot policy, per-slot existence, sex_acc, mask containment, contact_pair cohort (P3a §7)"
```

---

### Task 9: Pre-launch figure gates — sex labels, copy-paste, overlay contact case

**Files:**
- Create: `scripts/viz/mvq_sex_label_check.py` (repo root)
- Create: `scripts/viz/mvq_copy_paste_check.py`
- Modify: `scripts/viz/mvq_overlay.py` (add `contact_pair` case; row label shows the oracle slot index and `sex_prob` of that slot)
- Outputs: `figures/2026-09-mvq/p3a_gates/{sex_label_check.png, copy_paste_check.png}` (+ `.json` beside each)

Both scripts are CPU-only (labels drawn on decoded crops; the copy-paste one builds samples through the loader). They may run on this node (a few minutes, 4 threads) — set `OMP_NUM_THREADS=4`.

- [ ] **Step 1: `mvq_sex_label_check.py`**

Docstring (verbatim expectation): *"EXPECTATION: for each mixed recording, the fly the manifest calls MALE (orange) is the smaller body with the dark abdomen tip; the FEMALE (cyan) is larger with a pointed, pale-striped abdomen. In 2025_10_20_13_20_04 only the female is labelled: the unlabelled fly in the crop must be the smaller, darker one. If any recording shows the reverse, its `fly_sex` convention is wrong and the export must be fixed before training."*

Implementation sketch: for each of `["2025_10_20_13_20_04", "2026_04_02_17_28_34"]` (train) and `["2026_04_02_12_11_50", "2026_04_02_15_25_51"]` (val): `V12WindowDataset(root, split, T=1, train=False, recordings=[rec])`, pick 3 host-fly0 windows spread over the index, take the two cameras with the largest labelled-keypoint spread (by name via `ds.camera_names(i)`), draw fly 0 in cyan and fly 1 in orange (`viz.core.colors.PALETTE["fly0"]/["fly1"]`, BGR -> RGB), title `f"{rec} {cam}  fly0={sex0}  fly1={sex1 or 'unlabelled'}"`. 4 recordings x 3 frames rows, 2 cams columns.

- [ ] **Step 2: `mvq_copy_paste_check.py`**

Docstring: *"EXPECTATION: in every one of the 7 cameras the pasted donor (orange labels) sits at the same place relative to the host (cyan) — never floating, offset, or missing in one view — its labels lie on its own body, host keypoints under the donor are drawn hollow (occluded), and the contact rows (sep <= 15 units) show the two bodies touching or overlapping like a mounting pair."* Build `V12WindowDataset(root, "train", T=1, train=True, copy_paste=CopyPasteParams(p=1.0, max_tries=30))`, call `ds.paste_window(i, rng)` over random single-fly windows until 3 contact and 3 far pastes are collected (at least one same-sex), render 6 rows x 7 cams, camera names in titles, sep and sexes in the row label, save the info dicts as JSON beside the PNG.

- [ ] **Step 3: overlay `contact_pair` case** — in `mvq_overlay.py` add `elif case == "contact_pair": sel = [r for r in rows if r["contact"]]` where `r["contact"]` is computed from `ds.fly_centroids(i)` with the same 15-unit rule; add `slot` (oracle instance index) and `sex_prob` (`sigmoid(out["sex_logit"][0, inst])`) to each row dict and to the y-label: `f"#{i} {F/M} grp{g}\nslot{inst} pF={sex_prob:.2f}\n{mm:.2f}mm"`. Default `--cases female,two_fly,contact_pair,worst`.

- [ ] **Step 4: Run both gate scripts, read the PNGs with the Read tool, record what you saw** (against the expectations) in `docs/benchmark/2026-09-mvq/p3a-notes.md` (new file, section "Pre-launch gates"). If the sex-label gate shows a reversed recording: STOP and report — do not launch.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/mvq_sex_label_check.py scripts/viz/mvq_copy_paste_check.py scripts/viz/mvq_overlay.py
git add -f docs/benchmark/2026-09-mvq/p3a-notes.md
git commit -m "viz(mvq): P3a pre-launch gates (sex-label check, copy-paste check) and contact_pair overlay case"
```

---

### Task 10: Launch the warm-started P3a run on ckpt-all

**Files:**
- Modify: `docs/benchmark/2026-09-mvq/p3a-notes.md` (launch record)

Preconditions: Tasks 1-9 committed; the 30k run's `final/` exists (`ls /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_local8_20260904/final/mvq_run.json`). If it does not yet, wait for it (the run finishes ~11:15 on 2026-09-04) — do not warm-start from a `ckpt/` step (its EMA is a raw sum; `final/` is the debiased EMA).

- [ ] **Step 1: Submit**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
SRC=/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_local8_20260904/final
scripts/slurm/submit_task.sh --gpus 4 --cpus 32 --mem 128 --time 12:00:00 mvq_p3a \
  "export LD_PRELOAD=\$CONDA_PREFIX/lib/libstdc++.so.6 HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN= XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async && cd third_party/jarvis_jax && python -m jarvis_jax.scripts.train_mvq model=mvq train=mvq paths=hyak run_id=mvq_t1_b16_p3a_20260904 train.total_steps=10000 train.warmup_steps=500 train.lr=1e-4 train.prompt_p_start=0.5 train.prompt_p_end=0.5 train.copy_paste_p=0.5 train.warm_start=$SRC train.eval_every=2000 train.save_every=1000 train.batch_size=32 train.num_workers=24 \"paths.runs_root=\\\${paths.mvq_runs_root}\""
```

(`submit_task.sh` sets partition ckpt-all, constraint `h200|a100|l40s|l40|a40`, `--requeue`, `module load cuda`, `unset LD_LIBRARY_PATH JAX_PLATFORMS`. The a100 pool includes 40 GB cards; 8 samples/GPU needs ~41 GB on an L40S. If the job dies at startup with a CUDA OOM on an a100 node twice, resubmit with a copy of the script whose constraint is `h200|l40s|l40|a40`.)

- [ ] **Step 2: Verify the first 200 steps** in `slurm_logs/mvq_p3a-<jobid>.out`: the warm-start line lists exactly `decoder/e_inst (partial rows 0:3)` and the two `heads/sex` leaves; `resume @` is absent on the first launch; steps log `sex_acc` and `exist_acc`; record s/step next to the 30k run's 1.06 s/step (8 GPUs) / gate-1's 4-GPU number in the notes.

- [ ] **Step 3: Step-2000 val** — record `exist_prec_slot1/2`, `exist_rec_slot1/2`, `sex_acc`, `mask_containment`, `cohort_contact_pair`, `policy_miss_frac` for both modes in the notes.

- [ ] **Step 4: Commit the notes** (`git add -f docs/benchmark/2026-09-mvq/p3a-notes.md`).

The step-10k overlay gate (spec §7 item 3), the acceptance table (§8) and the same-joint DLT comparison are run when the job finishes (or at the 5k fallback check) — via `scripts/slurm/submit_task.sh` as for the step-14000 render, using `--step` with the run dir (no staging dir needed now).

---

## Self-review (done while writing)

- Spec §3 -> Task 1; §4 -> Tasks 2-3; §5 -> Tasks 4, 7; §6 -> Tasks 5-6; §7 -> Tasks 8-9; §8 -> Tasks 7, 10; §9 tests are spread over each task; §10 placement matches the file map.
- Names used across tasks: `assign_slots`, `slot_ignore`, `SEX_*`/`SLOT_*`/`N_SLOTS` (Task 1) are what Tasks 3, 6, 8 import; `fly_sex`/`unlabelled_sex`/`fly_centroids`/`paste_window` (Tasks 2, 6) are what Tasks 3, 8, 9 use; `CopyPasteParams` fields (Task 5) match the trainer config names in Task 7; `sex_logit`/`sex_prob` (Task 4) match Tasks 3 and 8.
- Known judgement calls left to the implementer, each flagged in its task: the fixture's mask path (Task 5), flax pure-dict helper names (Task 7), pass-through of new keys in `mv_augment` (Task 4).
