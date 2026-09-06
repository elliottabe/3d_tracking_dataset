# MVQ v2 — Plan B: T=2 training from scratch, calibration and acceptance

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train `mvq_t2_v2_20260905` FROM SCRATCH on T=2 windows drawn from the real v12 labels (weight 1.0) plus Plan A's pseudo export (weight 0.3) plus empty-window negatives: variable spacing Delta in {1, 4, 16}, copy-paste extended to two frames, an identity-persistence loss, existence negatives, doubled wing weights, camera dropout and flip, then temperature-calibrate the existence and visibility heads and run one acceptance script that writes the spec §5 scorecard.

**Architecture:** The loader keeps ONE `V12WindowDataset` per root and a thin `ConcatWindowDataset` mixes them, so cohorts, sampling weights and the copy-paste donor index stay per-root; T=2 becomes "labelled at both frames, spacing Delta" (a per-window `delta`) instead of "T consecutive labelled frames". The model is already T-aware (`crops (B,T,C,448,448,3)`, `e_frame[:T]`, per-T decoder readouts) — nothing in `models/mvq/` changes. The loss gains three things that only exist with T=2 or with pseudo data: assignment computed on FRAME 0 and reused on frame 1 plus a displacement hinge, a per-sample weight, and a per-keypoint weight vector. Acceptance is one script that runs every §5 row and writes a single scorecard.

**Tech Stack:** JAX 0.11 / Flax NNX 0.12.x, optax, orbax 0.11, numpy, OpenCV, matplotlib (Agg), Hydra; pytest from `third_party/jarvis_jax/` with `JAX_PLATFORMS=cpu`.

**Spec:** `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` §4 (model and training), §5 (acceptance), §6 items 4-9. Plan A (`docs/plans/2026-09-05-mvq-v2-plan-a-pseudolabels.md`) must be complete through its Task 6 before Task 1 here can run on real data; Tasks 1-4 can be built and unit-tested against fixtures without it.

## Global Constraints

- Package root `third_party/jarvis_jax/jarvis_jax/`; tests in `third_party/jarvis_jax/tests/` (bare `from mvq_fixtures import ...`); run pytest from `third_party/jarvis_jax/` with `JAX_PLATFORMS=cpu`. The whole mvq CPU suite must keep passing: `pytest tests/test_dinov3.py tests/test_mvq_*.py tests/test_v12_windows.py tests/test_mv_augment.py tests/test_mv_copy_paste.py tests/test_train_mvq_smoke.py tests/test_pseudo_*.py -q`.
- **T=2, Delta in {1, 4, 16}** sampled per example; real labelled pairs exist only at Delta = 1; real singles train as T=1 (`window_lengths: [1, 2]` alternates by step, as `run_training` already does).
- **Sample weights: real 1.0, pseudo 0.3** (spec §3.2), carried per window by Plan A Task 6's `sample_weight` and applied as a weight on every per-sample loss term.
- Gate thresholds that produced the pseudo set are FROZEN in its manifest (`exist >= 0.8`, step `<= 3.0 units` = 0.3 mm, reproj `<= 3 px` in `>= 5` views, containment `>= 0.9` in `>= 5` cameras, decorrelation 16 frames); the trainer reads them only to log them into `mvq_run.json`.
- **Batch 32 across 8 devices** (4/device). `run_training` raises unless `batch_size % len(jax.devices()) == 0`; the 7-device layout hung, so pin `CUDA_VISIBLE_DEVICES=0,…,7` and assert 8. Fallback on OOM: batch 16 with gradient accumulation 2 (spec §7).
- Recipe (spec §4): prompting OFF (`prompt_p_start=prompt_p_end=0`), jitter 10 units (1 mm), copy-paste p 0.8 / contact p 0.7 / contact sep (4, 25), female-host ratio 0.5 (`female_host_weight=4.27` on the real root — verify against the run log's realised ratio), cross-fly repulsion 20, camera dropout p 0.1, mirror p 0.5, wing keypoint and wing visibility weights x2, 40,000 steps, lr 3e-4, warmup 1,000, eval and save every 2,000.
- Acceptance thresholds (spec §5), all of which the Task 7 script asserts: val mpjpe `<= 0.085 mm`; policy misses 0; `sex_acc >= 0.99`; `contact_pair <= 0.146 mm`; `cross_fly_frac <= 0.036` overall and `<= 0.055` on contact pairs; `single_fly` cohort misses 0; centre shift at 1 mm within +10 % and misses `< 2 %`; masked bouts 1/4/28 (containment OFF) pose-jump 0 %, straddle `<= P3b`, female-missing `<= P3b`; mask-free bout 117 stride 1 pose-jump `<= 5 %` (P3b 28 %), straddle `<= 10 %` (P3b 23 %), no identity swap in the contact sheet; coarse pass 20_04 female trackable `>= 0.85` (jitter-10: 0.48), no `exist >= 0.5` on the stale-window bout 69 frames, gate bout 1 false peaks not asserted; single fly exactly one fly on `>= 99 %` of frames; existence reliability within 0.05 of the diagonal.
- **Never index a keypoint or camera axis by integer**: keypoint names from `ds.keypoint_names`, camera names from `ds.camera_names(i)` / `MVQRunner.cameras`; the L/R flip map is derived BY NAME and asserted against `configs/anatomy/v1.yaml` `model.KP_NAMES`.
- Figures under `figures/2026-09-mvq/v2_train/` (gitignored), JSON beside each PNG, expectation in the script docstring, **PNG read back with the Read tool before any claim**; notes and scorecards in `docs/benchmark/2026-09-mvq/v2-notes.md` and `docs/benchmark/2026-09-mvq/2026-09-05-mvq-v2/` (force-add).
- Commits: only the task's files, trailers `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01Npo4HiYC4t5M2xYjKUJsFP`.
- Compute: CPU tests here; every GPU step on an idle GPU node directly (`module load cuda/12.9.1; export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6; unset LD_LIBRARY_PATH JAX_PLATFORMS; export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN=`), else `scripts/slurm/submit_task.sh` (ckpt-all). Never on the login node.

## File map

| file | responsibility |
|---|---|
| `jarvis_jax/data/v12_windows.py` | per-window `delta`, `pair_deltas`, `window_index`, negatives, `window_batches` unchanged |
| `jarvis_jax/data/concat_windows.py` (new) | `ConcatWindowDataset` mixing the real, pseudo and negative roots |
| `jarvis_jax/data/mv_copy_paste.py` | `composite` over T frames (donor's own motion), same affine shift |
| `jarvis_jax/train/losses_mvq.py` | frame-0 assignment, `persist` term, `sample_weight`, `kp_weight`, wing multipliers |
| `jarvis_jax/train/train_mvq.py` | `pair_deltas`/`pseudo_root`/`pseudo_weight`/`negatives_root`/`wing_kp_mult` config + plumbing |
| `jarvis_jax/data/mv_augment.py` | T=2 correctness of dropout/mirror, `assert_lr_swap_covers` |
| `configs/train/mvq_v2.yaml` (new) | the §4 recipe |
| `scripts/benchmark/mvq_calibrate.py` (new) | temperature scaling -> `final/mvq_run.json`; applied by `MVQRunner` |
| `scripts/benchmark/mvq_perkp_bouts.py` (new) | per-keypoint bout metrics (promoted from the scratchpad) |
| `scripts/benchmark/mvq_v2_acceptance.py` (new) | the §5 scorecard |
| `configs/mvq/v2.yaml` (new), `scripts/slurm/mvq_p3a_campaign.sh` | promotion |
| tests | `test_v12_windows.py`, `test_concat_windows.py`, `test_mv_copy_paste.py`, `test_mvq_losses.py`, `test_mv_augment.py`, `test_train_mvq_smoke.py`, `test_mvq_calibrate.py`, `test_mvq_perkp.py` |

---

### Task 1: loader — T=2 with Delta, mixed roots, negatives

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py`
- Create: `third_party/jarvis_jax/jarvis_jax/data/concat_windows.py`
- Modify: `third_party/jarvis_jax/tests/test_v12_windows.py`; Create: `third_party/jarvis_jax/tests/test_concat_windows.py`

**Interfaces:**
- Produces: `V12WindowDataset(root, split, T=2, *, pair_deltas=(1,), …)`; `self.win_delta` (list, parallel to `self.windows`); `ds.window_index(rec, fly, f0, delta=None) -> int`; `ds.delta(i) -> int`; negatives (`fly_id == -1` framesets) yielding `fly_valid` all False, `has3d` all False, `unlabelled_sex == SEX_UNKNOWN`, `center3D` from the frameset, `is_negative` True.
- Produces: `ConcatWindowDataset(datasets, names=None)` with `__len__`, `__getitem__`, `epoch` (setter fans out), `keypoint_names`, `camera_names(i)`, `calib_group(i)`, `is_female(i)`, `n_flies(i)`, `fly_centroids(i)`, `source(i)`, `weight(i)`, `delta(i)`, `windows`, `which(i) -> (ds_index, local_index)`.
- Consumed by: Task 2 (`paste_window` runs per sub-dataset), Task 3 (`batch["sample_weight"]`, `fly_valid` all-False negatives), Task 5 (`run_training` builds the concat).

- [ ] **Step 1: Failing tests** (append to `tests/test_v12_windows.py`)

```python
def test_t2_windows_are_pairs_at_each_delta(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=6, two_fly_frame=1)
    ds = V12WindowDataset(root, "train", T=2, pair_deltas=(1, 4), train=False)
    got = {(w[0], w[1], w[2], d) for w, d in zip(ds.windows, ds.win_delta)}
    assert (REC, 0, 0, 1) in got and (REC, 0, 0, 4) in got
    assert (REC, 0, 3, 4) not in got            # frame 7 does not exist
    i = ds.window_index(REC, 0, 0, 4)
    s = ds[i]
    assert s["crops"].shape[0] == 2 and s["kp3d_local"].shape[1] == 2 and ds.delta(i) == 4
    # frame 1 of the pair really is frame 0+delta: its labels differ from frame 0's
    assert not np.allclose(s["kp3d_local"][0, 0], s["kp3d_local"][0, 1])
    # one origin per (sample, camera), shared by both frames (assemble() relies on it)
    np.testing.assert_array_equal(s["t_local"][0], s["t_local"][1])


def test_t1_still_means_single_frames(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=4)
    ds = V12WindowDataset(root, "train", T=1, train=False)
    assert all(d == 0 for d in ds.win_delta) and ds[0]["crops"].shape[0] == 1


def test_negative_frameset_has_no_flies_and_a_stored_centre(tmp_path):
    import json, os
    from jarvis_jax.data.v12_windows import V12WindowDataset
    from jarvis_jax.train.matching import SEX_UNKNOWN
    root = make_v12_root(tmp_path, n_frames=3)
    p = os.path.join(root, "annotations", "instances_train.json")
    coco = json.load(open(p))
    src = coco["framesets"][f"{REC}/Frame_0/fly0"]
    coco["framesets"][f"{REC}/Frame_0/neg0"] = {
        "recording": REC, "fly_id": -1, "negative": True, "center3D": [200.0, 50.0, 0.0],
        "subset": "negative", "frames": src["frames"], "ann_ids": src["ann_ids"]}
    json.dump(coco, open(p, "w"))
    ds = V12WindowDataset(root, "train", T=1, train=False)
    i = ds.window_index(REC, -1, 0)
    s = ds[i]
    assert s["fly_valid"].sum() == 0 and not s["has3d"].any() and bool(s["is_negative"])
    assert int(s["unlabelled_sex"]) == SEX_UNKNOWN
    np.testing.assert_allclose(s["center3D"], [200.0, 50.0, 0.0], atol=1e-4)
    assert s["crops"].shape == (1, 7, 448, 448, 3) and s["cam_valid"].any()
```

`tests/test_concat_windows.py`:

```python
import numpy as np
from mvq_fixtures import REC, make_v12_root


def test_concat_indexes_and_delegates(tmp_path):
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS, window_batches
    a = V12WindowDataset(make_v12_root(tmp_path / "a"), "train", T=1, train=False)
    b = V12WindowDataset(make_v12_root(tmp_path / "b"), "train", T=1, train=False)
    c = ConcatWindowDataset([a, b], names=["real", "pseudo"])
    assert len(c) == len(a) + len(b)
    assert c.which(len(a)) == (1, 0) and c.keypoint_names == a.keypoint_names
    np.testing.assert_array_equal(c[len(a)]["crops"], b[0]["crops"])
    assert c.source(len(a)) == b.source(0) and c.camera_names(0) == a.camera_names(0)
    c.epoch = 7
    assert a.epoch == 7 and b.epoch == 7
    batch = next(window_batches(c, 2, shuffle=False, num_workers=1, drop_last=False))
    assert set(batch) == set(WINDOW_KEYS) and batch["crops"].shape[0] == 2


def test_concat_refuses_mismatched_keypoint_orders(tmp_path):
    import json, os, pytest
    from jarvis_jax.data.concat_windows import ConcatWindowDataset
    from jarvis_jax.data.v12_windows import V12WindowDataset
    ra, rb = make_v12_root(tmp_path / "a"), make_v12_root(tmp_path / "b")
    names = json.load(open(os.path.join(rb, "annotations", "keypoint_names.json")))
    rev = list(reversed(names))
    json.dump(rev, open(os.path.join(rb, "annotations", "keypoint_names.json"), "w"))
    for split in ("train", "val"):
        p = os.path.join(rb, "annotations", f"instances_{split}.json")
        coco = json.load(open(p)); coco["keypoint_names"] = rev; json.dump(coco, open(p, "w"))
    with pytest.raises(ValueError, match="keypoint"):
        ConcatWindowDataset([V12WindowDataset(ra, "train", T=1, train=False),
                             V12WindowDataset(rb, "train", T=1, train=False)])
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py tests/test_concat_windows.py -q -k "t2 or negative or concat"` -> TypeError / ModuleNotFoundError.

- [ ] **Step 3: Implement the loader change.** In `__init__`, replace the window-building block with:

```python
        self.pair_deltas = tuple(int(d) for d in (pair_deltas or (1,)))
        self.windows, self.win_delta = [], []
        for (rec, frame, fly) in sorted(self._fs):
            if fly < 0:                                     # negative window: no partner needed
                self.windows.append((rec, fly, frame)); self.win_delta.append(0)
                continue
            if self.T == 1:
                self.windows.append((rec, fly, frame)); self.win_delta.append(0)
                continue
            for d in self.pair_deltas:                      # "labelled at both frames, spacing d"
                if all((rec, frame + k * d, fly) in self._fs for k in range(self.T)):
                    self.windows.append((rec, fly, frame)); self.win_delta.append(d)
        self._win_index = {(w[0], w[1], w[2], d): i
                           for i, (w, d) in enumerate(zip(self.windows, self.win_delta))}
```

and add

```python
    def delta(self, i):
        return int(self.win_delta[i])

    def window_index(self, rec, fly, f0, delta=None):
        """Index of ONE window BY KEY. `delta=None` matches whatever spacing
        that (rec, fly, f0) was built with (unique for T=1 and for negatives)."""
        if delta is not None:
            return self._win_index[(rec, int(fly), int(f0), int(delta))]
        hits = [i for (r, f, s, _), i in self._win_index.items() if (r, f, s) == (rec, int(fly), int(f0))]
        if len(hits) != 1:
            raise KeyError(f"{(rec, fly, f0)} matches {len(hits)} windows -- pass delta=")
        return hits[0]
```

In `_build`, `frames = [f0 + k * self.delta(i) for k in range(T)]` (unchanged for T=1), and a negatives branch right after `rec, host, f0 = self.windows[i]`:

```python
        negative = host < 0
        if negative:
            fsv = self._fs[(rec, f0, host)]
            center = np.asarray(fsv["center3D"], np.float64)     # no labels to derive one from
```

with `flies = []` (so `kp_full`/`X3d`/`has3d` stay zero), `fly_valid = np.zeros(F, bool)` (the `fly_valid[0] = True` line becomes `if not negative:`), `unlabelled_sex` forced to `SEX_UNKNOWN`, the prompt mask left all-zero, and `"is_negative": np.bool_(negative)` added to the returned dict and to `WINDOW_KEYS`. `cam_valid` for a negative comes from the frameset's own resolved slots exactly as for a labelled one.

- [ ] **Step 4: Implement `concat_windows.py`** — `ConcatWindowDataset.__init__` checks `keypoint_names` equality across datasets (ValueError naming both roots) and builds `self._offsets`; `which(i)` bisects; every delegating accessor is one line; `epoch` is a property whose setter writes through to each sub-dataset (`window_batches` sets `ds.epoch = seed`); `self.windows` is the concatenation (for logging and `_balanced_weights`); `manifest` is a merged dict (`{**a.manifest, **b.manifest}`) with a ValueError on a recording present in two roots with different `calib_group` — two roots that disagree about a recording's calibration would triangulate the same fly two ways.

- [ ] **Step 5: Run** — `JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py tests/test_concat_windows.py -q` -> all pass.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/data/v12_windows.py third_party/jarvis_jax/jarvis_jax/data/concat_windows.py \
        third_party/jarvis_jax/tests/test_v12_windows.py third_party/jarvis_jax/tests/test_concat_windows.py
git commit -m "feat(mvq-v2): T=2 windows at Delta in {1,4,16}, empty-window negatives, multi-root concat (spec §4)"
```

---

### Task 2: copy-paste for T=2

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/mv_copy_paste.py`
- Modify: `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` (`paste_window`'s T==1 guard)
- Modify: `third_party/jarvis_jax/tests/test_mv_copy_paste.py`

**Interfaces:**
- Produces: `composite(tgt, src, D, params)` accepting `T >= 1` — the donor is pasted into EVERY frame with the SAME 3D offset `D` (so the per-camera shift `view_shifts(M, D, t_src, t_tgt)` is computed once) but with its OWN frame-t pixels, mask and labels, i.e. the donor keeps its own two-frame motion. `paste_window(i, rng)` drops the `T == 1` restriction (both target and donor must have the same `T`; the donor's own delta may differ and is recorded in `info["donor_delta"]`).
- Consumed by: Task 5's config (`copy_paste_p: 0.8`) and Plan A's masks (a donor needs a real `prompt_mask`).

- [ ] **Step 1: Failing tests** (append to `tests/test_mv_copy_paste.py`)

```python
def _t2_samples(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=5, two_fly_frame=3, manifest_n_flies=1)
    ds = V12WindowDataset(root, "train", T=2, pair_deltas=(1,), train=False)
    return ds, ds[ds.window_index(REC, 0, 0, 1)], ds[ds.window_index(REC, 0, 2, 1)]


def test_composite_t2_is_the_t1_path_applied_per_frame(tmp_path):
    """Byte-equality guard (spec §7 risk 3): a T=2 window whose two frames are
    identical must composite to two frames identical to the T=1 result."""
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=4, manifest_n_flies=1)
    ds1 = V12WindowDataset(root, "train", T=1, train=False)
    tgt1, src1 = ds1[ds1.window_index(REC, 0, 0)], ds1[ds1.window_index(REC, 0, 2)]
    D = np.array([12.0, 0.0, 0.0], np.float32)
    out1 = composite(tgt1, src1, D, CopyPasteParams())
    dup = lambda s: {k: (np.concatenate([v, v], 0) if getattr(v, "ndim", 0) and k in
                         ("crops", "prompt_mask") else v) for k, v in s.items()}
    tgt2, src2 = _duplicate_frame_axis(tgt1), _duplicate_frame_axis(src1)   # helper in this file
    out2 = composite(tgt2, src2, D, CopyPasteParams())
    for k in ("crops", "prompt_mask"):
        np.testing.assert_array_equal(out2[k][0], out1[k][0])
        np.testing.assert_array_equal(out2[k][1], out1[k][0])
    for k in ("kp2d", "vis2d", "kp3d_local", "has3d"):
        np.testing.assert_array_equal(out2[k][:, 0], out1[k][:, 0])
        np.testing.assert_array_equal(out2[k][:, 1], out1[k][:, 0])


def test_donor_keeps_its_own_motion_across_the_pair(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams, composite
    from jarvis_jax.models.mvq.geometry import project_local
    import jax.numpy as jnp
    ds, tgt, src = _t2_samples(tmp_path)
    D = np.array([14.0, 0.0, 0.0], np.float32)
    out = composite(tgt, src, D, CopyPasteParams())
    assert out is not None and out["kp3d_local"].shape[1] == 2
    # the pasted fly moved by exactly the donor's own frame-to-frame motion
    np.testing.assert_allclose(out["kp3d_local"][1, 1] - out["kp3d_local"][1, 0],
                               src["kp3d_local"][0, 1] - src["kp3d_local"][0, 0], atol=1e-4)
    # and its 3D still reprojects onto its written 2D in BOTH frames
    for t in range(2):
        uv = np.asarray(project_local(jnp.asarray(out["kp3d_local"][1, t]), jnp.asarray(out["M"]),
                                      jnp.asarray(out["t_local"][t])))
        has = out["has3d"][1, t]
        for c in range(out["crops"].shape[1]):
            np.testing.assert_allclose(uv[has, c], out["kp2d"][1, t, c][has], atol=1e-3)


def test_paste_window_works_at_t2_and_records_the_donor_delta(tmp_path):
    from jarvis_jax.data.mv_copy_paste import CopyPasteParams
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path, n_frames=6, manifest_n_flies=1)
    ds = V12WindowDataset(root, "train", T=2, pair_deltas=(1,), train=True,
                          copy_paste=CopyPasteParams(p=1.0, max_tries=20))
    r = ds.paste_window(ds.window_index(REC, 0, 0, 1), np.random.default_rng(0))
    assert r is not None and set(r[1]) == {"donor", "D", "sep", "contact", "donor_delta"}
    assert r[0]["fly_valid"].tolist() == [True, True] and r[0]["crops"].shape[0] == 2
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_mv_copy_paste.py -q -k t2` -> AssertionError (`assert T == 1`).

- [ ] **Step 3: Implement.** In `composite`, replace `assert T == 1` with a frame loop, keeping the shift computation outside it (D is constant, so the affine shift is too):

```python
    T = crops.shape[0]
    if src["crops"].shape[0] != T:
        return None                                        # donor of a different window length
    shifts = view_shifts(tgt["M"], D, src["t_local"][0], tgt["t_local"][0])      # (C,2), frame-independent
    for ti in range(T):
        kp2d_d = src["kp2d"][0, ti] + shifts[:, None, :]                         # (C,K,2)
        vis_d = src["vis2d"][0, ti].copy()
        inside = (kp2d_d >= 0).all(-1) & (kp2d_d[..., 0] <= W - 1) & (kp2d_d[..., 1] <= H - 1)
        if np.any(vis_d & ~inside & tv[:, None]):
            return None                                    # reject on ANY frame, not just frame 0
        for c in range(C):
            if not tv[c]:
                continue
            m = _translate(src["prompt_mask"][ti, c].astype(np.uint8), shifts[c], nearest=True).astype(bool)
            if not m.any():
                continue
            donor = _translate(src["crops"][ti, c], shifts[c])
            gain = float(np.clip((np.median(crops[ti, c]) + 1.0) / (np.median(src["crops"][ti, c]) + 1.0), lo, hi))
            donor = np.clip(donor.astype(np.float32) * gain, 0, 255)
            alpha = cv2.GaussianBlur(m.astype(np.float32), (3, 3), 0)[..., None]
            out["crops"][ti, c] = (crops[ti, c] * (1 - alpha) + donor * alpha).astype(np.uint8)
            hk = np.round(tgt["kp2d"][0, ti, c]).astype(int)
            ok = (hk[:, 0] >= 0) & (hk[:, 0] < W) & (hk[:, 1] >= 0) & (hk[:, 1] < H)
            covered = np.zeros(hk.shape[0], bool); covered[ok] = m[hk[ok, 1], hk[ok, 0]]
            out["vis2d"][0, ti, c] &= ~covered
            out["prompt_mask"][ti, c] &= ~m
        out["kp2d"][1, ti] = kp2d_d.astype(np.float32)
        out["vis2d"][1, ti] = vis_d & inside & tv[:, None]
        out["kp3d_local"][1, ti] = (src["kp3d_local"][0, ti] + D) * src["has3d"][0, ti][:, None]
        out["has3d"][1, ti] = src["has3d"][0, ti]
```

`tv`/`sv` become `cam_valid.all(0)` (a camera the donor lacks in EITHER frame is not usable). In `v12_windows.paste_window`, drop `self.T == 1` from the `__getitem__` guard and the donor index build, restrict the donor pool to `self.win_delta[j] == self.win_delta[i]` is NOT required (the donor's own spacing is free — record it), and add `"donor_delta": int(self.delta(j))` to `info`.

- [ ] **Step 4: Run** — `JAX_PLATFORMS=cpu pytest tests/test_mv_copy_paste.py tests/test_v12_windows.py -q`.

- [ ] **Step 5: Figure gate.** `python scripts/viz/mvq_copy_paste_check.py --root <pseudo export> --out figures/2026-09-mvq/v2_train/copy_paste_t2.png --contact-sep 4 25 --contact-p 0.7` extended with a `--T 2` flag that renders BOTH frames of a pasted pair side by side. Expectation: *"the pasted donor (orange) sits in the same place relative to the host (cyan) in all 7 cameras in BOTH frames, and between frames it moves by its own body length at most — a donor frozen across the pair means the second frame reused frame 0's pixels."* Read the PNG back.

- [ ] **Step 6: Commit** (`mv_copy_paste.py`, `v12_windows.py`, the two tests, `scripts/viz/mvq_copy_paste_check.py`).

---

### Task 3: losses — identity persistence, negatives, wing weights, sample weights

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/losses_mvq.py`
- Modify: `third_party/jarvis_jax/tests/test_mvq_losses.py`

**Interfaces:**
- Produces: `LossWeights.persist: float = 0.5`, `LossWeights.persist_margin_units: float = 2.0`, `LossWeights.wing_kp_mult: float = 1.0`; `wing_kp_weight(kp_names, mult) -> (K,) float32` (2.0 on every name matching `^Wing`, else 1.0); `mvq_loss(out, batch, w, part_of_k, kp_weight=None)`; metrics gain `persist` and `n_negative`.
- Consumes: `batch["sample_weight"] (B,)`, `batch["is_negative"] (B,)` (Task 1), `assign_slots` on FRAME 0 only.
- Consumed by: Task 5 (`run_training` passes `kp_weight=wing_kp_weight(names, tcfg.wing_kp_mult)`), Task 7 (the scorecard reads the `persist` metric from the run log).

- [ ] **Step 1: Failing tests** (append to `tests/test_mvq_losses.py`; `_perfect_batch` gains `T=2` support — the existing helper already takes `T`, so only the new keys are needed: `"sample_weight": np.ones((B,), np.float32)`, `"is_negative": np.zeros((B,), bool)`)

```python
def test_sample_weight_scales_a_sample_out_of_the_loss():
    from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch(B=2)
    o = dict(out); o["xyz"] = out["xyz"].at[0, 1].add(20.0)        # sample 0 is badly wrong
    o["aux_pass1"] = {k: o[k] for k in out["aux_pass1"]}
    _, m_full = mvq_loss(o, batch, LossWeights(), pk)
    batch_w = dict(batch); batch_w["sample_weight"] = jnp.asarray([0.0, 1.0], jnp.float32)
    _, m_zero = mvq_loss(o, batch_w, LossWeights(), pk)
    assert float(m_zero["l3d"]) < 1e-3 < float(m_full["l3d"])       # weight 0 removes it entirely
    batch_h = dict(batch); batch_h["sample_weight"] = jnp.asarray([0.3, 1.0], jnp.float32)
    _, m_h = mvq_loss(o, batch_h, LossWeights(), pk)
    assert 0.0 < float(m_h["l3d"]) < float(m_full["l3d"])


def test_negative_window_targets_zero_existence_on_every_slot():
    from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch(B=2)
    b = dict(batch)
    b["fly_valid"] = jnp.zeros_like(batch["fly_valid"])            # nobody in the window
    b["has3d"] = jnp.zeros_like(batch["has3d"])
    b["unlabelled_sex"] = jnp.full((2,), -1, jnp.int8)
    b["is_negative"] = jnp.ones((2,), bool)
    o = dict(out); o["exist_logit"] = jnp.full_like(out["exist_logit"], -6.0)
    o["aux_pass1"] = {k: o[k] for k in out["aux_pass1"]}
    _, m_ok = mvq_loss(o, b, LossWeights(), pk)
    assert float(m_ok["exist"]) < 1e-2 and float(m_ok["exist_acc"]) == 1.0
    o2 = dict(o); o2["exist_logit"] = jnp.full_like(out["exist_logit"], 6.0)   # claims 4 flies
    o2["aux_pass1"] = {k: o2[k] for k in out["aux_pass1"]}
    _, m_bad = mvq_loss(o2, b, LossWeights(), pk)
    assert float(m_bad["exist"]) > 5.0 and float(m_bad["exist_acc"]) == 0.0
    assert float(m_ok["l3d"]) == 0.0 and float(m_ok["reproj"]) == 0.0          # no geometry to score


def test_identity_persistence_penalises_a_slot_that_teleports():
    from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch(B=2, T=2)
    w = LossWeights(persist=1.0)
    _, m0 = mvq_loss(out, batch, w, pk)
    assert float(m0["persist"]) < 1e-4                    # GT motion is fully explained
    o = dict(out); o["xyz"] = out["xyz"].at[:, 1, 1].add(30.0)      # slot 1 jumps on frame 1
    o["aux_pass1"] = {k: o[k] for k in out["aux_pass1"]}
    _, m1 = mvq_loss(o, batch, w, pk)
    assert float(m1["persist"]) > 20.0


def test_assignment_uses_frame_0_only():
    """With T=2 the two flies swap sides between frames; the slot a fly gets
    must be decided by frame 0, not by a centroid averaged over both."""
    from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch(B=1, T=2)
    b = dict(batch)
    x = np.asarray(batch["kp3d_local"]).copy()
    x[0, 0, 1] += 80.0; x[0, 1, 1] -= 80.0                # frame 1 swaps who is nearer the origin
    b["kp3d_local"] = jnp.asarray(x)
    _, m = mvq_loss(out, b, LossWeights(), pk)
    assert float(m["exist_acc"]) == 1.0                   # slots unchanged: female 1, male 2


def test_wing_keypoints_carry_double_weight():
    from jarvis_jax.train.losses_mvq import LossWeights, mvq_loss, wing_kp_weight
    names = ["EyeL", "EyeR", "Scutellum", "WingL_base", "WingR_base"]
    kpw = wing_kp_weight(names, 2.0)
    assert kpw.tolist() == [1.0, 1.0, 1.0, 2.0, 2.0]
    pk = np.arange(5, dtype=np.int32)
    out, batch = _perfect_batch()
    o_wing = dict(out); o_wing["xyz"] = out["xyz"].at[:, 1, 0, 3].add(5.0)     # error on a WING kp
    o_wing["aux_pass1"] = {k: o_wing[k] for k in out["aux_pass1"]}
    o_eye = dict(out); o_eye["xyz"] = out["xyz"].at[:, 1, 0, 0].add(5.0)       # same error, EyeL
    o_eye["aux_pass1"] = {k: o_eye[k] for k in out["aux_pass1"]}
    _, mw = mvq_loss(o_wing, batch, LossWeights(), pk, kp_weight=kpw)
    _, me = mvq_loss(o_eye, batch, LossWeights(), pk, kp_weight=kpw)
    assert float(mw["l3d"]) > 1.9 * float(me["l3d"])
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_mvq_losses.py -q -k "sample_weight or negative or persistence or frame_0 or wing"` -> TypeError / KeyError.

- [ ] **Step 3: Implement in `losses_mvq.py`**

```python
WING_RE = re.compile(r"^Wing")


def wing_kp_weight(kp_names, mult=2.0):
    """(K,) per-keypoint loss weight: `mult` on every wing landmark, 1.0
    elsewhere. BY NAME -- the wing landmarks are the weakest in the model
    (spec §1) and their indices differ between the detector and model orders."""
    return np.asarray([mult if WING_RE.match(str(n)) else 1.0 for n in kp_names], np.float32)
```

`_mmean` gains an optional weight, and every term routes through it:

```python
def _mmean(x, m, sw=None):
    """Masked mean; `sw` (B,) is a PER-SAMPLE weight folded into the mask, so a
    pseudo-label sample (0.3) contributes 0.3 of a real one's entries."""
    m = jnp.broadcast_to(m.astype(x.dtype), x.shape)
    if sw is not None:
        m = m * sw.reshape((sw.shape[0],) + (1,) * (x.ndim - 1)).astype(x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)
```

In `mvq_loss`: `sw = batch.get("sample_weight", jnp.ones((B,), jnp.float32))` passed to every `_mmean` call (including `_geo_terms`, which takes `sw` and `kp_weight`); in `_geo_terms` the per-keypoint weight multiplies on the K axis before the mean (`e2 * kpw[None, None, None, None, :]` for the (B,F,T,C,K) terms, `kpw[None, None, None, :]` for (B,F,T,K)) and the same vector weights the visibility BCE. The slot assignment's `dist` becomes frame-0 only:

```python
    has0 = batch["has3d"][:, :, 0].astype(jnp.float32)                              # (B,F,K)
    cen0 = (batch["kp3d_local"][:, :, 0] * has0[..., None]).sum(2) / jnp.maximum(has0.sum(2), 1.0)[..., None]
    dist = jnp.linalg.norm(cen0, axis=-1)          # frame 0 decides the slot; frame 1 REUSES it
```

(`cen`, used by term 7b, stays the all-frames centroid.) New term 9, after term 7b:

```python
    # term 9 identity persistence (spec §4): the slot assignment is made on frame 0
    # and reused on frame 1, so a slot that swaps animals between the two frames shows
    # up as a per-slot centroid displacement far beyond what the GT fly actually moved.
    persist = jnp.zeros(())
    if T > 1 and w.persist > 0:
        pc = pf["xyz"].mean(3)                                                   # (B,F,T,3) predicted centroid
        d_pred = jnp.linalg.norm(pc[:, :, 1] - pc[:, :, 0], axis=-1)             # (B,F)
        hasT = batch["has3d"].astype(jnp.float32)
        gc = ((batch["kp3d_local"] * hasT[..., None]).sum(3)
              / jnp.maximum(hasT.sum(3), 1.0)[..., None])                        # (B,F,T,3)
        d_gt = jnp.linalg.norm(gc[:, :, 1] - gc[:, :, 0], axis=-1)
        ok = fv_eff & (hasT[:, :, 0].sum(-1) > 0) & (hasT[:, :, 1].sum(-1) > 0)
        persist = _mmean(jax.nn.relu(d_pred - d_gt - w.persist_margin_units), ok, sw)
    total = total + w.persist * persist
```

Add `"persist": persist` and `"n_negative": batch.get("is_negative", jnp.zeros((B,), bool)).sum()` to `metrics`. Negatives need no branch: `fly_valid` all False makes `slot_target` all False, `unlabelled_sex == -1` makes `ignore` all False, so the existence target is 0 on every slot and every geometry mask is empty.

- [ ] **Step 4: Run** — `JAX_PLATFORMS=cpu pytest tests/test_mvq_losses.py tests/test_mvq_matching.py -q` -> all pass (the pre-existing `test_loss_near_zero_at_ground_truth_and_metrics` bound must NOT be loosened).

- [ ] **Step 5: Commit** (`losses_mvq.py`, `test_mvq_losses.py`).

---

### Task 4: augmentation — camera dropout p 0.1 and the named L/R flip

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/mv_augment.py`, `third_party/jarvis_jax/jarvis_jax/data/augment.py`
- Modify: `third_party/jarvis_jax/tests/test_mv_augment.py`

**Interfaces:**
- Produces: `augment.assert_lr_swap_covers(names, required=None)` — raises unless every `*L*`/`*R*` pair in `names` maps to a distinct partner and the covered set equals `configs/anatomy/v1.yaml` `model.KP_NAMES`'s L/R pairs (passed in by the caller as `required`); `MVAugParams.cam_drop_p` default unchanged (0.3), set to **0.1** in `configs/train/mvq_v2.yaml`.
- Consumed by: Task 5 (`run_training` calls `assert_lr_swap_covers(names, required=model_kp_names)` before step 0).

- [ ] **Step 1: Failing tests**

```python
def test_lr_swap_covers_every_anatomy_pair():
    from omegaconf import OmegaConf
    from jarvis_jax.data.augment import assert_lr_swap_covers, build_lr_swap
    names = [str(n) for n in OmegaConf.load("../../configs/anatomy/v1.yaml").model.KP_NAMES]
    swap = build_lr_swap(names)
    assert_lr_swap_covers(names, required=names)
    moved = [i for i in range(len(names)) if swap[i] != i]
    assert len(moved) == 2 * sum(1 for n in names if n.startswith(("WingL", "T1L", "T2L", "T3L", "EyeL")))
    for i in moved:
        assert names[i].replace("L", "R", 1) == names[swap[i]] or names[swap[i]].replace("L", "R", 1) == names[i]


def test_lr_swap_refuses_a_half_pair():
    import pytest
    from jarvis_jax.data.augment import assert_lr_swap_covers
    with pytest.raises(ValueError, match="WingR_base"):
        assert_lr_swap_covers(["WingL_base", "Scutellum"], required=["WingL_base", "WingR_base", "Scutellum"])


def test_t2_augmentation_keeps_gt3d_on_gt2d_in_both_frames(tmp_path):
    """Every geometric op must update M/t_local so the labels stay consistent --
    at T=2 the mirror's t_local broadcast and the camera dropout's all-frames
    AND are the two places that can silently break one frame only."""
    import jax
    from jarvis_jax.data.augment import build_lr_swap
    from jarvis_jax.data.mv_augment import MVAugParams, augment_window
    from jarvis_jax.data.v12_windows import V12WindowDataset, WINDOW_KEYS, window_batches
    from jarvis_jax.models.mvq.geometry import project_local
    root = make_v12_root(tmp_path, n_frames=5)
    ds = V12WindowDataset(root, "train", T=2, pair_deltas=(1,), train=False)
    b = next(window_batches(ds, 2, shuffle=False, num_workers=1, drop_last=False))
    jb = {k: jnp.asarray(v) for k, v in b.items()}
    p = MVAugParams(cam_drop_p=1.0, cam_drop_max=2, mirror_p=1.0)
    a = augment_window(jax.random.PRNGKey(0), jb, p, build_lr_swap(ds.keypoint_names))
    for t in range(2):
        uv = jax.vmap(lambda X, M, tl: project_local(X, M, tl))(a["kp3d_local"][:, 0, t], a["M"], a["t_local"][:, t])
        d = np.linalg.norm(np.asarray(uv) - np.moveaxis(np.asarray(a["kp2d"][:, 0, t]), 1, 2), axis=-1)
        m = np.asarray(a["vis2d"][:, 0, t])
        assert float(d[np.moveaxis(m, 1, 2)].max()) < 0.5      # px
    assert np.asarray(a["cam_valid"]).sum(1).min() >= 3 * 2 / 2   # >= 3 cameras survive, in every frame
    assert np.array_equal(np.asarray(a["cam_valid"])[:, 0], np.asarray(a["cam_valid"])[:, 1])
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_mv_augment.py -q -k "lr_swap or t2"` -> ImportError.

- [ ] **Step 3: Implement `assert_lr_swap_covers`** in `augment.py`:

```python
def assert_lr_swap_covers(names, required=None):
    """Raise unless every left/right landmark in `required` (default `names`)
    has its partner in `names` and `build_lr_swap` maps the two to each other.
    A silently unpaired landmark makes the horizontal flip relabel a left leg
    as itself -- a mirrored image with unmirrored labels, which trains the
    model to average the two sides."""
    swap = build_lr_swap(names)
    idx = {n: i for i, n in enumerate(names)}
    missing = []
    for n in (required or names):
        m = _mirror_name(n)
        if m is None:
            continue
        if m not in idx or n not in idx or swap[idx[n]] != idx[m]:
            missing.append(m if m not in idx else n)
    if missing:
        raise ValueError(f"lr_swap does not pair {sorted(set(missing))}: the horizontal flip "
                         f"would leave those labels unmirrored")
    return swap
```

- [ ] **Step 4: Run** — `JAX_PLATFORMS=cpu pytest tests/test_mv_augment.py -q`. If the T=2 projection test fails on `_mirror`, the cause is `t_local`'s frame-0-only broadcast: assert instead that `t_local` is frame-invariant on the input (it is, by construction in `_build`) and raise a ValueError in `_mirror` when it is not, rather than silently mirroring one frame's geometry onto the other.

- [ ] **Step 5: Commit** (`augment.py`, `mv_augment.py`, `test_mv_augment.py`).

---

### Task 5: config, trainer plumbing, smoke test, launch

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py`, `third_party/jarvis_jax/jarvis_jax/scripts/train_mvq.py`
- Create: `third_party/jarvis_jax/configs/train/mvq_v2.yaml`
- Modify: `third_party/jarvis_jax/tests/test_train_mvq_smoke.py`
- Modify: `docs/benchmark/2026-09-mvq/v2-notes.md`

**Interfaces:**
- Produces: `MVQTrainConfig` gains `pair_deltas: tuple = (1,)`, `pseudo_root: str | None = None`, `pseudo_weight: float = 0.3`, `negatives_root: str | None = None`, `negatives_frac: float = 0.05`, `wing_kp_mult: float = 1.0`, `persist` via `LossWeights`; `run_training` builds `ConcatWindowDataset([real, pseudo, negatives])` for the TRAIN split only (val stays the real root, spec §7: "validation on human labels only") and logs the realised real/pseudo/negative mix over the first `_RATIO_BATCHES` batches.
- Consumed by: Task 6 (reads `final/mvq_run.json`), Task 7 (reads `train` block for the scorecard header).

- [ ] **Step 1: Failing smoke tests** (append to `tests/test_train_mvq_smoke.py`)

```python
def test_t2_run_trains_and_mixes_roots(tmp_path):
    import dataclasses, json
    from jarvis_jax.data.mv_augment import MVAugParams
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    real = make_v12_root(tmp_path / "real", n_frames=6)
    pseudo = make_v12_root(tmp_path / "pseudo", n_frames=6)
    man = json.load(open(f"{pseudo}/manifest.json")); man["source"] = "pseudo"; man["weight"] = 0.3
    json.dump(man, open(f"{pseudo}/manifest.json", "w"))
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4,
                     n_local=1, n_global=1, dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0,
                     refine_passes=1, patch_rgb=3, fourier_bands=2, backbone="tiny", backbone_depth=1,
                     backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=2, batch_size=2, warmup_steps=1, eval_every=2, save_every=2,
                          log_every=1, num_workers=1, pretrained=False, window_lengths=(1, 2),
                          pair_deltas=(1, 4), pseudo_root=pseudo, pseudo_weight=0.3,
                          wing_kp_mult=2.0, smoke=True)
    run = tmp_path / "run"
    res = run_training(real, out_dir=str(run / "final"), ckpt_dir=str(run / "ckpt"), mcfg=mcfg, tcfg=tcfg,
                       aug=MVAugParams(enabled=False), weights=LossWeights(persist=0.5))
    assert np.isfinite(res["final_loss"])
    meta = json.load(open(run / "final" / "mvq_run.json"))
    assert meta["train"]["pair_deltas"] == [1, 4] and meta["train"]["pseudo_root"] == pseudo
    assert meta["val"]["unprompted"]["mpjpe3d_mm"] == meta["val"]["unprompted"]["mpjpe3d_mm"]   # not NaN


def test_batch_size_must_divide_the_device_count(tmp_path):
    import pytest
    from jarvis_jax.train.train_mvq import MVQTrainConfig, run_training
    ...   # existing pattern: expect ValueError naming batch_size and len(jax.devices())
```

- [ ] **Step 2: Run to verify failure** — `JAX_PLATFORMS=cpu pytest tests/test_train_mvq_smoke.py -q -k t2` -> TypeError (unexpected kwarg `pair_deltas`).

- [ ] **Step 3: Implement.** In `run_training`, after the existing `train_sets` build:

```python
    def _root(path, weight):
        return V12WindowDataset(path, "train", T=T, train=True, seed=tcfg.seed, copy_paste=copy_paste,
                                jitter_units=tcfg.jitter_units, sex_overrides=ov,
                                pair_deltas=tcfg.pair_deltas)
    parts, names_ = [train_sets[T]], ["real"]
    if tcfg.pseudo_root:
        parts.append(_root(tcfg.pseudo_root, tcfg.pseudo_weight)); names_.append("pseudo")
    if tcfg.negatives_root:
        parts.append(_root(tcfg.negatives_root, 0.0)); names_.append("negatives")
    train_sets[T] = ConcatWindowDataset(parts, names=names_) if len(parts) > 1 else parts[0]
```

`_balanced_weights` runs per sub-dataset and the concatenated weights are renormalised so the negatives take exactly `negatives_frac` of the mass and the rest splits by window count; print the realised real/pseudo/negative mix and the host-sex ratio over the first `_RATIO_BATCHES` batches (the existing print, extended). `assert_lr_swap_covers(names, required=names)` runs before the model build. `kp_weight = wing_kp_weight(names, tcfg.wing_kp_mult)` is built once and threaded into `make_train_step` and `evaluate`. Hydra: in `jarvis_jax/scripts/train_mvq.py` add `tnode["pair_deltas"] = tuple(int(v) for v in tnode["pair_deltas"])` beside the existing `window_lengths` conversion.

- [ ] **Step 4: Write `configs/train/mvq_v2.yaml`** (a full copy of `mvq.yaml` with these values changed — every line commented with the spec section it comes from):

```yaml
# @package train
# mvq v2 (spec 2026-09-05 §4): T=2 from scratch on real + pseudo + negatives.
lr: 3.0e-4
warmup_steps: 1000
total_steps: 40000
batch_size: 32            # 4/device on 8 GPUs; the 7-device layout hung (p3b-notes.md)
window_lengths: [1, 2]    # real singles as T=1, pairs as T=2 (spec §2 decision 3)
pair_deltas: [1, 4, 16]   # 1.25-20 ms at 800 fps
pseudo_root: /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_p3b_20260905
pseudo_weight: 0.3
negatives_root: /gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_pseudo_negatives_20260905
negatives_frac: 0.05
prompt_p_start: 0.0       # prompting OFF from step 0
prompt_p_end: 0.0
jitter_units: 10.0        # 1 mm, the mask-free centre budget (P4 §6)
copy_paste_p: 0.8
copy_paste_contact_p: 0.7
copy_paste_contact_sep: [4.0, 25.0]
female_host_weight: 4.27  # host-sex ratio 0.5 on the REAL root; verify in the run log
wing_kp_mult: 2.0         # wing keypoint AND wing visibility weight x2 (spec §4)
eval_every: 2000
save_every: 2000
num_workers: 24
pretrained: true
warm_start: null          # FROM SCRATCH (spec §2 decision 1)
val_cohorts: [female, two_fly, contact_pair, single_fly]
loss:
  other_fly_repulsion: 20.0   # spec §4 = the weight P3b actually converged with (0.5 did nothing; 20 from step 4000 moved every metric) -- see the launch check in Step 6
  persist: 0.5
  persist_margin_units: 2.0
  # (the remaining weights are mvq.yaml's, unchanged)
mv_aug:
  cam_drop_p: 0.1
  cam_drop_max: 2
  mirror_p: 0.5
  # (the remaining aug knobs are mvq.yaml's, unchanged)
```

- [ ] **Step 5: CPU smoke** — `JAX_PLATFORMS=cpu pytest tests/test_train_mvq_smoke.py tests/test_configs.py -q`, then a 20-step CPU run on the tiny fixture with `train=mvq_v2` overrides to prove the Hydra path parses the new keys.

- [ ] **Step 6: LAUNCH CHECK on `other_fly_repulsion`.** Spec §4 says 20 because that is what P3b converged with: at 0.5 the term contributed ~0.2 of a ~35 total and no metric moved (steps 0-4000); relaunched at 20 from step 4000, cross_fly_frac fell 0.052 -> 0.036 and contact_pair 0.178 -> 0.146 mm with the headline improving (p3b-notes.md). From scratch the loss balance differs (no warm start, T=2, negatives), so run 200 steps with 20.0 and read the log: `other_rep * 20` should be a visible but minority share of `total` (roughly 5-30 %); if it exceeds 50 %, halve to 10 and re-measure; if it is under 2 %, the term is inert -- raise to 40. Record the measured share in the launch record. Do NOT launch 40k steps on an unmeasured loss balance.

- [ ] **Step 7: Launch** (8 idle L40S; ~1 day at ~2.2 s/step for T=2 at batch 32):

```bash
cd third_party/jarvis_jax
module load cuda/12.9.1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN=
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
setsid nohup python -u -m jarvis_jax.scripts.train_mvq model=mvq train=mvq_v2 paths=hyak \
  run_id=mvq_t2_v2_20260905 "paths.runs_root=\${paths.mvq_runs_root}" \
  > slurm_logs/mvq_t2_v2_20260905.out 2>&1 &
```

Record in the notes: PID (the `python` process, not the `setsid` wrapper), log path, run dir, the realised host-sex ratio and real/pseudo/negative mix lines, s/step and GB/GPU at steady state, and the step-2000 val row. If it OOMs, resubmit at `train.batch_size=16` with gradient accumulation 2 (spec §7) and record that instead.

- [ ] **Step 8: Commit** (`train_mvq.py` x2, `configs/train/mvq_v2.yaml`, `test_train_mvq_smoke.py`, notes force-added).

---

### Task 6: post-training calibration

**Files:**
- Create: `scripts/benchmark/mvq_calibrate.py`
- Modify: `third_party/jarvis_jax/jarvis_jax/tracking/lift_mvq.py` (`MVQRunner` applies the temperatures)
- Create: `third_party/jarvis_jax/tests/test_mvq_calibrate.py`

**Interfaces:**
- Produces: `fit_temperature(logits, targets, mask) -> float` (1-D Newton/`scipy`-free bisection on the NLL derivative, 1e-4 tolerance, clamped to [0.25, 10]); `reliability(probs, targets, mask, bins=10) -> {"edges", "acc", "conf", "n", "max_gap", "ece"}`; a `"calibration"` block written into `<run>/final/mvq_run.json`: `{"exist_temperature": t_e, "vis_temperature": t_v, "reliability_exist": {...}, "reliability_vis": {...}, "n_val": 153}`.
- Produces: `MVQRunner.exist_temperature` / `vis_temperature` (read from `self.meta.get("calibration", {})`, default 1.0) applied in `infer` before every sigmoid, so `exist_thresh=0.5` means calibrated 0.5 everywhere downstream (`slot_read`, `pick_typed_pair`, `pick_mask_pair`, `coarse_track`).
- Consumed by: Task 7 (the "existence calibration" acceptance row reads `reliability_exist["max_gap"] <= 0.05`).

- [ ] **Step 1: Tests**

```python
def test_fit_temperature_recovers_a_known_scaling():
    from scripts_benchmark import mvq_calibrate as mc          # imported by path, see the file header
    rng = np.random.default_rng(0)
    z = rng.normal(size=20000) * 2.0
    p = 1 / (1 + np.exp(-z))
    y = rng.uniform(size=z.shape) < p
    t = mc.fit_temperature(z * 3.0, y, np.ones_like(y, bool))  # logits inflated 3x
    assert abs(t - 3.0) < 0.1


def test_reliability_of_perfectly_calibrated_probs_is_on_the_diagonal():
    from scripts_benchmark import mvq_calibrate as mc
    rng = np.random.default_rng(1)
    p = rng.uniform(size=50000)
    y = rng.uniform(size=p.shape) < p
    r = mc.reliability(p, y, np.ones_like(y, bool))
    assert r["max_gap"] < 0.05 and r["ece"] < 0.02


def test_runner_applies_the_stored_temperature(tmp_path):
    """A calibrated runner must not change WHICH slot is picked, only the
    reported probability -- the policy threshold moves with it."""
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    ...   # FakeRunner-style: set runner.meta["calibration"] = {"exist_temperature": 2.0};
          # assert the returned exist probs equal sigmoid(logit / 2) and the argmax slot is unchanged
```

- [ ] **Step 2-3: Implement.** `mvq_calibrate.py` runs ONE pass over the val split with the same forward `train_mvq.evaluate` uses (`normalize_crops`, unprompted), collecting `exist_logit (N,I)` with the label-driven `slot_target`/`ignore` from `assign_slots`/`slot_ignore` (only non-ignored slots are scored) and `vis_logit (N,F,T,C,K)` against `vis2d` masked by `cam_valid & fly_valid`. `fit_temperature` minimises the BCE of `sigmoid(z / t)` by bisection on `d NLL / d t`. The reliability curve is 10 equal-width bins. The script then rewrites `final/mvq_run.json` in place (read, add `"calibration"`, atomic write) and saves `figures/2026-09-mvq/v2_train/calibration.png` — two reliability panels (existence, visibility), before and after, with the diagonal and the +-0.05 band. Docstring EXPECTATION: *"before scaling the existence curve sits ABOVE the diagonal (the head is over-confident: 0.9-bin frames are right less than 90 % of the time); after scaling both curves lie inside the +-0.05 band. A curve that is already on the diagonal means T ~ 1 and nothing to do — record that, do not force a temperature."*

- [ ] **Step 4: Run on the v2 final checkpoint**, read the PNG back, record both temperatures and `max_gap` in the notes.

- [ ] **Step 5: Commit** (`scripts/benchmark/mvq_calibrate.py`, `lift_mvq.py`, the test, notes).

---

### Task 7: the acceptance suite

**Files:**
- Create: `scripts/benchmark/mvq_perkp_bouts.py` (promoted from the scratchpad `perkp_all_bouts.py`)
- Create: `scripts/benchmark/mvq_v2_acceptance.py`
- Create: `third_party/jarvis_jax/tests/test_mvq_perkp.py`
- Create: `docs/benchmark/2026-09-mvq/2026-09-05-mvq-v2/scorecard.{json,md}`

**Interfaces:**
- Produces: `perkp_bout_metrics(kp3d_by_fly, kp_names, *, male_fly=1, jump_mm=0.5, min_kp=5, contact_units=15.0) -> {"T", "pose_jump_frac", "straddle_frac", "contact_frac", "female_missing_frac", "head_tail_flips", "per_kp_jump_frac" (K,), "per_kp_names"}` and `bout_dir_metrics(bout_dir)` / `tracks_npz_metrics(path)` wrappers (the second reads `coarse_tracks.npz`/`fine_tracks.npz`'s `kp3d`+`kp_names`, keypoints looked up BY NAME).
- Produces: `mvq_v2_acceptance.py --run <run>/final --out docs/benchmark/2026-09-mvq/2026-09-05-mvq-v2/` writing ONE `scorecard.json` (+ `scorecard.md`) with a row per spec §5 check: `{check, metric, value, threshold, pass, evidence}` and a top-level `"accepted": all(pass)`.
- Consumed by: Task 8 (promotion is gated on `accepted == true`).

- [ ] **Step 1: Tests for the promoted metric** (`tests/test_mvq_perkp.py`): on a synthetic two-fly bout — (a) a bout with no motion scores `pose_jump_frac == 0`; (b) injecting a 0.8 mm one-frame step on 6 male keypoints scores exactly `1/T` (the jump frame only) and names those 6 in `per_kp_jump_frac`; (c) placing 6 male keypoints on the female's centroid scores `straddle_frac == 1/T`; (d) a NaN female frame counts in `female_missing_frac` and does NOT count as a straddle (the P3b bout-1 blindness lesson: an unmeasurable frame must not read as clean); (e) `head_tail_flips` counts an `Antenna_Base -> Abd_tip` axis reversal, found BY NAME with a permuted `kp_names`.

- [ ] **Step 2: Run to verify failure**, **Step 3: implement** `mvq_perkp_bouts.py` (the scratchpad script's body, with the by-name index lookups and a `--json` output; keep its docstring's definitions verbatim so the numbers stay comparable with `p3b-notes.md`'s tables).

- [ ] **Step 4: Implement `mvq_v2_acceptance.py`.** It runs, in order, and records each row with the command that produced it:

1. **val** — `train_mvq.evaluate` on the real v12 val split with `cohorts=_cohorts(val_ds)` and `val_cohorts` including `contact_pair` and `single_fly`; rows: `mpjpe3d_mm <= 0.085`, `policy_miss_frac == 0`, `sex_acc >= 0.99`, `cohort_contact_pair * MM_PER_UNIT <= 0.146`, `cross_fly_frac <= 0.036`, `cross_fly_frac_contact_pair <= 0.055`, `policy_miss_frac_single_fly == 0`.
2. **centre shift** — `scripts/benchmark/mvq_centre_shift.py`'s `run()` imported directly; rows: `mpjpe at 1 mm <= 1.1 x mpjpe at 0`, `miss_frac < 0.02`.
3. **masked bouts 1/4/28 of 20_04, containment OFF** — `scripts/mvq_lift_bout.py --identity mask --containment off` into `OutFiles/v2_accept/off/`, then `bout_dir_metrics`; rows: `pose_jump_frac == 0` in all three, `straddle_frac <= ` the P3b numbers (0.1793 / 0.0000 / 0.0000), `female_missing_frac <=` (0.1092 / 0.2563 / 0.1545).
4. **mask-free bout 117 stride 1** — `scripts/coarse_pass_mvq.py --start 111488 --end 114688 --stride 1 --num-animals 2 --out OutFiles/v2_accept/fine_bout117/fine_tracks.npz`, then `tracks_npz_metrics`; rows: `pose_jump_frac <= 0.05` (P3b 0.28), `straddle_frac <= 0.10` (P3b 0.23), plus the contact-sheet figure below for the identity-swap read.
5. **coarse pass 20_04** — `scripts/coarse_pass_mvq.py --stride 16` over the whole recording; rows: female `frac_trackable >= 0.85` (jitter-10: 0.48); `max(exist[:, frames of bout 69]) < 0.5` (the stale-window frames named in `gate-bouts-probe-2026-09-05.md`); gate bout 1's arena-edge frames not asserted (`exist < 0.5`).
6. **single fly** — `scripts/pseudo_labels/singlefly_p3b_pass.py` over 2 free_running bouts and Clip Session6; rows: exactly one slot `>= 0.5` on `>= 99 %` of frames; keypoint stability (median per-keypoint step) `<=` bout 28's; agreement with Clip's `data3D.csv` JARVIS reference within the courtship LOO band (median `<= 3.66 px`, the p90 in `session-driver-2026-09-05.md`).
7. **calibration** — `reliability_exist["max_gap"] <= 0.05` from Task 6's block.

Figures written and **read back with the Read tool** before the scorecard is called complete: `figures/2026-09-mvq/v2_train/{bout117_contact_sheet.png, masked_bouts_1_4_28.png, val_female_overlay.png, calibration.png}` — the first two via `scripts/viz/contact_sheet.py`/`scripts/viz/mvq_identity_overlay.py` on the worst frames each metric names (NOT the P3b-era frame picks), the third via `python -m viz overlay`-style `scripts/viz/mvq_overlay.py --cases female,contact_pair,worst`. Each row's `evidence` field names its figure and JSON.

- [ ] **Step 5: Run the suite** on the v2 checkpoint, read every figure, and write `docs/benchmark/2026-09-mvq/2026-09-05-mvq-v2/notes.md` with the scorecard table, the P3b column beside it, and one paragraph per FAILED row saying what the figure shows.

- [ ] **Step 6: Commit**

```bash
git add scripts/benchmark/mvq_perkp_bouts.py scripts/benchmark/mvq_v2_acceptance.py third_party/jarvis_jax/tests/test_mvq_perkp.py
git add -f docs/benchmark/2026-09-mvq/2026-09-05-mvq-v2/
git commit -m "bench(mvq-v2): per-keypoint bout metrics and the spec §5 acceptance scorecard"
```

---

### Task 8: promotion

**Files:**
- Create: `configs/mvq/v2.yaml`
- Modify: `scripts/slurm/mvq_p3a_campaign.sh`, `scripts/slurm/mvq_session_pipeline.sh` defaults
- Modify: `docs/benchmark/2026-09-mvq/v2-notes.md`

**Precondition:** `scorecard.json`'s `"accepted": true`. If any row fails, STOP and report — P3b stays production (spec §2 decision 1).

- [ ] **Step 1:** `configs/mvq/v2.yaml` = a copy of `configs/mvq/p3b.yaml` with `checkpoint` pointed at `/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t2_v2_20260905/final`, the header rewritten to name this plan and the scorecard, and a new `calibration: from_run_json` note (the temperatures live in `mvq_run.json`, not here, so a re-calibration does not change the gate signature). `identity: mask` and `containment: "on"` stay — v2 does not remove the mask route, it makes the mask-free one usable.
- [ ] **Step 2:** campaign defaults `RUN_NAME=pose_mvq_v2` / `MVQ_CONFIG=v2` (p3b still selectable), then `scripts/slurm/mvq_p3a_campaign.sh --dry-run --only 2025_10_20_13_20_04` and `scripts/slurm/mvq_session_pipeline.sh --dry-run --session .../Session1` — both must print the chain rooted at `pose_mvq_v2` with `mvq=v2` and submit nothing. **Do not edit the campaign script while a campaign is running** (`pgrep -f mvq_p3a_campaign.sh` must be empty — bash reads scripts incrementally).
- [ ] **Step 3:** re-lift both sessions once more (spec §6 item 9) via `scripts/slurm/mvq_session_pipeline.sh`, and record the `scripts/session_collect.py` roll-up (female-missing %, LOO median, reproj median) against the `pose_mvq_p3b` row.
- [ ] **Step 4:** `JAX_PLATFORMS=cpu pytest tests/test_lift_masked_bout.py tests/test_mvq_session_pipeline.py -q` and commit config + campaign defaults + notes.

---

## Self-review (done while writing)

- Spec coverage: §4 T=2 -> Task 1; copy-paste -> Task 2; identity persistence, existence negatives, wing weights -> Task 3; camera dropout and flip -> Task 4; the recipe, batch/devices and the run -> Task 5; calibration -> Task 6; §5 -> Task 7; §6 items 4-9 -> Tasks 1-8 in order.
- Names crossing tasks: `pair_deltas`/`win_delta`/`window_index`/`ConcatWindowDataset`/`is_negative` (Task 1) are what Tasks 2, 3 and 5 use; `composite`'s T loop and `info["donor_delta"]` (Task 2) are what Task 5's `copy_paste_p: 0.8` exercises; `wing_kp_weight`/`sample_weight`/`persist` (Task 3) are what Task 5 threads and Task 7 reads from the log; `assert_lr_swap_covers` (Task 4) is called by `run_training` (Task 5); `final/mvq_run.json["calibration"]` (Task 6) is read by `MVQRunner` and by Task 7's calibration row; `perkp_bout_metrics` (Task 7) is what Task 8's re-lift comparison reuses. Plan A's `sample_weight`, `source`, `role`, `partners` and the negatives' `center3D` are consumed in Tasks 1, 3 and 5 exactly as Plan A writes them.
- Ambiguities resolved, each flagged where it is decided: `other_fly_repulsion = 20` is P3b's converged weight (0.5 was inert; 20 moved every metric from step 4000) and is re-checked by a 200-step loss-share measurement because a from-scratch T=2 run balances differently (halve if > 50 % of the total, double if < 2 %); "real singles train as T=1" is implemented as `window_lengths: [1, 2]` (alternating by step, the mechanism `run_training` already has) rather than a mixed-T batch, which the model's `(B,T,…)` crops cannot express; validation stays on the real root only; negatives take a fixed 5 % of the sampling mass (the spec gives a count, not a rate — 2,000 windows against ~12,000 positives is ~14 %, and 5 % is the conservative starting point, logged and adjustable).
- Judgement calls left to the implementer: the exact T=2 step time and whether batch 32 fits on 8 L40S (Task 5 Step 6 measures before the 40k launch), whether `_mirror` needs the frame-invariance guard (Task 4 Step 4), and the bout-69/gate-bout-1 frame ranges, which must be read out of `gate-bouts-probe-2026-09-05.md` rather than retyped from memory.
