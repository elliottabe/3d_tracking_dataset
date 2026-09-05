# MVQ mask-free front end — Plan 1 (P4a spike + runner, P4b coarse pass)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how far off-centre an mvq window may be, extract one shared mvq runner from the bout-video script, and build the mask-free coarse pass (CenterDetect peaks -> 3D centres -> mvq at stride 16 -> 50 fps tracks -> bout table via the existing gates), validated on recording Session0/2025_10_20_13_20_04 with its 30 reviewed bouts.

**Architecture:** `jarvis_jax/tracking/lift_mvq.py::MVQRunner` owns "frames + 3D centres -> mvq outputs -> pipeline-format keypoints". `coarse_centres.py` turns CenterDetect heatmap peaks into 3D centres and window plans; `coarse_track.py` runs the runner at stride over a recording and writes `coarse_tracks.npz` in the SAM3 coarse-pass schema plus mvq fields; `scripts/coarse_pass_gates.py` gains an mvq-aware trackability/behaviour signal. Plan 2 (fine pass, pipeline switches, learned detector, validation) follows.

**Tech Stack:** JAX 0.11 / Flax NNX 0.12.x, orbax, numpy, OpenCV, PIL, Hydra; pytest from `third_party/jarvis_jax/` with `JAX_PLATFORMS=cpu -m "not gpu"`.

**Spec:** `docs/specs/2026-09-04-mvq-maskfree-frontend-design.md` (§2 facts, §4.1-4.3 components, §5.1 gates baseline, §6 spike, §7 budget, §8 gates 1-2 and 6, §9 tests). Also binding: `docs/specs/2026-09-04-mvq-p3a-identity-existence-design.md` §3 slot table.

## Global Constraints

- Package root `third_party/jarvis_jax/jarvis_jax/`; tests in `third_party/jarvis_jax/tests/` (bare `from mvq_fixtures import ...`); repo-root scripts in `scripts/`. Never `git add -A`; commit trailers `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01Npo4HiYC4t5M2xYjKUJsFP`.
- Slots: 0 prompted, 1 female, 2 male, 3 other (`jarvis_jax/train/matching.py` `SLOT_*`); sex codes 0 female, 1 male. The mask-free route uses UNPROMPTED inference only.
- Keypoint order: mvq emits `meta["keypoint_names"]` (= `configs/detector/vitpose_v3.yaml kp_names`); pipeline files are `configs/anatomy/v1.yaml` `model.KP_NAMES` order. Permute BY NAME with `jarvis_jax.tracking.predict_2d.detector_to_model_perm(det_names, model_names)` (`kp_model[..., k, :] = kp_det[..., perm[k], :]`). Cameras BY NAME (`ReprojectionTool(calib_dir).cameras.keys()` == `configs/recording/session0.yaml recording.cameras`, assert it).
- Confidence: pipeline-facing 2D conf = view-visibility sigmoid; conf3d = its mean over cameras; raw D4RT confidence stored as `conf3d_mvq_raw`.
- Window geometry must equal `V12WindowDataset._build` (`jarvis_jax/data/v12_windows.py`): `M, t = _affine_np(rt.camera_matrices)`; `u,v = M[c] @ centre + t[c]`; `origin[c] = crop_origin([u, v, 0, 0], W, H, 448)`; `t_local[c] = M[c] @ centre + t[c] - origin[c]`; crop `frame[c][y0:y0+448, x0:x0+448]`; `normalize_crops` from `jarvis_jax.train.train_mvq`.
- World units 0.1 mm (`MM_PER_UNIT = 0.1`); merge distance 30 units; CenterDetect input = full frame squashed to 320x320 (PIL bilinear or `cv2.resize(..., (320, 320), INTER_LINEAR)`), ImageNet normalisation (`jarvis_jax/data/device.py IMAGENET_MEAN_J/STD_J`), heatmap 160x160, `extract_top_k_peaks(heatmap, k=2, suppression_radius=15)` and `peaks_to_full_image(peaks, 160, W, H)` from `jarvis_jax/eval/centerdetect_decode.py`.
- Data: recording `/gscratch/portia/eabe/data/Johnson_lab/Video_recordings/courtship/Session0/2025_10_20_13_20_04` (7 mp4, 800 fps, 1936x448, ~497,993 frames; calibration `calibration/`; reviewed bouts in `courtship_bouts_fly0_summary.csv` (30 rows, columns `fly_id,bout_idx,start_frame,end_frame`; the unified CSV is a broken symlink)). mvq checkpoint: `/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3a_20260904/final`. CenterDetect checkpoint: `/gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004` (model `EfficientTrack(num_joints=1, in_channels=3, model_size="medium")`).
- Compute: this session's node g3102 has 8 idle L40S (the fine-tune finished) — run GPU steps directly here with `module load cuda/12.9.1; export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6; unset LD_LIBRARY_PATH JAX_PLATFORMS; export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN=`. If the node is busy again, `scripts/slurm/submit_task.sh` (ckpt-all). Never on the login node.
- Figures under `figures/2026-09-mvq/p4_maskfree/`, JSON beside each PNG, expectation in the script docstring, PNG read with the Read tool before any claim; notes in `docs/benchmark/2026-09-mvq/p4-maskfree-notes.md` (force-add).

---

## File map

| file | responsibility |
|---|---|
| `jarvis_jax/data/v12_windows.py` | `center_shift_units` option (eval-mode deterministic centre offset) — Task 1 |
| `scripts/benchmark/mvq_centre_shift.py` | §6 spike: policy error vs centre shift, figure + JSON — Task 1 |
| `jarvis_jax/tracking/lift_mvq.py` | `MVQRunner`: windows, infer, read_typed, to_pipeline, gates signature — Task 2 |
| `scripts/viz/mvq_bout_video.py` | refactored onto `MVQRunner` (behaviour unchanged) — Task 2 |
| `jarvis_jax/tracking/coarse_centres.py` | CenterDetect wrapper, peaks -> 3D centres, clustering, window plan — Task 3 |
| `jarvis_jax/tracking/coarse_track.py` + `scripts/coarse_pass_mvq.py` | stride-16 coarse pass -> `coarse_tracks.npz` (+ features) — Task 4 |
| `scripts/coarse_pass_gates.py` | mvq-aware trackability/behaviour signals — Task 5 |
| `scripts/viz/coarse_centres_check.py`, `scripts/viz/coarse_tracks_check.py` | figure gates 1-2 — Task 6 |
| tests: `test_v12_windows.py` (+), `test_lift_mvq.py`, `test_coarse_centres.py`, `test_coarse_track.py`, `test_coarse_pass_gates.py` | |

---

### Task 1: Centre-shift spike (spec §6)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` (`__init__` kwarg, `_build`)
- Create: `scripts/benchmark/mvq_centre_shift.py`
- Test: `third_party/jarvis_jax/tests/test_v12_windows.py`

**Interfaces:**
- Produces: `V12WindowDataset(..., center_shift_units: float = 0.0)` — in eval mode (`train=False`) every window's centre is displaced by exactly `center_shift_units` in a deterministic random direction in the x-y plane (seeded per `(seed, i)`); labels stay in the same world frame (ROI-local values shift accordingly, as `_build` already derives them from `center`). Train mode ignores it.
- Produces: `figures/2026-09-mvq/p4_maskfree/centre_shift.{png,json}` and a notes section with the decision (§6 rule).

- [ ] **Step 1: Failing test** (append to `tests/test_v12_windows.py`)

```python
def test_center_shift_moves_window_by_exact_amount(tmp_path):
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root = make_v12_root(tmp_path)
    ds0 = V12WindowDataset(root, "val", T=1, train=False)
    ds5 = V12WindowDataset(root, "val", T=1, train=False, center_shift_units=5.0)
    i = ds0.windows.index((REC, 0, 1))
    a, b = ds0[i], ds5[i]
    d = b["center3D"] - a["center3D"]
    assert abs(np.linalg.norm(d) - 5.0) < 1e-4 and abs(d[2]) < 1e-6          # exact magnitude, in-plane
    # the labels describe the same world points: local + centre is invariant
    np.testing.assert_allclose(b["kp3d_local"][0, 0] + b["center3D"], a["kp3d_local"][0, 0] + a["center3D"], atol=1e-3)
    assert np.array_equal(ds5[i]["center3D"], b["center3D"])                    # deterministic
    dtr = V12WindowDataset(root, "val", T=1, train=True, center_shift_units=5.0, jitter_units=0.0)
    np.testing.assert_allclose(dtr[i]["center3D"], a["center3D"], atol=1e-6)    # train mode ignores it
```

- [ ] **Step 2: Run to verify it fails** — `cd third_party/jarvis_jax && JAX_PLATFORMS=cpu pytest tests/test_v12_windows.py -q -k center_shift` -> TypeError (unexpected kwarg).

- [ ] **Step 3: Implement** — in `__init__` add `center_shift_units=0.0`, store `self.center_shift = float(center_shift_units)`. In `_build`, right after the jitter block (before `center = center.astype(np.float32)`):

```python
        if not self.train and self.center_shift > 0:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(i), 99]))
            ang = rng.uniform(0, 2 * np.pi)
            center = center + self.center_shift * np.array([np.cos(ang), np.sin(ang), 0.0])
```

- [ ] **Step 4: Run** — the test passes; the whole file passes.

- [ ] **Step 5: The spike script** `scripts/benchmark/mvq_centre_shift.py` (repo root; header pattern of `scripts/viz/mvq_overlay.py`: sys.path insert of `third_party/jarvis_jax` and repo root, `matplotlib.use("Agg")`).

Docstring EXPECTATION: "the unprompted typed-slot policy error stays within 10 % of its unshifted value and the miss fraction under 2 % up to a 1 mm (10 unit) centre shift; if it climbs steeply before 1 mm the interpolated coarse centres of the mask-free pass are not safe and the §6 retrain with 1 mm jitter is required."

```python
SHIFTS_MM = [0.0, 0.5, 1.0, 2.0, 3.0]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--step", default=None); ap.add_argument("--attn_impl", default=None)
    ap.add_argument("--root", default="/gscratch/portia/eabe/data/Johnson_lab/red_data/red_data_3d_v12_export0902")
    ap.add_argument("--out", default="figures/2026-09-mvq/p4_maskfree"); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()
    step = int(a.step) if (a.step is not None and a.step != "latest") else a.step
    model, meta = load_mvq_model(a.run, step=step, attn_impl=a.attn_impl)
    rows = []
    for mm in SHIFTS_MM:
        ds = V12WindowDataset(a.root, "val", T=1, train=False, center_shift_units=mm / MM_PER_UNIT)
        errs, miss, n = [], 0, 0
        for b in window_batches(ds, a.batch, shuffle=False, drop_last=False, num_workers=8):
            B = b["crops"].shape[0]
            out = model(normalize_crops(jnp.asarray(b["crops"])), jnp.asarray(b["cam_valid"]), jnp.asarray(b["M"]),
                        jnp.asarray(b["t_local"]), jnp.asarray(b["prompt_mask"]), prompt_on=jnp.zeros((B,), bool))
            xyz = np.asarray(out["xyz"]); ex = 1 / (1 + np.exp(-np.asarray(out["exist_logit"])))
            for bi in range(B):
                n += 1
                inst = policy_instance(ex[bi], xyz[bi], prompted=False, has_mask=False)
                if inst is None: miss += 1; continue
                gt, has = b["kp3d_local"][bi, 0, 0], b["has3d"][bi, 0, 0]
                if has.any(): errs.append(float(np.linalg.norm(xyz[bi, inst, 0][has] - gt[has], axis=-1).mean()))
        rows.append(dict(shift_mm=mm, mpjpe_policy_mm=float(np.mean(errs)) * MM_PER_UNIT, miss_frac=miss / max(n, 1), n=n))
        print(rows[-1], flush=True)
    # figure: two panels, error (mm) and miss fraction vs shift, with the 10 %/2 % decision lines
    ...
```

(`window_batches` from `jarvis_jax.data.v12_windows`; `normalize_crops`, `MM_PER_UNIT` from `jarvis_jax.train.train_mvq`; `policy_instance` from `jarvis_jax.models.mvq.policy`; `load_mvq_model` from `jarvis_jax.models.mvq.checkpoint`. Write the JSON beside the PNG. Complete the figure code: `fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))`, error curve with a dashed line at `1.1 * rows[0]["mpjpe_policy_mm"]`, miss curve with a dashed line at 0.02, x in mm, titles naming the checkpoint step.)

- [ ] **Step 6: Run it on a GPU** (node idle): `PYTHONPATH=third_party/jarvis_jax:. CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 python scripts/benchmark/mvq_centre_shift.py --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3a_20260904/final` (about 5 x 153 windows; a few minutes). READ the PNG. Write `docs/benchmark/2026-09-mvq/p4-maskfree-notes.md` with a "Centre-shift spike" section: the table, the reading, and the §6 decision (retrain needed: yes/no). If yes, STOP after this task and report — the controller schedules the retrain before Task 4 runs on real data (Tasks 2-3 can proceed).

- [ ] **Step 7: Commit** — `git add third_party/jarvis_jax/jarvis_jax/data/v12_windows.py third_party/jarvis_jax/tests/test_v12_windows.py scripts/benchmark/mvq_centre_shift.py && git add -f docs/benchmark/2026-09-mvq/p4-maskfree-notes.md && git commit -m "bench(mvq): centre-shift robustness spike (P4a §6)"`.

---

### Task 2: `MVQRunner` and the bout-video refactor

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/lift_mvq.py`
- Modify: `scripts/viz/mvq_bout_video.py` (use the runner for window building, inference, slot reading and pipeline-format output; rendering unchanged)
- Test: `third_party/jarvis_jax/tests/test_lift_mvq.py`

**Interfaces (Produces):**

```python
class MVQRunner:
    def __init__(self, run_dir_or_final, *, step=None, attn_impl=None, calib_dir, cameras, batch=32, exist_thresh=0.5)
    # attributes: model, meta, kp_names (mvq order), cameras (list[str]), rt (ReprojectionTool), M (C,2,3) f64, t (C,2) f64, cam_mats (C,4,3)
    def windows(self, frames, present, centres) -> dict
        # frames (C,H,W,3) uint8 RGB in `cameras` order; present (C,) bool; centres (B,3) world -> crops (B,1,C,448,448,3) u8,
        # cam_valid (B,1,C), M (B,C,2,3) f32, t_local (B,1,C,2) f32, origin (B,C,2) int32
    def infer(self, w) -> dict
        # unprompted forward on a windows dict (pads to `batch`), returns numpy: kp3d (B,I,K,3) world, kp2d (B,I,C,K,2) full-frame,
        # vis (B,I,C,K) sigmoid, exist (B,I) sigmoid, sex_prob (B,I), conf_raw (B,I,K)
    def read_typed(self, out, bi, want_sex) -> dict | None
        # slot 1 for want_sex==0, slot 2 for want_sex==1; None if exist < exist_thresh; dict(slot, kp3d (K,3), kp2d (C,K,2), vis (C,K), conf_raw (K,), exist, sex_prob)
    def to_pipeline(self, kp3d (T,K,3), kp2d (T,C,K,2), vis (T,C,K), conf_raw (T,K), model_names) -> dict
        # permutes K by name (detector_to_model_perm); returns dict(kp2d, conf=vis, kp3d, conf3d=vis.mean(axis=1), conf3d_mvq_raw, kp_names=model_names, gates=self.gates_signature())
    def gates_signature(self) -> dict   # {"lifter": "mvq", "checkpoint": path, "sha256": first 16 hex of the checkpoint's manifest/_METADATA, "step": step or "final", "exist_thresh": ...}
```

Important: read `scripts/run_bout.py::stage_b_gate_signature` (around line 371) and the `gates` check at lines ~1580-1608 FIRST, and make `gates_signature()` produce a dict that check accepts when `pipeline.lifter == "mvq"` — if the check compares against the detector's config verbatim, add an `lifter` key branch to `stage_b_gate_signature` so an mvq-produced `kp3d.npz` passes without `allow_stale_kp3d`. Plan 2 wires `pipeline.lifter`; here the minimum is that the signature round-trips through the check function in a unit test.

- [ ] **Step 1: Failing tests**

```python
# tests/test_lift_mvq.py
import numpy as np, jax.numpy as jnp
from mvq_fixtures import make_v12_root, CAMS, REC

def _tiny_final(tmp_path):
    """Train the tiny model for 1 step so a real final/ exists (reuse the smoke-test recipe)."""
    from jarvis_jax.models.mvq import MVQConfig
    from jarvis_jax.train.train_mvq import run_training, MVQTrainConfig
    from jarvis_jax.train.losses_mvq import LossWeights
    from jarvis_jax.data.mv_augment import MVAugParams
    root = make_v12_root(tmp_path)
    mcfg = MVQConfig(crop=448, patch=16, embed_dim=32, num_keypoints=50, num_cameras=7, n_instances=4, n_local=1, n_global=1,
                     dec_layers_3d=2, dec_layers_2d=1, dec_heads=4, mlp_ratio=2.0, refine_passes=1, patch_rgb=3, fourier_bands=2,
                     backbone="tiny", backbone_depth=1, backbone_heads=4, remat=False)
    tcfg = MVQTrainConfig(total_steps=1, batch_size=2, warmup_steps=1, eval_every=1, save_every=1, log_every=1, num_workers=1,
                          pretrained=False, window_lengths=(1,), smoke=True)
    run = tmp_path / "run"
    run_training(root, out_dir=str(run / "final"), ckpt_dir=str(run / "ckpt"), mcfg=mcfg, tcfg=tcfg, aug=MVAugParams(enabled=False), weights=LossWeights())
    return root, str(run / "final")

def test_runner_windows_match_dataset_geometry(tmp_path):
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    from jarvis_jax.data.v12_windows import V12WindowDataset
    root, final = _tiny_final(tmp_path)
    ds = V12WindowDataset(root, "val", T=1, train=False)
    i = ds.windows.index((REC, 0, 1)); s = ds[i]
    r = MVQRunner(final, calib_dir=f"{root}/calibrations/A", cameras=CAMS, batch=2)
    # rebuild the same window from the raw frames the dataset used
    frames = np.stack([ds._decode(ds._img[img]) for img, _ in ds._frame_infos(REC, 1)])   # helper added below; (C,H,W,3) in CAMS order
    w = r.windows(frames, np.ones(7, bool), s["center3D"][None])
    np.testing.assert_array_equal(w["origin"][0], s["crop_origin"])
    np.testing.assert_allclose(w["t_local"][0, 0], s["t_local"][0], atol=1e-4)
    np.testing.assert_array_equal(w["crops"][0, 0], s["crops"][0])

def test_runner_infer_and_read_typed_shapes(tmp_path):
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    root, final = _tiny_final(tmp_path)
    r = MVQRunner(final, calib_dir=f"{root}/calibrations/A", cameras=CAMS, batch=2)
    frames = np.zeros((7, 448, 1936, 3), np.uint8)
    out = r.infer(r.windows(frames, np.ones(7, bool), np.zeros((1, 3), np.float32)))
    assert out["kp3d"].shape == (1, 4, 50, 3) and out["kp2d"].shape == (1, 4, 7, 50, 2) and out["exist"].shape == (1, 4)
    res = r.read_typed({k: v for k, v in out.items()}, 0, want_sex=0)
    assert res is None or res["slot"] == 1

def test_to_pipeline_permutes_by_name_and_signs_gates(tmp_path):
    from jarvis_jax.tracking.lift_mvq import MVQRunner
    import yaml
    root, final = _tiny_final(tmp_path)
    r = MVQRunner(final, calib_dir=f"{root}/calibrations/A", cameras=CAMS)
    model_names = list(reversed(r.kp_names))                                     # any permutation of the same names
    T, C, K = 3, 7, 50
    kp3d = np.random.default_rng(0).normal(size=(T, K, 3)).astype(np.float32)
    kp2d = np.zeros((T, C, K, 2), np.float32); vis = np.full((T, C, K), 0.9, np.float32); raw = np.full((T, K), 0.05, np.float32)
    p = r.to_pipeline(kp3d, kp2d, vis, raw, model_names)
    iL, iR = r.kp_names.index("EyeL"), r.kp_names.index("EyeR"); jL, jR = model_names.index("EyeL"), model_names.index("EyeR")
    np.testing.assert_allclose(p["kp3d"][:, jL], kp3d[:, iL]); np.testing.assert_allclose(p["kp3d"][:, jR], kp3d[:, iR])
    assert list(p["kp_names"]) == model_names and p["conf3d"].shape == (T, K) and abs(p["conf3d"].mean() - 0.9) < 1e-6
    assert p["gates"]["lifter"] == "mvq" and len(p["gates"]["sha256"]) == 16
```

Add a small helper to `V12WindowDataset`: `_frame_infos(rec, frame) -> list[(img_id, ann_id)]` in `cameras` order for the host frameset (from `iter_resolved_slots` of `self._fs[(rec, frame, 0)]`, mapped to camera row via `self._rt(rec).cameras`), used only by tests and figure scripts.

- [ ] **Step 2: Run to verify failures** — ImportError.

- [ ] **Step 3: Implement `lift_mvq.py`** — move `build_window`-equivalent geometry from `scripts/viz/mvq_bout_video.py` into `MVQRunner.windows` (vectorised over B centres), `run_inference`'s forward/`assemble` into `infer` (jit the forward once for the fixed batch shape; pad with zeros and drop the padding), the slot choice into `read_typed`, and the npz writing into `to_pipeline`. `gates_signature`: hash the first 16 hex chars of `hashlib.sha256` over the checkpoint dir's `_METADATA` (final/) or `ckpt/<step>/_CHECKPOINT_METADATA` bytes, plus path and step. Read `scripts/run_bout.py` lines 360-420 and 1575-1610 and make the check accept `{"lifter": "mvq", ...}` (add the branch if needed, minimal, documented).

- [ ] **Step 4: Refactor `scripts/viz/mvq_bout_video.py`** to construct `MVQRunner` once and call `windows/infer/read_typed/to_pipeline`; keep `BoutMaskStore`, `centers_3d`, rendering and the CLI unchanged; the prompted mode stays available there (it is the only consumer): add `prompt_mask` as an optional argument to `windows`/`infer` (`prompt_on` True only when given). Re-run the script's smoke (`--n 4` on CPU, `--attn_impl xla`) and confirm `kp3d.npz` for fly0 is numerically identical (allclose 1e-4) to the previous smoke output at `/tmp/claude-398823/.../scratchpad/bout28_smoke/unprompted/fly0/kp3d.npz` if it still exists, else to a fresh run of the pre-refactor script checked out from git (`git show HEAD:scripts/viz/mvq_bout_video.py > /tmp/old_bout_video.py`).

- [ ] **Step 5: Measure the window cost** on a GPU: `MVQRunner(final P3a, batch=32).infer` on 32 windows of real frames from bout 28 (frames via `read_window`), 20 timed iterations after 2 warm-ups; record ms/window (fp32 and, if `attn_impl cudnn` requires it, bf16 crops) in `p4-maskfree-notes.md` under "Window cost" together with the §7 decision (stride 1 vs stride 2 for the fine pass).

- [ ] **Step 6: Run** `JAX_PLATFORMS=cpu pytest tests/test_lift_mvq.py tests/test_v12_windows.py tests/test_train_mvq_smoke.py -q -m "not gpu"` -> all pass.

- [ ] **Step 7: Commit** — `git add third_party/jarvis_jax/jarvis_jax/tracking/lift_mvq.py third_party/jarvis_jax/jarvis_jax/data/v12_windows.py scripts/viz/mvq_bout_video.py scripts/run_bout.py third_party/jarvis_jax/tests/test_lift_mvq.py && git add -f docs/benchmark/2026-09-mvq/p4-maskfree-notes.md && git commit -m "feat(mvq): MVQRunner -- shared frames->windows->pipeline-format runner; bout video refactored onto it (P4a)"`.

---

### Task 3: Coarse localiser — CenterDetect peaks to 3D centres and window plans

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/coarse_centres.py`
- Test: `third_party/jarvis_jax/tests/test_coarse_centres.py`

**Interfaces (Produces):**

```python
class CenterDetector:
    def __init__(self, ckpt_dir, *, min_score=0.2)          # EfficientTrack(num_joints=1, in_channels=3, model_size="medium"), Orbax StandardCheckpointer restore, model.eval()
    def peaks(self, frames) -> (peaks (C,2,2) float32 full-frame px, scores (C,2) float32)   # frames (C,H,W,3) u8 RGB; k=2 peaks per camera; score<min_score -> NaN peak
def lift_peaks_to_centres(peaks, scores, cam_mats, *, min_views=3, max_resid_px=25.0, max_animals=2) -> (centres (A,3), n_views (A,), score (A,))
    # greedy: seed candidates from every pair of cameras x their peaks (triangulate_dlt_batched), reproject to all cameras
    # (project_center_to_cameras), inliers = cameras with a peak within max_resid_px; take the candidate with most inliers
    # (ties: higher summed score), refine by DLT over its inliers, remove those peaks, repeat up to max_animals; NaN-padded
def cluster_centres(centres, *, min_sep_units=15.0) -> centres      # merge centres closer than 1.5 mm (mean), NaN-padded to the input length
def plan_windows(centres, *, merge_dist_units=30.0) -> (window_centres (W,3), assignment (A,) int)   # one window per centre; two centres within 3 mm share one window at their midpoint
```

- [ ] **Step 1: Failing tests** — synthetic affine cameras from the fixture (`ReprojectionTool(f"{root}/calibrations/A")`), two true 3D points, project with `rt.reproject_point`, add 1 px noise, feed as peaks:

```python
def test_lift_two_flies_from_perfect_peaks(tmp_path):
    from jarvis_jax.tracking.coarse_centres import lift_peaks_to_centres
    root = make_v12_root(tmp_path); rt = ReprojectionTool(f"{root}/calibrations/A")
    X = np.array([[0.0, 0.0, 0.0], [40.0, 5.0, 0.0]])
    peaks = np.stack([np.stack([rt.reproject_point(x) for x in X], 0) for _ in range(7)])        # (C,2,2) -- both flies in every camera
    peaks = peaks.astype(np.float32) + np.random.default_rng(0).normal(scale=1.0, size=peaks.shape).astype(np.float32)
    c, nv, sc = lift_peaks_to_centres(peaks, np.ones((7, 2), np.float32), rt.camera_matrices)
    order = np.argsort(c[:, 0]); np.testing.assert_allclose(c[order], X, atol=1.0); assert (nv[order] >= 6).all()

def test_lift_tolerates_missing_and_swapped_peaks(tmp_path):
    ...  # drop fly 1's peak in 3 cameras (NaN), swap the peak order in 2 cameras -> both centres still recovered within 1.5 units, n_views (7, 4)

def test_lift_rejects_inconsistent_peaks(tmp_path):
    ...  # one camera's peak moved by 80 px -> that camera is not an inlier; centre within 1 unit of truth using the other 6

def test_single_fly_and_min_views(tmp_path):
    ...  # only fly 0 present -> one centre, second row NaN; only 2 cameras have peaks -> no centre (min_views=3)

def test_cluster_and_plan_windows():
    from jarvis_jax.tracking.coarse_centres import cluster_centres, plan_windows
    c = np.array([[0, 0, 0], [8.0, 0, 0]], np.float32)             # 0.8 mm apart -> one cluster
    cc = cluster_centres(c); assert np.isnan(cc[1]).all() and np.allclose(cc[0], [4, 0, 0])
    w, a = plan_windows(np.array([[0, 0, 0], [20.0, 0, 0]], np.float32))      # 2 mm apart -> one shared window at the midpoint
    assert w.shape == (1, 3) and np.allclose(w[0], [10, 0, 0]) and a.tolist() == [0, 0]
    w, a = plan_windows(np.array([[0, 0, 0], [50.0, 0, 0]], np.float32))      # 5 mm apart -> two windows
    assert w.shape == (2, 3) and a.tolist() == [0, 1]
```

(`CenterDetector` itself is GPU/heavy: test only its `peaks()` post-processing on a hand-built heatmap via a `_peaks_from_heatmap(hm (C,160,160), W, H, min_score)` helper: two Gaussian blobs -> two full-frame peaks at the expected positions with the anisotropic x/y scale; a blob below `min_score` -> NaN.)

- [ ] **Step 2: Run to verify failures**; **Step 3: Implement** (`CenterDetector.peaks`: `cv2.resize(frame, (320, 320), interpolation=cv2.INTER_LINEAR)`, ImageNet normalise, `model(x, use_running_average=True)` -> `(C,160,160,1)`, `extract_top_k_peaks(k=2, suppression_radius=15)`, `peaks_to_full_image(peaks, 160, W, H)`; restore pattern from `jarvis_jax/scripts/viz_centerdetect_fp.py::_restore`); **Step 4: Run** `JAX_PLATFORMS=cpu pytest tests/test_coarse_centres.py -q`; **Step 5: Commit** `feat(mvq): coarse localiser -- CenterDetect peaks to 3D centres and window plans (P4b §4.1)`.

---

### Task 4: Coarse pass over a recording

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/coarse_track.py`, `scripts/coarse_pass_mvq.py`
- Test: `third_party/jarvis_jax/tests/test_coarse_track.py`

**Interfaces (Produces):**

```python
def coarse_pass(reader, runner: MVQRunner, detector: CenterDetector, *, frames, num_animals=2, merge_dist_units=30.0,
                min_views=3, max_resid_px=25.0, floor=None, progress=None) -> dict
    # reader: callable(frame_idx) -> (frames (C,H,W,3) u8, present (C,) bool); frames: iterable of absolute frame indices (the stride-16 sample)
    # per frame: detector.peaks -> lift_peaks_to_centres -> cluster -> plan_windows -> runner.windows/infer -> runner.read_typed(want_sex=0) and (want_sex=1)
    #   (single-fly recordings: read both typed slots, keep the one that exists; report its sex)
    #   with NO centre this frame: reuse the previous frame's window centres (flag `centre_source = 1`), NaN outputs if that also fails
    # returns arrays: frame (N,), and per fly f in (0=female, 1=male): kp3d (N,K,3) world, centroid (N,3), exist (N,), sex_prob (N,), slot (N,), centre_source (N,) int8
def coarse_features(tracks, kp_names, *, floor) -> dict
    # inter-fly distance (N,), male_to_female_heading_deg (N,), speed (2,N) units per coarse frame, wing_angle_deg (2,N) (angle between the body axis
    # Scutellum->Abd_tip and each wing vector WingX_base->WingX_V13, max over L/R), height (2,N) above the floor plane, trackable (2,N) = exist>=0.5 & finite
def write_coarse_tracks(path, tracks, features, cameras, *, session_dir, stride, num_animals, cam_mats)
    # coarse_tracks.npz in the SAM3 coarse-pass schema (coarse_frame (T,), cameras (C,), centroid (2,C,T,2) px = reprojected 3D centroids, valid (2,C,T),
    # border_dist (2,C,T) px, in_frame (2,C,T) int8 (1 inside image, 0 outside), area (2,C,T) = NaN (no masks), area_med, border_med, n_valid_cams,
    # X3d (2,T,3), sep3d (T,), sep2d_med (T,), filled (T,)) PLUS mvq fields (exist (2,T), sex_prob (2,T), wing_angle_deg (2,T), heading_deg (T,),
    # speed (2,T), height (2,T), kp3d (2,T,K,3) f16, kp_names) and the companion .meta.json (session_dir, stride, cameras, W, H, num_animals, n_coarse, coarse0, source="mvq")
```

Floor plane: fit once per recording from the first 2000 coarse centroids with finite values (least-squares plane; z-up sign chosen so that the median fly height is positive).

`scripts/coarse_pass_mvq.py` (Hydra-free argparse): `--session-dir --calib-dir --cameras (comma list; default from configs/recording/session0.yaml) --run --step --centerdetect --stride 16 --start 0 --end (default: video length from cv2) --out <dir>/coarse_tracks.npz --batch 32 --num-animals 2`; reads frames with `read_window(session_dir, cameras, load_plan(session_dir), slot, 1)` (from `jarvis_jax.predict.synced_reader`); prints throughput every 500 coarse frames; resumable via `--resume` (appends from the last written frame — write partial npz every 2000 coarse frames as `coarse_tracks.partial.npz`).

- [ ] **Step 1: Failing test** — a fake `reader` returning fixture frames, a fake `detector` returning the projected true centres as peaks, the tiny-model `MVQRunner` (from Task 2's `_tiny_final` helper, import it from `test_lift_mvq`), `frames=[0, 1, 2]`: output arrays have the documented shapes, `frame == [0,1,2]`, `centre_source` is 0 for frames with a detection, and `write_coarse_tracks` produces an npz that `scripts/coarse_pass_gates.py`'s loader (`load_tracks` or equivalent — read the script) opens without error (assert the SAM3-schema keys exist with the right shapes and `area` is all-NaN).

- [ ] **Step 2-4**: verify failure, implement, pass. **Step 5**: run the real coarse pass on 20_04 on this node (GPU 0): `PYTHONPATH=third_party/jarvis_jax:. python scripts/coarse_pass_mvq.py --session-dir /gscratch/.../Session0/2025_10_20_13_20_04 --calib-dir .../calibration --run .../mvq_t1_b16_p3a_20260904/final --centerdetect /gscratch/portia/eabe/data/Johnson_lab/jax_centerdetect_runs/cd_focal_bg30/ckpt/epoch_004 --stride 16 --out /gscratch/portia/eabe/data/Johnson_lab/processed/courtship/Session0/2025_10_20_13_20_04/coarse_mvq/coarse_tracks.npz` in the foreground with `run_in_background: true` and a completion watch (expected 30-60 min). Record wall clock per stage in the notes ("Coarse pass timing").

- [ ] **Step 6: Commit** `feat(mvq): mask-free coarse pass -- CenterDetect + mvq at stride 16 -> coarse_tracks.npz (P4b §4.3)`.

---

### Task 5: Gates read mvq tracks; bout table for 20_04

**Files:**
- Modify: `scripts/coarse_pass_gates.py`
- Test: `third_party/jarvis_jax/tests/test_coarse_pass_gates.py` (the script is importable from repo root; add `sys.path` insertion in the test like the other repo-root script tests, or move the gate functions into `jarvis_jax/tracking/bout_gates.py` and have the script import them — prefer the move if the script has no package-level dependencies on `scripts/`).

**Interfaces (Produces):** when the npz has `source == "mvq"` (meta) or an `exist` array: `per_fly_trackable = (exist >= 0.5) & (n_valid_cams >= min_cams)` (no area gate), `behaviour_ok = (wing_angle_deg[male] >= --wing-angle-min (default 30)) | (sep3d <= --proximity-max-units (default 30))`; everything else (separable, min duration, max gap, CSV writer, `--ground-truth` scoring) unchanged. CLI gains `--wing-angle-min`, `--proximity-max-units`; the SAM3 path is untouched (regression test: a synthetic SAM3-schema npz produces the same CSV before and after).

- [ ] **Step 1: Failing tests**: (a) mvq-schema npz with a hand-built 200-coarse-frame trace (flies far apart, then close with the male's wing angle 45 deg for 60 frames, then apart) -> exactly one bout covering the close stretch, boundaries within `--max-gap`; (b) the SAM3-schema regression; (c) `--ground-truth` scoring on (a) with a GT CSV equal to the truth -> recall 1, precision 1, offsets 0.
- [ ] **Step 2-4** verify failure, implement, pass.
- [ ] **Step 5: Run on 20_04**: `python scripts/coarse_pass_gates.py --tracks .../coarse_mvq/coarse_tracks.npz --out-csv .../coarse_mvq/bouts_mvq_gates.csv --session-tag Session0/2025_10_20_13_20_04_fly0 --ground-truth /gscratch/.../2025_10_20_13_20_04/courtship_bouts_fly0_summary.csv` and record recall / precision / boundary offsets in the notes next to the SAM3 coarse-pass calibration numbers from `docs/specs/2026-08-31-pipeline-schematic-and-inventory-design.md`.
- [ ] **Step 6: Commit** `feat(mvq): coarse-pass gates read mvq tracks (existence + wing angle); 20_04 bout table (P4b §5.1)`.

---

### Task 6: Figure gates 1 and 2

**Files:**
- Create: `scripts/viz/coarse_centres_check.py`, `scripts/viz/coarse_tracks_check.py`
- Modify: `docs/benchmark/2026-09-mvq/p4-maskfree-notes.md`

- [ ] **Step 1: `coarse_centres_check.py`** — docstring EXPECTATION (spec §8.1 verbatim): "every visible fly has a centre within its body in both cameras; when the flies touch, one centre between them is acceptable; no centre on the wall or reflection." 12 coarse frames spread over the recording (include 4 inside reviewed bouts, 4 outside, 4 at bout boundaries), two cameras named by `cameras` (one overhead `Cam2012630`, one side `Cam2012861`): draw CenterDetect peaks (small yellow squares) and the reprojected 3D centres (large circles, female cyan / male orange by the slot mvq assigned), full-frame crops 700 px wide around the centres. Output `figures/2026-09-mvq/p4_maskfree/coarse_centres_check.png` + JSON of the frames and centres.
- [ ] **Step 2: `coarse_tracks_check.py`** — docstring EXPECTATION (§8.2): "reviewed bouts coincide with close-distance, wing-extension episodes; a reviewed bout that does not is recorded." Four stacked time panels over the whole recording (x in minutes): inter-fly distance (mm), male wing angle (deg), both flies' speed (mm/s), existence (both flies); reviewed bouts shaded grey, gate-derived bouts drawn as a bar row; a zoom panel on bout 28. Output PNG + JSON.
- [ ] **Step 3: Run both** on the 20_04 outputs, READ both PNGs, and write the notes section "Figure gates 1-2" with what was seen against each expectation, plus the recall/precision table from Task 5 and the timing from Task 4. State plainly any reviewed bout the tracks do not explain.
- [ ] **Step 4: Commit** `viz(mvq): coarse-pass figure gates (centres, tracks) and 20_04 readings (P4b §8)`.

---

## Self-review

- Spec coverage: §6 -> Task 1; §4.2 -> Task 2; §4.1 -> Task 3; §4.3 -> Task 4; §5.1 -> Task 5; §8 gates 1, 2, 6 -> Tasks 6, 1; §7 window-cost measurement -> Task 2 step 5. §4.4-4.6, §5.2, §8 gates 3-5 are Plan 2.
- Names: `MVQRunner.windows/infer/read_typed/to_pipeline/gates_signature` (Task 2) are what Task 4's `coarse_pass` calls; `CenterDetector.peaks`, `lift_peaks_to_centres`, `cluster_centres`, `plan_windows` (Task 3) are what Task 4 calls; `write_coarse_tracks`' npz keys (Task 4) are what Task 5's gates read (`exist`, `wing_angle_deg`, `sep3d`, `n_valid_cams` plus the SAM3-schema keys).
- Judgement calls flagged to implementers: the `gates` signature acceptance in `run_bout.py` (Task 2), the CenterDetect restore pattern (Task 3), the gates loader function name (Task 4/5), the floor-plane sign (Task 4).
