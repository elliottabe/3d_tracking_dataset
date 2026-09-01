# Wing Orientation From SAM Masks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the IK's wing blade orientation — wings currently rotate inward through the abdomen — by refining wing PITCH against the SAM masks after STAC, without disturbing the courtship song.

**Architecture:** A post-STAC, wings-only, opt-in refinement stage. Precompute a per-(frame, camera) signed distance field from the masks; run Adam on `wing_pitch_left/right` only (everything else frozen) against a two-term cost: containment (wing vertices must not lie outside the fly mask) plus wing coverage (mask area the BODY does not explain must be explained by the WINGS). No new term in the main marker solve, no rest prior.

**Tech Stack:** NumPy/SciPy/cv2 for the offline SDF; JAX + optax for the differentiable refinement; MuJoCo/MJX for FK; Hydra config; pytest.

**Spec:** `docs/specs/2026-09-01-wing-orientation-from-masks-design.md` — read it first. It carries the root-cause measurements, four rejected alternatives, and the verification of every recovered file.

## Global Constraints

- **Wing pitch only.** Optimise `wing_pitch_left` and `wing_pitch_right`. Wing **yaw** and **roll** are frozen — yaw carries the song, roll is the strongest-observed wing DOF (per-column sensitivity 0.344 vs pitch 0.042). Root, thorax, abdomen and legs frozen.
- **Song guard is PER-DOF.** `hp_rms` of `wing_pitch_*` on the SINGING wing must stay within 20% of control (control 0.3615). A yaw-only guard is blind to the failure that killed the previous attempt.
- **Blade-normal is NEVER an acceptance gate.** It is a hypersensitive function of the null direction (a 12% marker-residual change swung it 59°). Report it; do not gate on it.
- **Opt-in.** `wing_mask_fit.enabled: false` by default. Absent config is a strict no-op.
- **Never index a keypoint or camera axis by integer.** Use `viz.core.bout_artifacts.load_bout_kp` / `centroids_canonical`. `kp2d/kp3d` are in `cfg.model.KP_NAMES` order; the camera axis is canonical `cfg.recording.cameras` order. Three index-space errors happened in one session; see CLAUDE.md.
- **Verify recovered code, do not trust docstrings.** Three docstrings in this repo were contradicted by measurement. Every recovered file gets a test asserting its documented behaviour.
- Never `git add -A` / `git commit -a`; explicit paths only. Never commit PNGs or videos; figures go under `figures/<date>-<topic>/`.
- **Performance is a requirement, not a follow-up.** The per-bout budget after
  `3eab485` is 1230 s, of which STAC IK is 746 s (61%). The new stage gets a
  **hard budget of 60 s per bout-fly** (< 5% of the bout). If it cannot be met,
  reduce `n_steps` or the vertex count -- do not ship a stage that doubles the bout.
- **Never regress the numbers for speed.** The discipline that made `3eab485`
  trustworthy: profile first (`cProfile`, real frames), then prove the outputs
  unchanged against the artifacts already on disk -- bit-identical, or a
  documented 1-2 ulp with the reassociation named. `np.einsum`, not `@` (BLAS
  reassociates a 4-term dot and drifts 2.3e-13 px); a float64 camera-matrix
  stack, not the float32 `camera_matrices` attribute.
- **Do not materialise the masks again.** `load_bout_masks` already builds a
  12.2 GB `(T,C,H,W)` bool array and Stage E peaks near 27 GB RSS. The SDF
  precompute must stream over it, not add a second copy.

- MuJoCo renders need `MUJOCO_GL=egl`; `unset LD_LIBRARY_PATH` before JAX. You are on a GPU node — run directly, do not sbatch.

---

### Task 1: Cropped SDF from masks — **ALREADY DONE (`c7df5bc`)**

**Files:** `third_party/jarvis_jax/jarvis_jax/tracking/mask_sdf.py`, `third_party/jarvis_jax/tests/test_mask_sdf.py`

**Interfaces:**
- Produces: `mask_bbox(mask, margin) -> (x0,y0,x1,y1)|None`; `mask_to_sdf_crop(mask, bbox, out_hw) -> (sdf (H,W) f32, grid_scale (2,), grid_offset (2,))|None`; `sdf_stack_from_masks(masks, valid, *, out_hw, bbox_margin) -> (sdf (T,C,H,W), grid_scale (T,C,2), grid_offset (T,C,2), present (T,C))`; `BIG`.
- Convention: `grid_xy = (orig_xy - grid_offset) * grid_scale`. SDF negative inside, positive outside, in ORIGINAL pixels.

Recovered from `silhouette_sdf.py` with its averaged-scale bug fixed (26.6% → 1.2% error on an anisotropic crop). 6 tests pass. Nothing to do; later tasks consume this interface.

---

### Task 2: Differentiable containment residual

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/mask_containment.py`
- Test: `third_party/jarvis_jax/tests/test_mask_containment.py`

**Interfaces:**
- Consumes: `mask_sdf`'s `grid_scale`/`grid_offset` convention.
- Produces: `bilinear_sample(img, xy) -> (M,)`; `containment_residual(proj_pts, sdf, grid_scale, grid_offset, conf, *, margin=0.0, present=True) -> (M,)`.

- [ ] **Step 1: Recover the file verbatim, then rename**

```bash
cd /mmfs1/gscratch/portia/eabe/Research/MyRepos/3d_tracking_dataset
git show 0bc36fe^:third_party/jarvis_jax/jarvis_jax/tracking/silhouette_containment.py \
  > third_party/jarvis_jax/jarvis_jax/tracking/mask_containment.py
```

Then edit its module docstring to say it is recovered from `silhouette_containment.py` (deleted in `0bc36fe`), that it is used for WING pitch refinement only, and — importantly — that it is **one-sided**: `relu(d + margin)` means vertices inside the mask cost nothing, so minimising it ALONE would tuck the wing inside the body silhouette, which is the bug being fixed. Task 4 supplies the opposing term.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np
import jax.numpy as jnp
import pytest
from jarvis_jax.tracking.mask_containment import bilinear_sample, containment_residual
from jarvis_jax.tracking.mask_sdf import mask_bbox, mask_to_sdf_crop


def _sdf_box(h=40, w=60):
    m = np.zeros((200, 300), bool)
    m[80:80 + h, 120:120 + w] = True
    sdf, gs, go = mask_to_sdf_crop(m, mask_bbox(m, 0.4), (64, 64))
    return jnp.asarray(sdf), jnp.asarray(gs), jnp.asarray(go), m


def test_inside_costs_nothing_outside_costs_distance():
    sdf, gs, go, _ = _sdf_box()
    pts = jnp.asarray([[150.0, 100.0],        # box centre, inside
                       [250.0, 100.0]])       # well to the right, outside
    r = containment_residual(pts, sdf, gs, go, jnp.ones(2))
    assert float(r[0]) == pytest.approx(0.0, abs=1e-6), "inside must be free"
    assert float(r[1]) > 5.0, "outside must be penalised in original px"


def test_present_false_zeroes_the_residual():
    sdf, gs, go, _ = _sdf_box()
    pts = jnp.asarray([[250.0, 100.0]])
    assert float(containment_residual(pts, sdf, gs, go, jnp.ones(1),
                                      present=False)[0]) == 0.0


def test_gradient_points_back_toward_the_mask():
    """The whole purpose: an outside vertex must feel an inward pull."""
    import jax
    sdf, gs, go, _ = _sdf_box()
    f = lambda x: float(containment_residual(
        jnp.asarray([[x, 100.0]]), sdf, gs, go, jnp.ones(1))[0])
    g = jax.grad(lambda x: containment_residual(
        jnp.stack([jnp.stack([x, jnp.asarray(100.0)])]),
        sdf, gs, go, jnp.ones(1))[0])(jnp.asarray(250.0))
    assert float(g) > 0.0, "moving further right must increase the cost"
    assert f(250.0) > f(200.0), "closer to the mask must cost less"


def test_far_outside_the_grid_keeps_a_finite_gradient():
    """The recovered code adds an `overflow` term so a vertex beyond the SDF crop
    does not sit on a clamp plateau with zero gradient."""
    sdf, gs, go, _ = _sdf_box()
    near = containment_residual(jnp.asarray([[250.0, 100.0]]), sdf, gs, go, jnp.ones(1))
    far = containment_residual(jnp.asarray([[900.0, 100.0]]), sdf, gs, go, jnp.ones(1))
    assert float(far[0]) > float(near[0]) + 50.0, "no plateau far from the crop"


def test_conf_scales_the_residual_linearly():
    sdf, gs, go, _ = _sdf_box()
    pts = jnp.asarray([[250.0, 100.0]])
    a = float(containment_residual(pts, sdf, gs, go, jnp.asarray([1.0]))[0])
    b = float(containment_residual(pts, sdf, gs, go, jnp.asarray([0.5]))[0])
    assert b == pytest.approx(0.5 * a, rel=1e-5)


def test_bilinear_sample_matches_the_array_at_integer_coords():
    img = jnp.asarray(np.arange(25, dtype=np.float32).reshape(5, 5))
    got = bilinear_sample(img, jnp.asarray([[2.0, 3.0]]))   # (x=2, y=3)
    assert float(got[0]) == pytest.approx(float(img[3, 2]))
```

- [ ] **Step 3: Run them; expect failures only from a wrong import path**

Run: `JAX_PLATFORMS=cpu python -m pytest third_party/jarvis_jax/tests/test_mask_containment.py -q`
Expected: collection error or failures naming `silhouette_*` imports.

- [ ] **Step 4: Fix imports in the recovered file**

Replace any `from jarvis_jax.tracking.silhouette_*` with the new names. `mask_containment` should import only `jax`/`jax.numpy`.

- [ ] **Step 5: Re-run to green**

Run: `JAX_PLATFORMS=cpu python -m pytest third_party/jarvis_jax/tests/test_mask_containment.py -q`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/tracking/mask_containment.py \
        third_party/jarvis_jax/tests/test_mask_containment.py
git commit -m "feat(mask-containment): recover the differentiable SDF-relu residual, with tests

Recovered from silhouette_containment.py (deleted in 0bc36fe). Verified rather
than trusted: one-sided relu(d+margin), present-gated, conf-linear, and the
overflow term really does keep a finite inward gradient far outside the crop.
ONE-SIDED BY DESIGN -- minimising it alone would tuck the wing inside the body
silhouette, which is the bug being fixed; the wing-coverage term opposes it."
```

---

### Task 3: Appendage DOF + wing vertex selection

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/appendage_dof.py`
- Test: `third_party/jarvis_jax/tests/test_appendage_dof.py`

**Interfaces:**
- Produces: `APPENDAGE_PATTERNS`; `build_appendage_dof_mask(model, include=("wing",)) -> (nq,) bool`; `appendage_vertex_indices(mesh_npz, subset="fps_300", include=("wing",)) -> (M,) int32` FULL-vertex-array indices.

- [ ] **Step 1: Recover and rename**

```bash
git show 0bc36fe^:third_party/jarvis_jax/jarvis_jax/tracking/silhouette_dof.py \
  > third_party/jarvis_jax/jarvis_jax/tracking/appendage_dof.py
git show 0bc36fe^:third_party/jarvis_jax/jarvis_jax/tracking/silhouette_targets.py \
  > third_party/jarvis_jax/jarvis_jax/tracking/mask_fk_indices.py
```

`appendage_dof` needs only `silhouette_fk_indices` from the second file; rename that function to `mask_fk_indices` and drop `build_silhouette_targets` (it indexes a COCO dataset the pipeline does not use). Fix imports.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np
import mujoco
import pytest
from omegaconf import OmegaConf
from jarvis_jax.tracking.appendage_dof import (build_appendage_dof_mask,
                                               appendage_vertex_indices)

ANATOMY = "configs/anatomy/v1.yaml"


def _model():
    from hydra import initialize, compose
    with initialize(config_path="../../../configs", version_base=None):
        c = compose(config_name="pipeline", overrides=["paths=hyak"])
    return mujoco.MjModel.from_xml_path(c.anatomy.mjcf_path), c


def test_wing_selects_exactly_the_six_wing_joints():
    mj, _ = _model()
    m = np.asarray(build_appendage_dof_mask(mj, include=("wing",)))
    sel = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(mj.njnt)
           if int(mj.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
           and m[int(mj.jnt_qposadr[j])]]
    assert len(sel) == 6 and all("wing" in s for s in sel), sel


def test_abdomen_pattern_does_not_catch_the_leg_coxa_abduct():
    """The documented reason the pattern is `abdomen` and not a bare `abd`.
    NOTE abdomen_abduct_* ARE abdomen joints -- the trap is `coxa_abduct`."""
    mj, _ = _model()
    m = np.asarray(build_appendage_dof_mask(mj, include=("abdomen",)))
    sel = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_JOINT, j)
           for j in range(mj.njnt)
           if int(mj.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE)
           and m[int(mj.jnt_qposadr[j])]]
    assert sel and not any("coxa" in s for s in sel), sel


def test_free_root_joint_is_never_selected():
    mj, _ = _model()
    m = np.asarray(build_appendage_dof_mask(mj, include=("wing", "leg", "abdomen")))
    assert not m[:7].any(), "the free joint's 7 qpos must stay frozen"


def test_wing_vertex_selection_is_wing_only_BOTH_wings():
    """INDEX-SPACE TRAP: vertex_segment stores seg_ids VALUES (1..67) while
    seg_names is positional (0..66). Indexing seg_names by a vertex_segment
    value is off by one and made an earlier check report 'abdomen, wing_right'.
    Build the map by zipping seg_ids with seg_names."""
    mj, c = _model()
    npz = c.anatomy.cse_mesh_npz
    z = np.load(npz, allow_pickle=True)
    id2name = dict(zip(np.asarray(z["seg_ids"]).tolist(),
                       [str(s) for s in z["seg_names"]]))
    vseg = np.asarray(z["vertex_segment"])
    idx = np.asarray(appendage_vertex_indices(npz, subset="fps_300", include=("wing",)))
    hit = sorted({id2name[int(vseg[int(v)])] for v in idx})
    assert hit == ["wing_left", "wing_right"], hit
    assert len(idx) == 100, f"expected 100 wing verts in fps_300, got {len(idx)}"


def test_returned_indices_are_FULL_array_space_not_subset_space():
    """The recovered docstring warns these are full-array indices (0..139352),
    unlike wing_side_vertices which returns fps-subset indices. Mixing them
    silently selects the wrong vertices."""
    mj, c = _model()
    npz = c.anatomy.cse_mesh_npz
    z = np.load(npz, allow_pickle=True)
    idx = np.asarray(appendage_vertex_indices(npz, subset="fps_300", include=("wing",)))
    assert idx.max() > 300, "full-array indices should exceed the subset size"
    assert idx.max() < len(z["vertex_segment"])
```

- [ ] **Step 3: Run; expect import failures**

Run: `JAX_PLATFORMS=cpu python -m pytest third_party/jarvis_jax/tests/test_appendage_dof.py -q`

- [ ] **Step 4: Fix imports and the renamed function; re-run to green**

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/tracking/appendage_dof.py \
        third_party/jarvis_jax/jarvis_jax/tracking/mask_fk_indices.py \
        third_party/jarvis_jax/tests/test_appendage_dof.py
git commit -m "feat(appendage-dof): recover wing DOF + vertex selection, with the index-space trap pinned"
```

---

### Task 4: Wing coverage residual (the opposing term)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/wing_coverage.py`
- Test: `third_party/jarvis_jax/tests/test_wing_coverage.py`

**Interfaces:**
- Consumes: `appendage_dof.appendage_vertex_indices`, `mask_sdf` conventions.
- Produces:
  - `wing_target_points(mask, body_uv, *, n_points, dilate_px=3, seed=0) -> (n_points, 2) float32` — sample mask pixels the BODY does not explain (NaN-padded if fewer exist).
  - `coverage_residual(target_pts, proj_wing_uv, *, beta=8.0, huber_delta=0.0) -> (n_points,)` — softmin distance from each target to the nearest projected wing vertex.

- [ ] **Step 1: Recover the chamfer core**

```bash
git show 0bc36fe^:third_party/jarvis_jax/jarvis_jax/tracking/silhouette_chamfer.py \
  > /tmp/chamfer_ref.py
```
Use its `softmin = -1/beta * logsumexp(-beta * d)` formulation and its NaN-safety and memory-chunking behaviour for `coverage_residual`. Do **not** copy `build_silhouette_targets`; targets here are wing-attributable mask pixels, computed by `wing_target_points`.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np
import jax.numpy as jnp
import pytest
from jarvis_jax.tracking.wing_coverage import wing_target_points, coverage_residual


def test_targets_exclude_pixels_the_body_already_explains():
    mask = np.zeros((100, 100), bool)
    mask[40:60, 20:80] = True                 # a wide bar
    body_uv = np.stack(np.meshgrid(np.arange(20, 50), np.arange(40, 60)),
                       -1).reshape(-1, 2).astype(np.float32)   # covers the LEFT half
    pts = np.asarray(wing_target_points(mask, body_uv, n_points=200, dilate_px=0, seed=0))
    good = pts[np.isfinite(pts).all(1)]
    assert len(good) > 20
    assert good[:, 0].min() > 45, "targets must avoid the body-covered left half"


def test_coverage_pulls_toward_an_uncovered_target():
    tgt = jnp.asarray([[50.0, 50.0]])
    near = coverage_residual(tgt, jnp.asarray([[52.0, 50.0]]))
    far = coverage_residual(tgt, jnp.asarray([[90.0, 50.0]]))
    assert float(near[0]) < float(far[0])


def test_softmin_approaches_the_hard_min_for_large_beta():
    tgt = jnp.asarray([[0.0, 0.0]])
    verts = jnp.asarray([[3.0, 0.0], [10.0, 0.0]])
    r = float(coverage_residual(tgt, verts, beta=200.0)[0])
    assert r == pytest.approx(3.0, abs=0.1)


def test_nan_targets_contribute_zero():
    tgt = jnp.asarray([[np.nan, np.nan], [0.0, 0.0]])
    r = coverage_residual(tgt, jnp.asarray([[1.0, 0.0]]))
    assert float(r[0]) == 0.0 and float(r[1]) > 0.0


def test_gradient_flows_to_the_wing_vertices():
    """A descent step must move the vertex TOWARD the target.

    The gradient itself is POSITIVE here: the cost is the distance from the
    target to the nearest vertex, so pushing the vertex further right (away)
    raises it. Descent (`v -= lr * grad`) therefore moves the vertex left,
    toward the target at x=0. Verified: cost 4.0/5.0/6.0 at x=4/5/6, analytic
    grad +1.0, finite-difference +0.99993.
    """
    import jax
    tgt = jnp.asarray([[0.0, 0.0]])
    g = jax.grad(lambda v: coverage_residual(tgt, v)[0])(jnp.asarray([[5.0, 0.0]]))
    assert float(g[0, 0]) > 0.0, "cost must rise as the vertex moves away"
    step = 5.0 - 0.5 * float(g[0, 0])
    assert step < 5.0, "a descent step must move the vertex toward the target"
```

- [ ] **Step 3: Run; expect ImportError**

- [ ] **Step 4: Implement `wing_coverage.py`**

`wing_target_points`: rasterise `body_uv` into a boolean image (round to int, optionally dilate by `dilate_px`), take `mask & ~body`, and randomly sample `n_points` of the surviving pixel coordinates with the given seed; NaN-pad if fewer. `coverage_residual`: chunked softmin over projected wing vertices, NaN-safe, optional Huber.

- [ ] **Step 5: Re-run to green** — expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/tracking/wing_coverage.py \
        third_party/jarvis_jax/tests/test_wing_coverage.py
git commit -m "feat(wing-coverage): mask area the BODY does not explain must be explained by the WINGS

Opposes mask_containment, which is one-sided (inside costs nothing) and would
otherwise be minimised by tucking the wing inside the body silhouette."
```

---

### Task 5: Wing pitch refinement (Adam)

**Files:**
- Create: `third_party/jarvis_jax/jarvis_jax/tracking/wing_mask_refine.py`
- Test: `third_party/jarvis_jax/tests/test_wing_mask_refine.py`

**Interfaces:**
- Consumes: `mask_sdf.sdf_stack_from_masks`, `mask_containment.containment_residual`, `wing_coverage.*`, `appendage_dof.*`, `fk.make_fk_repose`.
- Produces: `refine_wing_pitch(q_init, *, fk_repose, wing_vert_idx, body_vert_idx, cam_Ms, cam_ts, sdf, grid_scale, grid_offset, present, masks, bridge_s, bridge_R, bridge_t, opt_mask, lb, ub, containment_weight=0.3, coverage_weight=0.3, smooth_weight=0.005, limit_weight=10.0, n_steps=300, lr=1e-2, chunk_size=32) -> (T, nq)`.

**What the recovered code ALREADY does** (verified at `0bc36fe^` — do not
rebuild these, and do not "fix" them into something else):
- `_frame_cost` computes containment + chamfer-coverage for ONE frame: FK once
  via `fk_repose(q_t, 1.0, idx)` with vertex selection, bridge model->mm, then
  cameras iterated by `jax.lax.scan`. This is the right shape already.
- `refine_appendages_adam` vmaps `_frame_cost` over frames, runs the Adam steps
  in a `jax.lax.scan` (NOT a Python loop), gates gradients AND updates by
  `opt_mask` (`g = g * opt_mask[None,:]`, `q0 + where(opt_mask, dq, 0)`), and
  builds `lb_row`/`ub_row` with +-inf on frozen DOFs.

**`chunk_size` DOES NOT MEAN frames.** In the recovered code it is passed down
into `chamfer_residual` as its memory chunk over TARGET POINTS. An earlier draft
of this plan misread it as a frame chunk. Keep `chunk_size` with its real
meaning and add a SEPARATE `frame_chunk` parameter for the new frame batching.

**Performance requirements (part of the interface, not a later pass):**
- The `lax.scan` over `n_steps` already gives one trace per call — keep it.
  Do NOT rewrite it as `fori_loop`; `scan` also returns the loss history free.
- **Frame chunking is genuinely absent and must be added.** Today the whole
  `(T, nq)` trajectory goes in one call: at T=2007 x 7 cameras x 128x128 f32 the
  SDF stack alone is ~920 MB. Chunk over frames with `frame_chunk`, pad the last
  chunk to a constant shape, and pass the pad as a mask so all chunks share one
  trace.
- `jax.vmap` over frames inside a chunk. Frames are independent apart from the
  smoothness term, which applies within a chunk with an overlap of 1.
- float32 throughout the refinement. This is a pixel-scale cost, not a
  reprojection identity, so unlike Task 9's rules f32 is the correct choice here.
- FK only the wing vertices -- exactly 100 of 139 353 (`fps_300` holds 100 wing
  verts, and a dedicated `fps_wing` subset of the same 100 exists) -- selected
  once outside the loop.
- Build the next chunk's SDF on the CPU while the current chunk optimises on the
  GPU (`ThreadPoolExecutor(1)`; `distance_transform_edt` releases the GIL).


- [ ] **Step 1: Recover the Adam driver as the starting point**

```bash
git show 0bc36fe^:third_party/jarvis_jax/jarvis_jax/tracking/silhouette_refine.py \
  > third_party/jarvis_jax/jarvis_jax/tracking/wing_mask_refine.py
```

Keep its structure — verified correct: `opt_mask` gates both the parameter update and the limit rows, so all other DOFs stay exactly at `q_init`, and it uses Adam deliberately (jaxls' non-scale-invariant `λI` damping cannot navigate a pixel-scale cost with `|diag(JtJ)| ~ 1e6`). Replace its chamfer-boundary objective with containment + wing coverage; drop the anchor term.

- [ ] **Step 2: Write the failing tests**

```python
import numpy as np
import pytest


def test_only_the_masked_dofs_move():
    """The load-bearing guarantee: with opt_mask limited to wing pitch, every
    other qpos entry must come back bit-identical."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    # a synthetic 5-frame problem; see the test file for the fixture builder
    q0, kw = _tiny_problem()
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    moved = np.abs(q1 - q0).max(axis=0) > 1e-9
    assert moved[kw["opt_mask"]].any(), "the optimised DOFs should move"
    assert not moved[~np.asarray(kw["opt_mask"])].any(), "everything else must be frozen"


def test_joint_limits_are_respected():
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem()
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    m = np.asarray(kw["opt_mask"])
    assert (q1[:, m] >= np.asarray(kw["lb"])[m] - 1e-6).all()
    assert (q1[:, m] <= np.asarray(kw["ub"])[m] + 1e-6).all()


def test_a_frame_with_no_present_camera_is_left_at_q_init():
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q0, kw = _tiny_problem()
    kw["present"] = np.zeros_like(kw["present"])
    q1 = np.asarray(refine_wing_pitch(q0, **kw))
    assert np.allclose(q1, q0), "no evidence must mean no change"


def test_it_recovers_a_known_pitch_offset():
    """Synthesise masks FROM the model at a known pitch, perturb pitch, and check
    the refinement moves back toward the truth. This is the only test that shows
    the objective points the right way."""
    from jarvis_jax.tracking.wing_mask_refine import refine_wing_pitch
    q_true, q_pert, kw = _synthetic_from_model(pitch_offset_deg=25.0)
    q1 = np.asarray(refine_wing_pitch(q_pert, **kw))
    m = np.asarray(kw["opt_mask"])
    err0 = np.abs(q_pert[:, m] - q_true[:, m]).mean()
    err1 = np.abs(q1[:, m] - q_true[:, m]).mean()
    assert err1 < 0.5 * err0, f"pitch error {err0:.3f} -> {err1:.3f} rad"
```

Write `_tiny_problem()` and `_synthetic_from_model()` in the test file: build the real MuJoCo model, pick 5 frames, render or rasterise the model's own silhouette at a known pitch to serve as the "mask", and feed the identity bridge.

- [ ] **Step 3: Run; expect failures** — then implement until green (4 passed).

- [ ] **Step 4: Measure against the 60 s budget, and pin what causes blowups**

Time the real bout (2007 frames, 7 cameras), printing SDF precompute / compile /
optimise separately. A wall-clock assert would be flaky in CI, so pin the two
structural properties instead:

```python
def test_the_step_loop_compiles_once_not_per_chunk():
    """A re-trace per FRAME chunk is the difference between 40 s and 20 minutes."""
    from jarvis_jax.tracking import wing_mask_refine as R
    q0, kw = _tiny_problem(n_frames=40)          # 3 chunks at frame_chunk=16
    kw["frame_chunk"] = 16
    with _count_traces(R, "_refine_chunk") as n:
        R.refine_wing_pitch(q0, **kw)
    assert n.value == 1, f"retraced {n.value}x -- pad chunks to a constant shape"


def test_only_the_wing_vertices_are_fk_d():
    """FK over all 139 353 vertices per step per frame is a ~1400x waste."""
    q0, kw = _tiny_problem()
    assert len(kw["wing_vert_idx"]) < 400, "wing selection must be the fps subset"
```

Put the measured seconds in the commit message. If it exceeds 60 s, cut
`n_steps` until it fits and say so — that is a legitimate outcome, not a failure.

- [ ] **Step 5: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/tracking/wing_mask_refine.py \
        third_party/jarvis_jax/tests/test_wing_mask_refine.py
git commit -m "feat(wing-mask-refine): Adam refinement of wing pitch against the SAM masks

<measured> s per bout-fly against a 60 s budget: SDF <a> s, compile <b> s,
optimise <c> s. A single jitted fori_loop, vmapped over frames, FK-ing only the
~100 wing vertices; chunks padded to a constant shape so it traces once."
```

---

### Task 6: Wire into `run_bout` as an opt-in stage

**Files:**
- Modify: `scripts/run_bout.py` (after Stage C `stac_ik.h5`, before Stage D bridge)
- Modify: `configs/pipeline.yaml`
- Modify: `tests/test_run_bout_pipeline_structure.py`

**Interfaces:**
- Consumes: `stac_ik.h5`, `masks_dict`, `cfg.recording.calib_dir`.
- Produces: `<bout_dir>/qpos_wingfit.npz` with `qpos`; Stage D reads it in place of the STAC qpos when the stage ran.

- [ ] **Step 1: Add the config block**

```yaml
# Post-STAC refinement of WING PITCH against the SAM masks. Wing pitch is 99.6%
# of the marker Jacobian's null direction (per-column sensitivity yaw 0.271 /
# roll 0.344 / pitch 0.042), because each wing body carries only two markers --
# WingX_base maps to the THORAX. Unconstrained, the blade rotates into the
# abdomen. The masks DO contain wing pixels (wing landmarks project inside them
# on 75-91% of frames), and sweeping pitch shows a real silhouette optimum.
# Yaw and roll are NOT touched: yaw carries the song, roll is the
# strongest-observed wing DOF.
# See docs/specs/2026-09-01-wing-orientation-from-masks-design.md
wing_mask_fit:
  enabled: false
  containment_weight: 0.3
  coverage_weight: 0.3
  smooth_weight: 0.005
  limit_weight: 10.0
  n_steps: 300
  lr: 0.01
  out_hw: [128, 128]
  bbox_margin: 0.4
  n_target_points: 128
  # a frame with fewer than this many present cameras is left at its STAC pose
  min_present_cameras: 3
  # cameras to exclude for wings. Cam2012631 has the right wing inside its mask
  # on only 43-46% of frames (Abd_tip 20%), i.e. truncated masks.
  exclude_cameras: [Cam2012631]
```

- [ ] **Step 2: Extend the Stage-B/C gate signature**

Add a `wing_mask_fit` block to `stage_b_gate_signature` in `scripts/run_bout.py` so a resumed run cannot silently skip or stale-reuse the refinement, mirroring `wing_collapse`/`rigid_repair`. Extend `test_stage_b_gate_signature_changes_with_every_gate` with a `wingfit_on` variant and a parameter tweak.

- [ ] **Step 3: Add the stage**

Insert after `stac_ik.h5` is loaded and before Stage D, guarded by `if not stage_done(wingfit_path)`, printing a `[wing-mask-fit]` line with the number of frames refined, frames skipped for want of cameras, and median pitch change per wing. Stage D then reads `qpos_wingfit.npz` when present.

- [ ] **Step 4: Add a structure test**

```python
@pytest.mark.parametrize("marker", ["Stage C: STAC", "Stage D: model->mm bridge",
                                    "Stage E: outputs", "[wing-mask-fit]"])
def test_wingfit_stage_lives_inside_process_bout_fly(tree, marker):
    ...
```

- [ ] **Step 5: Run the structure tests to green; commit**

```bash
git add scripts/run_bout.py configs/pipeline.yaml tests/test_run_bout_pipeline_structure.py
git commit -m "feat(pipeline): opt-in post-STAC wing pitch refinement against the SAM masks"
```

---

### Task 7: Acceptance measurement and the 7-camera figure

**Files:**
- Create: `scripts/analysis/wing_mask_fit_ab.py`
- Create: `scripts/viz/wing_fit_7cam.py`
- Create: `docs/benchmark/2026-09-01-wing-mask-fit/notes.md`

- [ ] **Step 1: A/B script measuring every acceptance number**

Reuse `scripts/analysis/wing_pitch_rest_prior_ab.py`'s metrics: `pulse_stats` per DOF, `|yawL-yawR|`, wing-keypoint residual, `mj_geomDistance` penetration. Add the mask-optimal-pitch comparison. Run control vs treatment on bout 28 **fly1 and fly0**.

- [ ] **Step 2: Gate on the acceptance criteria, in this order**

1. singing-wing `wing_pitch_*` `hp_rms` within 20% of control (control 0.3615);
2. `|yawL-yawR|` mean and pulse stats within noise;
3. penetration median from -0.043 toward -0.0013 on BOTH wings;
4. wing-keypoint residual rise ≤ 20%;
5. folded-wing pitch within ~10° of the measured mask optimum (-20…-40°), not rest (-57°).

If (1) or (2) fails, STOP and report — that is the failure that killed the rest prior.

- [ ] **Step 3: The 7-camera figure**

`scripts/viz/wing_fit_7cam.py`: rows = the 7 rig cameras, columns = real frame | control render | treatment render, all in one crop, using the `rigcam` recipe from `viz/views/sidebyside.py` (`similarity_from_points` then `mujoco_camera_from_affine(cam_mat, (W,H), s, R, t, anchor, back_off=2.0)`, set `cam_pos/quat/fovy`, `mj_forward` AGAIN, render). Overlay the SAM mask outline. Run on **fly0 (female)** and on fly1's **non-singing** wing. State the expectation in the docstring before generating, then read the PNG back with the Read tool and report what it shows against that expectation.

- [ ] **Step 4: Write the notes and commit**

Commit only the small text artifacts (`notes.md`, the scorecard `.json`) under `docs/benchmark/2026-09-01-wing-mask-fit/`; reference the `figures/2026-09-01-wing-mask-fit/` directory and the exact command that regenerates it. Never commit the PNGs.

```bash
git add scripts/analysis/wing_mask_fit_ab.py scripts/viz/wing_fit_7cam.py \
        docs/benchmark/2026-09-01-wing-mask-fit/notes.md
git commit -m "test(wing-mask-fit): acceptance A/B + 7-camera figure"
```

---

### Task 8: Decide the default, and record the outcome

- [ ] **Step 1:** If all five acceptance criteria pass on BOTH flies, propose flipping `wing_mask_fit.enabled` to `true` — as a separate commit, with the scorecard in the message. If any fail, leave it `false` and record the negative result in the spec's rejected-alternatives section with its measurements, the way the four existing entries are recorded.
- [ ] **Step 2:** Either way, update `docs/specs/2026-09-01-wing-orientation-from-masks-design.md` with the measured outcome so the next reader does not re-derive it.

---

---

## Phase 2 — Optimization

`3eab485` already took QC + overlays from 1076 s to 215 s (5.0x) with every
number proven unchanged. That moved the bottleneck rather than removing it. The
per-bout budget now:

| stage | s | share |
|---|---|---|
| **STAC IK** | **746** | **61%** |
| overlays (7 videos) | 195 | 16% |
| bridge + FK | 154 | 13% |
| sidebyside.mp4 | 115 | 9% |
| qc.json + qc_perframe (shared) | 15 | 1% |
| triangulation | 5 | <1% |
| **total** | **1230** | |

These tasks are ordered by measured share, and each one **profiles before it
optimises**. Do not start Task 10 or 11 on a hunch about where the time goes;
that is exactly what the QC work disproved (the named suspect was real, but a
second bottleneck — QC computing the same two metrics twice — was worth as much
and nobody had named it).

---

### Task 9: Profile and cut STAC IK (the 61%)

**Files:**
- Profile script: scratchpad (throwaway)
- Modify: `third_party/jarvis_jax/jarvis_jax/tracking/stac.py` (`fit_offsets_once`
  / `ik_only_bout` — what Stage C actually calls) and/or
  `third_party/jarvis_jax/jarvis_jax/tracking/ik_solve.py` (`solve_ik`, the
  de-risked `JaxlsBatchSolver` wrapper) — only where the profile points.
- **`stac-mjx/stac_mjx/stac_core_jaxls.py` holds the actual LM solve, and
  `stac-mjx` is a GIT SUBMODULE.** A change there is a commit in a separate
  repo: commit it locally in the submodule and say so in your report, but do
  NOT push the submodule — that is the controller's call.
- Create: `docs/benchmark/2026-09-01-stac-ik-speedup/notes.md`

- [ ] **Step 1: Profile it before touching anything**

Instrument one real bout-fly (2007 frames): time spent in JIT compilation vs.
solve, per `jaxls` batch; iterations actually taken vs. `n_iter` (default 50 in
`solve_ik`); and **whether the pipeline chunks the clip at all**. Establish that
by reading the code, not by assuming: `solve_trajectory` is handed the whole
clip, and the 600-frame figure appearing in this repo's analysis output is the
`--nt` default of `scripts/analysis/wing_pitch_rest_prior_ab.py`, not a pipeline
batch size. If it does chunk and the last chunk differs in shape, that re-trace
is free to fix — the trap Task 5 pins a test against.

Report a table before proposing any change. Record it in `notes.md` even if the
answer is "it is genuinely solve-bound", because that result decides Step 2.

- [ ] **Step 2: Act only on what the profile showed**

Candidates, in the order they are cheap to test — take the ones the profile
supports and explicitly reject the others in `notes.md` with the measurement:

1. **Pad the last batch to a constant shape.** Free if it is re-tracing.
2. **Early termination.** If the LM solve converges well before
   `max_iterations`, tighten the tolerance rather than spending the iterations.
3. **`jax.block_until_ready` placement.** Confirm the timing is real and not an
   async-dispatch artifact before believing any of the above.
4. **Batch size.** 600 frames/batch was not chosen by measurement; sweep it.
5. **float32.** Try it, and gate it hard: the pose must not move by more than
   the smoothness prior's own noise floor. If it moves the fitted qpos
   measurably, reject it and write down the delta — accuracy is not for sale here.

- [ ] **Step 3: Prove the fit did not change**

Rerun the bout and diff `stac_ik.h5` against the copy already on disk. For a
pure scheduling/compile change, require **bit-identical** qpos. For anything
numerical (batch size, tolerance, dtype), report max abs and max rel qpos delta
plus the wing-keypoint residual, and state which acceptance threshold it clears.
A speedup that moves the fit is a different change and needs its own A/B.

- [ ] **Step 4: Commit**

```bash
git add <the files the profile actually led you to> \
        docs/benchmark/2026-09-01-stac-ik-speedup/notes.md
git commit -m "perf(stac-ik): <what the profile actually showed> -- 746 s -> <X> s

qpos <bit-identical | max abs delta ...>. Rejected: <candidates> because <measurement>."
```

---

### Task 10: Stop materialising 12.2 GB of masks

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/tracking/bout_masks.py` (`load_bout_masks`)
- Modify: `scripts/run_bout.py` (Stage E and the new wing-fit stage)
- Test: `third_party/jarvis_jax/tests/test_bout_masks_streaming.py`

This is a **throughput** task, not a latency one, and it may be worth more than
either: Stage E peaks near 27 GB RSS, which is why concurrent pipelines are
capped at ~4 on a 128 GB-cgroup node. Halving peak RSS raises the cap.

- [ ] **Step 1: Measure peak RSS per stage** (`resource.getrusage` +
  `tracemalloc`, or `/proc/self/status` VmHWM sampled in a thread) so the win is
  a number, not an argument.

**The real signature** (verified — do not trust any sketch over the file):
`load_bout_masks(npz_path, fly, *, expected_cameras=None) -> dict`, keys
`masks (T,C,H,W) bool`, `valid (T,C)`, `centroids (T,C,2)`, `T/C/H/W`. It returns
a **dict**, not an object, and raises `ValueError` on a camera mismatch — the
`CameraOrderError` in `viz/core/bout_artifacts.py` is a different layer.

- [ ] **Step 2: Add a streaming accessor** — `iter_bout_masks(npz_path, fly, *, expected_cameras=None)` yielding
  `(t, {cam: mask})` in canonical camera order, with the SAME name-based
  reordering `load_bout_masks(..., expected_cameras=…)` does. **Keep the
  reorder-by-name**: this is the camera-order trap, and a streaming path that
  drops it would reintroduce the exact bug CLAUDE.md documents.

- [ ] **Step 3: Test that streaming and bulk agree, camera for camera**

```python
def test_streaming_matches_bulk_in_canonical_camera_order():
    bulk = load_bout_masks(npz, fly, expected_cameras=cams)      # a dict
    for t, per_cam in iter_bout_masks(npz, fly, expected_cameras=cams):
        for ci, cam in enumerate(cams):
            assert (per_cam[cam] == bulk["masks"][t, ci]).all(), (t, cam)


def test_streaming_refuses_a_camera_list_it_cannot_satisfy():
    with pytest.raises(ValueError):        # bout_masks raises ValueError
        next(iter(iter_bout_masks(npz, fly, expected_cameras=cams + ["nope"])))
```

- [ ] **Step 4: Convert the consumers** — Stage E's QC (already reads each mask
  twice, not four times, after `3eab485`) and the Task 6 wing-fit stage. Leave
  `load_bout_masks` in place for callers that genuinely need random access.

- [ ] **Step 5: Prove qc.json is still identical** and report peak RSS before/after.

- [ ] **Step 6: Commit**

```bash
git add third_party/jarvis_jax/jarvis_jax/tracking/bout_masks.py scripts/run_bout.py \
        third_party/jarvis_jax/tests/test_bout_masks_streaming.py
git commit -m "perf(masks): stream masks instead of a 12.2 GB (T,C,H,W) array

Peak RSS <before> -> <after> GB. qc.json bit-identical. The streaming path keeps
load_bout_masks' reorder-BY-NAME; dropping it would reintroduce the camera-order
trap in CLAUDE.md."
```

---

### Task 11: bridge + FK (154 s) and sidebyside (115 s)

**Files:**
- Modify: `third_party/jarvis_jax/jarvis_jax/tracking/bridge.py`
- Modify: `viz/views/sidebyside.py`
- Create: `docs/benchmark/2026-09-01-bridge-sidebyside-speedup/notes.md`

- [ ] **Step 1: Profile both.** For the bridge, expect the same shape of problem
  the QC work found — per-frame Python over a scalar API. `compute_bridges`
  solves a per-frame Umeyama/SVD; those stack into one batched `np.linalg.svd`
  exactly as `loo_reproj` did. Confirm with the profile first.

- [ ] **Step 2: Batch the bridge**, using `np.einsum` and a float64 stack for
  the same reasons as `3eab485`. Require **bit-identical** `bridge_s/R/t`; if
  the stacked SVD differs in sign convention on a degenerate frame, handle it
  explicitly rather than accepting a delta.

- [ ] **Step 3: sidebyside.mp4** is MuJoCo/EGL-bound and already frame-capped at
  300. The cheap win is rendering the model once per frame instead of once per
  panel if it currently re-renders; check before assuming. Do **not** switch
  encoders — `h264_nvenc` changes the bytes for a ~1.2x ceiling, which
  `3eab485` already evaluated and rejected.

- [ ] **Step 4: Verify** the bridge outputs bit-identical and the video
  md5-identical (or, if the render genuinely changed, extract frames and Read
  them per CLAUDE.md before claiming it is fine).

- [ ] **Step 5: Commit** with the before/after table.

---

### Task 12: Report the end-to-end number

- [ ] Rerun one full bout-fly and rebuild the stage table: 1230 s before this
  phase, plus the wing-fit stage's measured cost, minus what Tasks 9-11 removed.
  Write it into `docs/benchmark/2026-09-01-qc-render-speedup/notes.md` as a
  follow-up section so the whole optimisation story lives in one place, and
  state the new dominant stage — whatever it turns out to be — so the next
  person starts from a measurement instead of a guess.

## Self-review

**Spec coverage:** §2 root cause → Tasks 2–5 (the cost acts on pitch only). §4.1 mask evidence → Task 7 criterion 5. §4.2 reuse → Tasks 2–5 all recover from `0bc36fe^`. §4.2b verification → every recovered task has tests asserting documented behaviour. §4.3 one-sidedness trap → Task 4 exists precisely to oppose it. §4.4 scope/staging → Task 6. §5 acceptance → Task 7. §6 risks: the broad optimum and modest contrast are why criterion 5 allows ~10°; `exclude_cameras` handles the weak-camera risk; cost is bounded by `n_steps`/`chunk_size`.

**Placeholder scan:** none — every step has a command, code, or a named file and function.

**Performance coverage:** the new stage is budgeted (60 s) and its two blowup
modes are pinned by tests in Task 5, not left to a later pass. Phase 2 attacks
the measured shares in order — STAC IK 61%, masks/RSS (throughput), bridge 13% +
sidebyside 9% — and every task profiles before it edits and proves the outputs
unchanged after. Tasks 9-12 are independent of Tasks 2-8 and can run in parallel
with them; only Task 10 touches a file (`run_bout.py`) that Task 6 also touches,
so land Task 6 first or expect a small conflict there.

**Type consistency:** `grid_xy = (orig_xy - grid_offset) * grid_scale` is used identically in Tasks 1, 2, 5. `appendage_vertex_indices` returns FULL-array indices in Tasks 3, 4, 5 (pinned by a test). `opt_mask` is the parameter name in Task 5 (verified; it is not `sil_qs`).
