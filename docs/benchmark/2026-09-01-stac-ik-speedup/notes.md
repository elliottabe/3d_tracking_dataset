# STAC IK (Stage C): where the 746 s actually goes

**Bout-fly:** `Session0/2025_10_20_13_20_04` `bout_00028`, both flies.
fly1 (male) = 2007 frames solved as one clip; fly0 (female) = one finite
segment of 1760 frames. 50 model keypoints, `anatomy=v1` (`nq=93`).
All wall clock on one GPU node (8x L40S-class, 46 GB each, 32 cores).
**Machine state before every timing run: `load average` 0.9-2.4 on 32 cores,
all 8 GPUs at 0% and 0 MiB.** Nothing else of mine ran concurrently with a
timed run.

Overrides taken verbatim from the run's own
`pose/hydra/analysis/.hydra/overrides.yaml`:
`paths=hyak bout_ids=28 wing_collapse.enabled=true rigid_repair.enabled=true`.

---

## 0. The two things the plan assumed that are NOT true

**(a) There is no chunking, so there is no ragged last chunk to pad.**
Read from the code, then confirmed at runtime:

* `configs/anatomy/v1.yaml: JAXLS_CHUNK_SIZE: 0`, so
  `compute_stac._pose_optimization_jaxls` takes its single-solve branch
  (`if chunk_size > 0 and T > chunk_size:` is false). The Python-loop chunker
  with its warm-start hand-off exists but is dead at this config.
* `jarvis_jax.tracking.stac.ik_only_bout` sets
  `cfg.stac.n_frames_per_clip = int(kp3d.shape[0])` before calling
  `stac_mjx.run_stac`, so `utils.batch_kp_data` computes
  `n_batches = T // T = 1` and yields exactly ONE clip of T frames.
  (The `n_frames_per_clip: 500` in `configs/stac/courtship.yaml` is
  overwritten and never used on this path.)

So candidate 1 (**pad a variable-length last chunk**) is not applicable:
nothing is chunked and nothing is padded. Per bout-fly the pipeline compiles
exactly two jaxls problems -- `T=1` (root phase) and `T=2007` (pose phase).

**(b) `n_iter` is 500, not 50.** `ik_solve.solve_ik`'s `n_iter=50` is the
de-risking probe's default and Stage C never calls it. Stage C goes
`ik_only_bout -> stac_mjx.run_stac -> Stac.ik_only -> compute_stac.
pose_optimization -> StacCore._jaxls_solver`, and `StacCore.__init__` passes
`n_iter=n_iter_q`, i.e. **`cfg.model.N_ITER_Q = 500`**. That number was chosen
for the ProjectedGradient/Optax solver it originally configured and inherited
by the LM path unexamined.

---

## 1. The profile (fly1, T=2007)

Instrumented by monkeypatching `Stac.__init__`, `Stac.ik_only`,
`compute_stac.root_optimization/pose_optimization`,
`JaxlsBatchSolver._get_analyzed/_solve_se3`, `io.save_data_to_h5` --
every timer closed with `jax.block_until_ready`, so **candidate 3
(async-dispatch artifact) is ruled out by construction**: the numbers below are
device-complete, not dispatch-complete. Compile and execute were separated by
calling the identical `analyzed.solve(...)` twice; the second call is a
compilation-cache hit, so `compile = t_first - t_second`. That extra second
solve (178.65 s) is profiling overhead and is subtracted from the table.

| phase | s | share |
|---|---|---|
| `Stac.__init__` (MjSpec parse + keypoint sites + rescale) | 1.7 | 0.5% |
| load `offsets.h5` | 0.1 | 0.0% |
| `batch_kp_data` (1 clip) | 0.0 | 0.0% |
| vmapped `mjx_setup` (`mjx_load` + `set_site_pos` + FK), 1 clip | 23.1 | 6.8% |
| **`root_optimization`** (T=1 LM) | **55.6** | **16.4%** |
| `pose_optimization` total | 233.6 | 68.8% |
| &nbsp;&nbsp;├ jaxls `analyze` (graph build, 6020 terms) | 2.5 | 0.7% |
| &nbsp;&nbsp;├ **LM XLA compile** | **32.1** | **9.5%** |
| &nbsp;&nbsp;├ **LM execute (500 / 500 iterations)** | **178.7** | **52.7%** |
| &nbsp;&nbsp;└ warm-start + FK vmaps + `frame_error` | 20.3 | 6.0% |
| `infer_qvels` (`compute_velocity_from_kinematics`) | 25.1 | 7.4% |
| `_package_data` + write h5 | 0.4 | 0.1% |
| **total (production-equivalent)** | **339.3** | |

x2 flies = **679 s**, against the plan's 746 s, the 744 s the artifact mtimes
give for this bout (fly0 391 s + fly1 353 s), and **752.85 s** measured directly
here as the A/B baseline (357.69 + 395.16). The profile's own total is a little
low because the instrumented process differs slightly from a clean one; the
shares are what this table is for, and the A/B walls in section 3 are the
headline numbers.

Problem shape: `T=2007`, `nq=93`, `tangent_dim = 6 + 86 = 92`,
`T*tangent_dim = 184 644` -> `conjugate_gradient` (the `auto` rule's
`_DENSE_THRESHOLD` is 5000). 6020 cost terms: 2007 marker + 2006 smoothness +
2007 `leq_zero` joint-limit constraints (so jaxls runs its augmented-Lagrangian
path).

### It is genuinely solve-bound, and it never converges

`iterations = 500 / 500`, `termination_criteria (cost, gradient, parameter) =
[False, False, False]`. The deltas at exit against the configured tolerances:

| criterion | value at exit | tolerance | ratio |
|---|---|---|---|
| relative cost change | 2.43e-3 | 1e-5 (`JAXLS_COST_TOLERANCE`) | 243x too large |
| gradient inf-norm | 1.36e+1 | 1e-8 (`JAXLS_GRADIENT_TOLERANCE`) | 1.4e9x too large |
| parameter 2-norm | 1.48e-6 | 1e-10 (`JAXLS_PARAMETER_TOLERANCE`) | 1.5e4x too large |

Per-iteration cost: 178.7 s / 500 = **0.357 s**.

`cost_history` (the non-augmented objective):

| iteration | cost | % of the total reduction reached | how far above the iteration-499 cost |
|---|---|---|---|
| 0 | 939.38 | 0% | -- |
| 10 | 41.19 | 96.4% | +415% |
| 50 | 29.84 | 97.7% | +273% |
| 100 | 24.05 | 98.3% | +201% |
| 200 | 17.29 | 99.0% | +116% |
| 300 | 10.94 | 99.7% | +37% |
| 400 | 9.36 | 99.85% | +17% |
| 499 | 8.00 | 100% | 0% |

Still descending at 500. The "96% by iteration 10" headline is the wrong way to
read it: the interesting residual is the last 4%, and iteration 100 still sits
at **3x** the final cost.

---

## 2. What the profile supports, and what it rejects

### TAKEN: `root_optimization` on the ik_only path is dead work (55.6 s, 16.4%)

`compute_stac.root_optimization` solves a **one-frame** LM problem for the
first `root_dims` qpos entries and touches nothing else. On the jaxls pose
path, `_pose_optimization_jaxls` then builds its own per-frame warm start out
of that same qpos and overwrites

* `q_init_all[:, :3]` from the root keypoint (`ROOT_OPTIMIZATION_KEYPOINT`), and
* `q_init_all[:, 3:7]` from the trunk-orientation keypoints
  (`JAXLS_ORIENTATION_KEYPOINTS`),

which for a FREE root is exactly all 7 entries `root_dims` covers. The FK
template's non-qpos fields are all rebuilt from qpos inside `marker_cost`
(`mjx_data_template.replace(qpos=full_q)` -> `kinematics` -> `com_pos`), and
`qs_to_opt` is all-True there so the template's own qpos never even reaches the
residual. Nothing the root phase produces survives to be read.

Cost of not skipping it: 55.6 s, of which the T=1 LM *executes* in 0.01 s --
it is almost entirely XLA compile for a problem that is thrown away.

Guarded by `stac_mjx.stac.root_optimization_is_discarded` (a named function
with a truth table, not an inline `and`) and
`stac-mjx/tests/unit/test_root_opt_skip.py`, which pins the *property* -- move
`mjx_data.qpos[:7]` and the warm start handed to the solver must not move --
not just the boolean. `fit_offsets` is deliberately left alone: it runs once
per recording, is not in the per-bout budget, and its root phase feeds
`offset_optimization`, so the same argument does not apply.

### REJECTED: 1. pad a variable-length last chunk

Not applicable. `JAXLS_CHUNK_SIZE=0` and `n_frames_per_clip = T`: there is one
clip, no chunk loop, no ragged tail. See section 0(a).

### REJECTED: 2. early termination / tighter tolerance

The opposite of the assumption. The solve does not converge early -- it uses
all 500 iterations and **no** termination criterion fires (table above);
tightening a tolerance can only ever make it later, never earlier. The
tolerances are already tightened 1e4x/1e6x below jaxls' defaults on purpose
(`docs/benchmark/2026-08-13-stac-weak-dof-convergence/notes.md`: weakly
conditioned DOFs such as wing blade-roll, whose Jacobian column is ~14x weaker
than the strong wing DOFs, stop far short on the loose defaults).

The only lever here is **lowering** `N_ITER_Q`, which buys 0.357 s per
iteration dropped and is a change to the FIT, not to scheduling. Measured
trade, for whoever wants it: 250 iterations would save ~89 s per bout-fly
(~178 s/bout) and leave the objective ~57% above its 500-iteration value.
Not taken here -- that is its own A/B against the benchmark suite, and it
pushes directly against the weak-DOF convergence work above.

### REJECTED: 3. `jax.block_until_ready` placement

Checked first, as instructed. Every timer in the profile closes with
`jax.block_until_ready`, and the totals reconcile with independent wall clock
(`ik_only_bout` 517.97 s vs the sum of its parts 517.96 s) and with the
artifact mtimes of the production run (744 s/bout for two flies vs 679 s
measured here). The timings are real.

### REJECTED: 4. batch size

There is no batch size to sweep on this path -- see 0(a) and REJECTED 1. The
one shape knob that exists, `JAXLS_CHUNK_SIZE`, is 0 (off), and turning it ON
would make things *slower*, not faster: its own config comment says it "solves
in chunks of frames to reduce memory usage at the cost of speed", and each
chunk is a fresh XLA compile plus a serial warm-start dependency on the
previous chunk. Memory was not a constraint here (the T=2007 solve fits in one
46 GB GPU with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`).

### REJECTED: 5. float32

Already float32 -- there is nothing to convert. `jax_enable_x64` is **False**
in this environment, `jarvis_jax.tracking.stac._flat_scaled` casts the
keypoints to `np.float32`, and every array in the shipped `stac_ik.h5`
(`qpos`, `qvel`, `xpos`, `xquat`, `marker_sites`, `kp_data`, `offsets`) is
`float32`. The only dtype move available is float64, which would be slower.

### NOT TAKEN, reported for the controller

* `infer_qvels` costs **25.1 s per bout-fly** (7.4%) and `qvel` is not read by
  anything in the Stage C -> F chain; its only consumer in the tree is
  `scripts/postprocess_stac_data.py` (the RL reference-clip path). It is a
  config toggle (`configs/stac/default.yaml: infer_qvels: True`), so turning it
  off is a one-line decision -- but it drops a dataset from `stac_ik.h5`, i.e.
  a schema change with an out-of-tree consumer, not a scheduling change.
* The vmapped `mjx_setup` costs **23.1 s** to `mjx.put_model` a model, set 50
  site positions and run FK -- under a `jax.vmap` over a clip axis of length
  **1**. Collapsing the vmap when `n_clips == 1` should be numerically
  identical, but it changes the pytree shape every downstream
  `jax.tree.map(lambda x: x[i], ...)` indexes, so it is a wider change than
  this task's remit.

---

## 3. Acceptance: did the fit move?

### Why the h5 diff alone cannot answer this, and what does

A 500-iteration LM solve that never converges is **chaotic**, and XLA's runtime
gemm autotuning can pick a different algorithm between two runs of the same
binary. So two `stac_ik.h5` files differing does not, by itself, distinguish
"the change moved the fit" from "the GPU re-autotuned". Evidence for that,
measured here before anything else:

| pair | qpos max abs | marker_sites max abs | notes |
|---|---|---|---|
| clean baseline rerun vs the **production** `stac_ik.h5` on disk | 0.0803 | 5.56e-4 | different node, different day, IDENTICAL code and inputs |
| the instrumented profile run vs the clean baseline, SAME node | 0.1504 | 1.05e-3 | identical code, only `return_summary=True` (which changes what XLA can dead-code) |

Both of those are reruns of **unchanged** code. So "bit-identical h5" is not a
criterion this solve can meet under any change, including the empty change.

The statement that IS exact for a pure scheduling change is one level upstream:
**identical program + identical inputs => identical computation.** So the
acceptance evidence is the hash of every argument the pose-phase
`solve_trajectory` receives, captured with the change and without it.

### The A/B walls

Two arms, same node, same session, same harness (`/tmp/task9/run_stage_c.py`,
uninstrumented), the ONLY difference being the `stac_mjx/stac.py` change,
stashed and popped between them.

| bout-fly | frames | baseline | with the skip | delta |
|---|---|---|---|---|
| fly1 (male) | 2007 | 357.69 s | **296.18 s** | −61.51 s (−17.2%) |
| fly0 (female) | 1760 | 395.16 s | **347.04 s** | −48.12 s (−12.2%) |
| **bout total** | | **752.85 s** | **643.22 s** | **−109.63 s (−14.6%)** |

The baseline reproduces the plan's budget line: 752.85 s measured here against
the 746 s in the table and 744 s from the artifact mtimes. (The female's clip is
12% SHORTER and her solve is 10% LONGER -- her data is worse conditioned, so
each LM step's CG needs more inner iterations. Worth knowing: the hard fly is
also the expensive one.)

### The wing-keypoint residual, by name

`||marker_sites - kp_data||` over `WingL_base, WingR_base, WingL_V12,
WingL_V13, WingR_V12, WingR_V13` (taken by NAME out of the h5's own
`kp_names`), converted to mm through each fly's own body scale:

| | baseline | with the skip | delta |
|---|---|---|---|
| fly1 wing mean | 0.8497 mm | 0.8497 mm | +0.0000 mm |
| fly1 wing p95 | 1.4821 mm | 1.4811 mm | −0.0010 mm |
| fly0 wing mean | 2.0358 mm | 2.0422 mm | +0.0064 mm (+0.31%) |
| fly0 wing p95 | 6.2684 mm | 6.2766 mm | +0.0082 mm (+0.13%) |

### The exact statement: what the pose solve is actually handed

Every argument of the pose-phase `solve_trajectory` was md5-hashed with the
change and without it (227 hashed items: the eight explicit arrays, plus every
leaf of `mjx_model` and `mjx_data_template`).

**fly0 (the female): `q_init` is EXACTLY equal, `max|Δ| = 0.0`.**
**fly1 (the male): `q_init` differs on the QUATERNION columns 3:7 only** — max
|Δ| 1.19e-07 (one float32 ULP) on 297 of 2007 frames. Columns 0:3 (root xyz) and
7:93 (all 86 hinges) are exactly equal on both flies.

The other differing hashes are the same set on both flies, and they are all
**dead inputs**. Mapping the leaf indices back to `mjx.Data` field names:

| differing `mjx_data_template` leaf | why it cannot reach the residual |
|---|---|
| `qpos` | `_pose_optimization_jaxls` sets `qs_to_opt = jp.ones(nq, bool)`, so `marker_cost`'s `full_q = jnp.where(qs_to_opt, q, template.qpos)` is `q` — the template's qpos is never read |
| `xpos`, `xipos`, `xanchor`, `xaxis`, `geom_xpos`, `site_xpos`, `subtree_com`, `cdof`, `cinert` | every one is an OUTPUT of `mjx.kinematics` / `mjx.com_pos`, which the cost calls on `full_q` before reading `site_xpos` |

That is the complete list — exactly `qpos` plus the nine FK/com-pos outputs, and
nothing else. Which is what "root_optimization's result is discarded" means,
stated as a measurement instead of an argument.

(`mjx_model::treedef` and `mjx_data_template::treedef` also hash differently.
That is an artifact of hashing `str(treedef)`, which embeds a `repr` containing
object ids — see the run-to-run control below, where two runs of the *same*
code differ the same way.)

### The run-to-run floor -- the control that makes the numbers readable

`base2` is the **baseline rerun with the baseline code**, same node, same
session, minutes apart. It is the yardstick.

| | wall (fly1) | `qpos` max abs | `marker_sites` max abs | wing mean |
|---|---|---|---|---|
| baseline | 357.69 s | — | — | 0.8497 mm |
| **baseline REPEAT (unchanged code)** | **357.29 s** | **0.149447** | **2.47e-4** | **0.8499 mm** |
| with the skip | 296.18 s | 0.109262 | 1.31e-3 | 0.8497 mm |
| (the instrumented profile run) | — | 0.150410 | 1.05e-3 | — |
| (production `stac_ik.h5`, other node/day) | — | 0.080294 | 5.56e-4 | — |

**The change moves `qpos` LESS than two runs of the UNCHANGED code move it**
(0.109 vs 0.149), and its wing residual lands on the baseline's value to four
decimal places while the unchanged rerun does not (0.8497 / 0.8497 / 0.8499 mm).
Wall clock, by contrast, is highly reproducible — 357.69 vs 357.29 s, 0.1% —
so the −61.5 s is 150x the timing noise.

This is what a 500-iteration LM solve that never converges does: a 1-ulp
difference at the start is amplified into the third decimal place of `qpos` by
iteration 500, and the GPU supplies 1-ulp differences on its own.

### The one input that is not bit-identical, and what it is

`fixA` vs `fixB` — two dumps of the **same** code — give `q_init` EXACTLY equal
(and the same 11 `mjx_data_template` / treedef hashes differ, confirming those
are the recompute-and-repr noise described above). So the solver-input capture
is itself reproducible, and the fly1 quaternion difference below is attributable
to the change rather than to rerun noise:

* `q_init[:, 0:3]` (root xyz) — exactly equal, both flies.
* `q_init[:, 7:93]` (all 86 hinges) — exactly equal, both flies.
* `q_init[:, 3:7]` (the warm-start quaternion) — exactly equal on **fly0**;
  on **fly1**, 1 float32 ulp (1.19e-07) on 297 of 2007 frames.

A data leak from the root phase is ruled out **by the code path, not by
assertion**: `_pose_optimization_jaxls` writes those four columns with
`q_init_all.at[:, 3:7].set(quats)`, and `quats` is
`_estimate_orientation_from_keypoints(kp_flat, ...)` — a pure function of
`kp_flat` (hashed identical) and three static indices, reading nothing from
`mjx_data`. The root phase has no path into them. What remains is an
execution-level difference in evaluating that pure function.

### Isolating it: the pure function itself moves by 1 ulp

Two more runs, capturing `_estimate_orientation_from_keypoints`'s **input and
output directly** in each arm:

```
kp_flat  identical: True   bitwise: True
quats    identical: False  max|d| = 1.19209e-07   rows differing = 297/2007
q_init   cols 0:3 max|d| = 0    cols 3:7 max|d| = 1.19209e-07 (297 rows)    cols 7: max|d| = 0
```

So the function is handed **bitwise-identical** input and returns output that
differs by exactly one float32 ulp on 15% of frames, and `q_init` inherits
precisely that and nothing else. `_estimate_orientation_from_keypoints` runs as
eager JAX ops (a `vmap` of norms/crosses/`sqrt`, then a `lax.scan`), and whether
a large MJX/jaxls workload has already run in the process changes how XLA
evaluates them at the last bit. It is an execution-state artifact, not a
semantic one — and it is stable *within* an arm (`fixA` vs `fixB`: exactly
equal).

**What this rules in and out.** It rules OUT the thing that would invalidate the
change: root_optimization's result leaking into the solve. Those four columns
are written verbatim from a pure function of data that is bitwise identical, so
there is no path for the root phase's pose to reach them, and the root xyz and
all 86 hinge columns are exactly equal. What is left is a 1-ulp perturbation of
the warm start, fed into a solve whose own rerun-to-rerun floor is 40% LARGER in
the same units.

### Verdict

`stac_ik.h5` is **not bit-identical, and cannot be** — the unchanged code is not
bit-identical to itself on this hardware. Against the only meaningful yardstick:

* every live solver input is bit-identical except the warm-start quaternion,
  which moves by **1 float32 ulp** on 297/2007 frames of fly1 and **not at all**
  on fly0;
* the fitted `qpos` moves **less** than an unchanged-code rerun moves it
  (0.109 vs 0.149 max abs);
* the wing-keypoint residual is unchanged to 4 decimal places on fly1
  (0.8497 mm both) and +0.31% on fly0, against a rerun floor of ±0.0002 mm;
* the speedup, −61.5 s / −48.1 s, is 150x the 0.4 s wall-clock rerun noise.

I would not describe this as "bit-identical". I would describe it as **the fit
does not move by more than the solver moves on its own**, which is the strongest
statement this solve admits.

---

## 4. A second, independent Phase 2 item: threading `sdf_stack_from_masks`

Own commit, own measurement -- nothing here touches STAC.

`sdf_stack_from_masks` walked (T, C) = 2007 x 7 = 14 049 mask crops serially
through `cv2.resize` + two `scipy.ndimage.distance_transform_edt` calls. Both
are C extensions that release the GIL, the outputs are preallocated, and each
job writes only its own `(t, c)` slot -- so threading the loop is pure
scheduling.

### Measured on the REAL masks, on a quiet node

`Session0/2025_10_20_13_20_04` bout 28 **fly0 (the FEMALE -- the hard fly)**,
`out_hw=(128,128)`, `bbox_margin=0.4`, 32 cores.
Immediately before: `load average 1.67`, all 8 GPUs `0 %, 0 MiB`.
Immediately after: `load average 3.75` (this benchmark's own threads), GPUs
`0 %` / `2 %`, 0 MiB. No other compute job of mine was running.

| workers | s | speedup |
|---|---|---|
| 1 (serial, the old code) | 34.02 | 1.0x |
| 2 | 17.50 | 1.9x |
| 4 | 9.36 | 3.6x |
| 8 | 5.62 | 6.1x |
| 16 | 5.15 | 6.6x |
| **auto (= 16 here)** | **4.79** | **7.10x** |

That reproduces the previously-quoted 34.4 s -> 4.8 s almost exactly, this time
against committed code. Scaling is near-linear to 8 and flattens after, which is
why `_MAX_AUTO_WORKERS = 16`.

### Bit-identical, checked rather than argued

At **every** worker count (2, 4, 8, 16, auto) all four returned arrays match the
`workers=1` output under BOTH `np.array_equal` and a raw `tobytes()` compare
(the byte compare is the one that actually says *bitwise*: `array_equal` alone
would be blind to `+0.0` vs `-0.0` and would call two NaNs unequal):

```
workers=2       17.50 s
   sdf          float32  array_equal=True bitwise=True
   grid_scale   float32  array_equal=True bitwise=True
   grid_offset  float32  array_equal=True bitwise=True
   present      bool     array_equal=True bitwise=True
... identical at 4, 8, 16 and auto ...
ALL BIT-IDENTICAL
```

### Worker sizing

`SLURM_CPUS_PER_TASK` first, then `os.sched_getaffinity`, capped at 16. The
pipeline runs up to four bout processes per node under `--cpus-per-task=8`
(`configs/slurm/*.yaml: cpus: 8`), so sizing off `os.cpu_count()` would have
each of them spawn a node-wide pool. Pinned by
`test_auto_worker_count_respects_the_slurm_allocation`.

### Tests

`third_party/jarvis_jax/tests/test_mask_sdf.py`, five new cases, **seen red
first** (`TypeError: unexpected keyword 'workers'`). Mutation-checked twice:

* results zipped back against submission order via `as_completed` instead of
  each job's own slot (the classic threading defect) -> all three equality
  tests red;
* `_n_workers` ignoring `SLURM_CPUS_PER_TASK` -> the allocation test red.

### And the number that actually matters: the STAGE total

The 7x above is the precompute in isolation. What the budget cares about is
Stage D2 end to end. Same bout-fly (28/fly0, the female), shipped config with
the validity gate **on**, `wing_mask_fit.enabled=true`:

| row | serial SDF | threaded SDF | delta |
|---|---|---|---|
| `load_bout_masks` (outside the stage fn) | 9.01 | 9.19 | +0.18 |
| `body_uv_track` (gate's body basis) | 12.04 | 11.45 | −0.59 |
| `wing_fit_validity_gate` | 2.05 | 2.07 | +0.02 |
| **`sdf_stack_from_masks`** | **24.14** | **3.80** | **−20.34** |
| `refine_wing_pitch` (the Adam solve) | 57.34 | 55.38 | −1.96 |
| **`wing_mask_fit_bout` TOTAL** | **97.41** | **74.22** | **−23.19** |

Two things to read off this.

1. **Nothing else was hiding behind the precompute.** The stage total falls by
   23.2 s against the SDF row's 20.3 s — the surplus is run-to-run noise on the
   other rows, not a second serialised cost appearing. That was the stated
   expectation before the run and it held.
2. **The stage still does NOT fit its 60 s budget.** 74 s, plus 9 s of
   `load_bout_masks` before it. Threading the SDF was worth 23 s and the
   remaining 74 s is `refine_wing_pitch` (55 s, 75% of it) plus the gate's
   `body_uv_track` (11 s). Saying so plainly: this change does not make the
   stage affordable, it makes the precompute stop being the reason it is not.

   (The SDF row is 24.1 s here rather than the standalone benchmark's 34.0 s
   because the validity gate has already zeroed the sliver cameras, so there
   are fewer `(t, c)` pairs to transform.)

**Downstream reproducibility, stated honestly.** The refined qpos comes back
identical in NaN pattern with 127 of 163 215 finite entries differing, max
|Δq| = 2.44e-4 rad = **0.014 deg** — exactly the GPU rerun noise floor already
recorded for this stage in the Task 10 notes. That is Adam-on-GPU variance
between two runs, not this change: the SDF arrays `refine_wing_pitch` consumes
were proven bitwise equal above, so the threading contributes nothing to it.
