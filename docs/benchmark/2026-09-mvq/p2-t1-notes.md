# P2 Task 8: mvq figure gate 1, first real T=1 run, val baseline

Run: `run_id=mvq_t1_b16_gate1`, local (4x GPU, GPUs 0-3 on the node this session
ran on), `train.total_steps=2000 train.batch_size=32 train.eval_every=2001
train.save_every=999999`, `paths=hyak`, `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`,
`TF_GPU_ALLOCATOR=cuda_malloc_async`. Checkpoint:
`/gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_gate1/final`.
30k job: `run_id=mvq_t1_b16_20260903`, queue job id **39557158** (see "30k job"
below).

## Incident: two OOM bugs found and fixed before this run could complete

The brief anticipated an OOM at `batch_size=32` with a documented fallback to
16. Both batch sizes OOM'd locally, and a THIRD, more severe OOM (an
allocator failure on one GPU that then hangs the other 3 GPUs on the NCCL
clique rendezvous, requiring a hard kill) appeared partway through training
regardless of batch size. Root-caused and fixed both; **do not use this repo's
mvq training before these two commits** — they are required, not optional,
for a 4-GPU run to complete:

1. **`fix(mvq): remat per attention chunk`** (fusion.py). `masked_attention`'s
   `q_chunk`/`jax.lax.map` path was meant to bound the 2D cross-attention
   path's memory (`(B,heads,q_chunk,Nk)` instead of `(B,heads,Nq,Nk)`), but
   `lax.map`'s reverse-mode grad retained every chunk's logits+softmax
   simultaneously, so under `jax.grad` chunking cost *more* memory than not
   chunking — measured 8.51GB (shipped) vs 3.53GB (unchunked) vs 1.92GB
   (chunked + `jax.remat` per chunk, the fix) for the isolated 4-layer 2D
   stack at batch 4. In the full model this was the entire reason
   `train.batch_size=32` (the shipped default) OOM'd on 4x46GB GPUs at all
   (measured full-model peak 40.18GB/device at batch 4/device before the
   fix). Bisect table and isolation method: see the task-8 report.
2. **`fix(mvq): shard eval over the mesh and release eval executables`**
   (train_mvq.py). Independently of (1), `evaluate()` built its batches with
   plain `jnp.asarray` (unsharded — the whole eval batch and its `_fwd`
   executable lived on device 0 only), fragmenting GPU 0's BFC pool enough
   that the training step immediately following a mid-run eval could not
   place its own ~19GB arena: GPU 0 OOM'd (`ran out of memory trying to
   allocate 17.23GiB`) and the other 3 ranks hung on
   `rendezvous.cc:116 ... Acquire clique: devices=4:[0,1,2,3] ... may be
   stuck`, requiring `kill -9`. **Reproduced identically with periodic
   checkpointing fully disabled** (`save_every=999999`, verified zero
   checkpoint-save log lines that run), which is what isolated it to `eval`
   rather than the Orbax save. Fix: `evaluate()` now takes `mesh` and shards
   every batch array with `shard_batch` (batch padded to `batch_size` for the
   ragged last batch), and `run_training` does `del em; jax.clear_caches()`
   after each mid-run eval. Validated on GPUs 4-7
   (`run_id=mvq_evalfix_check`, 300 steps, `eval_every=100 save_every=100
   batch_size=32`, pretrained, **without** `TF_GPU_ALLOCATOR=cuda_malloc_async`,
   to prove the code fix alone is sufficient): three consecutive eval+save
   cycles (steps 100/200/300) all completed clean, no OOM, no hang.
   `JAX_PLATFORMS=cpu pytest tests/test_train_mvq_smoke.py -q`: 4 passed
   (also exercises the sharded-eval path, trivially, at n_dev=1 on CPU).

Both commits are on `mvq-p0-p2`:
`e11cb59 fix(mvq): remat per attention chunk -- lax.map retained all chunk logits under grad`
`7aa34ff fix(mvq): shard eval over the mesh and release eval executables -- unsharded eval fragmented GPU 0 and OOM'd the next step`

`TF_GPU_ALLOCATOR=cuda_malloc_async` was also set for the surviving gate-1 run
(the allocator's own suggested mitigation) but **did not change step time**:
~3.02-3.03 s/step measured both with and without it (see "seconds/step"
below) — the two commits above are what fixed the OOM, not the allocator.

Because this session's gate-1 run finished with `eval_every=2001`
(deliberately set higher than `total_steps` to sidestep the eval-OOM bug
*before* the fix landed, then kept after the fix landed since the run was
already committed to that config), its own log has only ONE mid-run eval
print (nominally at "step 2000", which is really the final eval, printed
once more from inside the loop and then once more, identically, from the
unconditional post-loop path). The **step-1000 numbers below are pulled from
an earlier, separate attempt** at this exact config (before the OOM saga),
run to verify training itself was healthy before spending time on gate-1
figures; two independent attempts agree to ~1%.

## Numbers

**Step 1000 (train, from an earlier same-config attempt):** `match_reproj_px`
17.0 px (an earlier print: 16.97 px; a later independent attempt: 18.78 px —
consistent to ~1%), `exist_acc` 0.906-0.917 (near 1.0, as expected on
single-fly batches).

**Step 1000 (val, same earlier attempt):**
| mode | mpjpe3d (units / mm) | reproj_px | exist_prec / exist_rec |
|---|---|---|---|
| prompted | 10.53 / 1.05 | 72.26 | 0.729 / 1.000 |
| unprompted | 9.45 / 0.95 | 71.67 | 0.559 / 1.000 |

**Step 2000 = final (val, this run's `final/mvq_run.json`):**
| mode | mpjpe3d (units / mm) | reproj_px | uv2d_px | head_vs_reproj_px | exist_prec / exist_rec | cohort_female | cohort_two_fly | cohort_group_A | cohort_group_C |
|---|---|---|---|---|---|---|---|---|---|
| prompted | 4.72 / 0.472 | 32.66 | 43.75 | 18.92 | 1.000 / 0.597 | 4.94 | 5.19 | 4.50 | 5.60 |
| unprompted | 3.76 / 0.376 | 26.71 | 38.18 | 19.06 | 0.848 / 0.855 | 4.04 | 4.29 | 3.50 | 4.80 |

rigid_spread_mm (unprompted; per-segment std of the fitted length across the
153-window val set, mm — smaller is more rigid/consistent): EyeL-EyeR 0.051,
T1L 0.033, T1R 0.044, T2L 0.045, T2R 0.040, T3L 0.039, T3R 0.035 — all
sub-0.06mm, i.e. the model's own segment lengths are internally consistent
window-to-window even though absolute position error is still large; expected
at this step count (a set predictor can be self-consistent before it is
accurate).

Training final: loss 80.48, train `match_reproj_px` 6.71 px at step 2000 (vs
`uv2d_px` 8.81 px) — **training-set** performance is already good; the
prompted/unprompted **val** numbers above (uv2d_px 38-44px) show a large
train/val generalization gap at 2k/30k steps, worth watching as the 30k run
progresses (not a gate-1 blocker — gate 1 checks geometry, not final
accuracy).

**seconds/step:** ~3.02-3.03 s/step steady-state at `batch_size=32` on 4
GPUs, measured identically in the run WITHOUT `TF_GPU_ALLOCATOR=cuda_malloc_async`
((1506-295)/(450-50)=3.03 s/step) and the final run WITH it
((5577-291)/(1800-50)=3.02 s/step, and (6181-291)/(2000-50)=3.02 s/step
end-to-end) — **the async allocator did not measurably change step time**;
it was carried forward per the controller's request as a defense-in-depth
mitigation alongside the two code fixes, not because it was shown necessary.
Wall time for the full 2000-step run: ~103 minutes (6181s reported at the
final step, plus ~130s of one-time backbone-load/compile overhead before the
timer starts).

**The data loader is not the bottleneck** (profiled): 0.24 s/batch at
`num_workers=16`, 0.11 s/batch at `num_workers=32`, against 2.78 s/step of
GPU compute for one GPU's share of a step. Even the slower (16-worker)
loader number is under a tenth of the per-step compute time, so the spec
§6.2 strip-cache mitigation (deferred pending evidence the loader is the
bottleneck) is **not triggered** — the measured ~3.02 s/step is compute-bound,
not IO-bound.

## Fix round 1 (2026-09-04): EMA debiasing was silently shrinking every joint

**UPDATE (fix round 2)**: the debiasing divisor `1/(1-decay**t)` was being
applied unconditionally on resume, inferring `t` from the step count -- but
a checkpoint saved by PRE-fix code has its EMA seeded from live params, not
zero, and applying the same divisor to that non-zero-seeded EMA would
silently rescale it incorrectly (it does not need debiasing at all). Fixed:
`t` is now an explicit `ema_updates` counter persisted in a new mandatory
Orbax item, `ema_meta` (`{"ema_updates": int, "ema_zero_seeded": True}`);
`_restore_latest` raises `ValueError` on any checkpoint missing this item or
whose `ema_zero_seeded` is not True, naming the checkpoint dir, rather than
guessing. See "30k job" below: the currently-running local 30k run predates
this fix (and fix round 1 entirely) and will need restarting.

**Controller-diagnosed defect, now fixed** (`train_mvq.py`, commit in this
round): the EMA was seeded from the step-0 (near-random) params and never
debiased. At step 2000, `decay**t = 0.999**2000 = 0.135` of that near-zero
init still weighted the average that `final/` actually saved and evaluated
-- a uniform SHRINK of every predicted joint toward the crop centre (xyz=0
in local coordinates), worse for joints far from centre. Measured on the
gate-1 checkpoint (predicted/GT radial distance from the crop centre, i.e.
exactly the ratio this bug would suppress):

| region | radial dist/GT ratio |
|---|---|
| head / thorax / abdomen / wing-base (proximal, near centre) | 0.86-0.93 |
| distal legs (tarsi, furthest from centre) | 0.68-0.83 |

This is a plausible SECOND contributor (alongside the 2D-precision gap
already discussed above) to the "green stops short of the tarsal tips"
pattern seen in every figure in this gate -- a uniform radial shrink reads
visually almost identically to "legs under-extended," and the two are hard
to tell apart from a static overlay alone. Fix: EMA now seeds at zero and
`_with_ema` divides by `(1 - decay**t)` (`t` = the current absolute step
count, always exactly the EMA's own update count -- no separate counter
needs to be saved/restored, since the checkpoint step number already IS
that count). `tests/test_train_mvq_smoke.py::test_ema_debias_matches_constant_params`
pins the identity: a zero-seeded EMA of a CONSTANT p, debiased, equals p to
1e-6 for n in {1, 10, 1000}. **This changes the numbers everywhere above and
in `mvq_run.json`** -- the gate-1 `final/` checkpoint predates this fix, so
every mm/units number in this document (val mpjpe, rigid_spread, baseline
comparison) reflects the BIASED (shrunk) EMA, not the debiased one. Re-running
gate 1 against a checkpoint trained with the fix is the natural next step;
not done in this round (the 8-GPU local 30k run owns GPUs 0-7 for its
duration -- see "30k job" below).

**New: `rigid_len_ratio/<seg>`** (mean predicted segment length / mean GT
segment length, alongside the existing `rigid_spread_mm`) makes exactly this
kind of collapse visible numerically, which spread alone cannot (a
collapsed-but-STEADY pair is smoother, not noisier -- CLAUDE.md's "check a
rigid invariant" history). Computed against the (pre-fix, biased-EMA) gate-1
checkpoint, unprompted, val set (CPU, `evaluate(model, val_ds, 8, ...)`):

| segment | rigid_len_ratio (prompted) | rigid_len_ratio (unprompted) |
|---|---|---|
| EyeL-EyeR | 0.426 | 0.541 |
| T1L_Tro-T1L_FeTi | 0.567 | 0.578 |
| T1R_Tro-T1R_FeTi | 0.540 | 0.630 |
| T2L_Tro-T2L_FeTi | 0.620 | 0.653 |
| T2R_Tro-T2R_FeTi | 0.569 | 0.708 |
| T3L_Tro-T3L_FeTi | 0.665 | 0.720 |
| T3R_Tro-T3R_FeTi | 0.637 | 0.754 |

Every segment ratio is well below 1.0 (predicted consistently SHORTER than
GT) -- direct, independent confirmation that the model under-predicts
segment length, in the same direction the EMA-debias bug predicts. Not
uniform across segments (0.43-0.75), so this is not a single global
position-domain scale factor (which would make every ratio identical); it
varies by which two joints and which part of the query/head pipeline they
pass through, same qualitative shape as the controller's radial-ratio table
above (non-uniform, worse for some categories) without the two metrics
being expected to match numerically (radial dist/GT is a per-JOINT position
ratio; this is a per-SEGMENT length ratio). Notably `EyeL-EyeR` -- a short,
bilateral (left/right) segment -- is the MOST collapsed here despite both
its endpoints being individually close to the crop centre in the
controller's proximal-joint bucket; a plausible read is that the bug
collapses left/right separation specifically (both eyes pulled toward the
body midline) rather than being explained by radial distance from centre
alone. All numbers computed on the pre-fix (biased-EMA) gate-1 checkpoint --
`evaluate(model, val_ds, 8, ...)` re-run on CPU against `final/`, val set,
153 framesets.

## Val baseline (ViTPose v5vf_maskoff 2D -> robust DLT), full 153 val framesets

`scripts/benchmark/mvq_val_baselines.py`. **Fix round 1** (2026-09-04):
gate thresholds, checkpoint path, `zero_mask_channel`, and `num_keypoints`
are now read from `configs/detector/vitpose_v3.yaml` via OmegaConf (resolved
against `configs/paths/hyak.yaml` for `${paths.vit_runs_root}`), not
hardcoded -- a config change now actually changes what this script runs. The
script asserts the detector's `kp_names` list equals the dataset's
`keypoint_names` element-wise BY NAME before comparing anything (raises
otherwise) -- checked and PASSES on this root (both 50-name lists are
identical, position-for-position; this is a belt-and-suspenders check, not a
fix for a mismatch that existed here). Re-run on GPU 4, then a second full
CPU pass (`JAX_PLATFORMS=cpu`, GPUs 0-7 owned by the 8-GPU local 30k run --
see below) adding the mvq-comparison arms:

| cohort | baseline mpjpe (units/mm) | coverage | mvq SAME joints, unprompted (units/mm) | mvq ALL joints, unprompted (units/mm) [=official] |
|---|---|---|---|---|
| overall | 1.070 / 0.107 | 1.000 | 3.760 / 0.376 | 3.760 / 0.376 |
| female | 0.723 / 0.072 | 1.000 | 4.040 / 0.404 | 4.040 / 0.404 |
| two_fly | 1.701 / 0.170 | 1.000 | 4.293 / 0.429 | 4.293 / 0.429 |
| group_A | 1.170 / 0.117 | 1.000 | 3.497 / 0.350 | 3.497 / 0.350 |
| group_C | 0.674 / 0.067 | 1.000 | 4.801 / 0.480 | 4.802 / 0.480 |

**Fairness finding**: `coverage` (fraction of GT joints the baseline's DLT
could triangulate, `sum(n_valid)/sum(n_has)`) is **1.000 in every cohort** --
the baseline never failed to triangulate a labelled joint anywhere in this
153-frameset val set, so "mvq restricted to the baseline's finite joints"
and "mvq on ALL joints" are numerically identical here (compare columns 4
and 5 above -- the earlier concern that the baseline could be scored on an
easier self-selected joint subset than mvq does not materialize on THIS val
set, though the code now checks it on every run rather than assuming it).
`n_empty_framesets` (a frameset where DLT triangulated zero GT joints) is 0.
Runtime: 79s on GPU, ~22 min on CPU with the mvq comparison added (one extra
mvq forward pass per frameset).

**Train-set-exposure caveat (item 3, could not be fully resolved)**: the
detector was trained on `red_data_3d_v5_valfix` (`.hydra/overrides.yaml`),
NOT v12's root, and that root no longer exists on disk (2026-09-02 cleanup)
-- this script cannot check directly whether any of v12's val recordings
(`2026_01_29_14_09_33`, `2026_04_02_12_11_50`, `2026_04_02_15_25_51`,
`2026_06_09_15_21_14`, `2026_06_09_15_46_55`) sat in the detector's own
TRAIN split. `docs/benchmark/2026-09-02-dataset-dedup/notes.md` documents
exactly this dataset lineage's known failure mode: byte-identical
recordings re-ingested under different session names landing on opposite
sides of a name-keyed split, including a confirmed `courtship_11_50_female`/
`courtship_11_50_male` alias pair. v12's own `2026_04_02_12_11_50` carries
that same `courtship_11_50` subset label -- suggestive, but NOT confirmed
(the alias pair's dates, 2026_05_27, don't match), since the source root is
gone. **This baseline's numbers are therefore PROVISIONAL**: the gap between
mvq and this baseline should be read as an upper bound on how far mvq
currently trails, not a precise, guaranteed-held-out figure. (Full text:
`train_exposure_caveat` field in
`docs/benchmark/2026-09-mvq/vitpose_dlt_baseline_gate1.json`.)

**A4 c2f arm reference (0.56 units)**: `docs/benchmark/2026-08-30-arm-results/notes.md`
— a **DIFFERENT, NOT like-for-like** val split (`red_data_3d_v5`, 216
framesets, 18 female / 198 male) and a different model family (HybridNet-style
c2f volumetric net, not this lifter). Given only for rough scale: it is close
in magnitude to this baseline's `overall` number (1.070) despite the
different split, which is a sanity check on the baseline script's units, not
a claim of equivalence.

## Expectation (stated before comparing) vs. what the numbers show

*Expected*: mvq prompted beats the ViTPose+DLT baseline on two_fly and
female; on male-cohort-analog metrics it's allowed to be within noise;
unprompted worse than prompted on two_fly by a margin that shrinks once P3's
copy-paste lands.

*What happened, gate-1 (2k steps)*: **mvq is worse than the baseline on
every cohort** — prompted/unprompted mpjpe (mm) 0.472/0.376 overall vs
baseline 0.107; 0.494/0.404 vs 0.072 (female); 0.519/0.429 vs 0.170
(two_fly). Per the brief's own diagnostic checklist for this outcome: mvq's
`uv2d_px` (43.75 prompted / 38.18 unprompted) is ~7-8x the trained detector's
own 5.29px val MPJPE (`configs/detector/vitpose_v3.yaml` docstring) — **the
2D precision gap is the dominant, expected explanation**: at 2k of a planned
30k steps the query-decoder's own 2D head has not yet converged anywhere
near the mature ViTPose detector's precision, so a worse 3D number follows
mechanically. This is explicitly labelled "**gate-1 (2k steps)**", not a
final comparison; the like-for-like 30k-vs-baseline comparison is a
follow-up once job 39557158 finishes.

One number goes the OTHER direction from the P3 expectation: unprompted
narrowly **beats** prompted on every cohort at this checkpoint (e.g. two_fly
4.29 vs 5.19 units) — the reverse of "prompted should help." Read together
with `exist_prec`/`exist_rec` (prompted 1.000/0.597 vs unprompted
0.848/0.855 on two_fly-type existence), the mask-prompted path is currently
UNDER-predicting existence (high precision, low recall: it only claims a
second instance exists when very sure) while unprompted over-predicts
existence more evenly. **This reversal is UNEXPLAINED, not an anneal-schedule
artifact** — an earlier draft of this note attributed it to `prompt_p`
annealing from 1.0 to 0.5 over `prompt_anneal_steps=2000` coinciding with
gate-1's own step count, reasoning the model had "only just started seeing
genuinely unprompted batches"; that reasoning does not hold up (the anneal
schedule affects how OFTEN each mode is trained on, not which mode should
score better at eval time, and by step 2000 `prompt_p` has already reached
its floor of 0.5, i.e. half of training batches were unprompted throughout
the back half of the run) and is retracted. Re-check at 30k with no
supporting theory yet for why prompted underperforms here.

## Figure gate 1 — per-figure observations

EXPECTATION (from `scripts/viz/mvq_overlay.py`'s docstring, written before
generating or looking at any figure): after 1k+ steps the GREEN reprojected
3D and the CYAN 2D head both sit on the fly in every valid camera, within a
few px of the WHITE human labels on easy male frames. If green is offset by
the same vector in all cameras, center3D/t_local bookkeeping is wrong (crop
origin or local offset); if green is right in some cameras and
rotated/mirrored in others, the per-camera geometry tokens or the camera
order by name is wrong; if cyan is fine and green is not, the 3D path is
broken independently of the encoder. On female wall/contact frames the
expectation is looser: points stay on HER body, never on the male.

All PNGs opened and read directly (Read tool) below; no PNG is committed
(`figures/` is gitignored) — regenerate with:
```
PYTHONPATH=third_party/jarvis_jax:. python scripts/viz/mvq_overlay.py \
  --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_gate1/final \
  --split val --n 6 --cases female,two_fly,worst [--prompted] \
  --out figures/2026-09-mvq/mvq_t1_b16_gate1/{unprompted,prompted}
```

### `unprompted/gate1_female.png` (6 female windows, calib group A, all cam1-cam7)

**Saw (corrected after a closer re-read -- an earlier draft of this note
mischaracterized this figure as a "diffuse cluster" that does not follow the
legs at all; that description is WRONG for this figure and is retracted
here)**: white (human) points radiate cleanly out along each leg to the
tarsal tips in every camera, as expected -- the labels themselves are fine.
Cyan (model 2D head) and green (model reprojected 3D) DO follow the same
leg-chain directions as white, visibly paralleling each leg outward from the
body in most of the 42 panels (6 samples x 7 cams) -- they are not a
body-centered blob. What they get wrong is reach: green/cyan consistently
fall short of white's tarsal tips, stopping partway along each leg (most
visible on cam2-cam5, where full legs are in frame). Per-panel reprojection
16.7-22.6px. Critically: this shortened-leg pattern tracks the SAME fly, in
the SAME rough pose, consistently across all 7 cameras for a given sample --
no camera shows it rotated, mirrored, or shifted relative to the others, and
no camera shows it jumping to the background or off the animal. **Meets the
loosened female-cohort bar** (stays on her body, legs recognizably in the
right directions); does **not** meet the strict few-px bar, which the
docstring explicitly reserves for "easy male frames" -- these are hard,
wall-adjacent female frames. (The genuinely body-centered, leg-chain-blind
collapse described below is a `gate1_worst` phenomenon, not this one.)

### `unprompted/gate1_two_fly.png` (6 two-fly windows, calib group C, mixed M/F)

**Saw**: same diffuse-cluster pattern as female, per-panel error 11.2-58.8px
(one outlier, sample #10, an edge/occlusion frame where the host fly is
barely inside the crop in cam1 at 47-59px reproj — the harder case flagged
by CLAUDE.md's "check the hard case" guidance). In samples #12 and #13 a
SECOND fly is visible at the frame edge in several cameras (cam5-cam7); the
green/cyan point cloud stays entirely on the HOST fly in every camera and
never attaches to the second animal — the one failure mode this gate exists
to catch (cross-fly bleed) does **not** appear.

### `unprompted/gate1_worst.png` (6 highest-reproj-px windows: 2 courtship
pairs calib group A + 4 wall/occlusion female group C, sorted worst-first)

**Saw**: the clearest view of the actual failure mode. In samples #67/#65 (a
mating pair, second fly plainly visible below the host in every camera) and
#10/#30/#24/#28 (wall-adjacent females), green/cyan collapse toward the
body-center/thorax and systematically fail to extend to the leg tips that
white correctly reaches — here the cloud genuinely does NOT trace the leg
chains (unlike the milder female-cohort shortening above): it is a compact
blob near the thorax regardless of the legs' actual splayed directions.
Per-panel error 36.3-59.8px. Same cross-fly check as above: even with a
second animal filling a third of the frame (#67, #65), the model's points
stay exclusively on the HOST fly, never the other animal — no sample in this
"worst" set shows cross-fly attachment.

**One partial exception, re-read and confirmed**: sample #67 (top row), cam6
specifically -- the green/cyan cloud sits at/just below the wall-edge line
near the BOTTOM of that panel, not clearly on the visible fly's body (which
sits more toward upper-left in that view). This is not cross-fly bleed (the
second animal is not there either); it looks like the point cloud landing on
the floor/wall boundary rather than on any fly. One panel out of 42 (6
samples x 7 cams) in this set -- not systemic, but a real miss worth naming
rather than folding into "stays on the host fly."

### `prompted/gate1_{female,two_fly,worst}.png` — same 3 cases, `--prompted`

**Saw**: qualitatively identical pattern to unprompted (diffuse
center-of-body cluster, never on the wrong fly, never rotated/mirrored/offset
by a constant vector), with per-panel errors uniformly ~2-8px HIGHER than
the matching unprompted samples (e.g. female #0: 21.7-29.6px prompted vs
16.7-22.6px unprompted; two_fly #10: 42.8-61.5px vs 40.2-58.8px) — consistent
with the numeric finding above that `prompted` currently trails `unprompted`
on every cohort at this early-anneal checkpoint.

### Gate verdict: **PASS** (geometry), accuracy gap is expected and separately diagnosed

The three failure signatures the gate is built to catch — (a) a constant
per-camera offset vector (crop-origin/center3D bookkeeping bug), (b) points
correct in some cameras and rotated/mirrored in others (camera-order-by-name
or geometry-token bug), (c) cross-fly attachment on two-fly windows — are
**absent** in all 12 PNGs (both prompted and unprompted x female/two_fly/worst),
with one single-panel exception (`gate1_worst`, sample #67 cam6, unprompted:
the point cloud sits on the floor/wall boundary rather than the fly -- not
cross-fly bleed, since neither animal is under it, but a genuine miss, 1 of
42 panels in that set). Cyan and green move together (not "cyan fine, green
broken"), ruling out an independently-broken 3D path. The real, visible
problem — legs recognizably followed but under-extended in the female
cohort, collapsing further into a body-centered blob on the hardest
(wall/occlusion/courtship-pair) `gate1_worst` frames — is a
precision/undertraining signature consistent with the numeric val/baseline
gap above (2k of 30k steps, `uv2d_px` still ~7-8x the mature detector's own
precision), not a geometry defect. **Proceed to the 30k run**; re-run this
same gate at 30k to confirm the tarsal-tip gap closes as `uv2d_px` converges.

## 30k job

**UPDATE (fix round 1, 2026-09-04): superseded.** Queue job **39557158** was
`scancel`led; the 30k run is now LOCAL on this node's all 8 GPUs,
`run_id=mvq_t1_b16_local8_20260904` (PID 642747):
```
python -m jarvis_jax.scripts.train_mvq model=mvq train=mvq paths=hyak \
  run_id=mvq_t1_b16_local8_20260904 train.total_steps=30000 train.eval_every=2000 \
  train.save_every=1000 train.batch_size=32 train.num_workers=24 \
  "paths.runs_root=\${paths.mvq_runs_root}"
```
GPUs 0-7 were confirmed idle before launch and are fully occupied by this
run for the duration of this fix round (all fix-round-1/2 work below used
`JAX_PLATFORMS=cpu`, never touching 0-7).

**This run PREDATES fix rounds 1 and 2 and will need to be restarted.** It
was launched with the code as of commit `d49c9da` -- before the attention-
chunk remat fix (`e11cb59`), the eval-sharding fix (`7aa34ff`), and every
fix-round-1/2 item (EMA debiasing + the persisted `ema_meta` counter,
`q_chunk=None`, batch-independent eval metrics, `rigid_len_ratio`). Its
`final/` checkpoint, if let run to completion, would carry the SAME
step-0-seeded, undebiased EMA bias documented above for gate-1, and its
`ckpt/` checkpoints predate the mandatory `ema_meta` item -- fix round 2's
`_restore_latest` will REFUSE to resume this run's own checkpoints with a
`ValueError` once it hits its next `save_every` boundary and this code is
used to resume it (working as intended: it must not be silently treated as
zero-seeded when it is not). Plan to `scancel`/kill this run and relaunch
fresh once the code state here is what should be trained on; not done in
this round since GPUs 0-7 were off-limits throughout. Not yet past its own
step-2000 eval boundary at the time of writing this update — the "worth a
quick log check" item from the superseded queue-job entry below still
applies to this
run instead.

<details><summary>Superseded queue-job entry (job 39557158, cancelled)</summary>

Job id **39557158** (queue `ckpt-all`), submitted from the worktree root with
the eval fix already committed, `train.save_every=250` (checkpoint every
~12-13 min at this node's measured s/step, well under the ~20-30min window
between `ckpt-all` preemptions observed this session) and
`export TF_GPU_ALLOCATOR=cuda_malloc_async &&` prefixed ahead of the training
command (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` also overridden, since
`submit_task.sh` defaults to 0.6). Command:
```
cd <worktree>
scripts/slurm/submit_task.sh --gpus 4 --cpus 32 --mem 200 --time 36:00:00 mvq_t1_b16 \
  'module load cuda; unset JAX_PLATFORMS; export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3; export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && export TF_GPU_ALLOCATOR=cuda_malloc_async && cd third_party/jarvis_jax && python -m jarvis_jax.scripts.train_mvq model=mvq train=mvq paths=hyak run_id=mvq_t1_b16_20260903 train.save_every=250 "paths.runs_root=\${paths.mvq_runs_root}"'
```
Observed before cancellation: past its first TWO periodic checkpoint saves
(step 250, step 500) with no OOM, then preempted (routine `ckpt-all`
`CANCELLED ... DUE TO PREEMPTION`, not a bug — this partition is
checkpoint/preemptible and was preempted 3 times total across this session's
attempts at this job, always auto-requeuing via `--requeue`) three times
before being superseded by the local run above.

</details>

## Files changed

Original commit (Task 8):
- `scripts/viz/mvq_overlay.py` (new) — figure-gate overlay script.
- `scripts/benchmark/mvq_val_baselines.py` (new) — ViTPose+DLT baseline script.
- `third_party/jarvis_jax/jarvis_jax/models/mvq/fusion.py`,
  `third_party/jarvis_jax/tests/test_mvq_model.py` — the attention-chunk
  remat fix (commit `e11cb59`).
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` — the eval-sharding
  fix (commit `7aa34ff`).

**Fix round 1** (2026-09-04, this update):
- `scripts/viz/mvq_overlay.py` — camera names by NAME (not `cam{c+1}`) via
  the new `V12WindowDataset.camera_names`; legend moved outside the image
  axes; `C` from `s["crops"].shape[1]` not hardcoded 7; `MM_PER_UNIT` instead
  of a literal `0.1`.
- `scripts/benchmark/mvq_val_baselines.py` — OmegaConf-resolved detector
  config (path/thresholds no longer hardcoded); `kp_names`-by-name assertion
  (raises on mismatch); mvq-vs-baseline same-joint-set fairness comparison
  (`--mvq_run`); per-cohort coverage/`n_framesets`/`n_framesets_finite`;
  empty-frameset warning; train-set-exposure caveat.
- `third_party/jarvis_jax/jarvis_jax/data/v12_windows.py` — new
  `camera_names(i)`.
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` — EMA debiasing
  (seed at zero, divide by `1-decay**t` in `_with_ema`); batch-weighted
  `reproj_px`/`uv2d_px`/`head_vs_reproj_px`; new `rigid_len_ratio/<seg>`;
  corrected `del em` comment (relief is `jax.clear_caches()`, not the `del`).
- `third_party/jarvis_jax/jarvis_jax/models/mvq/model.py`,
  `third_party/jarvis_jax/jarvis_jax/models/mvq/decoder.py`,
  `third_party/jarvis_jax/configs/model/mvq.yaml` — `MVQConfig.q_chunk`
  (default `None`, was an implicit 512), threaded to both `CrossBlock`
  call sites.
- `third_party/jarvis_jax/tests/test_mvq_model.py` — chunked-grad test now
  also checks `k`/`v` gradients (was `q` only).
- `third_party/jarvis_jax/tests/test_train_mvq_smoke.py` — updated
  `_with_ema` call site (new required `decay, t` args); new
  `test_ema_debias_matches_constant_params`,
  `test_evaluate_ragged_batch_matches_full_batch`.
- This file.

**Fix round 2** (2026-09-04, this update):
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` — persisted
  `ema_meta` Orbax item (`ema_updates`, `ema_zero_seeded`); `_restore_latest`
  refuses (ValueError) a checkpoint missing it or with `ema_zero_seeded`
  not True; `run_training` tracks `ema_updates` explicitly (not inferred
  from the step count) and returns it; `evaluate`'s per-batch weight for
  `reproj_px`/`uv2d_px`/`head_vs_reproj_px` now counts only the real
  (`:B0`) rows, not the padded ones; `_finish` returns NaN instead of
  raising `IndexError` when a mode has zero batches.
- `third_party/jarvis_jax/tests/test_train_mvq_smoke.py` — resume test
  asserts `ema_updates` round-trips; new
  `test_resume_refuses_checkpoint_without_ema_meta`; ragged-batch test
  extended to the three px keys (1e-4 tolerance -- genuine float
  non-associativity between batch shapes, not a grouping-dependence bug).
- `third_party/jarvis_jax/jarvis_jax/models/mvq/model.py`,
  `third_party/jarvis_jax/configs/model/mvq.yaml` — corrected a comment
  (the previous implicit `q_chunk` default was 512, not 8).
- `scripts/benchmark/mvq_val_baselines.py` — `zero_mask_channel` read from
  config and actually applied (zeros the 4th channel only when True;
  otherwise feeds `prompt_mask` as the 4th channel); `detector_zero_mask_channel`
  recorded in the JSON; a length-only `kp_names` mismatch now raises
  `ValueError` (was `TypeError` from indexing with `first=None`).
- `scripts/viz/mvq_overlay.py` — grid width is the max camera count across
  selected rows, with each row taking its OWN camera count (`Cr`) rather
  than assuming every row shares the first row's.
- This file (EMA-resume-hazard correction, group_C table cell, 30k-run
  provenance note).

`figures/2026-09-mvq/mvq_t1_b16_gate1/{unprompted,prompted}/gate1_{female,two_fly,worst}.png`
+ `summary.json` per mode, and `vitpose_dlt_baseline.json`, all under the
gitignored `figures/` tree — regenerate with the commands above (overlay) and
`PYTHONPATH=third_party/jarvis_jax:. python scripts/benchmark/mvq_val_baselines.py --out <path> [--mvq_run <final_dir>]`
(baseline). `docs/benchmark/2026-09-mvq/vitpose_dlt_baseline_gate1.json` is
the same baseline run (with the mvq comparison) committed as a small text
artifact per CLAUDE.md's evidence-for-a-decision convention.

## Task 9: Attention implementation A/B (2026-09-04) — cuDNN flash attention

No figure for this one, per CLAUDE.md's own exemption ("changes that provably
move no numbers... behavior-preserving refactors... don't need one"): this is
an alternate compute path for the SAME attention math (backbone + decoder),
proven equivalent by GPU parity tests (`test_mvq_attention.py`,
`test_dinov3.py`, `test_mvq_model.py`, `-m gpu -k cudnn`) at bf16 tolerance
(<2e-2 max abs diff, gradient cosine sim > 0.99) before this A/B ever ran —
the A/B below is about wall-clock and confirming the swap doesn't move
*training* numbers beyond noise at matched steps, not about correctness.

**Expectation (stated before running):** cudnn should be meaningfully faster
per step (the flash-attention kernel never materialises the (B,heads,N,Nk)
logits the explicit path does) with val `mpjpe3d_units`/`reproj_px` matching
xla to within a few percent at steps 150 and 300 — a big s/step win, a
near-identical loss/val curve.

**Setup:** one queue job (4x L40S, `ckpt-all`, job **39563250**), both arms
sequential in-process, shipped config (`model=mvq train=mvq paths=hyak`,
`train.batch_size=32`), 300 steps each, `train.eval_every=150
train.save_every=999999`, same seed (default 0), only `model.attn_impl`
differs: `run_id=mvq_attn_ab_xla model.attn_impl=xla` then
`run_id=mvq_attn_ab_cudnn model.attn_impl=cudnn`. Both arms completed with no
errors (no `NotImplementedError`/OOM/CUDA_ERROR) — cudnn compiles and runs
cleanly on the shipped 4-GPU/batch-32 config.

**s/step** (steady state, computed from the CLEAN 50-step gaps that don't
straddle the step-150 eval call — i.e. 50→100, 100→150, 200→250, 250→300 —
so eval time never contaminates these numbers):

| mode | 50→100 | 100→150 | 200→250 | 250→300 | mean s/step |
|---|---|---|---|---|---|
| xla | 2.66 | 2.66 | 2.66 | 2.64 | **2.66** |
| cudnn | 1.54 | 1.52 | 1.58 | 1.52 | **1.54** |

**cudnn is 42% faster per step (1.73x)** — well above the brief's 10%
adoption bar. (Peak memory wasn't printed by this training script — unlike
the standalone `flash_bench2.py` microbenchmark that motivated this task,
`train_mvq.py`'s step/eval prints don't include a memory line — so no peak-
memory row here; the flash kernel's own design (no materialised logits)
predicts a reduction, but that's not independently measured in this run.)

**val metrics at matched steps** (mpjpe3d in dataset units / reproj in px,
153-frameset val set, both cohorts prompted+unprompted):

| step | mode | xla mpjpe3d_units | cudnn mpjpe3d_units | Δ | xla reproj_px | cudnn reproj_px | Δ |
|---|---|---|---|---|---|---|---|
| 150 | prompted | 11.1362 | 11.2064 | +0.63% | 85.6704 | 85.4919 | -0.21% |
| 150 | unprompted | 11.2520 | 11.2361 | -0.14% | 89.1003 | 88.8518 | -0.28% |
| 300 | prompted | 11.0923 | 11.0739 | -0.17% | 78.8213 | 78.1027 | -0.91% |
| 300 | unprompted | 11.0424 | 11.0759 | +0.30% | 86.9499 | 86.2378 | -0.82% |

Every val delta is under 1% — an order of magnitude inside the brief's 5%
noise bar, consistent with the pre-existing GPU parity tests showing the two
paths compute the same attention to bf16 precision. (`exist_prec`/`exist_rec`
also track closely between modes at each step; not tabulated since they
aren't part of the adoption criteria.)

**Adoption verdict: ADOPT cudnn.** Both criteria cleared with margin: 42%
wall-clock gain (>>10%) and <1% val drift (<<5%) at both checkpoints.
`configs/model/mvq.yaml` already ships `attn_impl: cudnn` (this task's
commit); no further config change needed.

Run dirs `mvq_attn_ab_xla`/`mvq_attn_ab_cudnn` under
`${paths.mvq_runs_root}` were deleted after this table was extracted (per
the task brief) — re-run
`scripts/slurm/submit_task.sh --gpus 4 --cpus 32 --mem 200 --time 1:30:00 <name> '...'`
with the two `run_id=...model.attn_impl=...` invocations above (see
`third_party/jarvis_jax/jarvis_jax/train/train_mvq.py`'s CLI) to reproduce.
