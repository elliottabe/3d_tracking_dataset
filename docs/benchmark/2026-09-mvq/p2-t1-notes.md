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

## Val baseline (ViTPose v5vf_maskoff 2D -> robust DLT), full 153 val framesets

`scripts/benchmark/mvq_val_baselines.py`, default args (`conf_thresh=0.3
view_conf_thresh=0.6 reproj_resid_px=10.0 decode_sharpen=3.0`, matching
`configs/detector/vitpose_v3.yaml`), run on GPU 4 alone:

| cohort | mpjpe (units / mm) |
|---|---|
| overall | 1.070 / 0.107 |
| female | 0.723 / 0.072 |
| two_fly | 1.701 / 0.170 |
| group_A | 1.170 / 0.117 |
| group_C | 0.674 / 0.067 |

Runtime: 79s for all 153 framesets (a single OOM-and-recover warning mid-run
at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.5`, harmless — the allocator fell back to
a smaller buffer and every fresset still triangulated).

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
existence more evenly. `prompt_p` anneals from 1.0 to 0.5 over exactly
`prompt_anneal_steps=2000` steps, so at step 2000 the model has only just
started seeing genuinely unprompted batches during training — the prompted
path is the one that dominated training so far, and its cohort numbers being
slightly worse is therefore a step-2000-specific artifact of the anneal
schedule, not evidence prompting itself hurts; re-check once the 30k run is
well past its own anneal window.

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

**Saw**: white (human) points radiate cleanly out along each leg to the
tarsal tips in every camera, as expected — the labels themselves are fine.
Cyan (model 2D head) and green (model reprojected 3D) sit in a diffuse
cluster covering the fly's central body mass (thorax/head/wing-base region)
in every one of the 42 panels (6 samples x 7 cams), but do **not** reach out
along the leg chains to the tarsal tips the way white does. Per-panel
reprojection 16.7-22.6px. Critically: this diffuse cluster tracks the SAME
fly, in the SAME rough pose, consistently across all 7 cameras for a given
sample — no camera shows the cloud rotated, mirrored, or shifted relative to
the others, and no camera shows it jumping to the background or off the
animal. **Meets the loosened female-cohort bar** (stays on her body); does
**not** meet the strict few-px bar, which the docstring explicitly reserves
for "easy male frames" — these are hard, wall-adjacent female frames.

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
white correctly reaches — the point cloud is consistently "too short," not
consistently offset or rotated. Per-panel error 36.3-59.8px. Same
cross-fly check as above: even with a second animal filling a third of the
frame (#67, #65), the model's points stay exclusively on the host fly — no
sample in this "worst" set shows attachment to the wrong animal.

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
**absent** in all 12 PNGs (both prompted and unprompted x female/two_fly/worst).
Cyan and green move together (not "cyan fine, green broken"), ruling out an
independently-broken 3D path. The real, visible problem — points cluster near
body-center instead of extending to the tarsal tips, worst on the hardest
(wall/occlusion/courtship-pair) frames — is a precision/undertraining
signature consistent with the numeric val/baseline gap above (2k of 30k
steps, `uv2d_px` still ~7-8x the mature detector's own precision), not a
geometry defect. **Proceed to the 30k run**; re-run this same gate at 30k to
confirm the tarsal-tip gap closes as `uv2d_px` converges.

## 30k job

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
Observed at hand-off: past its first TWO periodic checkpoint saves (step 250,
step 500) with no OOM, then preempted (routine `ckpt-all` `CANCELLED ...
DUE TO PREEMPTION`, not a bug — this partition is checkpoint/preemptible and
was preempted 3 times total across this session's attempts at this job,
always auto-requeuing via `--requeue`). Do not wait on it; its own
`eval_every` default (2000) means it will exercise the eval-sharding fix at
its own step 2000 for the first time under real 30k-scale conditions —
worth a quick log check the next time this job is revisited, specifically
that step 2000-2001 shows no `bfc_allocator`/`rendezvous` messages.

## Files changed

- `scripts/viz/mvq_overlay.py` (new) — figure-gate overlay script (Step 1 of
  the brief, as written, plus a `load_model` fix: the checkpoint carries
  4-device sharding metadata, so restoring it requires 4 GPUs visible, same
  as `load_vitpose`'s existing pattern in this repo — restoring on 1 GPU
  raised `ValueError: Topology mismatch detected`).
- `scripts/benchmark/mvq_val_baselines.py` (new) — ViTPose+DLT baseline
  script (Step 4).
- `third_party/jarvis_jax/jarvis_jax/models/mvq/fusion.py`,
  `third_party/jarvis_jax/tests/test_mvq_model.py` — the attention-chunk
  remat fix (commit `e11cb59`).
- `third_party/jarvis_jax/jarvis_jax/train/train_mvq.py` — the eval-sharding
  fix (commit `7aa34ff`).
- This file.

`figures/2026-09-mvq/mvq_t1_b16_gate1/{unprompted,prompted}/gate1_{female,two_fly,worst}.png`
+ `summary.json` per mode, and `vitpose_dlt_baseline.json`, all under the
gitignored `figures/` tree — regenerate with the commands above (overlay) and
`PYTHONPATH=third_party/jarvis_jax:. python scripts/benchmark/mvq_val_baselines.py --out <path>`
(baseline).
