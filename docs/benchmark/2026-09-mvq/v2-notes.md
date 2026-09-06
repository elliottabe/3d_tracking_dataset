# mvq v2 -- benchmark notes

## Task 6: existence/visibility temperature calibration (2026-09-06)

Spec: `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` §5 "existence
calibration" / §4 "Calibration after training". Implements
`scripts/benchmark/mvq_calibrate.py` (`fit_temperature`, `reliability`) and
`MVQRunner.infer` applying the stored temperatures
(`third_party/jarvis_jax/jarvis_jax/tracking/lift_mvq.py`) so
`exist_thresh=0.5` means calibrated 0.5 for every downstream reader
(`slot_read`, `read_typed`, `policy_slot`/`pick_typed_pair`, `pick_mask_pair`,
`coarse_track`).

### What the script does

One UNPROMPTED forward pass over the v12 val split (153 framesets,
`V12WindowDataset(root, "val", T=1, train=False)`, the same forward
`train_mvq.evaluate` uses), collecting `exist_logit` against the
label-driven `assign_slots`/`slot_ignore` targets and `vis_logit` (gathered
per labelled fly via `losses_mvq._gather_inst`, the SAME gather `mvq_loss`
uses) against `vis2d`, masked by `cam_valid & fly_valid & (assign >= 0)`.
`fit_temperature` finds one scalar T per head by bisection on d(NLL)/dT
(1e-4 tolerance, clamped to [0.25, 10]); `reliability` bins into 10
equal-width [0,1] bins and reports `max_gap` (over populated bins only),
`max_gap_min_n` (over bins with `n >= min_n`, default 20 -- `None` if none
qualify) and `ece` (bin-size-weighted mean gap, over populated bins). Fix
round 1 added `max_gap_min_n`: a single near-empty val bin (n=1) can pin the
raw `max_gap` anywhere in [0,1] by chance (see the existence table below),
so `scripts/benchmark/mvq_v2_acceptance.py::check_calibration` now gates
PASS/FAIL on `max_gap_min_n`, not the raw number (which is still reported,
for reference, but never fails a row on its own).

### Step 4: proof run on the P3b checkpoint

No v2 checkpoint exists yet (Task 6 lands before the v2 training run,
`docs/deliverables` item 7). Per the controller's instruction, the script
was proven end-to-end against the existing production courtship checkpoint,
`mvq_t1_b16_p3b_contact_20260905/final`, on this node
(g3102, `CUDA_VISIBLE_DEVICES=5`, `module load cuda/12.9.1` +
`LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6`, `unset LD_LIBRARY_PATH
JAX_PLATFORMS`). The calibration block was written to a STAGING copy,
`OutFiles/v2_calib_check/p3b/mvq_run.json`, never to the production run's
own `final/mvq_run.json` (which the live p3b campaign reads).

Command:
```
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export PYTHONPATH=third_party/jarvis_jax:. HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3
CUDA_VISIBLE_DEVICES=5 python scripts/benchmark/mvq_calibrate.py \
  --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3b_contact_20260905/final \
  --out figures/2026-09-mvq/v2_train \
  --figure-name calibration_p3b.png \
  --stage-out OutFiles/v2_calib_check/p3b/mvq_run.json \
  --batch 16
```

n_val = 153 (every val fresh fresh window; `window_batches(drop_last=False)`
neither drops nor pads).

| head | T | before max_gap | before max_gap_min_n | before ECE | after max_gap | after max_gap_min_n | after ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| existence | 0.2655 | 0.4368 | 0.0026 | 0.0101 | 0.2775 | 0.0036 | 0.0027 |
| per-view visibility | 0.7172 | 0.2698 | 0.2698 | 0.0140 | 0.2693 | 0.2693 | 0.0076 |

Figure: `figures/2026-09-mvq/v2_train/calibration_p3b.png` (JSON alongside:
`calibration_p3b.json`).

### Per-bin table (re-derived from `OutFiles/v2_calib_check/p3b/mvq_run.json`)

**Existence** (T=0.2655; `nan`/`-` = empty bin, n=0):

| bin | conf range | n before | conf before | acc before | gap before | n after | conf after | acc after | gap after |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | [0.0,0.1) | 397 | 0.0009 | 0.0000 | -0.0009 | 408 | 0.0006 | 0.0000 | -0.0006 |
| 1 | [0.1,0.2) | 4 | 0.1688 | 0.0000 | **-0.1688** | 0 | - | - | - |
| 2 | [0.2,0.3) | 4 | 0.2457 | 0.0000 | **-0.2457** | 0 | - | - | - |
| 3 | [0.3,0.4) | 3 | 0.3276 | 0.0000 | **-0.3276** | 0 | - | - | - |
| 4 | [0.4,0.5) | 0 | - | - | - | 0 | - | - | - |
| 5 | [0.5,0.6) | 1 | 0.5632 | 1.0000 | +0.4368 | 0 | - | - | - |
| 6 | [0.6,0.7) | 7 | 0.6567 | 0.8571 | +0.2005 | 0 | - | - | - |
| 7 | [0.7,0.8) | 2 | 0.7356 | 1.0000 | +0.2644 | 1 | 0.7225 | 1.0000 | +0.2775 |
| 8 | [0.8,0.9) | 2 | 0.8367 | 1.0000 | +0.1633 | 3 | 0.8736 | 1.0000 | +0.1264 |
| 9 | [0.9,1.0] | 192 | 0.9974 | 1.0000 | +0.0026 | 200 | 0.9986 | 0.9950 | -0.0036 |

max_gap (raw) = bin 5 before (n=1), bin 7 after (n=1) -- both near-empty.
max_gap_min_n (bins with n>=20: bin 0 and bin 9 only, both before and
after) = 0.0026 before, 0.0036 after -- both tiny.

**Per-view visibility** (T=0.7172; every bin has n well over min_n=20):

| bin | conf range | n before | conf before | acc before | gap before | n after | conf after | acc after | gap after |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | [0.0,0.1) | 8374 | 0.0134 | 0.0099 | -0.0035 | 8934 | 0.0081 | 0.0162 | +0.0081 |
| 1 | [0.1,0.2) | 728 | 0.1464 | 0.1250 | -0.0214 | 556 | 0.1463 | 0.2212 | +0.0749 |
| 2 | [0.2,0.3) | 528 | 0.2472 | 0.2670 | +0.0198 | 359 | 0.2474 | 0.3844 | +0.1370 |
| 3 | [0.3,0.4) | 401 | 0.3488 | 0.4663 | +0.1175 | 269 | 0.3469 | 0.5688 | +0.2218 |
| 4 | [0.4,0.5) | 307 | 0.4487 | 0.6840 | +0.2353 | 220 | 0.4484 | 0.6955 | +0.2471 |
| 5 | [0.5,0.6) | 316 | 0.5530 | 0.8228 | +0.2698 | 219 | 0.5526 | 0.8219 | +0.2693 |
| 6 | [0.6,0.7) | 434 | 0.6524 | 0.8387 | +0.1863 | 291 | 0.6537 | 0.8316 | +0.1780 |
| 7 | [0.7,0.8) | 583 | 0.7531 | 0.8645 | +0.1114 | 400 | 0.7539 | 0.8525 | +0.0986 |
| 8 | [0.8,0.9) | 1179 | 0.8562 | 0.9262 | +0.0700 | 692 | 0.8584 | 0.8786 | +0.0202 |
| 9 | [0.9,1.0] | 58100 | 0.9909 | 0.9996 | +0.0087 | 59010 | 0.9969 | 0.9985 | +0.0017 |

max_gap = max_gap_min_n here (every bin qualifies) -- bin 5 both before and
after, essentially unchanged (0.2698 -> 0.2693).

### Figure read back (CLAUDE.md), rewritten from the tables above

Existence panel (T=0.2655, a SHARPENING temperature: T<1 divides logits by
a fraction, inflating their magnitude): the pre-scaling (red) curve is
**miscalibrated in BOTH directions at once, at different confidence
levels**, not uniformly under- or over-confident. Bins 1-3 (conf 0.17,
0.25, 0.33 -- n=4, 4, 3) sit **BELOW** the diagonal: `acc=0` every time even
though the head reports 17-33%, i.e. the head is over-confident about
existence being possible on frames that are always actually absent. Only
bins 5-8 (conf 0.56-0.84 -- n=1, 7, 2, 2) sit **ABOVE** the diagonal:
`acc` is 0.86-1.00 while `conf` is only 0.56-0.84, i.e. under-confident on
frames that are actually present -- the direction the earlier "~0.45
sigmoid on present flies" note describes, but that note is about THIS
region only, not the whole curve. Bin 9 (conf=0.997, n=192, the dominant
bin) is already almost exactly on the diagonal (gap=+0.0026) before any
scaling at all. After scaling (blue), nearly all mass moves to bin 0 (still
on the diagonal, correctly-denied) and bin 9 (now gap=-0.0036, still
essentially on the diagonal) -- the temperature sharpened the WELL-POPULATED
region into two tight, well-calibrated clusters. Only bins 7-8 remain
populated in between, with n=1 and n=3 -- exactly the near-empty bins that
set the raw `max_gap` (0.2775) both before and after scaling; `max_gap_min_n`
(bin 0 + bin 9 only) is 0.0026 before and 0.0036 after -- both far inside
the 0.05 band, showing the WELL-POPULATED part of this head was already
close to calibrated and stayed that way; the raw `max_gap` was never
measuring that.

Visibility panel (T=0.7172, also sharpening): the raw curve is close to the
diagonal at the low end (bins 0-1, gap magnitude < 0.03) and increasingly
**above** the diagonal (under-confident) from bin 2 through bin 8, peaking
at bin 5 (conf=0.55, acc=0.82, gap=+0.27 -- the raw `max_gap`). After
scaling, bins 0-4 (conf < 0.5) get **measurably WORSE** -- the sharpening
temperature was fit mainly to correct the LARGER gaps in bins 5-9 and, in
doing so, over-corrects the smaller low-conf gaps past the diagonal in the
other direction (bin 0: -0.0035 -> +0.0081; bin 1: -0.0214 -> +0.0749; bin
2: +0.0198 -> +0.1370; bin 3: +0.1175 -> +0.2218; bin 4: +0.2353 -> +0.2471,
all bigger in magnitude after scaling). Bins 6-9 (conf >= 0.6) get
**measurably BETTER** (bin 6: 0.1863 -> 0.1780; bin 7: 0.1114 -> 0.0986;
bin 8: 0.0700 -> 0.0202; bin 9: 0.0087 -> 0.0017). Bin 5 itself -- the bin
that sets `max_gap` both before (0.2698) and after (0.2693) -- barely moves,
which is why the headline `max_gap`/`max_gap_min_n` number looks unchanged
even though ECE improves ~2x (0.0140 -> 0.0076): a single global temperature
cannot simultaneously fix a curve that needs LESS sharpening at low
confidence and MORE at bin 5, so it settles on a compromise that helps the
bin-size-weighted average without moving the single worst bin.

### Conclusion for Task 6 / handoff to Task 7

The script and `MVQRunner` integration work end-to-end: `n_val=153` (matches
the known val split size), and the `"calibration"` block round-trips through
`mvq_run.json` in the exact shape
`scripts/benchmark/mvq_v2_acceptance.py::check_calibration` reads
(`calibration.reliability_exist.max_gap_min_n` is now the gated metric;
`.max_gap` is carried for reference only, per fix round 1). **Existence,
gated on `max_gap_min_n`, already clears the spec's 0.05 line on this P3b
proof run** (0.0026 before, 0.0036 after) -- the raw `max_gap` (0.4368/
0.2775) looked far worse only because it was reading a 1-3-sample bin, which
`max_gap_min_n` now excludes by construction. **Per-view visibility does
NOT clear 0.05** either before (0.2698) or after (0.2693) scaling -- a real,
broad miscalibration (not a sparsity artifact: every bin has hundreds to
tens of thousands of samples) that one global temperature per head cannot
fully correct, because the direction and magnitude of the raw gap differs
across the confidence range (see the per-bin table). This is expected and
non-blocking for THIS run specifically -- P3b was never trained with
calibration in mind and this run's whole purpose was to prove the script
works, not to certify P3b -- but it is a real finding for whoever calibrates
the v2 FINAL checkpoint: if v2's visibility head shows the same
bin-5-dominated pattern, a single scalar temperature may not be enough and
the spec may need a revisit (e.g. per-confidence-region or per-camera
temperatures) rather than another calibration re-run with the same
one-parameter fit. The real acceptance gate is the v2 FINAL checkpoint
(deliverable 7 in the design spec), not yet trained; when it exists, re-run
this exact command against `<v2 run>/final` with `--stage-out` OMITTED (the
default in-place write, now the correct target because it IS the production
run) and read `max_gap_min_n` there. Because the P3b run's own
`final/mvq_run.json` was never touched, `check_calibration` against it still
reports SKIP ("no 'calibration' block yet"), unchanged from before this
task.

## Task 5: v2 training config and run_training wiring (2026-09-06)

Spec: `docs/specs/2026-09-05-mvq-v2-pseudolabel-t2-design.md` §2 decisions, §4
training, §3.2/§3.4/§3.5 data, §7 "validation on human labels only".
Files: `third_party/jarvis_jax/configs/train/mvq_v2.yaml` (new),
`third_party/jarvis_jax/jarvis_jax/train/train_mvq.py`,
`third_party/jarvis_jax/jarvis_jax/scripts/train_mvq.py`,
`third_party/jarvis_jax/tests/test_train_mvq_smoke.py`.

### What `mvq_v2.yaml` changes vs `mvq.yaml`

| key | mvq.yaml | mvq_v2.yaml | why (spec) |
|---|---|---|---|
| `window_lengths` | `[1]` | `[1, 2]` | §2 decision 3: T=1 and T=2 batches alternate |
| `pair_deltas` | — (new key) | `[1, 4, 16]` | §2 decision 3: 1.25-20 ms at 800 fps |
| `warmup_steps` / `total_steps` | 500 / 30000 | 1000 / 40000 | §4, from scratch |
| `save_every` | 1000 | 2000 | §4 |
| `prompt_p_start` / `_end` | 1.0 / 0.5 | 0.0 / 0.0 | §4 "prompting off", from step 0 |
| `jitter_units` | 3.0 | 10.0 | §4 / P4 §6: 1 mm mask-free centre budget |
| `copy_paste_p` / `contact_p` / `contact_sep` | 0.0 / 0.3 / [8,30] | 0.8 / 0.7 / [4,25] | §4 contact-heavy |
| `female_host_weight` | 1.0 | 4.27 | §4 host-sex ratio 0.5 on the real root |
| `val_cohorts` | female, two_fly | + contact_pair, single_fly | §5 grades those cohorts |
| `num_workers` | 16 | 24 | T=2 doubles the JPEG decode per batch |
| `pseudo_root` / `pseudo_weight` | — | pseudo export / 0.3 | §3.2 |
| `singlefly_root` | — | single-fly export | §3.4 |
| `negatives_root` / `negatives_frac` | — | negatives export / 0.05 | §3.5 |
| `wing_kp_mult` | — | 2.0 | §4 wing keypoint AND visibility x2 |
| `loss.other_fly_repulsion` | 0.0 | 20.0 | §4 = the weight P3b converged with |
| `loss.persist` / `persist_margin_units` | — | 0.5 / 2.0 | §4 identity persistence (T=2 only) |
| `mv_aug.cam_drop_p` | 0.3 | 0.1 | §4 "each camera absent with p 0.1" |

`warm_start: null` and `batch_size: 32` are unchanged from `mvq.yaml` but are
load-bearing for v2 (§2 decision 1 from scratch; 4/device on 8 GPUs -- the
7-device layout hung P3b). The three exports live directly under
`/gscratch/portia/eabe/data/Johnson_lab/` (NOT under `red_data/`).

### How the mix is built, and where to read it back

`run_training` builds one `ConcatWindowDataset([real, pseudo, singlefly,
negatives])` **per T**, for the TRAIN split only; `val_ds` stays the human root
alone (§7). `_mix_weights` runs `_balanced_weights` SEPARATELY on each root --
so each root's behaviour balance and host-sex ratio are restored inside itself
-- and then sets the roots' total masses so the negatives take exactly
`negatives_frac` and the rest splits by window count. The pseudo export's
manifest `balance.female_host_weight` (n_male/n_female, ~2.1) is therefore
**information, logged next to the realised ratio, never applied a second
time**: the per-root balance already restores 0.5, and re-applying it would
over-sample female hosts by that factor.

Three log families make it verifiable (all `flush=True`, all in the run log):

1. the per-T mass table plus each root's `source`/`role`/mean `sample_weight`,
   printed before step 0;
2. `realised host-sex ratio ...` over the first 200 batches (or the whole run
   if shorter -- lowered from a hard 200 so a 20-step smoke run prints it too),
   reported both over everything drawn and over the non-negative windows alone
   (a negative has no host sex);
3. `T=<T> realised mix over N windows drawn: real=... pseudo=... singlefly=...
   negatives=...`, **counted** at `__getitem__` by a `_MixCounter` wrapper --
   not inferred from `sample_weight` (two roots can carry the same value) and
   not reproduced from the sampler's own RNG (which would drift silently the
   moment `window_batches` changed how it draws).

From the CPU fixture run (four tiny roots, batch 2, 20 steps):

```
[mvq] T=1 sampler mix: real 7 windows -> mass 0.3325  pseudo 7 windows -> mass 0.3325  singlefly 6 windows -> mass 0.2850  negatives 6 windows -> mass 0.0500
[mvq]   real: source='real' role='anchor' mean sample_weight=1.0000
[mvq]   pseudo: source='pseudo' role='anchor' mean sample_weight=0.3000
[mvq]   singlefly: source='pseudo' role='anchor' mean sample_weight=0.3000
[mvq]   negatives: source='pseudo' role='negative' mean sample_weight=1.0000
[mvq] realised host-sex ratio over the first 20 batches: female 37/40 = 0.925 (of the 38 non-negative windows: 0.974); negatives 2/40 = 0.050
[mvq] T=1 realised mix over 26 windows drawn: real=0.346 pseudo=0.308 singlefly=0.231 negatives=0.115
[mvq] T=2 realised mix over 25 windows drawn: real=0.320 pseudo=0.360 singlefly=0.280 negatives=0.040
```

The realised fractions track the masses to within the sampling noise of ~25
draws (the fixture's own host-sex ratio is extreme because its synthetic
recording is nearly all female-host; the real root's 4.27 is verified from the
launch log's ratio line, not from this fixture).

`mean sample_weight` is read from each ROOT's manifest, and the trainer prints
a `<-- WARNING: expected 0.3` marker if a pseudo/single-fly root's own weight
disagrees with `train.pseudo_weight` -- a mislabelled export would otherwise
train at a weight nobody configured. Negatives keep weight 1.0 on purpose:
`sample_weight` multiplies every term, so a down-weighted negative would
assert nothing to the existence head.

Two fail-fast guards came with it: a root contributing **0 windows at some T**
raises (naming the root and pointing at `pair_deltas`) instead of silently
dropping out of the mix, and `assert_lr_swap_covers(names, required=names)`
runs before the model build -- verified to pass on the real
`red_data_3d_v12_export0902/annotations/keypoint_names.json` (all 50 names
paired), so the launch will not trip on it at step 0.

### Wing weights

`kp_weight = wing_kp_weight(names, train.wing_kp_mult)` is built ONCE from the
model's keypoint NAMES and threaded into both `make_train_step` and
`evaluate`. On the real 50-name order it selects exactly
`['WingL_base', 'WingL_V12', 'WingL_V13', 'WingR_base', 'WingR_V12',
'WingR_V13']` -- printed at startup so the six landmarks are named in the log
rather than left as indices (CLAUDE.md's keypoint-order history). It
multiplies the reproj, l3d, uv2d **and the visibility BCE** terms, which is
what §4 asks for ("wing keypoint loss weight x2 and wing visibility weight
x2"). Passing it to `evaluate` moves no reported eval number (the px metrics
and every per-sample statistic are unweighted L2); it is threaded through so
eval scores the same objective training optimises. Setting the multiplier in
both `train.wing_kp_mult` and `train.loss.wing_kp_mult` to different values
now raises instead of silently picking one.

### Loss-share launch check (§4 `other_fly_repulsion`)

Spec §4 sets `other_fly_repulsion: 20` because that is what P3b converged with
-- but P3b was a WARM-STARTED T=1 run without negatives, so v2's loss balance
is a different question. `+share_check_steps=N` runs N steps of the REAL
pipeline (same concat mix, sampler, augmentation and T alternation), prints
every term's weighted share of `total`, and exits without eval, checkpoint or
run-dir json. Reusing `run_training` is the point of the check: a standalone
harness would measure a different balance than the one the launch optimises.

Proof on the CPU fixture (20 steps, tiny model, batch 2 -- this proves the
MECHANISM, not the launch decision):

```
[mvq] loss shares over 20 steps (mean total 1407.3488):
[mvq]   reproj            w=1      mean=527.5      weighted=527.5      share= 37.48%
[mvq]   deep_supervision  w=0.5    mean=nan        weighted=423.6      share= 30.10%
[mvq]   uv2d              w=0.5    mean=550.4      weighted=275.2      share= 19.55%
[mvq]   other_rep         w=20     mean=4.933      weighted=98.66      share=  7.01%
[mvq]   conf              w=1      mean=44.04      weighted=44.04      share=  3.13%
[mvq]   l3d               w=0.5    mean=73.07      weighted=36.53      share=  2.60%
[mvq]   exist             w=1      mean=1.275      weighted=1.275      share=  0.09%
[mvq]   sex               w=0.5    mean=0.9005     weighted=0.4503     share=  0.03%
[mvq]   rep               w=0.5    mean=0.1364     weighted=0.06818    share=  0.00%
[mvq]   vis               w=0.1    mean=0.5071     weighted=0.05071    share=  0.00%
[mvq]   persist           w=0.5    mean=0          weighted=0          share=  0.00%
[mvq] other_fly_repulsion (weight 20) share = 7.01% -- in the 5-30 % launch band
```

`deep_supervision` is the pass1 + aux-layer remainder (`mvq_loss` folds those
into `total` without itemising them), so the shares sum to 1 and no part of
`total` is left unexplained. `persist = 0` here is EXPECTED, not a wiring
failure: the hinge is `relu(d_pred - d_gt - margin)`, and a random-init model
predicts almost no inter-frame centroid motion, so it sits below the GT
displacement plus the 0.2 mm margin. It becomes informative only once the
model predicts real motion.

**The real check must run on the GPU against the real roots** -- the fixture's
absolute term values are synthetic, so its 7.01 % is not the launch number.
Exact command:

```bash
cd third_party/jarvis_jax
module load cuda/12.9.1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN=
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -u -m jarvis_jax.scripts.train_mvq model=mvq train=mvq_v2 paths=hyak \
  run_id=mvq_t2_v2_share_check "paths.runs_root=\${paths.mvq_runs_root}" \
  +share_check_steps=200 2>&1 | tee slurm_logs/mvq_t2_v2_share_check.out
```

(Keep all 8 devices exposed: `batch_size: 32` must divide `len(jax.devices())`,
and `run_training` refuses otherwise before any dataset load.)

Decision rule: share > 50 % -> halve to 10 and re-measure; share < 2 % -> raise
to 40 and re-measure; 5-30 % -> launch as configured. The printed verdict line
states which applies. **Record the measured share here before launching.**

### Launch command (Step 7, deferred to the controller)

Gated on the user's pseudo-label review AND on the three exports existing (when
this task ran the `p3b` root was still a stale provisional export, the
negatives root was being written, and the single-fly root did not exist yet --
`mvq_v2.yaml` names its final path). 8 idle L40S, ~1 day at ~2.2 s/step for
T=2 at batch 32:

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

Record: the PID of the `python` process (not the `setsid` wrapper), the log
path, the run dir, the mass-table + realised-mix + host-sex lines, s/step and
GB/GPU at steady state, and the step-2000 val row.

**OOM fallback** (§7 risk: T=2 doubles per-step memory): resubmit at
`train.batch_size=16` with gradient accumulation 2 and record that instead.
Caveat for whoever does it: gradient accumulation is NOT implemented in
`train_mvq.py` today, so `train.batch_size=16` alone HALVES the effective
batch -- either add accumulation first or record the reduced effective batch
explicitly. Do not work around an OOM by exposing 7 GPUs: that layout hung
P3b, and `run_training` now refuses `batch_size % len(jax.devices()) != 0`
before any dataset load.

### Verification

- `JAX_PLATFORMS=cpu pytest tests/test_train_mvq_smoke.py tests/test_configs.py -q`
  from `third_party/jarvis_jax`, run in three chunks against the current
  loader fix round: **4 passed** (the new T=2/device/share tests, 99 s),
  **15 passed** (the rest of `test_train_mvq_smoke.py`, 346 s), **18 passed /
  1 failed** (`test_configs.py`, 32 s) = 37 passed, 1 failed. The one failure
  is `test_configs.py::test_sam3_main_from_cfg_maps_config`
  (`FileNotFoundError: No bout source for dataset '' under /s/rec`, raised in
  `jarvis_jax/predict/bouts_resolve.py`) -- pre-existing and unrelated:
  `sam3=default` leaves `sam3.dataset: null`, which `main_from_cfg` tries to
  derive from the test's fake `session_dir=/s/rec`; none of the five files
  this task touched is on that path, and neither `test_configs.py`,
  `scripts/sam3_masks.py` nor `bouts_resolve.py` is modified in the working
  tree.
- Four new smoke tests, all four failing BEFORE the change with
  `TypeError: MVQTrainConfig.__init__() got an unexpected keyword argument
  'pair_deltas'`: `test_t2_run_trains_and_mixes_roots` (four roots, T=(1,2),
  Delta=(1,4), mass table, realised mix, the `mvq_run.json` train block),
  `test_t2_val_split_stays_the_real_root_only` (§7: the extra roots reach
  TRAIN only, and every train root is built at the same T and spacings),
  `test_batch_size_must_divide_the_device_count` (32 on a monkeypatched
  7-device layout), `test_loss_share_check_reports_every_term`.
- 20-step Hydra CPU run with `train=mvq_v2`, every root pointed at a tiny
  fixture: composes, trains, evaluates both modes, exits 0, and every new key
  reaches `MVQTrainConfig` (read back from `final/mvq_run.json`:
  `window_lengths=[1,2] pair_deltas=[1,4,16] pseudo_weight=0.3
  negatives_frac=0.05 wing_kp_mult=2.0 prompt_p_start=0.0 jitter_units=10.0
  female_host_weight=4.27 copy_paste_contact_sep=[4.0,25.0] warm_start=None`).
