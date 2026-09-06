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

### Fix round 1 (2026-09-06)

Four rulings from review, plus two defects the CPU smoke log exposed.

**1. `mvq_run.json` now records the OBJECTIVE, not just the schedule.** Both writes (the
run-dir copy before step 0 and `final/mvq_run.json` at the end) carry
`"loss": dataclasses.asdict(weights)` and `"aug": dataclasses.asdict(aug)` alongside
`"model"`/`"train"`. Before this, a v2 run's file could not say what `persist`,
`persist_margin_units` or `other_fly_repulsion` were — the three weights the whole round is
about — so no scorecard header or A/B could reconstruct the objective. Asserted in the T=2
smoke test (`meta["loss"]["persist"] == 0.5`, `meta["aug"]["cam_drop_p"]` present, and the
same block on disk before step 0).

**2. The female-host multiplier is SOLVED per root, not hand-set.** `female_host_weight:
4.27` was solved for one census (`red_data_3d_v12_export0902` train: 202 female-host windows
of 2661). v2 balances four roots separately, each with its own census, so the one number
STACKS and the aggregate lands wherever the roots average out to — measured ~0.75 female
across these four, not the 0.5 spec §4 asks for. New knob `train.female_host_target`
(mvq_v2.yaml: `0.5`): `_balanced_weights` computes each root's own post-balance female mass
`f` and applies `target*(1-f) / ((1-target)*f)`, which lands that root's female-host mass
exactly on the target. It prints the multiplier it solved, per root:

```
[mvq] T=1 root 'real': female_host_target=0.500, post-balance female mass 0.7101 -> solved female-host multiplier 0.4082 (train.female_host_weight=4.27 is unused while a target is set)
[mvq] T=1 root 'singlefly': female_host_target=0.500 UNATTAINABLE -- this root has no male/other-host window (post-balance female mass 1.0000); multiplier left at 1.0 and the root's sampling weights are unchanged
[mvq] T=1 root 'negatives': female_host_target=0.500 UNATTAINABLE -- this root has no female-host window (post-balance female mass 0.0000); multiplier left at 1.0 and the root's sampling weights are unchanged
```

`female_host_weight` stays as the fallback when no target is set, so `mvq.yaml`/P3b runs are
bit-identical (pinned by a test). `mvq_v2.yaml` keeps it at 4.27 but documents it as unused.

**Reading the ratio back — updated guidance.** The aggregate line
(`T=<T> sampler: ... weight mass female ...`) equals the target only when EVERY root could
reach it. A one-sided root is left unchanged by design and pulls the aggregate off the target
by its own mass: the fixture's single-fly root is all-female and the negatives root has no
host sex at all, so its aggregate is 0.6175 at T=1, not 0.5, and that is correct. **Read the
per-root lines first**, then the aggregate; and read the realised ratio over the NON-NEGATIVE
windows (the trainer prints both). On the real launch, expect the aggregate to sit near 0.5
only if the single-fly export contains both sexes — check its per-root line and record it.

**Two defects the CPU smoke log exposed (neither had a failing test before):**

- *One-sided-root guard was float-blind.* An all-female root's post-balance female mass comes
  back as `0.9999999999999998`, not `1.0` (measured on the T=2 fixture root, where only fly0
  has labelled pairs). The first guard was `f >= 1.0`, so it sailed past that and "solved" a
  multiplier of ~4e-16, printing the plausible-looking `-> solved female-host multiplier
  0.0000`. On a root that is merely NEARLY one-sided that would have zeroed the female mass
  outright. Now guarded on the window COUNT (exact) with a toleranced mass backstop, and
  pinned by `test_female_host_target_on_a_root_that_is_all_female_by_a_float_hair`. Found only
  by reading the run log of the 20-step CPU run — the four-root smoke test passed either way.
- *The aggregate line named the wrong knob*, printing `female_host_weight=4.27` while a target
  was in force. It now names whichever knob is actually active.

**3. Negatives train at `sample_weight = 1.0`.** The negatives export ships the pseudo-label
weight (0.3) like the rest of the campaign data, but `sample_weight` multiplies EVERY loss
term per sample — including the existence BCE, which is the only thing an empty window
carries — so training a negative at 0.3 down-weights its single purpose. `run_training` now
wraps the negatives root in `_ForceSampleWeight(d, 1.0)` and announces it:

```
[mvq] T=1 negatives root <path>: export sample_weight 0.3 OVERRIDDEN to 1.0 -- sample_weight
multiplies every loss term, and the existence target is the only thing an empty window
carries (spec §3.5); how OFTEN a negative is drawn is set by negatives_frac=0.05, not by its
loss weight
```

How often a negative is drawn is unchanged — that is `negatives_frac`'s job. The old
"expected 1" WARNING path is gone (the weight is forced, so there is nothing to warn about);
the pseudo/single-fly roots keep their `expected train.pseudo_weight` warning. The test
fixture now writes manifest weight 0.3 to match the real export, so the override is what the
test exercises.

**4. Smaller items.** A `test_configs.py` case composes `train=mvq_v2` and runs the same
Hydra→dataclass conversion the entrypoint does (`pair_deltas == (1,4,16)`,
`female_host_target 0.5`, `negatives_frac 0.05`, `warm_start None`, both export paths
absolute and outside `red_data/`). `scripts/train_mvq.py`'s duplicated `run_training` call is
collapsed to one. `LossWeights.wing_kp_mult` is REMOVED (dead: `mvq_loss` takes the built
`(K,)` vector, so the multiplier lives only at `train.wing_kp_mult`) along with the
now-pointless raise that compared the two. `negatives_frac` is validated to `[0, 1)`.
`share_sums` accumulates only the metrics `_loss_shares` reads (`_SHARE_KEYS`), so the share
check no longer forces a host sync on per-batch diagnostics nothing reports.

**5. Gradient accumulation does NOT exist** in `train_mvq.py`. The documented OOM fallback
`train.batch_size=16` therefore HALVES the effective batch rather than preserving it — no
accumulation step is applied. Ruling: acceptable and recorded. If v2 OOMs, either accept the
halved effective batch (and say so in the launch record) or implement accumulation first;
do not silently treat `batch_size=16` as equivalent to 32.

**Verification (foreground, `JAX_PLATFORMS=cpu`, from `third_party/jarvis_jax`).**
`pytest tests/test_train_mvq_smoke.py tests/test_configs.py tests/test_mvq_losses.py -q`
→ **64 passed, 1 failed in 454 s**; the failure is the same pre-existing, unrelated
`test_configs.py::test_sam3_main_from_cfg_maps_config`. Three new tests
(`test_female_host_target_solves_the_multiplier_per_root` — exact 0.5 mass on a 1:3 census
plus a 20k-draw realised check within 0.02, both one-sided cases, and the no-target path
pinned bit-identical; `test_female_host_target_on_a_root_that_is_all_female_by_a_float_hair`;
`test_negatives_train_at_sample_weight_one`) and one new config test. The 20-step Hydra CPU
run was re-run after the fixes and exits 0 with the corrected log lines quoted above.

### Share check (2026-09-06, 6 GPUs)

Plan B Task 5 Step 6, run on the REAL data roots (not the CPU fixture), against
`f395ec2` (`allow_calib_mismatch: true` in `mvq_v2.yaml` + the concat's
mismatch/disagreement recording). Node g3102, GPUs 0,1,2,3,6,7 (6 devices;
4,5 held by other agents), `train.batch_size=24` (4/device):

```bash
cd third_party/jarvis_jax
module load cuda/12.9.1
export LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6
unset LD_LIBRARY_PATH JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 TF_GPU_ALLOCATOR=cuda_malloc_async
export HF_HOME=/gscratch/portia/eabe/data/Johnson_lab/sam3 HF_TOKEN=
export CUDA_VISIBLE_DEVICES=0,1,2,3,6,7
python -u -m jarvis_jax.scripts.train_mvq model=mvq train=mvq_v2 paths=hyak \
  run_id=mvq_t2_v2_sharecheck train.batch_size=24 \
  "paths.runs_root=\${paths.mvq_runs_root}" +share_check_steps=200 \
  2>&1 | tee ../../slurm_logs/mvq_t2_v2_sharecheck_6gpu.out
```

Exit code 0. Full log: `slurm_logs/mvq_t2_v2_sharecheck_6gpu.out` (63 lines).

**1. Composes and loaders build.** Window counts per root, T=1 and T=2:

```
[mvq] T=1 sampler mix: real 2661 windows -> mass 0.0861  pseudo 24805 windows -> mass 0.8022  singlefly 1909 windows -> mass 0.0617  negatives 7302 windows -> mass 0.0500
[mvq] T=2 sampler mix: real 2543 windows -> mass 0.0594  pseudo 35099 windows -> mass 0.8192  singlefly 3059 windows -> mass 0.0714  negatives 5302 windows -> mass 0.0500
```

Calibration mismatch/disagreement lines (`allow_calib_mismatch: true` let the run
proceed on both, per root, each printed once for T=1 and once for T=2):

```
WARNING: recording '2026_04_02_15_25_51': manifest field 'calib_group' is 'A' in root real (.../red_data_3d_v12_export0902) and '2026_04_02_15_25_51' in root pseudo (.../red_data_3d_v12_pseudo_p3b_20260905). The calibration CONTENT differs by max |diff| 15.2278 (not a serialisation rounding difference) -- which one is right is undetermined here, so this refuses to silently pick one. Pass allow_calib_mismatch=True to proceed with each root using its OWN calibration per sample.
WARNING: recording '2026_04_02_17_28_34': manifest field 'calib_group' is 'A' in root real (.../red_data_3d_v12_export0902) and '2026_04_02_17_28_34' in root pseudo (.../red_data_3d_v12_pseudo_p3b_20260905). The calibration CONTENT differs by max |diff| 15.2278 (not a serialisation rounding difference) -- which one is right is undetermined here, so this refuses to silently pick one. Pass allow_calib_mismatch=True to proceed with each root using its OWN calibration per sample.
```

`2025_10_20_13_20_04` (the identical-content alias between real's calib-group letter
and pseudo's recording-id naming) printed no mismatch warning, matching the yaml
comment (max |diff| 1e-15, a serialisation difference — silently accepted either
way). Only these two recordings (`2026_04_02_15_25_51`, `2026_04_02_17_28_34`)
disagree in content and were let through under the override; that disagreement is
still under investigation per "Calibration witness test" and is unchanged by this
run. Other non-fatal warnings seen: the negatives-root sample-weight override
(`export sample_weight 1 OVERRIDDEN to 1.0`, T=1 and T=2) and a `[v12_windows]`
annotation-vs-manifest sex disagreement on `2025_10_20_13_20_04 fly0`
(`{'female': 15, 'male': 677}`, annotation wins per window, printed twice — once
per window_length pass). No other warnings, no load errors.

**2. Realised mix and per-root host-sex.**

```
[mvq] T=1 root 'real': female_host_target=0.500, post-balance female mass 0.1897 -> solved female-host multiplier 4.2705
[mvq] T=1 root 'pseudo': female_host_target=0.500, post-balance female mass 0.5006 -> solved female-host multiplier 0.9975
[mvq] T=1 root 'singlefly': female_host_target=0.500 UNATTAINABLE -- this root has no male/other-host window (post-balance female mass 1.0000); multiplier left at 1.0
[mvq] T=1 root 'negatives': female_host_target=0.500 UNATTAINABLE -- this root has no female-host window (post-balance female mass 0.0000); multiplier left at 1.0
[mvq] T=1 sampler: 14545/36677 female-host windows, female_host_target=0.5 -> weight mass female 0.5059 male/other 0.4941 (F/M 1.024)
[mvq] T=2 root 'real': female_host_target=0.500, post-balance female mass 0.0854 -> solved female-host multiplier 10.7069
[mvq] T=2 root 'pseudo': female_host_target=0.500, post-balance female mass 0.5256 -> solved female-host multiplier 0.9025
[mvq] T=2 root 'singlefly': female_host_target=0.500 UNATTAINABLE -- this root has no male/other-host window (post-balance female mass 1.0000); multiplier left at 1.0
[mvq] T=2 root 'negatives': female_host_target=0.500 UNATTAINABLE -- this root has no female-host window (post-balance female mass 0.0000); multiplier left at 1.0
[mvq] T=2 sampler: 22442/46003 female-host windows, female_host_target=0.5 -> weight mass female 0.5107 male/other 0.4893 (F/M 1.044)
[mvq] realised host-sex ratio over the first 200 batches: female 2385/4800 = 0.497 (of the 4548 non-negative windows: 0.524); negatives 252/4800 = 0.052
[mvq] T=1 realised mix over 2472 windows drawn: real=0.086 pseudo=0.803 singlefly=0.057 negatives=0.053
[mvq] T=2 realised mix over 2424 windows drawn: real=0.057 pseudo=0.817 singlefly=0.076 negatives=0.050
```

Confirms on the REAL exports: the single-fly root is all-female
(`post-balance female mass 1.0000`, UNATTAINABLE) exactly as the negatives root
has no host sex at all (`0.0000`, UNATTAINABLE) — both left at multiplier 1.0 by
design. `real`'s own female-host mass differs sharply between T=1 (0.1897, solved
multiplier 4.2705 — matches the yaml's recorded P3b value) and T=2 (0.0854, solved
multiplier 10.7069 — pairs skew more male-host at this delta set). The aggregate
realised negative fraction reads 0.052-0.053, matching the configured
`negatives_frac=0.05` to within sampling noise; per Task 5's "reading the ratio
back" guidance, the per-root lines are the ones to trust, and every one is close
to or an expected one-sided extreme of its target.

**3. Per-term loss shares at step ~200 (200 steps, real backbone, real data):**

```
[mvq] loss shares over 200 steps (mean total 2618.2531):
[mvq]   deep_supervision  w=0.5    mean=nan        weighted=1443       share= 55.12%
[mvq]   reproj            w=1      mean=604.2      weighted=604.2      share= 23.08%
[mvq]   uv2d              w=0.5    mean=629.2      weighted=314.6      share= 12.02%
[mvq]   other_rep         w=20     mean=10.26      weighted=205.2      share=  7.84%
[mvq]   l3d               w=0.5    mean=92.62      weighted=46.31      share=  1.77%
[mvq]   conf              w=1      mean=3.321      weighted=3.321      share=  0.13%
[mvq]   exist             w=1      mean=0.793      weighted=0.793      share=  0.03%
[mvq]   sex               w=0.5    mean=0.932      weighted=0.466      share=  0.02%
[mvq]   rep               w=0.5    mean=0.1981     weighted=0.09905    share=  0.00%
[mvq]   vis               w=0.1    mean=0.8819     weighted=0.08819    share=  0.00%
[mvq]   persist           w=0.5    mean=0          weighted=0          share=  0.00%
[mvq] other_fly_repulsion (weight 20) share = 7.84% -- in the 5-30 % launch band
```

`other_fly_repulsion` (weight 20) is a clearly visible 7.84 % of total — well above
the P3b (weight 0.5) reading of ~0.6 % / inert, and inside the launch decision
band (5-30 % -> launch as configured; the printed verdict says exactly that).
`persist=0` is the expected fresh-model reading (Task 5: the hinge needs the
model to predict real inter-frame motion first). `deep_supervision`'s `mean=nan`
is the known logging artifact (only the weighted total is accumulated for that
term, not a raw per-step mean; matches the CPU-fixture reading) — every OTHER
term has a finite mean, and the eleven shares sum to 100.00 % (rounding), so no
part of `total` (mean 2618.25) is unexplained by a real NaN.

**4. Throughput and memory.** Steps 50->200 (post-warmup/compile): 632s-316s =
316s over 150 steps = **2.11 s/step** steady state at T=2 (per the per-step log
timestamps: step 50 @316s, 100 @415s, 150 @519s, 200 @632s — 99s, 104s, 113s per
50-step block, i.e. 1.98-2.26 s/step). `nvidia-smi` sampled 3x during the run on
GPUs 0,1,2,3,6,7 (never touched 4,5): all six pinned at **~41.85 GB/GPU**
(41843-41859 MiB of 46068 MiB, ~91 %) and **100 % utilization** throughout the
step-50..200 window; all six released back to 0 MiB immediately on exit.

**5. NaN/inf/crash.** None. `grep -inE "nan|inf|error|traceback|oom"` over the
full log matches only the expected `deep_supervision mean=nan` line (item 3
above) and benign startup INFO lines (TPU backend absent, checkpoint handler
config) — no exception, no OOM, no crash. Process exited 0.

**Verdict:** other_fly_repulsion share 7.84 % is in the 5-30 % launch band ->
launch `mvq_v2.yaml` as configured (no reweighting needed). Concerns: (a) the
two calibration-content disagreements (`2026_04_02_15_25_51`,
`2026_04_02_17_28_34`) are let through by `allow_calib_mismatch: true` but remain
UNRESOLVED as to which root is right — this check does not settle that, only
confirms the loader no longer dies on it; (b) `real`'s female-host multiplier
differs 4x between T=1 (4.27) and T=2 (10.7), i.e. the real root's T=2 pairs are
more male-host-skewed than its T=1 singles — informational, not a blocker, since
`female_host_target` re-solves per T already. The 40k launch itself is still NOT
started (out of scope for this check).

---

## Calibration: ViTPose LOO + affine bundle adjustment (2026-09-06)

**Question.** Two calibrations disagree for the 2026-04-02 (Session1) courtship
recordings: **A** = human root group A
(`red_data_3d_v12_export0902/calibrations/A`, assigned to `15_25_51` and
`17_28_34` by the 2026-09-03 export fix) and **C** = the video-folder
calibration (byte-identical in every Session1 recording dir, and identical to
human-root group C, which the export assigns to `12_11_50`). Every existing
proof is circular: the human 2-D are reprojections of the labelled 3-D through
the labels' own DLT. Control, run here: the human 2-D reproduce **their own**
group and no other — `12_11_50` fit 0.389 px under C vs 3.49 (A) / 1.89 (B);
`15_25_51` and `17_28_34` fit 0.394/0.397 px under A vs ~2.6 under B and C
(0.4 px, not 0.001, because COCO keypoints are integer-rounded). A bundle
adjustment on those 2-D returns the calibration that made them. Hence an
independent observation was needed: the **ViTPose detector's per-view heatmap
peaks**.

**Method.** `scripts/run_bout.py … pipeline.stop_after=triangulate` (Stage A +
B only) on 11 bouts across the three recordings, both flies, all 7 cameras
(`OutFiles/calib_ba/<rec>/cal{A,C}/`), plus a mask-free run on the **exact
human-labelled frames** (`detect_human_frames.py`: crop centred on the human
2-D centroid per view, mask channel zeroed — the checkpoint is
`zero_mask_channel: true` and provably invariant to it — distractor gray-filled
from the other fly's body-landmark hull; detector lands 2.8-3.6 px from the
human labels, conf 0.96). Points enter with conf ≥ 0.5 in ≥ 5 views that pass
the pipeline's own gates (SAM mask valid, view median conf ≥ 0.6,
`view_mask_agreement` ≤ 3 fly-lengths); ungated, a bout whose female has 3/7
usable cameras contributes 140 px "residuals" that say nothing about geometry.
Cameras and keypoints indexed **by name** throughout (the mask npz's camera
order is not the canonical one).

### LOO table (leave-one-camera-out reprojection, px)

| recording (source) | n pts | A med / p90 | C med / p90 | paired A−C [95 % CI] | BA med | worst camera |
|---|---|---|---|---|---|---|
| `12_11_50` bouts | 31 896 | 5.39 / 28.7 | **4.90** / 28.3 | **+0.19** [+0.16,+0.22] → C | 3.65 | A: Cam2012861 6.6 |
| `12_11_50` human | 1 499 | 4.58 / 11.4 | **3.45** / 10.7 | **+0.75** [+0.54,+0.97] → C | 2.68 | A: Cam2012861 6.1 |
| `15_25_51` bouts | 66 657 | **3.44** / 10.3 | 5.06 / 11.6 | **−1.49** [−1.51,−1.46] → A | 2.76 | C: Cam2012855 6.5 |
| `15_25_51` human | 2 200 | **2.32** / 11.9 | 4.19 / 12.6 | **−1.41** [−1.55,−1.27] → A | 2.15 | C: Cam2012855 5.8 |
| `17_28_34` bouts | 42 627 | **3.31** / 12.0 | 4.73 / 12.8 | **−1.17** [−1.20,−1.14] → A | 2.93 | C: Cam2012855 6.2 |
| `17_28_34` human | 4 391 | **2.66** / 8.9 | 4.46 / 10.3 | **−1.53** [−1.63,−1.43] → A | 2.43 | C: Cam2012855 6.3 |

Figure `figures/2026-09-mvq/v2_train/calib_ba/loo_by_camera.png` (read back):
the C boxes sit visibly above the A boxes in Cam2012853/855/857/862 for both
afternoon recordings — largest at **Cam2012855**, the predicted worst camera —
and the ordering **reverses** on `12_11_50`, where C is at or below A in
Cam2012631/857/861. Cam2012630 separates least, as predicted (A and C differ
by only 0.8-1.7 px there over the real fly volume). Both BA arms are visually
identical in every panel and at or below both shipped calibrations everywhere.

**Not an artefact of the crop placement.** Stage A places its crop by
triangulating the mask centroids, the one path by which the calibration could
leak into the "independent" 2-D. Matched control on bouts 6+12 of `15_25_51`:
crops placed by A → paired A−C = −1.455 px; crops placed by C → −1.459 px.
No effect.

**Per-bout, across each recording's whole frame axis** (rig-move test): the
winner never flips *within* a recording — `15_25_51` bouts 1/6/12/26/30
(frames 60 k → 791 k) all → A (−0.63 to −1.65 px); `17_28_34` bouts 1/6/10
(465 k → 723 k) all → A; `12_11_50` bouts 2/3/10 (181 k → 526 k) all → C.

### Bundle adjustment

Alternating affine DLT / robust (Huber + 3× median trim) per-camera 2×4
resection, gauge pinned by re-aligning the refined cloud to the initialisation
with a 12-dof affine each iteration. Converges in 5-10 iterations. **Both
initialisations reach the same cameras**: after gauge alignment the max
per-camera reprojection difference between BA(init A) and BA(init C) is
**0.0008-0.006 px** in every recording — the data fully constrain the affine
cameras. Distance of the converged solution: 0.4-2.5 px per camera from **A**
and 1.2-4.7 px from **C** on the afternoon recordings; 0.9-4.0 px from A and
1.2-2.4 px from C on `12_11_50`. Held-out (BA fitted on half the bouts, scored
on the other half): `15_25_51` A 3.51 / C 4.96 / **BA 2.88**; `17_28_34`
A 6.10 / C 6.85 / **BA 5.76** — the gain is not an overfit. Refined YAMLs (input
format, one `Cam*.yaml` per camera) at
`OutFiles/calib_ba/<rec>/calibration_ba/`.

**Cross-recording transfer** (LOO median px on the row's data) is the decisive
result:

| evaluated on | A | C | BA:12_11_50 | BA:15_25_51 | BA:17_28_34 |
|---|---|---|---|---|---|
| `12_11_50` | 5.39 | 4.90 | **3.65** | 4.50 | 4.54 |
| `15_25_51` | 3.44 | 5.06 | 3.83 | **2.76** | 2.98 |
| `17_28_34` | 3.31 | 4.73 | 4.15 | 3.20 | **2.93** |

The two afternoon solutions transfer to each other (2.98 / 3.20 px, better than
either shipped calibration) and agree to ≤ 1.31 px per camera; the morning
solution does not transfer to them (4.50 / 4.54) and differs from both by
2.15-2.54 px — against an A-vs-C reference difference of 4.45 px.

### Verdict

**The rig moved between 12:11:50 and 15:25:51, and neither shipped calibration
is exactly right for either side.** A is the better of the two for `15_25_51`
and `17_28_34`; C is the better of the two for `12_11_50`. That reproduces the
2026-09-03 export's per-recording assignment from data it never saw, so the
export fix is confirmed and **the video-folder calibration C is wrong for every
afternoon Session1 recording**. But BA beats A by 0.4-0.7 px and C by
1.2-2.4 px on their own recordings, so both are ~1-3 px off the geometry the
images actually show.

### Practical consequence

Smaller than the pixel numbers suggest, and worth stating precisely. Triangulating
the *same* 2-D under A and under C moves the 3-D by 1.10-1.45 world units
(3.1-4.3 % of the fly's body extent), but **after the best similarity that
collapses to 0.018-0.045 u (0.05-0.13 % of body)**: the A↔C difference is
almost purely a world-frame rigid motion plus a uniform scale of
**0.47-0.58 %** (C reconstructs everything ~0.5 % larger). A rigid-invariant
check (14 thorax/leg segments, CV of length) is consistent with this and
**cannot discriminate** the two: CVs agree to the third digit because the
5-20 % detector-noise CV swamps a 0.5 % scale — recorded as an honest negative.

So the p3b lifts of `15_25_51` and `17_28_34` (and every other afternoon
Session1 recording, all lifted with C) are **not shape-scrambled**; they carry a
~0.5 % body-scale error, a small world-frame rotation/translation, and ~1.5 px
more multi-view inconsistency, which is what the STAC/IK residuals and
reprojection overlays see. Anything comparing 3-D across the morning/afternoon
boundary, or reusing `scale.json` across it, is affected at the 0.5 % level;
per-recording IK is affected mainly through the extra 1.5 px.

### For a proper fix: the wand recordings exist

`Video_recordings/courtship/NewCalibration/` holds six 7-camera wand/checkerboard
captures from that same day — **2026_04_02_11_39_34, _11_39_57, _11_40_21,
_11_42_59** (morning, before the 11:52 first recording) and **2026_04_02_18_15_05,
_18_15_22** (evening, after the 17:52 last one). They bracket the day on both
sides of the move. Every one of them ships calibration **C** in its own
`calibration/` dir (byte-identical to human-root C), i.e. only one calibration
was ever derived from that session and copied everywhere — which is exactly how
the afternoon recordings ended up with the morning geometry. These videos are
the bias-free source for a proper recalibration of both sides; **not** done here.

Scripts + JSON artifacts: `figures/2026-09-mvq/v2_train/calib_ba/`
(`analyse.py`, `calib_core.py`, `detect_human_frames.py`, `plot_loo.py`,
`consequence.py`, `rigid_invariant.py`, `human2d_selfcheck.py`,
`ac_over_real_volume.py`, `loo_ba_results.json`, `consequence.json`).

## Loader: process workers (2026-09-06)

**Symptom.** On the first 8-GPU / batch-32 v2 run (`mvq_t2_v2_20260906`) GPU
utilisation cycled ~6 s at 100 % then ~3 s at 0 %: the depth-2 prefetch drained
and every device waited on data. The trainer process used 150-410 % CPU -- 1.5
to 4 of the node's 32 cores -- with 24 loader threads. `V12WindowDataset.
__getitem__` only releases the GIL inside the JPEG decode; the window assembly,
centre jitter and copy-paste compositing around it are Python/numpy and
serialise on one interpreter. At 6 GPUs / batch 24 the same loader kept up at
2.11 s/step, which is 11.4 samples/s -- exactly its ceiling, so that run was
already loader-bound and simply had less to feed.

**Fix.** `window_batches(..., workers="processes")` dispatches to a **spawn**
`multiprocessing` pool (`jarvis_jax/data/loader_workers.py`). Fork is not an
option: the parent holds eight initialised CUDA contexts. Each worker rebuilds
the roots from a picklable spec (`V12WindowDataset.worker_spec()` /
`ConcatWindowDataset.worker_spec()`) in the pool initializer; sampling stays in
the parent (`_balanced_weights`, the epoch permutation, `_MixCounter`'s tally)
and the epoch travels with every task, so the per-sample RNG --
`SeedSequence([seed, index, epoch, ...])`, no worker-local state -- lands on the
same jitter and the same copy-paste donor. Batches are byte-identical to the
thread path with train=True and copy-paste on
(`tests/test_loader_workers.py`). One pool serves both the T=1 and T=2
streams, keyed by T, so two streams cost 24 processes rather than 48.

**Loader-only benchmark**, real four-root T=2 train concat as `run_training`
builds it (46 003 windows, copy_paste 0.8, jitter 10, pair_deltas 1/4/16),
batch 32, 12 timed batches, on g3102:

| loader | samples/s | s/batch(32) | vs threads |
|---|---|---|---|
| threads(24) | 9.18 | 3.49 | 1.00x |
| processes(24) | 23.10 | 1.39 | **2.52x** |
| processes(32) | 26.80 | 1.19 | 2.92x |

Worker memory: 2.25 GB resident per worker per dataset (54 GB over 24 workers
for one; ~133 GB for the two-T training pool). processes(32) buys 16 % more
throughput for 33 % more memory, so the config uses 24.

**150-step GPU check**, the real command with `+share_check_steps=150
train.loader_workers=processes` on all 8 L40S at batch 32:

* **1.44-1.49 s/step**, steady (steps 37/74/111/148 at 280/335/388/442 s) vs
  **1.82 s/step** on the thread loader (killed run, steps 200-250) -- 20 %
  faster wall-clock.
* `nvidia-smi dmon -s u`, 30 samples: mean SM **86 %**, 22 of 30 samples at
  100 % on all eight devices, the rest brief dips (two to 0). The 6 s/3 s
  100 %-then-0 % cycle is gone.
* Trainer + workers together now draw ~620 % CPU (was 150-410 %).
* Loss shares unchanged and still in band (`other_fly_repulsion` 8.08 %).

**Still loader-bound, and why the dips remain.** 32 samples / 1.45 s = 22
samples/s, which is the processes(24) benchmark rate, not a GPU limit -- the
residual 14 % of idle SM is the loader, not the model. Two things cap it: (a)
this node is shared, and another user held 595 processes and a load average of
140-320 on its 32 cores throughout both the benchmark and the 150-step run, so
every number here is a *contended* number; (b) 32 workers measured 16 % faster
than 24 and is a one-line config change if the extra ~44 GB is acceptable.

**Thread-pool lifetime.** The reported 9 634 OS threads are not a
`window_batches` leak: a fully drained epoch returns the thread count to its
baseline (measured, 20 epochs), and a bare 8-device JAX program already sits at
174 threads before any collective -- the bulk is XLA/NCCL. There *was* a real
leak on the abandoned path: the `with ThreadPoolExecutor(...)` wrapped the
generator body, so a consumer that stopped mid-epoch held `num_workers` threads
until the generator was garbage collected, and the GC at interpreter shutdown
died inside `Thread.join` (`TypeError: 'NoneType' object is not callable`).
`window_batches` now joins on a clean drain and releases without waiting on
abandonment; two tests count `/proc/self/status` Threads across 20 epochs and
across 10 explicitly closed mid-epoch generators.

**Regenerate.** The loader table: build the four-root T=2 concat exactly as
`run_training` does (the `pseudo_root`/`singlefly_root`/`negatives_root` of
`configs/train/mvq_v2.yaml`, `train=True`, `copy_paste_p=0.8`,
`jitter_units=10`, `pair_deltas=(1,4,16)`, `allow_calib_mismatch=True`), then
time 12 batches of `window_batches(ds, 32, seed=7)` after discarding the first,
once with `num_workers=24` and once each with
`workers="processes", pool=ProcessSampleLoader({None: dataset_spec(ds)}, n)`
for n in (24, 32). The driver must sit behind `if __name__ == "__main__":` --
spawn re-imports `__main__`. The GPU check: `/tmp/perfcheck150.sh`, i.e. the
launch script's environment plus
`run_id=mvq_t2_v2_perfcheck +share_check_steps=150
train.loader_workers=processes`, with `nvidia-smi dmon -s u -c 30` alongside.
The run itself: `bash slurm_logs/launch_mvq_t2_v2.sh`.
