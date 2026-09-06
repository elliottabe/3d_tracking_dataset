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
equal-width [0,1] bins and reports `max_gap` (over populated bins only) and
`ece` (bin-size-weighted mean gap).

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

| head | T | before max_gap | before ECE | after max_gap | after ECE |
|---|---:|---:|---:|---:|---:|
| existence | 0.266 | 0.437 | 0.0101 | 0.278 | 0.0027 |
| per-view visibility | 0.717 | 0.269 | 0.0140 | 0.268 | 0.0076 |

Figure: `figures/2026-09-mvq/v2_train/calibration_p3b.png` (JSON alongside:
`calibration_p3b.json`).

### Figure read back (CLAUDE.md)

Existence panel (T=0.266, a SHARPENING temperature, since T<1 divides
logits by a fraction and inflates them): the pre-scaling (red) curve is
**above** the diagonal at low-to-mid confidence and step-like -- conf~0.17
to conf~0.33 bins are all `acc=0` (existence correctly denied) but the very
next populated bin, conf~0.56, is `acc=1.0` (existence correctly asserted),
i.e. the raw head is markedly **under-confident**: frames it called "55%
likely present" were present 100% of the time. This matches the earlier
memory note ("~0.45 sigmoid on present flies") and is the OPPOSITE sign
from the module docstring's originally-written hypothesis (over-confidence)
-- the docstring has been corrected in the script to say "read the sign off
the figure, do not assume it". After scaling (blue), the curve sits much
closer to the diagonal through the populated low/mid range and lands almost
exactly on it at the dominant high-confidence bin (conf=0.999, n=200,
gap=0.005) -- most of the val mass is now well calibrated. `max_gap` stays
at 0.278 only because of two near-empty bins the temperature could not move
enough: conf-bin [0.7,0.8) has **n=1** (acc=1.0, conf=0.72, gap=0.28) and
[0.8,0.9) has n=3 (gap=0.13) -- a single val sample dominating a "gap"
statistic is exactly the case `reliability`'s own docstring warns about;
the bin-size-weighted ECE (0.0027) shows the aggregate improvement the raw
`max_gap` number does not.

Visibility panel (T=0.717, also sharpening): the raw curve sits mildly
above the diagonal across the whole range (broad, gentle under-confidence,
n in the hundreds-to-tens-of-thousands per bin -- no sparsity issue here).
After scaling, the low/mid bins (conf 0-0.5) move measurably closer to the
diagonal (this is where the fitted T mainly helps, and ECE drops
accordingly) but the high bins (0.75-0.95) end up *slightly further* from
the diagonal than before -- a single global temperature cannot correct a
curve that is under-confident at one end and roughly correct at the other
simultaneously, so `max_gap` (driven by a high bin) barely moves
(0.269 -> 0.268) even though ECE improves by ~2x.

### Conclusion for Task 6 / handoff to Task 7

The script and `MVQRunner` integration work end-to-end: `n_val=153` (matches
the known val split size), the fitted temperatures are physically sane
(both < 1, consistent with the previously-documented ~0.45-sigmoid
under-confidence on present flies), ECE improves for both heads, and the
`"calibration"` block round-trips through `mvq_run.json` in the exact shape
`scripts/benchmark/mvq_v2_acceptance.py::check_calibration` reads
(`calibration.reliability_exist.max_gap` etc.). **This P3b proof run does
NOT clear the spec's 0.05 max_gap acceptance line** -- expected and
non-blocking, since P3b was never trained with calibration in mind and this
run's whole purpose was to prove the script works, not to certify P3b. The
real acceptance gate is the v2 FINAL checkpoint (deliverable 7 in the design
spec), not yet trained; when it exists, re-run this exact command against
`<v2 run>/final` with `--stage-out` OMITTED (the default in-place write,
now the correct target because it IS the production run) and read
`max_gap` there. Because the P3b run's own `final/mvq_run.json` was never
touched, `check_calibration` against it still reports SKIP ("no
'calibration' block yet"), unchanged from before this task.
