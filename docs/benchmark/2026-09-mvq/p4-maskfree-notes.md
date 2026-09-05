# P4a mask-free front end -- benchmark notes

## Centre-shift spike (2026-09-04)

Task 1 of `.superpowers/sdd/2026-09-04-mvq-maskfree-p4a-p4b` (spec §6). The
mvq lifter (`mvq_t1_b16_p3a_20260904/final`, val loss/step: EMA at
`total_steps=10000`) was trained with window centres jittered by only
+-0.3mm (`jitter_units=3.0`, `MM_PER_UNIT=0.1`) around the host fly's own
labelled-3D bbox midpoint. The mask-free front end (P4b) will instead place
windows from a coarse track interpolated to 800fps, whose centre can be
~0.6mm off the true fly centre. This spike measures how the UNPROMPTED
typed-slot policy (`policy_instance(..., prompted=False, has_mask=False)` --
the real inference-time instance choice, no ground truth) degrades as the
window centre is displaced by a known, exact amount on the full val split
(153 windows), via the new `V12WindowDataset(..., center_shift_units=...)`
loader option (eval-mode only, deterministic per window, in-plane;
`jarvis_jax/data/v12_windows.py`).

Command:
```
PYTHONPATH=third_party/jarvis_jax:. CUDA_VISIBLE_DEVICES=0 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
  python scripts/benchmark/mvq_centre_shift.py \
  --run /gscratch/portia/eabe/data/Johnson_lab/jax_mvq_runs/mvq_t1_b16_p3a_20260904/final
```

### Results (n=153 val windows at every shift)

| shift (mm) | policy MPJPE (mm) | vs unshifted | miss fraction |
|---:|---:|---:|---:|
| 0.0 | 0.0934 | -- (baseline) | 0.000 |
| 0.5 | 0.0970 | +3.9% | 0.0065 |
| 1.0 | 0.1084 | **+16.1%** | 0.0065 |
| 2.0 | 0.4166 | +346% (4.46x) | 0.0654 |
| 3.0 | 0.8715 | +833% (9.33x) | 0.2222 |

Decision lines (per the docstring EXPECTATION / §6 rule): error within 10% of
the unshifted value (0.1027mm) and miss fraction under 2% (0.02), both **up
to and including the 1mm (10 unit) shift** -- the magnitude the coarse-track
front end is expected to sit near.

### Figure reading

`figures/2026-09-mvq/p4_maskfree/centre_shift.png` (JSON alongside):
two panels, error (mm, left) and miss fraction (right) vs centre shift (mm),
each with its decision line dashed in gray, title naming the checkpoint
(`mvq_t1_b16_p3a_20260904`, `step=final(total_steps=10000)`).

- **Left (error):** flat from 0 to 0.5mm, then the point at 1.0mm sits
  visibly just above the 1.1x dashed line -- the numeric check above confirms
  it (+16.1%, over the +10% allowance). From 1mm to 2mm the curve turns into
  a steep, near-linear climb (0.108mm -> 0.417mm, +0.31mm for the next 1mm of
  shift alone) and keeps climbing to 0.87mm at 3mm.
- **Right (miss fraction):** both the 0.5mm and 1.0mm points sit clearly
  below the 0.02 line (0.0065, no change between them at all -- the same
  10/153 windows miss at both). The curve crosses the 0.02 line between 1mm
  and 2mm and is at 0.065 (>3x the line) by 2mm, 0.222 by 3mm.

Both panels show the same shape: near-flat through 0.5mm, a borderline miss
of the error line right at 1mm, then a cliff starting immediately past 1mm.
This is the failure signature the docstring's EXPECTATION describes as
"climbs steeply" near the 1mm boundary, not a gradual degradation that stays
inside tolerance out to 3mm.

### Decision (§6 rule)

**Retrain needed: YES.** At the boundary shift the coarse-track front end is
expected to operate near (~0.6mm nominal, with 1mm named in the spec as the
shift the retrain should cover), the policy MPJPE already exceeds the 10%
tolerance (+16.1% at 1mm vs the +10% allowance, 0.1084mm vs the 0.1027mm
line) even though the miss fraction is still comfortably under 2% (0.65%) at
that point. Immediately beyond 1mm the degradation is not a soft margin but
a cliff: by 2mm the error is 4.46x the unshifted value and the miss fraction
(6.5%) is already more than 3x over its own 2% line; by 3mm error is 9.33x
and 22% of windows return no instance at all. A model trained on +-0.3mm
jitter has essentially no headroom past a 1mm centre error, which is exactly
the displacement the mask-free coarse track can produce -- the §6 retrain
with 1mm jitter (`jitter_units` an order of magnitude larger than today's
3.0) is warranted before Task 4 runs the mask-free front end on real
interpolated coarse-track centres. Per the task brief: STOP after Task 1 and
report this; Tasks 2-3 (which do not depend on the retrained checkpoint) can
proceed, and the controller should schedule the retrain before Task 4.
