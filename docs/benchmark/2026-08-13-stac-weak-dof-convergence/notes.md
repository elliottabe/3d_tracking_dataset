# STAC weak-DOF convergence — investigation, outcome, and what to run on Hyak

**Date:** 2026-08-13
**Branch:** `elliottabe/stac-weak-dof-convergence` (parent + `stac-mjx` submodule)
**Spec:** the design doc this investigation started from
(`docs/superpowers/specs/2026-08-13-stac-weak-dof-convergence-design.md`) is **not
committed** to either repo and lives under a gitignored `docs/superpowers/` path, so it
is unresolvable post-merge. The specific claims from it that D1/D2 below critique,
stated inline so this note stands on its own:
  - it mapped stac-mjx's existing `FTOL` config key onto jaxls' `cost_tolerance`, while
    separately listing `JAXLS_COST_TOLERANCE: 1.0e-5  # unchanged from jaxls default` —
    a direct self-contradiction (`FTOL` is `5.0e-03`, not `1.0e-5`);
  - it set a "polish" reselection threshold of `0.2` on normalized Jacobian column norm
    with no empirical grounding, validated only on synthetic two-hinge geometry;
  - it inferred from single-`pose_optimization` sweeps that production `fit_offsets`
    "leaves weakly-conditioned DOFs far short of their optimum" — a claim that does not
    hold once the full seven-call production path is measured (see Summary).
**Status:** fix B kept (corrected); fix C removed; 13-bout benchmark gate NOT satisfied.

## Summary

The investigation started from wings that render "rolled inward" in
`Session6/2025_10_12_15_06_46/ik_explainer/predictions/05_stac_ik.h5`. That symptom is
real, but the diagnosis generalised from the wrong thing and validation overturned it.

**Production IK was never broken.** A full `fit_offsets` run with the pre-change
tolerance and no polish converges the wings to `wing_pitch_left = -38.61 deg`,
`wing_pitch_right = -40.36 deg`. The unconverged wings (`-4.7 deg`) exist only in the
**explainer's** output, because `scripts/viz/ik_explainer/stage_ik.py` deliberately
runs a single `pose_optimization`, while production runs seven
(root + 6x[pose, offset] + final pose). The alternating loop overcomes the premature
termination on its own.

Every diagnostic sweep in the design doc measured that single-call path. The doc's
claim that production "leaves weakly-conditioned DOFs far short of their optimum" does
not hold for the production pipeline and should be read as applying to the explainer's
truncated solve only.

## What was measured (all on Session6/2025_10_12_15_06_46, 921 frames)

### Conditioning is real, and the arithmetic is right

Jacobian of the two wing-tip sites w.r.t. the three wing joint angles, frame 414:

| | yaw | roll | pitch (blade roll) |
|---|---|---|---|
| dP/dtheta (units/rad) | 0.193 | 0.342 | **0.029** |

Singular values `[0.343, 0.192, 0.0243]`, condition number **14.1**. Non-degenerate —
hinge, `V12` and `V13` form a real triangle with 8.09 deg of bearing separation, so all
three DOF are observable. The blade-roll direction is simply 14x weaker, mapping a
0.010 keypoint error to ~24 deg of angle versus ~1.7 deg on the well-determined ones.

This part of the diagnosis stands. What does not stand is the inference that the
production solver fails to converge it.

### Single `pose_optimization` (the explainer's path) — the sweeps that misled

| setting | pitchL | wing res | all-kp res |
|---|---|---|---|
| shipped (`lambda=1.0`, jaxls default tolerances) | -4.72 deg | 0.0222 | 0.0158 |
| `lambda=1e-2` | -19.06 deg | 0.0175 | 0.0124 |
| `gradient_tolerance=1e-8` | **-22.23 deg** | **0.0168** | **0.0122** |
| `gradient_tolerance=1e-8` + `lambda=1e-2` | -22.67 deg | 0.0168 | 0.0122 |

`lambda` was never a real fix — the `1e-2` result is knife-edge (both neighbours revert
to baseline) and disappears once termination is addressed. Independent runs at
identical settings vary ~1.2 deg, so acceptance thresholds must sit outside that band.

### Full `fit_offsets` (production) — what actually matters

| arm | wing_pitch L / R | all-kp res (median) | legs | wing tips |
|---|---|---|---|---|
| polish off, `gradient_tolerance=1e-4` | -38.61 / -40.36 | 0.00407 | 0.00454 | 0.00216 |
| polish on, `gradient_tolerance=1e-8` | -38.61 / -40.36 | 0.00407 | 0.00454 | 0.00216 |

Identical — see the two defects below for why.

## Defects found (all originate in the design doc / plan, not the implementations)

### D1 — `FTOL` mapped onto `cost_tolerance` made fix B inert AND caused a regression

The doc contradicted itself: it listed
`JAXLS_COST_TOLERANCE: 1.0e-5  # unchanged from jaxls default` *and* said `FTOL` maps
onto `cost_tolerance`. `FTOL` is `5.0e-03` — **500x looser** than jaxls' own `1e-5`
default. Effective tolerances as first shipped:

| | cost | gradient | parameter |
|---|---|---|---|
| jaxls library default | 1.0e-5 | 1.0e-4 | 1.0e-6 |
| as first merged | **5.0e-3** | 1e-4 or 1e-8 | 1.0e-10 |

The cost criterion therefore fired first and the gradient tolerance never engaged.
Proof: an A/B differing *only* in gradient tolerance produced **bit-identical** offset
errors for all six `fit_offsets` iterations. It was also a regression — before this
work stac-mjx passed no tolerances at all, so jaxls' `1e-5` applied; the merged code
terminated *earlier* than the pre-change pipeline.

**Fixed** by adding `JAXLS_COST_TOLERANCE` (default `1.0e-5`) as its own key and no
longer mapping `FTOL`. `FTOL` now documents that it governs the ProjectedGradient path
only.

### D2 — polish never selected the wings, and moved what it did select too freely

On the real rig (86 scalar DOFs) the median Jacobian column norm is `0.1322`, so the
threshold-0.2 cutoff is `0.0264`. `wing_pitch_*` sits at `0.0291` — **above** the
cutoff by ~10%, so it was never selected. What was selected: 8 distal tarsus DOFs
(lever arms 0.005-0.025), which polish moved by **up to 41 deg** to buy ~1-3% residual.
Those angles are barely constrained by the data; polishing them injects noise into any
downstream analysis of tarsus kinematics.

The `0.2` threshold was invented in the design doc with no empirical grounding, and the
plan's tests validated it only on synthetic two-hinge geometry.

**Resolution:** fix C removed entirely (Task 9). The conditioning-analysis tooling
survives in git history as `stac-mjx` **submodule** commits `e7aa9a9`, `136dfe8`,
`f69b2f6` (not parent-repo commits) if it is ever wanted as a diagnostic. These SHAs
are only reachable through `stac-mjx`'s own history and vanish if that submodule's
branch is ever squash-merged.

### D3 — derived arrays went stale after polish

`fit_offsets` takes `(qposes, xposes, xquats, marker_sites)` from `pose_optimization`,
then mutated `qposes` via polish, then passed the **unrecomputed** `xposes`/`xquats`/
`marker_sites` to `_package_data`. Proof: `marker_sites` was bit-identical across the
two A/B arms while `qpos` differed by up to `0.723 rad` (41 deg). Any residual computed
from `marker_sites` was blind to polish — which is exactly why both arms reported
identical numbers. Moot now that fix C is removed, but it is the reason the A/B table
above shows no difference.

**Unresolved and worth a separate look:** even in the polish-off arm, marker sites
recomputed from the saved `qpos` differ from the saved `marker_sites` by median
`0.0059`, max `0.0235`. That is independent of polish and was not chased down here.

## Outcome

**Kept — fix B (corrected).** jaxls termination tolerances are now explicit and
configurable (`JAXLS_GRADIENT_TOLERANCE` 1e-8, `JAXLS_PARAMETER_TOLERANCE` 1e-10,
`JAXLS_COST_TOLERANCE` 1e-5) across all six anatomy configs, and `FTOL` is no longer
silently discarded on the jaxls path — its real scope is documented instead. This is a
correctness and transparency fix. **Its effect on production output has not been
measured**, because the only A/B run was invalidated by D1; it must be measured on
Hyak before anyone claims a quality improvement.

**Removed — fix C (polish).** See D2.

**Not done — the 13-bout benchmark gate.** See below.

## Known divergence: JaxlsBatchSolver vs. the jarvis_jax silhouette/bundle solvers

This branch changed `JaxlsBatchSolver`'s constructor **defaults** (gradient
1e-4→1e-8, parameter 1e-6→1e-10; cost stays at jaxls' own 1e-5). Two other jaxls call
sites were knowingly **left on jaxls' un-overridden defaults** and were not touched:

- `third_party/jarvis_jax/jarvis_jax/tracking/silhouette_joint_ik.py:347`
  (`SilhouetteJaxlsBatchSolver`) builds a bare
  `jaxls.TerminationConfig(max_iterations=self.n_iter)`.
- `third_party/jarvis_jax/jarvis_jax/tracking/bundle_adjust.py:180` likewise builds a
  bare `jaxls.TerminationConfig(max_iterations=n_iter)`.

This was a deliberate scope decision, not an oversight: changing another subsystem's
solver with no way to validate it from this workstation would be unvalidated scope
creep. The consequence is that `JaxlsBatchSolver` and `SilhouetteJaxlsBatchSolver` now
terminate on different criteria, and
`third_party/jarvis_jax/tests/test_silhouette_joint_solve.py:48-53`
(`test_no_silhouette_matches_jaxls_batch_solver`) asserts they stay `atol=1e-5`
byte-close. That test is `skipif`-gated on an IK fixture under `/gscratch/...`
(Hyak-only), so it cannot run on this workstation and stays silently green here
regardless of whether the two solvers actually agree.

**Required pre-merge check on Hyak:** run
`third_party/jarvis_jax/tests/test_silhouette_joint_solve.py` where the `/gscratch`
fixture is present. If it still skips there too, the divergence remains unvalidated and
should block treating this as settled. If it fails, the fix is to give
`SilhouetteJaxlsBatchSolver` matching (or at least equally-explicit) tolerances —
**not** to loosen the test's `atol=1e-5` assertion to make it pass again.

## Coverage limitations — read before trusting anything above

- Everything was measured on **one clip, one fly**
  (`Session6/2025_10_12_15_06_46`), a single-animal reconstruction whose identity could
  not be confirmed. `CLAUDE.md` is explicit that the **female** fly (walls, occlusion,
  OOD poses) is where this pipeline fails and the male "usually looks fine regardless".
  No female, occlusion, or wall frame was examined.
- The frozen 13-bout benchmark A/B — the design doc's stated acceptance gate for a
  change that touches every STAC fit — **was not run**. `configs/benchmark/bouts.yaml`
  points `benchmark_root` at `/gscratch/portia/eabe/data/Johnson_lab/processed/benchmark`,
  a Hyak path not mounted on the workstation this ran on.

## What to run on Hyak

The fixed tolerance values ship as defaults in the anatomy configs, so the variant arm
needs no overrides; only a pre-change baseline does.

```bash
# 1. Baseline scorecard, computed off existing outputs (no re-run)
python -m scripts.benchmark.run_variant baseline \
  --out docs/benchmark/2026-08-13-stac-weak-dof-convergence/scorecard-baseline.json

# 2. Variant arm — current configs already carry the fixed tolerances
python -m scripts.benchmark.run_variant build    --variant weakdof
python -m scripts.benchmark.run_variant commands --variant weakdof | bash
python -m scripts.benchmark.run_variant collect  --variant weakdof \
  --out docs/benchmark/2026-08-13-stac-weak-dof-convergence/scorecard-fixed.json
```

To run an explicit pre-change arm instead of relying on stored baselines, add
`--override model.JAXLS_GRADIENT_TOLERANCE=1.0e-4 --override model.JAXLS_COST_TOLERANCE=1.0e-5 --override model.JAXLS_PARAMETER_TOLERANCE=1.0e-6`
to the `commands` step. Omitting `JAXLS_PARAMETER_TOLERANCE` here would leave it at the
new default `1.0e-10` instead of jaxls' pre-change `1.0e-6`, so the "baseline" arm would
still be partially tightened rather than a true pre-change recipe. Verify the override
key path resolves (`cfg.model.*`) with a single-bout dry run first — `--override`
values are passed through verbatim as Hydra overrides to `scripts/run_bout.py`.

The `--out` paths above (`docs/benchmark/.../scorecard-*.json`) land under
`docs/benchmark/`, which is gitignored; committing the scorecards as evidence for this
decision needs `git add -f`.

**Acceptance:**
- All-keypoint residual improves or is flat; no bout regresses beyond the ~1.2 deg /
  run-to-run noise band; the female cohort improves or is flat.
- **Wall-clock per bout, both arms.** Tightening `gradient_tolerance` 1e-4→1e-8 and
  `parameter_tolerance` 1e-6→1e-10 makes early termination far harder, so solves may
  now run out toward `N_ITER_Q: 500` instead of exiting early, and `fit_offsets` calls
  `pose_optimization` seven times per bout — this cost multiplies. Runtime impact was
  never measured in this investigation (only residuals and angles were), and slower
  wall-clock is the most likely production-visible effect of this branch. Record
  seconds/bout for baseline vs. fixed and flag any large increase even if quality is
  flat.
- Run `third_party/jarvis_jax/tests/test_silhouette_joint_solve.py` (see "Known
  divergence" above) as a **required** pre-merge check wherever the `/gscratch` IK
  fixture is available.
- If the fixed arm is indistinguishable from baseline on quality, fix B is a pure
  correctness change with no quality effect — which is a perfectly good result, and
  should be recorded as such rather than dressed up.

## Separate follow-up: the explainer

`scripts/viz/ik_explainer/stage_ik.py` runs one `pose_optimization` by design, which is
why Acts 3/4 render unconverged wings. Options, in preference order:

1. Have the `pose` stage iterate `pose_optimization` the same number of times
   production does (without `offset_optimization`, so it stays a strict subset of
   `fit_offsets`), and re-check the module's monotonic-residual assertion at
   `stage_ik.py:335`.
2. If the four-stage narrative depends on showing exactly one solver call, keep it and
   add a comment stating that the wings are unconverged **by design** — so the next
   person does not re-diagnose it as a production bug, as this investigation did.

Not applied here: this branch is based on `f14d592`, and `acts/act3_align.py` /
`acts/act4_solve.py` were being actively edited elsewhere at the time.

## Other findings, unrelated to this change

- `wing_yaw_{left,right}` saturates at its MJCF limit of ±85.94 deg on essentially
  every frame, std 0.00, on the production path (unconstrained optimum ~90 deg).
- The model's wing is ~4% shorter than the observed wing (hinge→tip 0.2151 / 0.2613
  model units vs observed 0.2233 / 0.2713), setting a residual floor.
- `WingL_V13` / `WingR_V13` learned marker offsets drift mostly **perpendicular** to
  the wing long axis (perp 0.0194 / 0.0184 vs along 0.0056 / 0.0050), near-symmetric
  across independently-fit wings — suggesting the model's `V13` site is mis-placed
  rather than that offsets absorbed pose error.

All three need MJCF edits and would invalidate existing fits; each deserves its own
ticket.
