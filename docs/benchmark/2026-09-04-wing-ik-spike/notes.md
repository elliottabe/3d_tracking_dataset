# Why the IK wing twists on the mvq keypoints, and per-fly offsets

*2026-09-04. Session0 `2025_10_20_13_20_04`, `pose_mvq_ik` bout 28 (mvq lifter
keypoints, checkpoint `mvq_t1_b16_p3a_20260904` step 7000). Figures and the
generating scripts: `figures/2026-09-04-wing-stop-spike/` (gitignored;
`wing_refit.py`, `wing_landscape.py`, `render_compare.py`,
`resolve_experiment.py`, `compare_perfly_offsets.py`). Male = fly1.*

## Question

The mvq keypoints and 3D look much better than ViTPose, but the STAC fit still
renders the male's FOLDED wing edge-on (twisted) and his abdomen straight.
Is it offsets, scale, a joint stop, or the solver?

## What the mvq keypoints changed

| quantity | ViTPose runs (prior notes) | this mvq run |
|---|---|---|
| WingL V12-V13 rigid length CV | ~30% male, 80-120% female | 2-3% both wings |
| implied blade-rotation noise / frame | ~24 deg | ~2 deg |
| measured wing kp reproj (median) | ~2 px | 0.9-1.7 px |
| IK-fitted wing kp reproj (median) | -- | 3.1-5.4 px |

Observability of blade rotation is no longer the bottleneck. The fitted sites
land 4-5 px from points measured to 1 px: the solver is not reaching markers
it could reach.

## Ruled out

- **Yaw stop.** Right wing sits at its 85.94 deg yaw stop on 40% of frames,
  and its V13 residual there is 1.45 mm vs 0.65 mm off the stop -- but
  widening the stop by 50 deg changes that residual by <0.1 mm and only 7% of
  frames want past it (`wing_refit_bout28_fly1.npz`, condition B).
- **Hinge position.** Freeing the wing body's position (condition C) fits
  exactly by construction (6 params, 6 equations); the implied shift is only
  partly consistent (mean 0.7 mm forward, sd 0.3-0.9 mm). Not the cause.
- **Sites "more separated" on the model.** Only valid if the detector's
  landmarks are separated too; V13 already carries the blade rotation with a
  3.6 mm lever arm (V12 sits 0.3 mm from the axis). Offsets are fitted, so the
  data already placed them.

## Cause: the batch solve stops early on the weak pitch DOF

`pitch_landscape_bout28_fly1.png`: sweeping `wing_pitch_right` with yaw and
roll re-optimised, the folded wing's marker cost has a clear minimum at
**-50 deg** (rest -57.3). STAC left it at **-19 deg**, 0.6 mm/marker up the
slope. The extended left wing sits within 10 deg of its optimum. Re-solving
ONLY the wing DOFs from STAC's pose (scipy LM, hard bounds) reaches -50 deg
and, rendered from Cam2012631, lays the wing flat over the abdomen exactly as
the video shows (`stac_vs_wingrefit_bout28_fly1.png`, frames 150 and 586).

Production jaxls solver re-run on frames 100-400 (`resolve_experiment.py`,
queue job 39606823):

| variant | wing_pitch_right @150 / @314 | all-kp resid | time |
|---|---|---|---|
| shipped (cost_tol 1e-5, grad 1e-8, param 1e-10, N_ITER_Q 500) | -40.3 / -39.9 | 0.410 mm | 188 s |
| tight (1e-12 / 1e-14 / 1e-16, 5000 iters) | **-49.4 / -48.0** | **0.356 mm** | 1161 s |
| smoothing 0 (shipped tolerances) | did not finish in 66 min | -- | -- |

Shipped settings on a 300-frame window give -40, the 1500-frame production
solve gave -19: the termination point also depends on T. Tight termination
reaches the marker optimum at ~6x the solve time.

**Not the fix:** priors on wing pitch/roll, site re-mapping, or the mask-based
wing fit (all tried before, see `2026-08-28-wingroll`, `2026-09-01-wing-mask-fit`).
**The fix is in termination** -- per-DOF-aware or tighter `cost_tolerance`,
or a final weak-DOF polish. Not implemented here.

## Second defect: shared offsets were fit on the female

`offsets.h5` was fit once per run root on whichever bout-fly arrived first;
its 500 sample frames match fly0 (female) frames 0..499 to 6e-4 model units.
The male then ran IK with her marker offsets: fitted/measured segment length
wings 1.01-1.06x, abdomen 1.10-1.12x, EyeL-EyeR 1.09x, and every abdomen joint
within 5 deg of rest (2 markers, 14 DOF, too long to fold).

### Implemented (this change)

- `scripts/offsets_sample.py` + `scripts/run_bout.py`: `offsets_fly<f>.h5`
  per fly, pooled over every triangulated bout of that fly, frames gated on
  all-finite, per-frame min conf3d >= `stac.offsets_min_conf` (0.7), implied
  body scale within 3 MAD of the fly's pooled median; stratified round-robin
  across bouts; `.json` provenance beside the h5. Per-fly requires canonical
  identity; otherwise refuse unless `stac.allow_shared_offsets`. On bout 28
  the female sample drops 365 low-conf + 125 scale-outlier frames; the male
  drops none.
- The sample is not a time series, so the fit runs with
  `JAXLS_SMOOTH_WEIGHT=0`. That exposed the solver stall above:
  `JaxlsBatchSolver` now solves independent frames (smooth_weight == 0) as a
  vmapped T=1 problem with per-frame dense factorisation and termination
  (`stac_core_jaxls._solve_independent`, tests in
  `stac-mjx/tests/unit/test_jaxls_independent_frames.py`). Timings for one
  500-frame pose solve of the offsets fit, same GPU class:

  | path | time |
  |---|---|
  | jaxls CG, smoothing 0 (auto) | stalled, >80 min |
  | chunked dense Cholesky, 50-frame chunks | 5-9 min |
  | vmapped per-frame (this change) | 3.6 s (+1.2 min one-off compile) |

  On 24 real frames from the production warm start, the vmapped path, the
  batch dense path and the production smoothed batch agree to ~2 deg of root
  orientation and 0.31-0.37 mm residual (`indep_diag3.py`).
- **Latent cache bug fixed on the way.** `_get_analyzed` keyed problems by
  (T, nq, n_kp_dim, ...) but NOT by `qs_to_opt` / `kps_to_opt` / bounds /
  sites, which are baked into the cost closures. `root_optimization` builds a
  T=1 problem that optimises only the root against the trunk keypoints; the
  first full-body T=1 solve of the independent path then reused it, froze
  every hinge and fit the root to the trunk alone (first-iteration mean frame
  error 0.81 vs 0.0036, whole body 180 deg off, offsets fit residual 1.41 mm).
  The key now carries those arrays; regression test
  `test_problem_cache_distinguishes_dof_and_keypoint_masks`. After the fix the
  per-fly fly0 offsets fit reaches 0.203 mm median residual (shared: 0.167 mm
  on the first 500 consecutive frames) with wing/abdomen offsets within 1% of
  the shared values.
- `scripts/slurm/submit_task.sh`: `XLA_PYTHON_CLIENT_PREALLOCATE=false`, no
  fixed memory fraction (user decision).

### A/B, bout 28, shared vs per-fly offsets (queue job 39607661, 46 min both flies)

`compare_perfly_offsets.py` -> `perfly_offsets_ab.json`,
`perfly_offsets_segment_ratios.png`, `perfly_offsets_stills.png`. Fitted /
measured segment length (1.0 = model matches this fly), IK stage:

| segment | female shared | female per-fly | male shared | male per-fly |
|---|---|---|---|---|
| WingL_base-WingL_V13 | 1.032 | 1.027 | 1.052 | **0.989** |
| WingR_base-WingR_V13 | 1.008 | 1.008 | 1.025 | 1.006 |
| Scutellum-Abd_A4 | 1.128 | 1.127 | 1.119 | 1.069 |
| Abd_A4-Abd_tip | 1.039 | 1.042 | 1.100 | 1.063 |
| EyeL-EyeR | 0.975 | 0.981 | 1.093 | **0.984** |
| T1L_Tro-T1L_FeTi | 1.013 | 0.994 | 1.131 | 1.021 |
| T2L_FeTi-T2L_TiTa | 0.993 | 0.992 | 1.173 | 1.043 |
| T3R_TaT1-T3R_TaT3 | 0.989 | 0.996 | 0.848 | 0.969 |
| marker residual, median mm | 0.714 | 0.726 | 0.576 | **0.414** |
| wing kp residual, median mm | 1.061 | 1.053 | 0.773 | **0.501** |
| abdomen kp residual, median mm | 0.660 | 0.613 | 0.697 | 0.702 |
| wing_pitch_right, median deg | -10.5 | -9.9 | -12.1 | -13.5 |

Read-back against the expectation:

* **Male**: every segment moves toward 1.0 (wings 1.05 -> 0.99, eyes 1.09 ->
  0.98, legs 1.13/1.17/0.85 -> 1.02/1.04/0.97); median marker residual
  **-28%**, wing residual -35%. Offsets were carrying the female's anatomy.
* **Abdomen only partly**: 1.12/1.10 -> 1.07/1.06 and the abdomen residual is
  unchanged (0.70 mm). The abdomen segment LENGTHS are in the model; site
  offsets can only slide two markers, so the remaining 6-7% is a body-model
  proportion question, not an offsets one (see segment_calibration, opt-in).
* **Female**: unchanged within noise (0.714 -> 0.726 mm), as predicted -- the
  shared file was hers. Her gated sample (365 low-conf + 125 scale-outlier
  frames removed) neither helped nor hurt this bout.
* **Wing twist unchanged** (pitch -12 -> -13.5 deg; stills from Cam2012631
  still show the folded wing edge-on in both arms). Expected: that is the
  termination issue in the section above, not offsets.
* Stills (`perfly_offsets_stills.png`, both flies x Cam2012855/2630/2631,
  incl. the female's wall frame): the two arms are visually near-identical at
  still-frame scale; the change is in the marker fit, not the silhouette.

Regenerate: `python figures/2026-09-04-wing-stop-spike/compare_perfly_offsets.py`
(run roots `pose_mvq_ik` = shared, `pose_mvq_ik_perfly` = per-fly).
