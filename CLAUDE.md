# CLAUDE.md

Pipeline that turns raw multi-camera fly video into 3D keypoints and an
articulated STAC/IK fit: SAM3 masks → ViTPose 2D keypoints → triangulation →
smoothing/bridging → STAC IK → polish. For how to run it, see
`docs/running_the_pipeline.md`; batch details in `BATCH_PROCESSING.md`.

## Visualize what you change

If a change moves numbers — detector training or heatmap decode, SAM3 masking
or identity assignment, triangulation/DLT, keypoint filtering or bridging,
body-scale estimation, STAC/IK, marker offsets or anatomy configs, silhouette
polish, sexing/canonicalization, camera sync, calibration — it is not done
until a figure (or reprojection video) shows what it did on real frames.
Metrics confirm a number matches; a figure is how you notice it matches for
the wrong reason. This pipeline's history is exactly that: LOO/IoU QC was
fully blind to a keypoint-order bug that scrambled anatomy, residual and NaN
checks were blind to marker offsets absorbing a 38x body-scale error — both
were caught only by looking at rendered overlays. Changes that provably move
no numbers (config plumbing, path/IO utilities, behavior-preserving
refactors, docs, tests-only edits) don't need one.

**State the expectation before generating the figure.** Write down what the
plot should look like if the change is correct, in the script docstring or
the commit note: *"after the bridge fix, cyan fitted markers sit ON the mesh
through the occlusion gap; markers detaching from the leg chain there means
the bridge is still following the mask centroid."* A figure you can't be
wrong about proves nothing.

**Then make it comparative.** Put the change next to something — before
vs. after, fitted vs. observed, variant vs. the frozen benchmark baseline.
The tools are built for this: `python -m viz overlay|legskel --compare
<other_run>`, `scripts/viz/compare_stac_fits.py --fit baseline=… --fit
fixed=…`, `scripts/viz/scale_check.py` for candidate scales side by side,
and the benchmark suite (`scripts/benchmark/`, frozen 13-bout set) for the
numeric A/B — a scorecard delta plus a figure of *why* it moved is the full
story. A lone render of new behavior is decoration.

**Check the hard case, not the flattering one.** The female fly (walls,
occlusion, OOD poses) is where this pipeline fails; the male usually looks
fine regardless. A figure that only shows fly1/male, one easy camera, or one
mid-bout frame overstates the change — include the female, an occlusion or
wall frame, and more than one camera.

**Never index a keypoint or camera axis by integer.** Two different index
spaces exist for each, they disagree, and picking the wrong one produces
confident, self-consistent, completely wrong numbers that every jitter,
confidence and residual metric rates as GOOD. Both traps were hit in one
sitting on 2026-08-31 and took four rounds of analysis plus a human pointing
at the picture to find:

- *Keypoint order.* `kp2d.npz`/`kp3d.npz` are written AFTER
  `reorder_detector_to_model` (`scripts/run_bout.py`), so their keypoint axis
  is `cfg.model.KP_NAMES` (XML/model order), NOT `cfg.detector.kp_names`. The
  two differ. Indexed with detector indices, `WingR_base/V12/V13` (28/29/30)
  read model `T2L_TiTa/TaT1/TaT3` — the middle-LEFT LEG. That yielded a
  "collapsed right wing vein, 2.53u vs 21.58u, CV 4.1%, confidence 0.96":
  a correct measurement of the wrong thing, and the *best-looking* keypoint in
  the table.
- *Camera order.* `kp2d`'s camera axis is the canonical
  `cfg.recording.cameras` order (== the calibration glob order), because
  `load_bout_masks(..., expected_cameras=...)` reorders the mask npz BY NAME.
  The npz's own stored `cameras` array is a DIFFERENT order. Using the npz's
  index against `kp2d` or `ReprojectionTool.camera_matrices` plots one
  camera's keypoints on another camera's image — and the wing points still
  looked *almost* plausible, which is what made it survive a first fix.

Use `viz.core.bout_artifacts.load_bout_kp(...)`, which takes keypoints and
cameras BY NAME and refuses a detector config; `centroids_canonical()` for
mask-npz arrays. Guarded by `tests/test_bout_artifacts_order.py`.

**Check a rigid invariant, not just smoothness.** The thing that finally
exposed the above was that a rigid wing vein's length has to be CONSTANT.
Jitter, spike rate, confidence and multi-view residual were all excellent on
the mislabelled points — a collapsed pair is *smoother* than a real landmark.
Every quantity with a known physical invariant (rigid segment lengths, bone
lengths, left/right symmetry) should be checked against it before any
conclusion, and reported with the invariant named.

**Label with real names and units.** Reprojection error in px, positions in
mm, angles in deg, time in frames; keypoint names (`T1L_FeTi`), camera names
(`cam1`…`cam7`), and fly identities (`fly0`/`fly1`, male = fly1 after
canonicalization) in legends — never bare array indices, which is how the
keypoint-order bug stayed invisible. Reuse the shared visual language in
`viz/core/colors.py` (`PALETTE`, `keypoint_groups`, `leg_chains`) so figures
across the repo stay readable together: white/cyan = observed/detector,
green = fit, orange = fly1, grey = mask/mesh.

**Read the figure back before claiming anything.** Open the PNG with the
Read tool, look at it, and report what it shows against the stated
expectation — especially when it disagrees. For videos, extract and Read a
few frames. A figure that was generated but never viewed is not evidence,
and "the plot confirms it" without having looked is a false claim about
verification.

**Where things land.** All figures, renders, and videos go under
`figures/<topic>/` (e.g. `figures/2026-08-07-scale-ab/`), which is
gitignored along with `OutFiles/` — never commit PNGs or videos to the repo.
Save the generating `.json`/`.npz` beside the PNG so it can be re-examined
without re-running. Since the images themselves aren't versioned, the
generating script is the durable artifact:

- One-off diagnostics: keep the script in the scratchpad, outputs under
  `figures/<topic>/`.
- A figure worth regenerating (verification suites, acceptance gates):
  promote the script to `scripts/viz/` or a `python -m viz` subcommand
  (`viz/views/`) and commit it with the change. Existing examples:
  `scripts/viz/compare_stac_fits.py`, `scripts/viz/scale_check.py`.
- Evidence for a decision (A/B results, retrain evaluations): commit only
  the small text artifacts — scorecard `.json`/`.md`, comparison notes —
  under `docs/benchmark/<date>-<topic>/`, and reference the matching
  `figures/<date>-<topic>/` directory (plus the command that regenerates
  it) from the notes so the images can be recreated on demand.

Rendering practicalities: MuJoCo renders need `MUJOCO_GL=egl` (the promoted
scripts set it) and a GPU node — never run renders or other heavy work on
the Hyak login node; when already on a GPU node, run directly rather than
sbatch-ing.
