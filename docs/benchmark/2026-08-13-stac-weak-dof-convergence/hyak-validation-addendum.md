# Hyak validation addendum — 2026-08-15

Context: the stac-mjx commits carrying fix B (`ed071ec..866d209`) were never
pushed from the workstation and are unrecoverable (the workstation submodule
was later moved back to `ed071ec`, orphaning them). Fix B was therefore
**reimplemented on Hyak** from the recorded contract — the parent test
`tests/test_stac_config_tolerances.py` (asserts the real forwarding through
`Stac.__init__` → `StacCore` → `JaxlsBatchSolver` attributes), the anatomy
config comments, and this notes file. All 7 contract tests pass.

## The required silhouette-divergence check: run, and the premise is dead

`test_no_silhouette_matches_jaxls_batch_solver`'s fixture
(`cse_work/2026_03_18_15_31_22/Fruitfly_ik_v1_cse.h5`) no longer exists, so
the same comparison was run inline against a fresh v1 free-running IK h5
(NewBouts `session1/2025_10_07_17_15_30`), T=6 frames, n_iter=30,
smooth_weight=0.1:

| comparison | max abs qpos diff (rad) |
|---|---|
| JaxlsBatchSolver vs SilhouetteJaxlsBatchSolver (bare sibling) | 6.599e-03 |
| same, after giving the sibling matching explicit tolerances   | 6.599e-03 |
| **JaxlsBatchSolver vs a second fresh JaxlsBatchSolver**       | **6.599e-03** |
| SilhouetteJaxlsBatchSolver vs a second fresh one              | 6.557e-07 |
| cross at n_iter=200                                           | 1.912e-01 |

The solver does not reproduce **itself** across separate `analyze()` calls:
jaxls vectorizes the cost groups in a different (hash-random) order on every
analyze — observed orders across six analyzes in one process:
`[marker, limit, smooth]`, `[smooth, limit, marker]`, `[marker, smooth,
limit]`, … Different summation order → different float paths → LM walks the
near-flat weak-DOF valley (wing blade-roll: 0.010 kp error ≈ 24 deg) to
different, equally-optimal points.

Conclusions:

1. The `atol=1e-5` agreement the test asserts is **not a well-defined
   property** of these solvers on ill-conditioned real data. Its historical
   pass was an artifact of the loose pre-fix tolerances early-terminating
   both solvers at the same iterate before float-path chaos could amplify.
2. `SilhouetteJaxlsBatchSolver` was nonetheless given the same explicit
   tolerances (`third_party/jarvis_jax/jarvis_jax/tracking/
   silhouette_joint_ik.py`), per this file's "Known divergence" prescription —
   the two solvers' *configurations* stay in lock-step even though exact
   numerical agreement is unattainable.
3. The formulations were verified identical line-by-line (variable classes,
   marker/reg/limit/smoothness factories, init normalization, linear-solver
   selection, trust region).
4. Follow-up if exact reproducibility is ever needed: pin jaxls' cost-group
   ordering (sort by name before analyze) upstream; run-to-run weak-DOF
   variation (~1.2 deg documented in the main notes) has the same root cause.
