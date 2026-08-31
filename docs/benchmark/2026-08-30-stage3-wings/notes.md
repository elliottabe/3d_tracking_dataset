# Does stage-3 gated refine remove the male WingL/R_V13 mirror-confusion artifact? (bout 28)

Diagnosis only, no pipeline changes. Full methodology, trade-off table, and
figure-by-figure readback are in the (untracked) task report; this is the
committed summary + pointer.

Data: Session0 `2025_10_20_13_20_04` bout 28, archived arrays at
`figures/2026-08-29-c2f-3d/phase0-baseline/arrays/bout_00028/fly{0,1}/`
(fly1 = male). Tested mechanism: `refine_from_seed` in
`third_party/jarvis_jax/jarvis_jax/tracking/triangulate.py` (unmodified),
seeded with the archived `kp3d.npz`. Figures (gitignored, regenerate via
the scratch scripts named in the full report):
`figures/2026-08-30-stage3-wings/`.

## Headline result

**No usable margin at any `gate_px` from 5-30 px.** Artifact suppression
is 0% (bit-for-bit identical refined-vs-raw `WingL_V13` angle at the
confirmed artifact peak frames, for every `gate_px` from 8-30; a
negligible -0.3% at `gate_px=5`) while real bilateral-flick/sweep
amplitude is correspondingly ~100% preserved. The trade-off curve is flat
on both axes — there is no crossing point because neither side moves.

**Why:** the artifact is a **majority-vote flip**, not a lone rogue
camera. Measured directly: at the confirmed artifact peak frames, the
seven cameras split into two clusters ~170-205 px apart, and 4-5 of 7
agree with whichever cluster the seed (built from the same per-frame
consensus) already picked — membership flips frame to frame (e.g.
`Cam2012853`/`Cam2012855` agree with the majority at offset 1 but are the
rejected minority at offsets 3, 5, 7, 9). `refine_from_seed`'s gate can
only reject views that disagree with the seed's own reprojection; since
the seed always reflects that frame's majority, and `gate_px` 5-30 sits
far below the ~170-205 px cluster separation, no setting in that range —
or arguably any range short of ~200 px, which would defeat the gate
entirely — changes which cluster wins. Stage 3 re-derives the same answer
Stage B's own consensus rejection already gave it.

## Female (fly0)

Briefly checked: her wing keypoints are NaN in the seed 20.6% of the time
bout-wide (vs 0.0% for the male), and the `min_views` fallback rate on
`WingL_V13` sits at ~29% and is nearly flat across the same `gate_px`
sweep (29.4%→28.6%, gate 5→30). Flat-with-gate-width is itself the
diagnostic: her binding constraint is genuine view scarcity (too few
cameras clear `conf_thresh` at all), not gate mistuning — a structurally
different failure mode from the male's majority-vote flip, and stage 3
cannot address either one here.

## Bottom line

Stage-3 gated refine, run standalone on the currently-available Stage-B
seed, does not make the male's wing trace frame-precise. It is a clean,
measured negative result, not a tuning problem: the artifact and the
usable gate range never overlap because the artifact isn't the kind of
disagreement (single-outlier-vs-majority) this gate is built to catch.
