# Does 2D leg/wing mirror confusion survive into 3D? (bout 28 diagnosis)

Diagnosis only, no pipeline changes. Full methodology, per-task numbers, and
figure-by-figure readback are in the (untracked) task report; this is the
committed summary + pointer.

Data: Session0 `2025_10_20_13_20_04` bout 28 (2007 frames, 7 cams), archived
arrays at `figures/2026-08-29-c2f-3d/phase0-baseline/arrays/bout_00028/`.
fly0=female, fly1=male. fly0 offsets >=1500 (coverage dropout) excluded from
headline stats. Figures (gitignored, regenerate via the scratch scripts
listed in the full report): `figures/2026-08-30-mirror-in-3d/`.

## Headline numbers

- **2D mirror-confusion rate** (among 2D errors >30px, the repo's own
  `TRIGGER_FACTOR x reproj_resid_px` swap cut): fly0 (female) 53.3%
  land closer to the mirror keypoint's reprojection than their own
  (legs 53.6%, wings 49.1%, n=45535); fly1 (male) 63.9% (legs only 46.0%,
  **wings 94.4%**, n=5839 — legs dominate the *count* of large errors for
  both flies but not the mirror *rate*).
- **Stage-B gate (`reproj_resid_px=10`, already shipped) already rejects**
  77.4% of the female's mirror-confused views and 100.0% (0/3732) of the
  male's. **22.6% of the female's mirror-confused views still survive into
  the DLT solve.**
- **3D cost of survivors**: a surviving mirror-confused view moves the
  female's fitted 3D point a median **6.0 mm** (mean 8.9, p90 19.1, max
  122 mm) vs. a **0.16 mm** ordinary one-view-dropout noise floor for clean
  points — **~38x**. Worst on wings (median 20.9 mm). Zero survivors for
  the male, so zero measured 3D cost there.
- **Stage-3 `refine_from_seed`** (defaults `gate_px=15, min_views=2`,
  seeded from current `kp3d`) would reject ~6x more views than the
  mirror-specific ones alone (34360 vs 5497 for the female), but **89.2%
  (2880/3229) of the female's contaminated points fall back to the seed
  unchanged** because too few clean views remain under the default
  `min_views=2` — real headroom exists but isn't realized by the defaults
  as configured.
- **Per-fly split**: female's disadvantage is only partly mirror confusion
  — she also has an ~11x higher base rate of large 2D errors overall
  (9.8% vs 0.9%) and 46.7% of her large errors are NOT mirror-type. Male's
  residual error is essentially untouched by mirror confusion (gate already
  handles it).

## Verdict

A meaningful amount of the female's mirror confusion **does** survive into
3D (median 6 mm/instance, ~38x the noise floor) and stage-3 gating has real
but currently-unrealized headroom to catch more of it — so 3D-side gating
work here is not wasted. But mirror confusion explains only about half of
her large-2D-error rate and essentially none of the male's residual error,
so **2D-side effort is not misdirected either** — both matter, and neither
this bout's data nor this analysis says to drop one for the other.

Full report: `.superpowers/sdd/2026-08-29-coarse-to-fine-3d/mirror-in-3d.md`
(untracked/gitignored — this file is the durable, committed pointer).
