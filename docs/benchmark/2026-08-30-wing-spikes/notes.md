# Are the sharp excursions in the male's wing-angle timeline real or artifacts? (bout 28)

Diagnosis only, no pipeline changes. Full methodology, per-spike evidence,
and figure-by-figure readback are in the (untracked) task report; this is
the committed summary + pointer.

Data: Session0 `2025_10_20_13_20_04` bout 28 (2007 frames, 7 cams), archived
arrays at `figures/2026-08-29-c2f-3d/phase0-baseline/arrays/bout_00028/fly1/`
(fly1 = male). Figures (gitignored, regenerate via the scratch scripts
listed in the full report): `figures/2026-08-30-wing-spikes/`.

## Headline result

The six named spikes in `fig4_wing_extension_timeline_male.png` split into
**two genuinely different phenomena**, distinguished by checking raw
per-camera 2D pixel traces and the actual video frames (not just the
reprojected 3D angle):

- **Offset ~0-20 (WingL 34°→85° "instant" jump): ARTIFACT.** The 2D
  detector alternates, frame to frame, between two real, sharp,
  high-confidence (0.91-0.94) candidate points that sit on **two different
  visible wing structures already present in the same frame** — direct
  frame-diff confirms the wing itself does not move between the "low" and
  "high" reads. This is a fine-grained (per-frame, tip-landmark) version
  of the 2D wing mirror-confusion already characterized in
  `male-wing-mirror.md`; it recurs through roughly offset 0-130, the fast
  onset of the male's first wing-extension bout.
- **Offset ~360-375, ~680-700 (smoother swings): REAL.** Continuous,
  single-valued 2D pixel drift (no bimodal toggle) plus a visible,
  coherent wing-edge displacement in the frame-diff. This is the settled,
  ~10-30-frame-period oscillation that dominates most of offset 130-750
  once the initial-transition artifact above has resolved — plausibly
  wing-vibration song content or a slow postural adjustment.
- **Offset ~1500-1560, ~1810-1870, ~1930-1990 (erratic bilateral bursts):
  REAL.** A recurring, distinctive 3-4-frame signature — both `WingL` and
  `WingR` angle jump up together, then undershoot below baseline together
  — that frame-diff confirms is a genuine brief (~4-5 ms) **bilateral
  wing flick**, both wings visibly spreading and retracting together in
  the dorsal-view cameras. One instance also shows visible front-leg
  motion in the same frames.

## The proximity/female-view-loss lead: refuted for the male

The male's own post-gate view count for every wing keypoint stays **≥5/7,
almost always 7/7**, through all six windows, including the entire
1500-1990 span where the female (fly0) is frequently lost to <2 cameras
(NaN). The one real view-count dip (3-5/7) sits at offset 0-130 — the
artifact window — and tracks which wing is actively extended, not
inter-fly distance. The three real bilateral-flick clusters happen while
the female is mostly far (up to 68 mm) or fully NaN — i.e. the male is
tracked *better*, not worse, exactly when these occur. The two phenomena
are temporally near each other in this bout by coincidence, not causation.

## Does smoothing or IK remove either class?

No, for either class. The pre-STAC savgol `keypoint_filter` barely changes
either class of spike at this timescale (within 0.3-1.5° of raw at every
tested frame). The IK output (`outputs.h5`'s `kp3d_mm`) preserves the real
bilateral flicks' shape/amplitude closely (Δ17.7° raw vs Δ17.8° IK for one
event) but does not cleanly remove the artifact spike either — it reshapes
it inconsistently (sometimes overshoots raw, sometimes undershoots by
~20°) rather than eliminating it. Both classes reach the qpos-derived IK
output essentially intact.

## Verdict

The male's wing-angle trace's **coarse structure is trustworthy**
(consistent with the prior identity diagnosis's 0/2007 L/R swap result):
which wing is extended, and roughly when, is correct. But at **fine
(few-frame) timescale**, a real, identifiable, uncorrected 2D
tracking artifact (near/far-wing landmark confusion, concentrated at the
fast onset of active wing extension) inflates the apparent angle by up to
~50° in short bursts, and this is **not** removed by the pipeline's
existing smoothing or IK steps — so the trace is not usable as-is for any
analysis needing frame-level amplitude/timing precision (counting or
timing individual wing events), though it remains usable for coarse
behavioral classification.

Full report: `.superpowers/sdd/2026-08-29-coarse-to-fine-3d/wing-spikes.md`
(untracked/gitignored — this file is the durable, committed pointer).
