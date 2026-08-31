# 8-arm 3D training results: figures and what they show

Val set: `red_data_3d_v5` val split, 216 framesets (18 female fly-samples,
198 male). All numbers below are **val MPJPE in world units** (the
reprojection grid's own coordinate system, `grid_spacing=1`); **1 world
unit ≈ 0.17 mm** (`jarvis_jax/hybridnet/refine.py` docstring). All 8 arms
ran the full 20000 steps, zero NaN, all plateaued.

Figures: `figures/2026-08-30-arm-results/` (gitignored; `.json` companion
beside every `.png` with the exact numbers). Generating scripts:
`scripts/viz/voxel_resolution.py` (fig 1, already committed) and three
one-off scripts kept in this session's scratchpad (figs 2-4 — no numbers
in the pipeline moved, these are read-only analyses of finished runs, so
per CLAUDE.md they don't need to be promoted into the repo).

## Figure 1 — the falsification gate (`fig1_voxel_resolution_{A1_base,A4_c2f_aug}.png`)

**Verdict: the resolution hypothesis HOLDS. Error is NOT flat across
segment length** — it falls off sharply and monotonically from short
(tarsal) to long (body/wing) segments, in both arms:

| arm | short segments (<2.2 vox, n=3) mean err/length | long segments (≥4 vox, n=5) mean err/length | ratio |
|---|---|---|---|
| A1_base | 0.2215 | 0.0688 | 3.22x |
| A4_c2f_aug | 0.2011 | 0.0627 | 3.21x |

The three tarsal links (`T1L_TiTa->T1L_TaT1` 1.97 vox, `T1L_TaT1->T1L_TaT3`
1.95 vox, `T1L_TaT3->T1L_TaTip` 1.68 vox) sit far above every other segment
in both plots; the body (`Scutellum->Abd_tip`, 12.9 vox) and wing
(`WingL_base->WingL_V12`, 17.2 vox) segments sit at the bottom. That is the
premise the project was built on, and it is confirmed on real held-out data.

**But coarse-to-fine does NOT selectively close that gap.** A1→A4 lowers
both bins by about the same fraction (short −9.2%, long −8.9%), so the
short/long RATIO barely moves (3.22x → 3.21x). Refine is buying a broadly
uniform ~9% relative improvement, not a targeted fix of the tarsal
quantization problem specifically — see Figure 2 for where the gain
actually concentrates.

## Figure 2 — per-joint error, A1 vs A4 vs A7, grouped (`fig2_per_joint_groups.png`)

Panel A (5 anatomical groups; n is the number of keypoints in each group —
head/thorax/abdomen are thin bins and read noisier than legs/wings):

| group (n) | A1_base | A4_c2f_aug | A7_c2f_aug_femwt |
|---|---|---|---|
| head (3) | 0.381 | 0.368 (−3.6%) | 0.363 |
| thorax = Scutellum only (1) | 0.427 | 0.535 (**+25.3%, worse**) | 0.492 |
| abdomen (2) | 0.558 | 0.613 (**+9.9%, worse**) | 0.586 |
| wings (6) | 0.526 | 0.502 (−4.5%) | 0.513 |
| legs (38) | 0.441 | 0.423 (−4.2%) | 0.423 |

**Finding, stated plainly because it disagrees with the "all the gain is in
the tarsi" framing:** the ~3% overall A1→A4 gain is not a tarsus story at
all. Legs and wings improve modestly and fairly *uniformly* (Panel B: every
leg segment from `ThxCx` to `TaTip` improves by about the same 3-6%, no
distal concentration), while `Scutellum` (+25%) and `Abd_A4` (+22%,
`Abd_tip` flat) get **worse** under both A4 and A7. The net ~3% is legs/wings
gains partly offset by a thorax/abdomen regression. `EyeL` is the single
biggest per-joint improver (−16.9%); `Scutellum` is the single biggest
regressor (+25.3%). A7 (female-upweighted) sits between A1 and A4 on
thorax/abdomen — it doesn't fix the regression, just softens it slightly.

## Figure 3 — 8-arm comparison with the noise floor (`fig3_arm_comparison.png`)

Noise floor `|A4 − A8| = 0.00556` world units. Panel A (log scale, all 8)
makes A5_hires's 5.7x blowup (3.218 vs ~0.56) immediately obvious against
every other arm — a single-stage 96³/spacing-0.5 grid is not a viable
substitute for coarse-to-fine. Panel B (7 arms excluding A5, linear, noise
band shaded) shows which of the remaining deltas are real:

- **Within noise of A4** (band, marked `*`): A6_c2f_aug_norecover
  (Δ=0.0002, 0.04x noise) and A8_c2f_aug_seed2 (Δ=0.0056, 1.0x noise by
  definition). Dropping the 264 recovered second-fly framesets (A6) makes
  **no measurable difference**.
- **Outside noise, real**: A1_base (Δ=+0.0168, 3.0x noise), A2_base_aug
  (Δ=+0.0277, 5.0x noise — rotation aug alone, *without* refine, is
  actually the single worst non-A5 arm), A3_c2f (Δ=+0.0106, 1.9x noise).
  Coarse-to-fine (A3→better than A1/A2) and adding aug on top of it (A4)
  are both real, if small, effects.
- **A7_c2f_aug_femwt**: Δ=−0.0057 vs A4, **1.02x the noise floor** using
  full-precision numbers (the task brief's "1.2x" uses the 3-decimal
  rounded values 0.555/0.561/0.005) — a marginal, borderline-real
  improvement from female upweighting, not a clear win.

## Figure 4 — 3D skeleton overlays, worst-error frames (`fig4_skeleton_overlay_worst.png`)

Per sex, the val frameset with A1_base's highest mean per-joint error
(not a flattering pick):

- **Female** — `2026_06_19_11_09_36` frame `431686` fly0 (val idx 215).
  A1_base mean err 2.21u → A4_c2f_aug 2.18u (−1.6%). Dominated by **leg**
  mistracking: `T2R_TaTip` 7.7u, `T2R_TaT3` 7.0u, `T1L_TaTip` 6.9u — a
  right-mid-leg and left-front-leg tarsus snapped to the wrong place in
  both arms; A4 barely moves it.
- **Male** — `2026_06_15_12_12_33` frame `464871` fly0 (val idx 171).
  A1_base mean err 3.86u → A4_c2f_aug 3.94u (**+2.1%, A4 is worse here**).
  Dominated by a **wing mirror**: `WingL_V12` 41.6u, `WingR_V12` 39.9u,
  `WingL_V13` 37.5u, `WingR_V13` 20.3u — all four wing-tip joints are
  off by tens of world units (bigger than the whole wing, 17u), visible in
  the render as a green/orange marker flying far off the body. This matches
  the standing wing-mirror finding in
  `docs/benchmark/2026-08-30-mirror-in-3d/` (male wing mirror rate 94.4%
  among >30px 2D errors); this frame is a 3D instance of that same failure,
  and coarse-to-fine does not fix it (it isn't a resolution problem, it's a
  wrong-correspondence problem upstream).

I opened all 5 PNGs with the Read tool before writing any of the above.

## Bottom line

The core resolution hypothesis is confirmed on real data (short segments
carry ~3.2x the normalized error of long ones, in both arms) — Figure 1
does not refute the project's premise. But coarse-to-fine's actual ~3%
mean win (Figures 2-3) is a small, broadly-distributed leg/wing
improvement partly offset by a thorax/abdomen regression, not a
concentrated fix of the tarsal quantization problem, and it does nothing
for the two largest real failure modes on record (leg mistracking on the
female, wing mirroring on the male — Figure 4). A5_hires (single-stage
high-res) is decisively worse, not a viable alternative. A6 shows the
second-fly recovery was free. A7's female upweighting is a marginal,
near-noise-floor win at best.
