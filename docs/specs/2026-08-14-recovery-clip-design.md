# Recovery Clip — Design

## Context

The IK explainer (`docs/specs/2026-08-13-ik-explainer-animation-design.md`,
delivered as `<CLIP>/ik_explainer/ik_explainer.mp4`) shows the pipeline working.
It does not show the pipeline *recovering* — a moment where 2D detection fails
and the downstream stages absorb the error.

That moment exists in this bout and is worth its own clip, because "seven
cameras plus temporal filtering plus anatomical constraints make the fit robust"
is a claim the explainer asserts structurally but never demonstrates on a real
failure.

`<CLIP>` = `/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46`.

## The event — located by measurement, not by eye

Three independent measures pick out the same instant, which is what makes this a
causal story rather than an anecdote.

**1. The worst 2D failure in the bout.** Comparing each camera's detector output
against the reprojection of the raw triangulated 3D over all 921 frames × 7
cameras × 50 keypoints: median disagreement **2.52 px**, p99 **18.9 px**,
max **122 px**. The maximum is `T1R_TaTip` on `Cam2012853` (**Camera 3**) at
**frame 441**, with detector confidence **0.49**.

Every one of the twelve largest disagreements is a distal tarsal point
(`*_TaTip`, `*_TaT3`, `*_TaT1`) at confidence 0.35–0.63. The detector knows it
is lost; it does not fail silently.

**2. The worst raw-3D spike.** `*_TaTip` frame-to-frame acceleration:

| stage | median | p99 | max |
|---|---|---|---|
| raw triangulation | 0.0158 | 0.304 | **1.627** |
| filtered | 0.0063 | 0.061 | 0.116 |
| IK (FK'd marker sites) | 0.0057 | 0.057 | 0.092 |

(mm/frame²). The maximum is `T1R_TaTip` at **frame 443**.

**3. At that spike, the recovery is measurable:**

| stage | accel at f443 | suppression |
|---|---|---|
| raw | 1.627 | — |
| filtered | 0.022 | **75×** |
| IK | 0.015 | **110×** |

Sustained over frames **422–462**: raw mean 0.078, filtered 0.019, IK 0.017.

**Window for the clip: frames 420–465** (45 source frames), centred on the
failure.

### What the data does NOT support

At this instant the other six cameras also disagree with the consensus. Each
camera's own 2D detection vs the reprojection of the **raw** triangulation, for
`T1R_TaTip` (px, measured at render time by `recovery_event.load_tracks`'s
`cam_disagree`; recomputed independently in
`tests/test_recovery_event.py::test_cam_disagree_matches_a_direct_recomputation`):

| slice | …630 | …631 | **…853** | …855 | …857 | …861 | …862 |
|---|---|---|---|---|---|---|---|
| at the peak frame **441** | 44.4 | 10.8 | **122.2** | 41.2 | 19.6 | 21.0 | 32.6 |
| per-camera peak over the window 420–465 | 76.9 | 20.7 | **122.2** | 120.8 | 54.9 | 52.1 | 41.8 |

`Cam2012853` (bold) is the camera on the clip's left panel. **At frame 441 the
other six span 11–44 px** — that is the range the clip states on screen, and it
is computed at render time, not quoted from here. (An earlier revision of this
document claimed "peaks **83–121 px**"; no slice of the data yields that. It
overstated the disagreement. Note also that neither row is a like-for-like
comparison with the 162.6 px quoted elsewhere for this camera, which is
detector-vs-**filtered** reprojection rather than detector-vs-raw.)

Even so, none of the six is *on* the consensus. This is **not** "one camera
failed and six rescued it." The
recovery comes from triangulation absorbing part of the error and then the
temporal filter and IK's anatomical constraints absorbing the rest. The clip
must say so on screen; the cleaner story would be a more satisfying one and is
not the true one.

## Deliverable

`<CLIP>/ik_explainer/recovery_clip.mp4` — **1920×1080, 30 fps, H.264 `yuv420p`,
faststart, silent, 360 frames = 12.0 s**. Standalone; plays on its own or after
the explainer.

Generator committed at `scripts/viz/ik_explainer/recovery_clip.py`. Frames to
`<CLIP>/ik_explainer/frames/recovery_clip/`. The mp4 and frames are artifacts and
are not committed.

**Playback rate:** 45 source frames over 360 output frames = 8 output frames per
source frame. At 800 fps source and 30 fps output that is **1/213 real time**.
The spike spans source frames 441–443, i.e. **24 output frames ≈ 0.8 s** — slow
enough to read. The on-screen speed label must state the computed value, not a
copied one.

## Layout — 2-up (left column is a 3-camera stack)

**Left panel: what the detectors saw.** A vertical stack of **three** camera
views of the same instant, enhanced footage, each cropped to the right foreleg.
Two markers on `T1R_TaTip` in every panel:

- the **detector's own 2D**, in its JARVIS chain colour
- the **reprojection of the filtered 3D**, in white

The top panel is the event camera, `Camera 3` (`Cam2012853`). The two below are
chosen by measurement (`recovery_event.select_panel_cams`) as the **worst** and
the **cleanest** of the remaining six by peak disagreement over the window —
here `Camera 2` (`Cam2012855`, peak 120.8 px) and `Camera 6` (`Cam2012631`,
peak 20.7 px). One camera alone can only show *that* the detector failed; the
stack is what shows the failure is **shared but uneven**, which is what the
caveat claims. Each panel carries its own live detector confidence: at frame
441 they read **0.49 / 0.92 / 0.95**. Both markers are legended by name once,
on the top panel.

All three panels **share one crop size and therefore one px/mm**. This is a
correctness requirement, not a cosmetic one: sizing each panel to its own
markers zooms the cleanest camera ~4× against the event camera's ~1×, drawing
its ~20 px miss *larger* on screen than the event camera's 122 px one and
inverting the comparison the panel exists to make. Asserted by
`test_all_camera_panels_share_one_zoom`.

At frame 441 the top panel's detector marker leaps **162.6 px** off the foot
while its reprojected marker stays on it; the middle panel shows a visible but
smaller separation; the bottom panel's two markers are still essentially
together. Note 162.6 px is a different measurement from the 122.2 px above: the
panel draws the detector's 2D against the reprojected **filtered** 3D, while
the event was selected on detector vs the **raw** triangulation.

**Right panel: what the pipeline did about it.** A time-series over frames
420–465 with three traces — **raw triangulation**, **filtered**, **IK-fitted** —
and a playhead locked to the left panel's frame.

Plot the **3D coordinate with the largest raw excursion**, chosen automatically
and named on screen, in mm. Position rather than acceleration: a position kink is
legible without explanation, where an acceleration curve needs one. The
acceleration numbers are annotated at the spike instead —
`1.627 → 0.022 → 0.015 mm/frame²`, labelled `75× / 110×`.

## Architecture

One module, `scripts/viz/ik_explainer/recovery_clip.py`, plus one new drawing
primitive.

| Concern | Source |
|---|---|
| DLT, enhanced video, camera names/labels | `clip_io` |
| markers, labels, fades, type scale | `draw` |
| per-chain keypoint colours | `kp_colors` |
| detector 2D + confidence | `predictions/02_kp2d.npz` |
| raw 3D | `predictions/03_kp3d.npz` |
| filtered 3D | `predictions/04_kp3d_filt.npz` |
| IK marker sites | `ik_production/stac_ik_full.h5` (`marker_sites`, ÷ `shared_scale` → mm) |

**No new computation.** Every number in this spec is already measured from
artifacts on disk; the clip reads and renders them.

**The one new primitive** is the time-series panel — a small line-chart renderer
(axes, three traces, playhead, annotation). It belongs in `recovery_clip.py`
unless a second consumer appears; promoting it to `draw.py` speculatively would
be building for a caller that does not exist.

### Conventions inherited from the explainer

- Camera **display** labels are `Camera 1`–`Camera 7` by arc position;
  `Cam2012853` is **Camera 3** (60°). All indexing stays keyed to the real
  `Cam20128xx` names — display order must never reach data selection.
- Keypoint arrays are **MODEL order**; colours map by keypoint **name**.
- Display footage is `enhanced/`; detection ran on the raw videos.
- Shared type scale: title 1.6, caption 0.65, small 0.5.

## Verification

Stated before rendering, checked by reading the frames afterwards.

| Check | Expectation | Falsification |
|---|---|---|
| Failure is visible | at f441 the detector marker sits ~163 px off the foot (detector vs reprojected filtered 3D) while the white reprojected marker stays on it | both markers together ⇒ wrong camera, keypoint, or frame mapping |
| Recovery is visible | the raw trace spikes at f443; filtered and IK stay smooth through it | all three spiking ⇒ wrong arrays plotted |
| Panels agree | playhead frame == left panel's source frame at every output frame | drift ⇒ two independent indices; use one |
| Caveat present | the "other six cameras also disagree" line is on screen | absent ⇒ the clip overstates the mechanism |
| Encoding | `yuv420p`, faststart, 1920×1080, 30 fps, 360 frames | — |

Rendering needs `MUJOCO_GL=egl` only if a 3D view is added later; as specified
this clip is OpenCV-only and CPU-bound, so it can render alongside GPU work.

## Risks

1. **The chart is a different visual language** from the rest of the video, which
   is markers-on-footage throughout. Mitigation: match the type scale and the
   per-chain colour for the traces so it reads as the same family. If it still
   looks alien, the fallback is the zoomed-leg variant considered during design
   (three overlaid markers, no chart) — that loses the quantitative punch.
2. **One keypoint, one moment.** This is a case study, not evidence about the
   pipeline in general. The clip should not be captioned as if it were.
3. **Easy-case caveat inherited from the explainer.** A single fly in an open
   arena is not where this pipeline struggles; CLAUDE.md records the female on
   walls as the hard case. A recovery shown here is real but not representative
   of the hardest conditions.

## Open decisions (assumed, not confirmed)

- 12 s at 1/213 real time.
- Position rather than acceleration on the trace.
