# Recovery Clip Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A 12-second standalone clip showing a real 2D detection failure and the triangulation + filter + IK recovery that absorbs it.

**Architecture:** One module reads already-computed artifacts and renders a 2-up composite — `Camera 3` footage with two markers on the left, a three-trace time-series with a playhead on the right — then encodes it. No new computation: every figure is measured and on disk.

**Tech Stack:** Python 3.12, NumPy, OpenCV, h5py. CPU-only (no MuJoCo, no GPU).

**Spec:** `docs/specs/2026-08-14-recovery-clip-design.md` — read it first. It records why the event was chosen and what the data does *not* support.

## Global Constraints

- **`CLIP` = `/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46`.** Raw inputs (`calibration/`, `Cam*.mp4`, `enhanced/`, `data3D*.csv`, `ik_production/`) are **read-only**. All writes under `<CLIP>/ik_explainer/`.
- **Output:** `<CLIP>/ik_explainer/recovery_clip.mp4` — 1920×1080, 30 fps, H.264, **`yuv420p`**, faststart, no audio, **360 frames = 12.0 s**. `yuv420p` is what makes it play in VSCode; do not change it.
- **Frames** to `<CLIP>/ik_explainer/frames/recovery_clip/f%05d.png`. Write PNGs with `[cv2.IMWRITE_PNG_COMPRESSION, 1]` (the convention the acts use).
- **Event:** keypoint `T1R_TaTip`, camera `Cam2012853` (**display name `Camera 3`**, arc 60°), source frames **420–465** (45 frames). Failure peak at source frame **441** (122 px off consensus, detector confidence 0.49); raw 3D spike at **443** (1.627 mm/frame²).
- **Playback:** 45 source frames over 360 output frames = **8 output frames per source frame** = **1/213 real time**. Compute the speed factor in code and put the computed value on screen — never a copied literal.
- **Keypoint order is MODEL order** everywhere (`clip_io.model_kp_names()`); colours map by keypoint **NAME**. Mixing the two orders scrambles anatomy while numeric checks stay green — CLAUDE.md records exactly that bug.
- **Camera display names come from `clip_io.display_names(cam_names, clip)`** (`Camera 1`–`Camera 7` by arc position). **All indexing uses the real `Cam20128xx` names.** Display order must never reach data selection.
- **Display footage is `enhanced/`** — `clip_io.video_path()` already defaults to `enhanced=True`.
- **Use `draw.py` helpers** for markers and text; do not hand-roll `cv2.putText`/`cv2.circle`. Several tasks in the sibling project were sent back for that.
- **Shared type scale:** `draw.TITLE_SCALE = 1.6`, `draw.CAPTION_SCALE = 0.65`, `draw.SMALL_SCALE = 0.5`.
- **Colours** from `kp_colors.jarvis_kp_colors(kp_names)` → `{name: (B,G,R)}`.
- Labels use real names and units — keypoint names, camera display names, mm, px, frames. Never bare array indices.
- **Tests** live flat in `tests/test_recovery_clip*.py`, run with `JAX_PLATFORMS=cpu python -m pytest`. Skip cleanly if the clip is absent.
- Run with `env -u JAX_PLATFORMS`. This clip is CPU-only, so it can render alongside GPU work.

## Available interfaces (already built, do not reimplement)

`scripts/viz/ik_explainer/clip_io.py`:
- `CLIP_DEFAULT: str`
- `load_dlt(calib_dir) -> (cam_mats (C,4,3), cam_names)`
- `project(cam_mats, pts_mm) -> (C,K,2)`
- `video_path(clip, cam_name, *, enhanced=True) -> str`
- `read_frames(path, indices) -> (n,H,W,3)` BGR uint8
- `out_dirs(clip) -> {root, predictions, qc, frames}`
- `model_kp_names() -> list[str]`
- `display_names(cam_names, clip) -> {cam_name: "Camera N"}`
- `arc_positions_deg(cam_mats, names, tol=1.5) -> {cam_name: degrees}`

`scripts/viz/ik_explainer/draw.py`:
- `TITLE_SCALE`, `CAPTION_SCALE`, `SMALL_SCALE`
- `draw_keypoints(img, uv, kp_names, conf=None, alpha=1.0, radius=3, kp_colors=None)`
- `label(img, text, xy, scale=0.5, color=(255,255,255), thickness=1)`
- `stage_title(img, title, subtitle="")`
- `fade(img_a, img_b, t)`
- `scale_bar_mm(img, px_per_mm, mm=1.0, origin=None)`

`scripts/viz/ik_explainer/kp_colors.py`:
- `jarvis_kp_colors(kp_names=None) -> {name: (B,G,R)}`

Data on disk:
- `<CLIP>/ik_explainer/predictions/02_kp2d.npz` — `kp2d (921,7,50,2)` px, `conf (921,7,50)`, `cam_names`, `kp_names`
- `<CLIP>/ik_explainer/predictions/03_kp3d.npz` — `kp3d (921,50,3)` mm **raw**, `kp_names`
- `<CLIP>/ik_explainer/predictions/04_kp3d_filt.npz` — `kp3d (921,50,3)` mm **filtered**, `kp_names`
- `<CLIP>/ik_production/stac_ik_full.h5` — `marker_sites (921,50,3)` in **model units**; divide by `shared_scale = 0.1261` for mm. `kp_names` is MODEL order.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/viz/ik_explainer/recovery_event.py` | Load the three 3D tracks + detector 2D for one keypoint/window; compute the event statistics. Pure data, no drawing. |
| `scripts/viz/ik_explainer/trace_panel.py` | The one new drawing primitive: a line chart with axes, three traces, playhead and an annotation. Pure drawing, no data loading. |
| `scripts/viz/ik_explainer/recovery_clip.py` | Compose the 2-up frames and encode. Uses both of the above. |
| `tests/test_recovery_event.py` | Task 1 tests |
| `tests/test_trace_panel.py` | Task 2 tests |

Splitting data from drawing keeps each file testable on its own: `recovery_event` is verifiable against known numbers with no rendering, and `trace_panel` is verifiable on synthetic arrays with no clip data.

---

### Task 1: `recovery_event.py` — the data behind the event

**Files:**
- Create: `scripts/viz/ik_explainer/recovery_event.py`
- Test: `tests/test_recovery_event.py`

**Interfaces:**
- Consumes: `clip_io` (`CLIP_DEFAULT`, `out_dirs`, `model_kp_names`, `load_dlt`, `project`).
- Produces:
  - `SHARED_SCALE: float = 0.1261`
  - `EVENT: dict` with keys `kp` (`"T1R_TaTip"`), `cam` (`"Cam2012853"`), `t0` (420), `t1` (465), `peak_2d` (441), `peak_3d` (443)
  - `load_tracks(clip, kp_name, t0, t1) -> dict` with keys:
    - `raw (n,3)`, `filt (n,3)`, `ik (n,3)` — mm, model order, for that keypoint over `[t0,t1)`
    - `det2d (n,2)`, `conf (n,)` — that camera's detector 2D and confidence
    - `rep2d (n,2)` — the filtered 3D reprojected into that camera
    - `frames (n,)` — the source frame indices
  - `accel(track (n,3)) -> (n-2,)` — per-frame acceleration magnitude, mm/frame²
  - `worst_axis(raw (n,3)) -> int` — index (0/1/2) of the coordinate with the largest raw excursion

- [ ] **Step 1: Write the failing test**

```python
# tests/test_recovery_event.py
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import clip_io, recovery_event as ev

CLIP = clip_io.CLIP_DEFAULT
pytestmark = pytest.mark.skipif(
    not Path(CLIP).exists(), reason="source clip not present")


def test_event_constants_match_the_spec():
    assert ev.EVENT["kp"] == "T1R_TaTip"
    assert ev.EVENT["cam"] == "Cam2012853"
    assert (ev.EVENT["t0"], ev.EVENT["t1"]) == (420, 465)
    assert ev.EVENT["peak_2d"] == 441
    assert ev.EVENT["peak_3d"] == 443


def test_load_tracks_shapes_and_frames():
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    n = e["t1"] - e["t0"]
    for k in ("raw", "filt", "ik"):
        assert t[k].shape == (n, 3), f"{k} has shape {t[k].shape}"
    assert t["det2d"].shape == (n, 2)
    assert t["rep2d"].shape == (n, 2)
    assert t["conf"].shape == (n,)
    assert np.array_equal(t["frames"], np.arange(e["t0"], e["t1"]))


def test_the_detector_really_fails_at_the_peak_frame():
    """The clip's whole premise: at frame 441 this camera is ~122 px off the
    consensus and reports low confidence. If this stops holding, the event
    moved and the clip would be showing nothing."""
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    i = e["peak_2d"] - e["t0"]
    gap = np.linalg.norm(t["det2d"][i] - t["rep2d"][i])
    assert gap > 80.0, f"detector-vs-reprojection gap only {gap:.0f} px"
    assert t["conf"][i] < 0.7, f"confidence {t['conf'][i]:.2f} not low"


def test_recovery_is_present_and_ordered():
    """raw spikes; filtered and IK do not. This is the claim the clip makes."""
    e = ev.EVENT
    t = ev.load_tracks(CLIP, e["kp"], e["t0"], e["t1"])
    ar, af, ai = (ev.accel(t[k]) for k in ("raw", "filt", "ik"))
    assert ar.max() > 1.0, f"raw peak accel {ar.max():.3f} — expected a spike"
    assert af.max() < 0.2, f"filtered peak accel {af.max():.3f} — not smooth"
    assert ai.max() < 0.2, f"IK peak accel {ai.max():.3f} — not smooth"
    assert ar.max() / af.max() > 10.0


def test_accel_is_zero_for_constant_velocity():
    t = np.arange(10)[:, None] * np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(ev.accel(t), 0.0, atol=1e-9)


def test_worst_axis_picks_the_largest_excursion():
    n = 20
    a = np.zeros((n, 3))
    a[10, 1] = 5.0                      # a big kick on y only
    assert ev.worst_axis(a) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/eabe/Research/MyRepos/3d_tracking_dataset && JAX_PLATFORMS=cpu python -m pytest tests/test_recovery_event.py -v`
Expected: FAIL — `ImportError: cannot import name 'recovery_event'`

- [ ] **Step 3: Write the implementation**

```python
"""Data for the recovery clip: one keypoint's three 3D tracks plus the
detector's own 2D, over the window where detection failed.

The event was chosen by measurement (see
docs/specs/2026-08-14-recovery-clip-design.md): T1R_TaTip on Cam2012853 is
simultaneously the bout's worst detector-vs-consensus disagreement (122 px at
frame 441, confidence 0.49) and its worst raw-3D spike (1.627 mm/frame^2 at
frame 443). Nothing here is computed fresh -- every array is read from disk.
"""
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from scripts.viz.ik_explainer import clip_io   # noqa: E402

# marker_sites in the production h5 are in model units; mm = value / SHARED_SCALE
SHARED_SCALE = 0.1261

EVENT = {
    "kp": "T1R_TaTip",
    "cam": "Cam2012853",
    "t0": 420,
    "t1": 465,
    "peak_2d": 441,   # worst detector-vs-consensus disagreement in the bout
    "peak_3d": 443,   # worst raw triangulation spike
}


def accel(track):
    """(n,3) positions -> (n-2,) acceleration magnitude per frame."""
    v = np.diff(np.asarray(track, np.float64), axis=0)
    return np.linalg.norm(np.diff(v, axis=0), axis=-1)


def worst_axis(raw):
    """Index of the coordinate with the largest excursion about its median.

    Used to choose which of x/y/z to plot: the one where the failure is most
    legible. Returned rather than hardcoded so the choice stays correct if the
    window or keypoint changes.
    """
    a = np.asarray(raw, np.float64)
    return int(np.argmax(np.nanmax(np.abs(a - np.nanmedian(a, axis=0)), axis=0)))


def load_tracks(clip=clip_io.CLIP_DEFAULT, kp_name=EVENT["kp"],
                t0=EVENT["t0"], t1=EVENT["t1"], cam_name=EVENT["cam"]):
    """Three 3D tracks + the detector's 2D for one keypoint over [t0, t1)."""
    d = clip_io.out_dirs(clip)
    kp_names = clip_io.model_kp_names()
    k = kp_names.index(kp_name)          # by NAME -- never a positional guess

    z2 = np.load(d["predictions"] / "02_kp2d.npz", allow_pickle=True)
    cam_names = [str(c) for c in z2["cam_names"]]
    if [str(n) for n in z2["kp_names"]] != kp_names:
        raise ValueError("02_kp2d.npz is not in MODEL keypoint order")
    c = cam_names.index(cam_name)        # by NAME

    raw = np.load(d["predictions"] / "03_kp3d.npz", allow_pickle=True)["kp3d"]
    flt = np.load(d["predictions"] / "04_kp3d_filt.npz", allow_pickle=True)["kp3d"]
    with h5py.File(Path(clip) / "ik_production" / "stac_ik_full.h5", "r") as f:
        if [s.decode() for s in f["kp_names"][:]] != kp_names:
            raise ValueError("stac_ik_full.h5 is not in MODEL keypoint order")
        ik = f["marker_sites"][t0:t1, k] / SHARED_SCALE     # -> mm

    cam_mats, dlt_names = clip_io.load_dlt(str(Path(clip) / "calibration"))
    if dlt_names != cam_names:
        raise ValueError("calibration and kp2d disagree on camera order")

    # reproject the FILTERED 3D into this camera: what the pipeline believes
    rep = np.stack([clip_io.project(cam_mats, flt[t])[c, k] for t in range(t0, t1)])

    return {
        "raw": raw[t0:t1, k].astype(np.float64),
        "filt": flt[t0:t1, k].astype(np.float64),
        "ik": np.asarray(ik, np.float64),
        "det2d": z2["kp2d"][t0:t1, c, k].astype(np.float64),
        "conf": z2["conf"][t0:t1, c, k].astype(np.float64),
        "rep2d": rep.astype(np.float64),
        "frames": np.arange(t0, t1),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_recovery_event.py -v`
Expected: 6 passed.

If `test_the_detector_really_fails_at_the_peak_frame` or
`test_recovery_is_present_and_ordered` fails, **do not relax the thresholds** —
they encode the clip's premise. Report instead; it would mean the artifacts on
disk no longer contain the event.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/recovery_event.py tests/test_recovery_event.py
git commit -m "feat(recovery-clip): load the three 3D tracks behind the event

Reads raw/filtered/IK tracks plus the detector's own 2D for T1R_TaTip over
frames 420-465. Tests assert the premise rather than the plumbing: the detector
really is >80px off consensus at frame 441 with low confidence, and raw really
does spike >10x harder than filtered or IK. If the artifacts stop containing
the event those fail loudly instead of rendering an empty story."
```

---

### Task 2: `trace_panel.py` — the chart primitive

**Files:**
- Create: `scripts/viz/ik_explainer/trace_panel.py`
- Test: `tests/test_trace_panel.py`

**Interfaces:**
- Consumes: `draw` (`label`, `CAPTION_SCALE`, `SMALL_SCALE`).
- Produces:
  - `render_trace_panel(w, h, series, frames, cursor, *, ylabel, title=None, annotation=None) -> np.ndarray`
    - `series`: list of `(name, values (n,), colour (B,G,R))`
    - `frames`: `(n,)` source frame numbers for the x axis
    - `cursor`: source frame number where the playhead sits
    - returns an `(h, w, 3)` uint8 BGR panel

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trace_panel.py
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.viz.ik_explainer import trace_panel as tp


def _series(n=45):
    x = np.linspace(0.0, 1.0, n)
    return [("raw", x + 0.5 * (np.arange(n) == 20), (0, 0, 255)),
            ("filt", x, (0, 255, 0)),
            ("ik", x, (255, 255, 0))]


def test_panel_has_the_requested_shape_and_dtype():
    p = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 441,
                              ylabel="z (mm)")
    assert p.shape == (600, 800, 3)
    assert p.dtype == np.uint8


def test_each_series_colour_actually_appears():
    """A chart that silently drops a trace would still look plausible."""
    p = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 441,
                              ylabel="z (mm)")
    for _name, _vals, col in _series():
        hit = np.all(p == np.array(col, np.uint8), axis=-1).sum()
        assert hit > 0, f"colour {col} never drawn"


def test_playhead_moves_with_the_cursor():
    a = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 425,
                              ylabel="z (mm)")
    b = tp.render_trace_panel(800, 600, _series(), np.arange(420, 465), 460,
                              ylabel="z (mm)")
    assert not np.array_equal(a, b), "playhead did not move"


def test_flat_series_does_not_divide_by_zero():
    n = 20
    flat = [("a", np.full(n, 3.0), (255, 255, 255))]
    p = tp.render_trace_panel(400, 300, flat, np.arange(n), 5, ylabel="mm")
    assert np.isfinite(p).all() and p.shape == (300, 400, 3)


def test_returns_a_new_array_each_call():
    s, f = _series(), np.arange(420, 465)
    a = tp.render_trace_panel(400, 300, s, f, 430, ylabel="mm")
    b = tp.render_trace_panel(400, 300, s, f, 430, ylabel="mm")
    assert a is not b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_trace_panel.py -v`
Expected: FAIL — `ImportError: cannot import name 'trace_panel'`

- [ ] **Step 3: Write the implementation**

```python
"""A minimal line-chart panel for the recovery clip.

Deliberately local to this clip rather than promoted into draw.py: it has one
consumer, and draw.py's helpers are markers-on-footage primitives. Promote it
if a second caller appears.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from scripts.viz.ik_explainer import draw   # noqa: E402

_BG = (0, 0, 0)
_AXIS = (110, 110, 110)
_PLAYHEAD = (255, 255, 255)
_PAD_L, _PAD_R, _PAD_T, _PAD_B = 110, 40, 70, 60


def render_trace_panel(w, h, series, frames, cursor, *, ylabel,
                       title=None, annotation=None):
    """Line chart: several series over `frames`, with a playhead at `cursor`.

    series: [(name, values (n,), (B,G,R)), ...]
    Returns a fresh (h, w, 3) uint8 BGR image.
    """
    panel = np.zeros((h, w, 3), np.uint8)
    panel[:] = _BG
    frames = np.asarray(frames)
    x0, x1 = _PAD_L, w - _PAD_R
    y0, y1 = _PAD_T, h - _PAD_B

    vals = np.concatenate([np.asarray(v, np.float64).ravel() for _n, v, _c in series])
    lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        lo, hi = lo - 1.0, lo + 1.0        # flat series: give it a visible band
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fx = lambda f: int(round(x0 + (f - frames[0]) / max(frames[-1] - frames[0], 1) * (x1 - x0)))
    fy = lambda v: int(round(y1 - (v - lo) / (hi - lo) * (y1 - y0)))

    cv2.rectangle(panel, (x0, y0), (x1, y1), _AXIS, 1, cv2.LINE_AA)
    for frac in (0.0, 0.5, 1.0):
        v = lo + frac * (hi - lo)
        y = fy(v)
        cv2.line(panel, (x0 - 6, y), (x0, y), _AXIS, 1, cv2.LINE_AA)
        panel = draw.label(panel, f"{v:.2f}", (8, y + 5), scale=draw.SMALL_SCALE,
                           color=_AXIS)
    for f in (frames[0], frames[len(frames) // 2], frames[-1]):
        x = fx(f)
        cv2.line(panel, (x, y1), (x, y1 + 6), _AXIS, 1, cv2.LINE_AA)
        panel = draw.label(panel, str(int(f)), (x - 18, y1 + 26),
                           scale=draw.SMALL_SCALE, color=_AXIS)

    for name, v, col in series:
        v = np.asarray(v, np.float64)
        pts = np.array([[fx(frames[i]), fy(v[i])] for i in range(len(v))
                        if np.isfinite(v[i])], np.int32)
        if len(pts) > 1:
            cv2.polylines(panel, [pts], False, tuple(int(c) for c in col), 2,
                          cv2.LINE_AA)

    xc = fx(cursor)
    cv2.line(panel, (xc, y0), (xc, y1), _PLAYHEAD, 1, cv2.LINE_AA)

    panel = draw.label(panel, ylabel, (8, y0 - 18), scale=draw.CAPTION_SCALE)
    if title:
        panel = draw.label(panel, title, (x0, 34), scale=draw.CAPTION_SCALE)
    if annotation:
        panel = draw.label(panel, annotation, (x0, h - 16),
                           scale=draw.SMALL_SCALE, color=(190, 190, 190))

    # legend, in each series' own colour
    for i, (name, _v, col) in enumerate(series):
        panel = draw.label(panel, name, (x1 - 150, y0 + 24 + 26 * i),
                           scale=draw.CAPTION_SCALE,
                           color=tuple(int(c) for c in col))
    return panel
```

Note `cv2.polylines`/`rectangle`/`line` are used directly here because this is a
chart primitive, not marker-on-footage drawing — `draw.py` has no chart helpers.
Text still goes through `draw.label` so the type scale stays shared.

- [ ] **Step 4: Run tests to verify they pass**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_trace_panel.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/viz/ik_explainer/trace_panel.py tests/test_trace_panel.py
git commit -m "feat(recovery-clip): line-chart panel with playhead

Kept local to the recovery clip rather than promoted into draw.py -- one
consumer, and draw.py's helpers are markers-on-footage primitives. Text routes
through draw.label so the shared type scale still applies. Tests assert each
series' colour actually reaches the canvas, since a chart that silently drops a
trace still looks plausible."
```

---

### Task 3: `recovery_clip.py` — compose and encode

**Files:**
- Create: `scripts/viz/ik_explainer/recovery_clip.py`

**Interfaces:**
- Consumes: `recovery_event` (`EVENT`, `load_tracks`, `accel`, `worst_axis`, `SHARED_SCALE`), `trace_panel.render_trace_panel`, `clip_io`, `draw`, `kp_colors`.
- Produces: `<CLIP>/ik_explainer/recovery_clip.mp4` and `frames/recovery_clip/f%05d.png`.

- [ ] **Step 1: Write the renderer**

Structure, with the constraints that matter spelled out:

```python
"""Recovery clip: a real 2D detection failure, absorbed downstream.

Left  : Camera 3 footage, detector marker vs the reprojected filtered 3D.
Right : raw / filtered / IK traces for the same keypoint, with a playhead.

See docs/specs/2026-08-14-recovery-clip-design.md. The event is measured, not
chosen by eye, and the on-screen caveat is load-bearing: the other six cameras
also disagree here (83-121 px), so this is NOT "one camera failed and six
rescued it".
"""
```

Required behaviour:

1. **Timing.** `N_OUT = 360`; source frames `EVENT["t0"] … EVENT["t1"]` (45). Output frame `f` shows source frame `t0 + f * (t1 - t0) // N_OUT`. Compute the speed factor as `src_fps / out_fps / ((t1 - t0) / N_OUT)` with `src_fps = 800`, `out_fps = 30`, and render it as `800 fps -> 1/<value> speed`. It should come out ≈213 — but print the computed number, do not hardcode it.
2. **Left panel** (960×1080): the `enhanced/` frame from `EVENT["cam"]`, cropped around the reprojected marker so the foreleg fills the panel, at a fixed crop size so it does not jitter. Draw:
   - the detector's 2D marker in `jarvis_kp_colors()[EVENT["kp"]]`
   - the reprojected filtered 3D marker in white
   - a 1 mm scale bar via `draw.scale_bar_mm(..., px_per_mm=80.7)`
   - a `draw.label` readout: `Camera 3   detector conf 0.49` using the live per-frame confidence and `clip_io.display_names()`
3. **Right panel** (960×1080): `render_trace_panel` with the three tracks' `worst_axis` coordinate in mm, `cursor` = the current source frame, `ylabel` naming the axis (e.g. `T1R_TaTip  z (mm)`), and `annotation` = `raw 1.627 -> filtered 0.022 -> IK 0.015 mm/frame^2  (75x / 110x)` with those numbers **computed from `accel()`**, not typed in.
4. **Title** via `draw.stage_title` and the caveat line via `draw.label` at `CAPTION_SCALE`:
   `all 7 cameras disagree here (83-121 px) -- the recovery is triangulation + filter + IK, not one camera`
5. Write PNGs with `[cv2.IMWRITE_PNG_COMPRESSION, 1]`, then encode to `recovery_clip.mp4` with H.264 / `yuv420p` / faststart / 30 fps / no audio.
6. **Assert 360 frames were written** before encoding, and fail loudly otherwise — a truncated render must never silently ship.

- [ ] **Step 2: Render**

Run:
```bash
cd /home/eabe/Research/MyRepos/3d_tracking_dataset
env -u JAX_PLATFORMS python scripts/viz/ik_explainer/recovery_clip.py
ls /data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46/ik_explainer/frames/recovery_clip | wc -l
```
Expected: 360.

- [ ] **Step 3: READ the frames and report**

```
EXPECTATION at source frame 441 (output frame ~187): the coloured detector
marker sits far off the fly's foot while the WHITE reprojected marker stays on
it, and the confidence readout shows ~0.49. At source frame 443 the raw trace
shows a clear spike while the filtered and IK traces pass through smoothly. The
playhead sits at the current source frame in every frame.
FALSIFICATION: both markers together => wrong camera/keypoint/frame mapping.
All three traces spiking => wrong arrays plotted. Playhead not tracking =>
two independent frame indices; there must be one.
```

Open output frames **0**, **~187** (the failure) and **359** with the Read tool and report what they show against this.

- [ ] **Step 4: Verify the encoding**

Run:
```bash
CLIP=/data2/users/eabe/datasets/3d_tracking/clips/Session6/2025_10_12_15_06_46
ffprobe -v error -show_entries stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate \
  -of default=nw=1 $CLIP/ik_explainer/recovery_clip.mp4
```
Expected: `h264`, `yuv420p`, `1920`, `1080`, `360`, `30/1`.

Confirm faststart (`moov` before `mdat`), and confirm `ik_explainer.mp4` and the four act frame directories are untouched — this clip is additive.

- [ ] **Step 5: Run the full suite and commit**

Run: `JAX_PLATFORMS=cpu python -m pytest tests/test_recovery_*.py tests/test_trace_panel.py tests/test_ik_explainer_*.py -q`
Expected: all pass (34 existing + 11 new).

```bash
git add scripts/viz/ik_explainer/recovery_clip.py
git commit -m "feat(recovery-clip): 2-up failure-and-recovery clip

Camera 3 footage with the detector marker leaving the foot at frame 441 beside
raw/filtered/IK traces with a playhead. Speed factor and the acceleration
annotation are computed, not hardcoded, so they stay true if the window moves.
Carries the caveat that all seven cameras disagree at this instant -- the
recovery is triangulation plus filter plus IK, not one camera being outvoted."
```

---

## Verification Summary

Gates that must be passed by eye, not by number alone:

| Gate | Task | Stop condition |
|---|---|---|
| Detector really fails at f441 (>80 px, conf <0.7) | 1 | test fails ⇒ the event moved; do not relax thresholds |
| raw spikes >10× filtered and IK | 1 | test fails ⇒ wrong arrays |
| Every trace colour reaches the canvas | 2 | a silently dropped series still looks plausible |
| Detector marker off the foot, white marker on it | 3 | both together ⇒ wrong camera/keypoint/frame |
| Playhead tracks the left panel | 3 | drift ⇒ two frame indices instead of one |
| Caveat line on screen | 3 | absent ⇒ the clip overstates the mechanism |
| `yuv420p` + faststart + 360 frames | 3 | otherwise it will not play in VSCode |
