"""A minimal line-chart panel for the recovery clip.

Deliberately local to this clip rather than promoted into draw.py: it has one
consumer, and draw.py's helpers are markers-on-footage primitives. Promote it
if a second caller appears.
"""
import sys
import warnings
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

    The last series is drawn dashed to remain visible when it coincides exactly
    with a prior series (filtered and IK are nearly identical in the real clip,
    and that agreement is the message, but both must still be distinguishable).

    series: [(name, values (n,), (B,G,R)), ...]
    Returns a fresh (h, w, 3) uint8 BGR image.
    """
    panel = np.zeros((h, w, 3), np.uint8)
    panel[:] = _BG
    frames = np.asarray(frames)
    x0, x1 = _PAD_L, w - _PAD_R
    y0, y1 = _PAD_T, h - _PAD_B

    vals = np.concatenate([np.asarray(v, np.float64).ravel() for _n, v, _c in series])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='All-NaN slice encountered')
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi):
        lo, hi = 0.0, 1.0                # nothing finite: draw empty axes, don't crash
    elif hi - lo < 1e-9:
        lo, hi = lo - 1.0, lo + 1.0      # flat series: give it a visible band
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

    for si, (name, v, col) in enumerate(series):
        v = np.asarray(v, np.float64)
        pts = [(fx(frames[i]), fy(v[i])) for i in range(len(v))
               if np.isfinite(v[i])]
        if len(pts) > 1:
            dashed = (si == len(series) - 1)          # last series dashed
            for idx, (a, b) in enumerate(zip(pts[:-1], pts[1:])):
                if dashed and (idx // 3) % 2:  # skip alternating runs
                    continue
                cv2.line(panel, a, b, tuple(int(c) for c in col), 2, cv2.LINE_AA)

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
