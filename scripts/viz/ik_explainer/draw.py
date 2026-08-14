"""Shared 2D drawing for the explainer acts.

Colours come from viz/core/colors.py so this video shares the repo's visual
language. Every helper returns a COPY -- acts composite one base frame at several alphas
and in-place drawing would smear across fades.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from viz.core.colors import PALETTE, keypoint_groups, leg_chains   # noqa: E402

_GROUP_COLOR = {"head": PALETTE["head"], "abdomen": PALETTE["tail"],
                "thorax": PALETTE["thorax"], "legs": PALETTE["fly0"]}

# --- shared type scale (task-19, "consistent, correctly sized typography") -
# ONE size per role, imported by every act instead of each picking its own ad
# hoc `scale`/`thickness` per call site. Calibrated for legibility on a
# projected 1920x1080 slide, verified by reading rendered frames with the
# Read tool (not guessed):
#   TITLE_SCALE/_THICKNESS   -- the act's on-screen name (`stage_title`'s
#                               `title` arg). Must read as clearly dominant
#                               over any other on-screen text.
#   CAPTION_SCALE/_THICKNESS -- any remaining secondary/caption line (e.g.
#                               Act 2's camera-rig fact line, `stage_title`'s
#                               optional `subtitle`).
#   SMALL_SCALE/_THICKNESS   -- the bottom-left frame counter and per-panel
#                               labels (camera name + elevation in Acts 1-2).
TITLE_SCALE = 1.6
TITLE_THICKNESS = 3
CAPTION_SCALE = 0.65
CAPTION_THICKNESS = 1
SMALL_SCALE = 0.5
SMALL_THICKNESS = 1


def fade(img_a, img_b, t: float):
    t = float(np.clip(t, 0.0, 1.0))
    if t <= 0.0:
        return np.asarray(img_a).copy()
    if t >= 1.0:
        return np.asarray(img_b).copy()
    return cv2.addWeighted(np.asarray(img_a), 1.0 - t, np.asarray(img_b), t, 0.0)


def draw_keypoints(img, uv, kp_names, conf=None, alpha=1.0, radius=3, kp_colors=None):
    """`kp_colors`: optional {name: BGR tuple} (e.g.
    `kp_colors.jarvis_kp_colors`) for a per-keypoint colour scheme. When
    omitted, falls back to the legacy 4-group anatomical colouring
    (head/thorax/abdomen/legs) for backward compatibility."""
    out = np.asarray(img).copy()
    if alpha <= 0.0:
        return out
    layer = out.copy()
    names = list(kp_names)
    if kp_colors is not None:
        for i, name in enumerate(names):
            p = np.asarray(uv)[i]
            if not np.all(np.isfinite(p)):
                continue
            r = radius if (conf is None or conf[i] >= 0.3) else max(1, radius - 2)
            cv2.circle(layer, tuple(np.round(p).astype(int)), r,
                       kp_colors[name], -1, cv2.LINE_AA)
    else:
        groups = keypoint_groups(names)
        for g, idxs in groups.items():
            for i in idxs:
                p = np.asarray(uv)[i]
                if not np.all(np.isfinite(p)):
                    continue
                r = radius if (conf is None or conf[i] >= 0.3) else max(1, radius - 2)
                cv2.circle(layer, tuple(np.round(p).astype(int)), r,
                           _GROUP_COLOR[g], -1, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0.0)


def draw_leg_chains(img, uv, kp_names, alpha=1.0, thickness=1, kp_colors=None):
    """`kp_colors`: optional {name: BGR tuple}; each leg chain is drawn in its
    own colour (its first joint's colour -- under the JARVIS scheme a whole
    chain shares one colour) instead of the legacy flat `PALETTE["fly0"]`."""
    out = np.asarray(img).copy()
    if alpha <= 0.0:
        return out
    layer = out.copy()
    names = list(kp_names)
    for _leg, chain in leg_chains(names).items():
        pts = np.asarray(uv)[chain]
        color = kp_colors[names[chain[0]]] if kp_colors is not None else PALETTE["fly0"]
        for a, b in zip(pts[:-1], pts[1:]):
            if np.all(np.isfinite([a, b])):
                cv2.line(layer, tuple(np.round(a).astype(int)),
                         tuple(np.round(b).astype(int)), color,
                         thickness, cv2.LINE_AA)
    return cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0.0)


def label(img, text, xy, scale=0.5, color=(255, 255, 255), thickness=1):
    out = np.asarray(img).copy()
    cv2.putText(out, text, (int(xy[0]), int(xy[1])), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)
    return out


def stage_title(img, title, subtitle=""):
    """`title` uses the shared `TITLE_SCALE` (dominant on every act);
    `subtitle`, when given, uses `CAPTION_SCALE` -- both sizes come from the
    ONE shared type scale above so all four acts read consistently
    (task-19)."""
    out = np.asarray(img).copy()
    cv2.putText(out, title, (48, 72), cv2.FONT_HERSHEY_SIMPLEX, TITLE_SCALE,
                (255, 255, 255), TITLE_THICKNESS, cv2.LINE_AA)
    if subtitle:
        cv2.putText(out, subtitle, (48, 122), cv2.FONT_HERSHEY_SIMPLEX,
                    CAPTION_SCALE, (190, 190, 190), CAPTION_THICKNESS, cv2.LINE_AA)
    return out


def scale_bar_mm(img, px_per_mm, mm=1.0, origin=None):
    out = np.asarray(img).copy()
    h, w = out.shape[:2]
    x0, y0 = origin if origin is not None else (w - int(mm * px_per_mm) - 40, h - 30)
    x1 = x0 + int(mm * px_per_mm)
    cv2.line(out, (x0, y0), (x1, y0), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, f"{mm:g} mm", (x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                SMALL_SCALE, (255, 255, 255), SMALL_THICKNESS, cv2.LINE_AA)
    return out
