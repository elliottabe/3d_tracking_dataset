"""Derive the rig's principal camera roles (top / left / right) from the
calibration geometry alone -- never from a hardcoded camera name. Camera IDs
differ per session; "camera order/identity comes from the calibration
directory glob, never from a config" is a documented trap in this repo.

The rig is ORTHOGRAPHIC (telecentric lenses): every DLT matrix's projected
homogeneous `w` is exactly 1 for all X (see viz.core.mjcam), so there is no
finite camera centre and the usual "third row of the 3x4 camera matrix is the
principal ray" trick returns NaN. The view direction is instead the null
direction of the image plane's own basis: in the `ph @ M` convention used by
viz.core.reproject (M is (4,3), P = M.T is the standard (3,4) camera matrix),
the first two rows of P are `m0 = M[:3, 0]` and `m1 = M[:3, 1]` -- the two
image axes, in world units. Both are, by construction, perpendicular to the
camera's viewing direction, so ``v = normalize(cross(m0, m1))`` recovers it.

Measured on Session0 2025_10_20_13_20_04 (7 cameras, ~180 deg arc in the Y-Z
plane, 30 deg apart): X ~ 0 for every camera; |v_z| is largest (~1) for the
one camera looking straight down, and falls off towards the two ends of the
arc, which look ~horizontal (|v_z| ~ 0).

Role rule (elevation-targeted, not "most opposed" -- an exactly horizontal
lateral view sees the fly edge-on: the wing blade is roughly horizontal, so a
horizontal view resolves wing pose poorly and the body self-occludes; a
mildly elevated view resolves both horizontal and vertical structure):

  - top   = the camera with the largest |v_z|.
  - left / right = one camera per side of the dominant horizontal axis
    (opposite sign), each the camera on its side whose |v_z| is closest to
    `elevation_target` (default 0.5, i.e. ~30 deg above horizontal). "Side"
    and "dominant axis" are read off the data (whichever of the two
    non-vertical axes actually varies across the rig), never hardcoded to Y.
"""
from __future__ import annotations

from collections import namedtuple

import numpy as np

from viz.core import reproject

RigView = namedtuple("RigView", ["name", "direction", "elevation_deg"])


def view_directions(calib_dir):
    """{camera_name: unit view-direction (3,)} for every camera in calib_dir.

    Raises ValueError on a camera whose calibration is not the rig's
    orthographic/affine kind (third homogeneous column != [0,0,0,1]) or whose
    two image-axis rows are degenerate (parallel) -- both would silently give
    a nonsense direction otherwise.
    """
    cam_mats, names = reproject.camera_matrices(calib_dir)
    out = {}
    for M, name in zip(np.asarray(cam_mats, float), names):
        third = M[:, 2]
        if not (np.allclose(third[:3], 0, atol=1e-4) and np.isclose(third[3], 1, atol=1e-4)):
            raise ValueError(
                f"camera {name!r} is not an affine/orthographic camera "
                f"(homogeneous column {third}, expected [0,0,0,1]); rigviews "
                "only supports this rig's telecentric calibration")
        m0, m1 = M[:3, 0], M[:3, 1]
        cr = np.cross(m0, m1)
        n = float(np.linalg.norm(cr))
        if n < 1e-9:
            raise ValueError(f"camera {name!r}: degenerate view direction (image axes parallel)")
        out[name] = cr / n
    return out


def classify_views(calib_dir, elevation_target=0.5):
    """Return {"top", "left", "right"} -> RigView(name, direction, elevation_deg).

    `elevation_target` is |v_z| for the lateral cameras (0 = exactly
    horizontal, 1 = straight down/up); default 0.5 is ~30 deg above the
    horizontal plane. Tune it if a rig's arc spacing makes 0.5 land far from
    any real camera.

    Degrades with a clear ValueError -- never an arbitrary silent pick -- for:
    fewer than 3 cameras, or no camera on one side of the dominant horizontal
    axis (so no left/right pair can be formed).
    """
    v = view_directions(calib_dir)
    names = list(v)
    if len(names) < 3:
        raise ValueError(
            f"need >=3 cameras to derive a top/left/right triple, got "
            f"{len(names)}: {names}")

    top = max(names, key=lambda n: abs(v[n][2]))
    others = [n for n in names if n != top]

    # Dominant horizontal axis: whichever of the two non-vertical (x, y) axes
    # actually varies across this rig's cameras -- read off the data so a rig
    # whose arc lies in the X-Z plane (instead of Y-Z, as measured here)
    # still resolves correctly.
    horiz = np.stack([v[n][:2] for n in others])           # (N-1, 2): [x, y]
    axis = int(np.argmax(np.abs(horiz).max(axis=0)))

    left_side = [n for n in others if v[n][axis] < 0]
    right_side = [n for n in others if v[n][axis] > 0]
    if not left_side or not right_side:
        raise ValueError(
            "no camera on one side of the dominant horizontal axis "
            f"(axis={('x', 'y')[axis]}); left_side={left_side} "
            f"right_side={right_side}; cannot form a left/right pair")

    def _pick(side):
        return min(side, key=lambda n: abs(abs(v[n][2]) - elevation_target))

    left, right = _pick(left_side), _pick(right_side)

    def _view(n):
        elev = float(np.degrees(np.arcsin(np.clip(abs(v[n][2]), -1.0, 1.0))))
        return RigView(n, v[n], elev)

    return {"top": _view(top), "left": _view(left), "right": _view(right)}


def format_view_label(role, rv):
    """Human-readable panel label, e.g. 'Cam2012855 [left, oblique 31 deg, view (-0.01,-0.85,-0.52)]'."""
    x, y, z = rv.direction
    kind = "top-down" if role == "top" else f"oblique {rv.elevation_deg:.0f} deg elev"
    return f"{rv.name} [{role}, {kind}, view ({x:.2f},{y:.2f},{z:.2f})]"
