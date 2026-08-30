"""Group recordings by the CONTENT of their DLT calibration.

The V2VNet's voxel grid is axis-aligned to the world frame (see
hybridnet/reproject.py: `grid = base_grid + center3D`), so a recalibration that
rotates the world frame invalidates the model's learned orientation prior.
Calibration identity is therefore a first-class property of a recording, not
an incidental file path.

Fingerprints are computed from ROUNDED NUMERIC coefficients, never file bytes:
20_04_female_climbing's calibration is numerically identical to Session0's but
written with fewer decimal places, and a byte hash wrongly splits them.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from collections import Counter

CAM_GLOB = "Cam*.yaml"
NUM_CAMERAS = 7
_ROUND = 6


def _projection_matrix(path: str) -> list[float]:
    text = open(path).read()
    m = re.search(r"data:\s*\[(.*?)\]", text, re.S)
    if m is None:
        raise ValueError(f"no projectionMatrix data block in {path}")
    vals = [float(x) for x in m.group(1).replace("\n", " ").split(",") if x.strip()]
    if len(vals) != 12:
        raise ValueError(f"expected 12 projection coefficients in {path}, got {len(vals)}")
    return [round(v, _ROUND) for v in vals]


def calib_fingerprint(calib_dir: str) -> str:
    """8-char hex fingerprint of a calibration directory's 7 camera matrices."""
    files = sorted(glob.glob(os.path.join(calib_dir, CAM_GLOB)))
    if len(files) != NUM_CAMERAS:
        raise ValueError(
            f"expected {NUM_CAMERAS} cameras in {calib_dir}, found {len(files)}")
    coeffs: list[float] = []
    for f in files:
        coeffs.extend(_projection_matrix(f))
    return hashlib.md5(repr(coeffs).encode()).hexdigest()[:8]


def group_calibrations(dirs: dict[str, str]) -> dict[str, str]:
    """Map recording name -> group label ('A', 'B', ...), largest group = 'A'."""
    fp = {rec: calib_fingerprint(d) for rec, d in dirs.items()}
    counts = Counter(fp.values())
    ordered = sorted(counts, key=lambda h: (-counts[h],
                                            min(r for r in fp if fp[r] == h)))
    label = {h: chr(ord("A") + i) for i, h in enumerate(ordered)}
    return {rec: label[h] for rec, h in fp.items()}
