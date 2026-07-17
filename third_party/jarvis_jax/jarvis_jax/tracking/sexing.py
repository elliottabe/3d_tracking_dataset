"""Deterministic fly sexing + identity canonicalization for the pose pipeline.

The male is the fly whose wing-splay FLUCTUATES most over a courtship bout
(unilateral wing-extension song), measured as a coefficient of variation. This
module is pure numpy + stdlib (no jax/torch/mujoco) so it runs on the login
node. See docs/superpowers/specs/2026-07-17-fly-sexing-canonicalization-design.md.
"""
from __future__ import annotations

import io
import json
import os
import zipfile

import numpy as np

WING_SIDES = ("L", "R")


def _kp_index(kp_names):
    """Map keypoint name -> column index (model KP order)."""
    return {n: i for i, n in enumerate(kp_names)}


def wing_song_cv(kp3d, conf3d, kp_index, *, conf_min=0.2, min_frames=20):
    """Per-fly wing-song signal: CV (std/mean) of the per-frame max wing-splay angle.

    Body axis = Abd_tip - Scutellum. Wing vectors = WingX_V13 - WingX_base for
    X in {L,R}; per-frame splay = angle(wing, body axis) in degrees, taken as the
    max over the two wings. A frame's splay is valid only when Abd_tip, Scutellum,
    and that side's base+tip all have conf > conf_min. Returns nan if fewer than
    min_frames valid frames or a nonpositive mean."""
    kp3d = np.asarray(kp3d, float)
    conf3d = np.asarray(conf3d, float)

    def col(n):
        return kp_index[n]

    axis = kp3d[:, col("Abd_tip"), :] - kp3d[:, col("Scutellum"), :]
    axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-9)
    body_ok = (conf3d[:, col("Abd_tip")] > conf_min) & (conf3d[:, col("Scutellum")] > conf_min)

    splays = []
    for s in WING_SIDES:
        w = kp3d[:, col(f"Wing{s}_V13"), :] - kp3d[:, col(f"Wing{s}_base"), :]
        w = w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-9)
        cosang = np.clip((w * axis).sum(axis=1), -1.0, 1.0)
        a = np.degrees(np.arccos(cosang))
        ok = (body_ok
              & (conf3d[:, col(f"Wing{s}_V13")] > conf_min)
              & (conf3d[:, col(f"Wing{s}_base")] > conf_min))
        splays.append(np.where(ok, a, np.nan))

    with np.errstate(invalid="ignore"):
        splay = np.nanmax(np.stack(splays, axis=1), axis=1)   # (T,) nan where both wings invalid
    valid = splay[np.isfinite(splay)]
    if valid.size < min_frames:
        return float("nan")
    mean = float(valid.mean())
    if mean <= 0:
        return float("nan")
    return float(valid.std() / mean)


def sex_bout_from_pose(cv0, cv1, *, mask_sex_meta=None, ratio_thr=1.5, high_ratio=2.5):
    """Decide the male slot from the two flies' wing-song CVs (4-branch rule)."""
    cv = [cv0, cv1]
    male_pose = None
    ratio = None
    finite = [c for c in cv if c is not None and np.isfinite(c)]
    if len(finite) == 2 and min(cv) > 0:
        male_pose = int(np.argmax(cv))
        ratio = float(max(cv) / min(cv))

    if ratio is not None and ratio >= ratio_thr:
        conf = "high" if ratio >= high_ratio else "medium"
        return dict(male_fly=male_pose, confidence=conf, method="pose_wing_cv",
                    cv=[cv0, cv1], cv_ratio=ratio, note="")
    if mask_sex_meta is not None:
        return dict(male_fly=1, confidence="low", method="mask_fallback",
                    cv=[cv0, cv1], cv_ratio=ratio,
                    note="weak pose signal; mask sexed bout to slot 1")
    return dict(male_fly=None, confidence="unknown", method="unresolved",
                cv=[cv0, cv1], cv_ratio=ratio,
                note="weak pose signal, no mask sex_meta")
