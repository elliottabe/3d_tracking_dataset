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


_SWAP_TMP = ".fly_swap_tmp"


def read_sex_meta(npz_path):
    """Parse the SAM3 mask `sex_meta` (0-d JSON-string 'sex_meta.npy') from a
    sam3_masks.npz. Returns the dict, or None if the file/entry is absent or
    unreadable."""
    if not npz_path or not os.path.exists(npz_path):
        return None
    try:
        with zipfile.ZipFile(npz_path) as zf:
            if "sex_meta.npy" not in zf.namelist():
                return None
            arr = np.load(io.BytesIO(zf.read("sex_meta.npy")), allow_pickle=True)
        return json.loads(str(arr))
    except (zipfile.BadZipFile, OSError, ValueError):
        return None


def _swap_fly_dirs(bout_dir):
    """Swap fly0 <-> fly1 crash-safely via a temp name. On entry, complete a
    partial swap left by a prior crash (temp dir present)."""
    f0 = os.path.join(bout_dir, "fly0")
    f1 = os.path.join(bout_dir, "fly1")
    tmp = os.path.join(bout_dir, _SWAP_TMP)

    if os.path.exists(tmp):                       # recover partial swap; tmp holds original fly0
        if not os.path.exists(f0) and os.path.exists(f1):     # crashed after step 1
            os.rename(f1, f0)
            os.rename(tmp, f1)
            return
        if os.path.exists(f0) and not os.path.exists(f1):     # crashed after step 2
            os.rename(tmp, f1)
            return
        raise RuntimeError(
            f"ambiguous partial-swap state in {bout_dir}: fly0/fly1 both present "
            f"with {_SWAP_TMP}; resolve manually")

    os.rename(f0, tmp)      # step 1
    os.rename(f1, f0)       # step 2
    os.rename(tmp, f1)      # step 3


def _jsonnum(x):
    """float for JSON, or None for None/non-finite."""
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def canonicalize_bout(bout_dir, kp_names, *, mask_sex_meta=None, male_slot=1,
                      ratio_thr=1.5, high_ratio=2.5, conf_min=0.2, min_frames=20,
                      dry_run=False):
    """Sex one bout from its two flies' pose and canonicalize so male == fly{male_slot}.

    Loads fly0/fly1 kp3d.npz, computes wing_song_cv, decides via sex_bout_from_pose,
    physically swaps the fly dirs when the male is not already at male_slot (unless
    dry_run), and writes sex.json (unless dry_run). Idempotent. Returns the decision
    dict written to sex.json."""
    kp_index = _kp_index(kp_names)

    def load_cv(fly):
        p = os.path.join(bout_dir, f"fly{fly}", "kp3d.npz")
        if not os.path.exists(p):
            return None
        with np.load(p) as d:
            return wing_song_cv(d["kp3d"], d["conf3d"], kp_index,
                                conf_min=conf_min, min_frames=min_frames)

    cv0, cv1 = load_cv(0), load_cv(1)
    if cv0 is None or cv1 is None:
        decision = dict(male_fly=None, confidence="unknown", method="missing_kp3d",
                        cv=[cv0, cv1], cv_ratio=None, note="a fly kp3d.npz is missing")
    else:
        decision = sex_bout_from_pose(cv0, cv1, mask_sex_meta=mask_sex_meta,
                                      ratio_thr=ratio_thr, high_ratio=high_ratio)

    male = decision["male_fly"]
    applied_swap = male is not None and male != male_slot
    if applied_swap and not dry_run:
        _swap_fly_dirs(bout_dir)

    out = dict(
        male_fly=(male_slot if male is not None else None),
        original_male_fly=male,
        applied_swap=bool(applied_swap),
        confidence=decision["confidence"],
        method=decision["method"],
        wing_cv_original={"fly0": _jsonnum(cv0), "fly1": _jsonnum(cv1)},
        cv_ratio=_jsonnum(decision.get("cv_ratio")),
        mask_song_cv=(mask_sex_meta.get("song_cv") if mask_sex_meta else None),
        note=decision["note"],
    )
    if not dry_run:
        with open(os.path.join(bout_dir, "sex.json"), "w") as f:
            json.dump(out, f, indent=2)
    return out
