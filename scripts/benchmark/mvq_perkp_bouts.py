#!/usr/bin/env python
"""Per-keypoint bout stability metrics for the mvq courtship-pair lifter,
promoted from the scratchpad diagnostics `perkp_all_bouts.py` /
`perkp_jump_probe.py` (see docs/benchmark/2026-09-mvq/p3b-notes.md, "P3b on
the mask-free route" and "re-lift bouts 1, 4, 28 of Session0/2025_10_20_13_20_04"
-- the tables `scripts/benchmark/mvq_v2_acceptance.py` compares a v2
checkpoint's numbers against).

DEFINITIONS (kept verbatim with the scratchpad scripts so numbers stay
comparable to the P3b tables -- read these before changing anything below):

  * **pose_jump**: a frame has >= `min_kp` (default 5) MALE keypoints whose
    frame-to-frame step exceeds `jump_mm` (default 0.5mm = 5 world units,
    `MM_PER_UNIT`=0.1) -- a real landmark cannot move that far in one frame
    at 800fps; this many at once means the identity/keypoint track jumped.
  * **straddle**: a frame has >= `min_kp` MALE keypoints nearer the FEMALE's
    3D body centroid than the male's own centroid -- i.e. sitting on her,
    the cross-fly identity-tangling failure. Scored ONLY on frames where the
    female's own centroid is finite: an unmeasurable frame (her centroid
    NaN) must NOT read as a clean ("not straddling") frame just because the
    comparison could not be made -- CLAUDE.md's history and the P3b bout-1
    finding (her centroid was NaN on 76% of frames under the r2 checkpoint,
    silently making that bout's straddle number blind rather than good).
  * **contact**: the male<->female centroid separation is below
    `contact_units` (default 15.0 units = 1.5mm).
  * **female_missing**: the female's 3D centroid is NaN (she was not
    localised at all that frame).
  * **head_tail_flips**: count of frames where the `Antenna_Base`->`Abd_tip`
    unit vector reverses direction (cos angle to the previous frame < 0) --
    a >90deg turn of the whole body axis in a single frame, i.e. a
    head/tail identity swap. Found BY NAME (`kp_names.index(...)`), never a
    bare integer -- CLAUDE.md's keypoint-order-trap history.

Everything here is by-NAME (`kp_names` is always read alongside the arrays,
never assumed) and in named units (mm/units are labelled, not bare floats) --
see CLAUDE.md's "never index a keypoint or camera axis by integer" and "label
with real names and units".

CLI:

    python scripts/benchmark/mvq_perkp_bouts.py \\
        --bouts-root .../pose_mvq_p3a_r2/bouts --bouts 1 4 28 --out perkp.json

    python scripts/benchmark/mvq_perkp_bouts.py \\
        --tracks .../coarse_mvq_p3b/fine_bout117/fine_tracks.npz --out perkp.json

`--bouts-root` mode reads the per-bout on-disk courtship schema
(`<bouts_root>/bout_NNNNN/fly{0,1}/kp3d.npz`, fly0=female, fly1=male,
MODEL/XML keypoint order per CLAUDE.md -- `kp_names` is read from each npz,
never assumed, and fly0/fly1 are required to agree on it). `--tracks` mode
reads a `coarse_tracks.npz`/`fine_tracks.npz` written by
`jarvis_jax.tracking.coarse_track.write_coarse_tracks` (`kp3d` shape
(F=2,T,K,3), fly index 0=female/1=male, `kp_names` alongside it).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import warnings

import numpy as np

# Mirrored from `jarvis_jax.train.train_mvq.MM_PER_UNIT` / `jarvis_jax.data.
# pseudo_gates.MM_PER_UNIT` (both 0.1) -- duplicated here (not imported) so
# this module stays pure numpy/argparse, importable on CPU with no jax/flax
# dependency, matching `pseudo_gates.py`'s own precedent for this constant.
MM_PER_UNIT = 0.1


def perkp_bout_metrics(kp3d_by_fly, kp_names, *, male_fly=1, jump_mm=0.5,
                       min_kp=5, contact_units=15.0):
    """One bout's per-frame + per-keypoint stability metrics.

    Args:
        kp3d_by_fly: {fly_index: (T,K,3) float array}, exactly two flies --
            `male_fly` and exactly one other (read as the "female"). NaN
            marks a missing keypoint/frame.
        kp_names: (K,) sequence of keypoint name strings, in the SAME order
            as `kp3d_by_fly`'s keypoint axis (never assumed -- callers must
            resolve this by name from whatever wrote the arrays).
        male_fly: key into `kp3d_by_fly` for the male track (default 1,
            matching CLAUDE.md's "male = fly1 after canonicalization").
        jump_mm: frame-to-frame step (mm) above which a keypoint counts as
            "jumping" (converted to world units via `MM_PER_UNIT`).
        min_kp: number of keypoints that must jump/straddle in the same
            frame for that frame to count.
        contact_units: male<->female centroid separation (world units)
            below which a frame counts as "contact".

    Returns dict with T, pose_jump_frac, straddle_frac, contact_frac,
    female_missing_frac, head_tail_flips, per_kp_jump_frac (K,) float array,
    per_kp_names (the same `kp_names`, so a caller never has one without the
    other).
    """
    names = [str(n) for n in kp_names]
    fly_ids = sorted(kp3d_by_fly)
    if male_fly not in fly_ids:
        raise ValueError(f"male_fly={male_fly!r} not in kp3d_by_fly keys {fly_ids!r}")
    female_ids = [f for f in fly_ids if f != male_fly]
    if len(female_ids) != 1:
        raise ValueError(
            f"perkp_bout_metrics needs exactly one other (female) fly besides "
            f"male_fly={male_fly!r}; got fly ids {fly_ids!r}")
    female_fly = female_ids[0]

    male = np.asarray(kp3d_by_fly[male_fly], dtype=np.float64)     # (T,K,3)
    female = np.asarray(kp3d_by_fly[female_fly], dtype=np.float64)  # (T,K,3)
    if male.shape != female.shape:
        raise ValueError(f"male {male.shape} and female {female.shape} kp3d shapes disagree")
    T, K = male.shape[0], male.shape[1]
    if len(names) != K:
        raise ValueError(f"kp_names has {len(names)} entries but kp3d has K={K} keypoints")

    if T == 0:
        return {"T": 0, "pose_jump_frac": float("nan"), "straddle_frac": float("nan"),
                "contact_frac": float("nan"), "female_missing_frac": float("nan"),
                "head_tail_flips": 0, "per_kp_jump_frac": np.zeros(K, np.float64),
                "per_kp_names": names}

    jump_units = jump_mm / MM_PER_UNIT

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN rows are expected (missing fly)
        c1 = np.nanmean(male, axis=1)      # (T,3) male centroid
        c0 = np.nanmean(female, axis=1)    # (T,3) female centroid

        # --- pose jump: >= min_kp male keypoints step > jump_units in one frame ---
        step = np.linalg.norm(np.diff(male, axis=0), axis=-1)   # (T-1,K); NaN where either side is NaN
        jump_kp = np.zeros((T, K), bool)
        jump_kp[1:] = step > jump_units                          # NaN comparisons are False (no false positives)
        n_jump = jump_kp.sum(axis=1)
        pose_jump = n_jump >= min_kp

        # --- straddle: >= min_kp male keypoints nearer the female centroid than
        # their own, but ONLY on frames where the female centroid itself exists ---
        fem_ok = np.isfinite(c0).all(axis=1)
        d_own = np.linalg.norm(male - c1[:, None], axis=-1)      # (T,K)
        d_fem = np.linalg.norm(male - c0[:, None], axis=-1)      # (T,K)
        strad_kp = (d_fem < d_own) & np.isfinite(d_fem) & np.isfinite(d_own)
        n_strad = strad_kp.sum(axis=1)
        # explicit `& fem_ok` (redundant with the NaN propagation above when
        # the female is missing) kept as an intentional, self-documenting
        # safety gate -- the P3b bout-1 lesson is exactly that an
        # unmeasurable frame must never silently read as "not straddling".
        straddle = (n_strad >= min_kp) & fem_ok

        # --- female missing / contact ---
        female_missing = ~fem_ok
        sep = np.linalg.norm(c1 - c0, axis=-1)
        contact = np.isfinite(sep) & (sep < contact_units)

        # --- head-tail flip: Antenna_Base->Abd_tip axis reversal, BY NAME ---
        if "Antenna_Base" not in names or "Abd_tip" not in names:
            raise ValueError(
                f"kp_names is missing 'Antenna_Base' and/or 'Abd_tip' -- cannot "
                f"compute head_tail_flips (names present: {names})")
        i_a, i_t = names.index("Antenna_Base"), names.index("Abd_tip")
        axis_vec = male[:, i_t] - male[:, i_a]                    # (T,3)
        axis_norm = np.linalg.norm(axis_vec, axis=-1, keepdims=True)
        unit = np.where(axis_norm > 0, axis_vec / np.where(axis_norm == 0, 1.0, axis_norm), np.nan)
        cosang = np.full(T, np.nan)
        cosang[1:] = np.sum(unit[1:] * unit[:-1], axis=-1)
        flips = int(np.nansum(cosang < 0))

    return {
        "T": int(T),
        "pose_jump_frac": float(pose_jump.mean()),
        "straddle_frac": float(straddle.mean()),
        "contact_frac": float(contact.mean()),
        "female_missing_frac": float(female_missing.mean()),
        "head_tail_flips": flips,
        "per_kp_jump_frac": jump_kp.mean(axis=0),
        "per_kp_names": names,
    }


def bout_dir_metrics(bout_dir, *, male_fly=1, **kwargs):
    """Wrapper over the on-disk courtship per-bout schema:
    `<bout_dir>/fly0/kp3d.npz` (female) + `<bout_dir>/fly1/kp3d.npz` (male),
    each carrying `kp3d` (T,K,3) and `kp_names` (K,) -- the MODEL/XML
    keypoint order (CLAUDE.md: `kp3d.npz` is written AFTER
    `reorder_detector_to_model`). Refuses to compare if the two flies'
    `kp_names` disagree (never silently assume position-for-position
    agreement)."""
    flies = {}
    names_ref = None
    for f in (0, 1):
        p = os.path.join(bout_dir, f"fly{f}", "kp3d.npz")
        z = np.load(p, allow_pickle=False)
        names = [str(n) for n in z["kp_names"]]
        if names_ref is None:
            names_ref = names
        elif names != names_ref:
            raise ValueError(
                f"{p}: kp_names differ from fly0's -- refusing to compare "
                f"mismatched keypoint order (CLAUDE.md keypoint-order trap)")
        flies[f] = np.asarray(z["kp3d"])
    return perkp_bout_metrics(flies, names_ref, male_fly=male_fly, **kwargs)


def tracks_npz_metrics(path, *, male_fly=1, **kwargs):
    """Wrapper over a `coarse_tracks.npz`/`fine_tracks.npz`
    (`jarvis_jax.tracking.coarse_track.write_coarse_tracks` schema): `kp3d`
    (F=2,T,K,3, fly index 0=female/1=male) + `kp_names` (K,)."""
    z = np.load(path, allow_pickle=False)
    kp3d = np.asarray(z["kp3d"])
    names = [str(n) for n in z["kp_names"]]
    F = kp3d.shape[0]
    if F != 2:
        raise ValueError(f"{path}: expected exactly 2 flies (F=2) in kp3d, got F={F}")
    flies = {f: kp3d[f] for f in range(F)}
    return perkp_bout_metrics(flies, names, male_fly=male_fly, **kwargs)


def _bout_dir_for(bouts_root, bout_idx):
    exact = os.path.join(bouts_root, f"bout_{bout_idx:05d}")
    if os.path.isdir(exact):
        return exact
    hits = sorted(glob.glob(os.path.join(bouts_root, f"bout_*{bout_idx}")))
    if len(hits) == 1:
        return hits[0]
    raise FileNotFoundError(f"no unique bout dir for bout_idx={bout_idx} under {bouts_root}")


def _jsonable(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bouts-root", default=None, help="dir of bout_NNNNN/fly{0,1}/kp3d.npz")
    ap.add_argument("--bouts", type=int, nargs="*", default=None, help="bout indices under --bouts-root")
    ap.add_argument("--all-bouts", action="store_true", help="use every bout_* dir under --bouts-root")
    ap.add_argument("--tracks", default=None, help="a coarse_tracks.npz/fine_tracks.npz path instead")
    ap.add_argument("--male-fly", type=int, default=1)
    ap.add_argument("--jump-mm", type=float, default=0.5)
    ap.add_argument("--min-kp", type=int, default=5)
    ap.add_argument("--contact-units", type=float, default=15.0)
    ap.add_argument("--out", default=None, help="JSON output path (also printed to stdout)")
    a = ap.parse_args()

    if bool(a.bouts_root) == bool(a.tracks):
        ap.error("pass exactly one of --bouts-root or --tracks")

    kw = dict(male_fly=a.male_fly, jump_mm=a.jump_mm, min_kp=a.min_kp, contact_units=a.contact_units)
    result = {}
    if a.tracks:
        result["tracks"] = {a.tracks: _jsonable(tracks_npz_metrics(a.tracks, **kw))}
    else:
        if a.all_bouts:
            dirs = sorted(glob.glob(os.path.join(a.bouts_root, "bout_*")))
            bout_ids = [int(os.path.basename(d).split("_")[-1]) for d in dirs]
        else:
            if not a.bouts:
                ap.error("--bouts-root needs --bouts <idx ...> or --all-bouts")
            bout_ids = a.bouts
        bouts = {}
        for b in bout_ids:
            d = _bout_dir_for(a.bouts_root, b)
            bouts[str(b)] = _jsonable(bout_dir_metrics(d, **kw))
        result["bouts"] = bouts

    text = json.dumps(result, indent=1)
    print(text)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as f:
            f.write(text)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
