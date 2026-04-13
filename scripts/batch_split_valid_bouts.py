#!/usr/bin/env python3
"""Classify per-fly preprocessed bouts into valid-bucket files for STAC.

Replaces the old paired-bout merge/split flow (``batch_pair_bouts.py`` →
``merge_paired_bouts.py`` + ``run_stac_paired.py``).

For each ``Predictions_3D_*`` folder we read the per-fly preprocessed h5s
created by ``preprocess_keypoints_for_ik.py`` (which already carry a per-frame
``valid_fly`` mask in each bout) and emit up to three bucket files used as
STAC inputs:

  preprocessing/preprocessed_bout_<a>_<d>_fly0_only.h5  (fly0 valid, fly1 not)
  preprocessing/preprocessed_bout_<a>_<d>_fly1_only.h5  (fly1 valid, fly0 not)
  preprocessing/preprocessed_bout_<a>_<d>_both.h5       (both valid; one entry
                                                          per fly, fly0 first)

A bout is the **union** of the three conditions: if both flies are valid in
the same bout it lands in ``_both`` (not in either ``_only`` file). If only
one fly clears its threshold, the bout lands in that fly's ``_only`` file.
Empty buckets are not written. Folders that produce no bouts at all are
reported and skipped without crashing the pipeline.

A small ``preprocessed_bout_<a>_<d>_split_index.json`` manifest is written
alongside the bucket files for debugging / downstream alignment.

Usage:
    python scripts/batch_split_valid_bouts.py --dataset courtship --anatomy v2 \\
        --base-dir /path/to/Johnson_lab/courtship
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from utils import io_dict_to_hdf5 as ioh5  # noqa: E402
from utils.identity_relink import RelinkConfig, relink_pair_bouted  # noqa: E402
from utils.keypoint_filter import despike_isolated_spikes, medfilt_despike  # noqa: E402
from utils.pair_validity import (  # noqa: E402
    PairValidityConfig,
    compute_pair_validity,
    pair_validity_config_from_dict,
)
from utils.song_analysis import SongAnalysisConfig, analyze_fly_song  # noqa: E402
from utils.sex_id import SexIdConfig, identify_male_female  # noqa: E402


BUCKETS = ("fly0_only", "fly1_only", "both")


def _fix_interfly_scale(b0: dict, b1: dict) -> None:
    """Correct inter-fly distance distortion from per-fly Procrustes scaling.

    Per-fly Procrustes applies ``kp *= scale_i`` which distorts centroid
    positions when scale_i differs between flies (common: 5-20% difference).
    We undo each fly's per-fly scale on the centroid and reapply the mean
    scale so that body shape stays per-fly correct but inter-fly distances
    use a shared scale.  Mutates ``b0`` and ``b1`` keypoints in place.
    """
    ai0 = b0.get("alignment_info")
    ai1 = b1.get("alignment_info")
    if ai0 is None or ai1 is None:
        return
    s0 = float(ai0.get("scales", 1.0))
    s1 = float(ai1.get("scales", 1.0))
    if s0 == 0 or s1 == 0 or s0 == s1:
        return

    s_mean = (s0 + s1) / 2.0

    # Undo per-fly scale and reapply the shared mean scale so both flies
    # are in identical units.  Body shape changes are negligible (~4% per
    # fly) and inter-fly distances are now exact.
    b0["keypoints"] = b0["keypoints"] * (s_mean / s0)
    b1["keypoints"] = b1["keypoints"] * (s_mean / s1)

    # Update alignment_info so downstream knows the shared scale was applied
    ai0["shared_scale"] = s_mean
    ai1["shared_scale"] = s_mean


def _resolve_kp_names(d0: dict, d1: dict) -> list[str]:
    """Find a kp_names list. Prefer info-level, fall back to per-bout."""
    for d in (d0, d1):
        info = d.get("info", {}) or {}
        v = info.get("kp_names")
        if v is not None:
            if isinstance(v, dict):
                v = [v[k] for k in sorted(v.keys(), key=lambda x: int(x))]
            return list(v)
    for d in (d0, d1):
        for k, v in d.items():
            if k == "info" or not isinstance(v, dict):
                continue
            if "kp_names" in v:
                return list(v["kp_names"])
    raise KeyError("kp_names not found in either per-fly h5 (info or bouts)")


def _per_frame_keys(bout: dict, T: int) -> list[str]:
    """Names of arrays in a bout dict whose leading dim equals T."""
    out: list[str] = []
    for k, v in bout.items():
        if not isinstance(v, np.ndarray):
            continue
        if v.ndim == 0 or v.shape[0] != T:
            continue
        out.append(k)
    return out


def _swap_rows_between(b0: dict, b1: dict, mask: np.ndarray,
                       skip_keys: set[str]) -> list[str]:
    """Swap rows of every per-frame array (matching shapes) between b0 / b1
    wherever ``mask`` is True. ``skip_keys`` lists keys that the caller has
    already overwritten and must not be touched. Returns the keys swapped."""
    if not mask.any():
        return []
    T = mask.shape[0]
    keys0 = set(_per_frame_keys(b0, T)) - skip_keys
    keys1 = set(_per_frame_keys(b1, T)) - skip_keys
    swapped: list[str] = []
    for k in sorted(keys0 & keys1):
        a, c = b0[k], b1[k]
        if a.shape != c.shape:
            continue
        tmp = a[mask].copy()
        a[mask] = c[mask]
        c[mask] = tmp
        swapped.append(k)
    return swapped


def joint_relink_shared_bouts(
    d0: dict,
    d1: dict,
    shared: list[str],
    kp_names: list[str],
    relink_cfg: RelinkConfig,
    pv_cfg: PairValidityConfig,
    verbose: bool = True,
) -> dict:
    """Run a single bout-aware joint relink across every shared bout, then
    overwrite ``keypoints``, swap any other per-frame arrays, and replace
    ``valid_fly`` / install ``valid_fly0`` / ``valid_fly1`` / ``valid_both``
    via the *real* (fly0, fly1) ``compute_pair_validity``.

    Mutates d0 and d1 in place. Returns a per-bout summary dict.

    IMPORTANT: detection runs on ``orig_keypoints`` (raw, pre-Procrustes
    world frame). The per-fly Procrustes alignment in
    ``preprocess_keypoints_for_ik.py`` scales each fly *independently* to
    the MuJoCo model size, which breaks inter-fly distances on the
    post-Procrustes ``keypoints`` arrays — you'd be comparing fly0 in
    fly0-units against fly1 in fly1-units, and the position-aware Viterbi
    transition cost would be miscalibrated. ``orig_keypoints`` is stored
    PER FLY by the preprocessing stage but in the *shared raw camera frame*
    (same units for both flies), which is what the relink algorithm needs.
    The discovered swap_state is identity-equivariant, so we can apply it to
    both ``orig_keypoints`` and the post-Procrustes ``keypoints`` (and any
    other per-frame arrays) without recomputing anything.
    """
    fly0_kp: dict[str, np.ndarray] = {}
    fly1_kp: dict[str, np.ndarray] = {}
    trimmed: list[str] = []
    detection_source: dict[str, str] = {}
    for bk in shared:
        b0 = d0[bk]
        b1 = d1[bk]
        # Prefer orig_keypoints (raw world frame, joint-fly comparable). Fall
        # back to keypoints if orig_keypoints is missing — that's only safe
        # for datasets without per-fly Procrustes, but we keep it as a
        # graceful degradation.
        if "orig_keypoints" in b0 and "orig_keypoints" in b1:
            src = "orig_keypoints"
        elif "keypoints" in b0 and "keypoints" in b1:
            src = "keypoints"
        else:
            continue
        a = np.asarray(b0[src])
        c = np.asarray(b1[src])
        if a.shape != c.shape:
            T = min(a.shape[0], c.shape[0])
            if T == 0 or a.shape[1:] != c.shape[1:]:
                continue
            a = a[:T]
            c = c[:T]
            # Trim every per-frame array on both sides so the bouts stay
            # internally consistent before we swap rows below.
            for tgt, raw in ((b0, d0[bk]), (b1, d1[bk])):
                for k, v in list(raw.items()):
                    if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] >= T:
                        raw[k] = v[:T]
            trimmed.append(bk)
        fly0_kp[bk] = a
        fly1_kp[bk] = c
        detection_source[bk] = src

    if not fly0_kp:
        return {"n_pairs": 0, "n_pairs_swap": 0, "trimmed": trimmed}

    rl0_dict, rl1_dict, logs = relink_pair_bouted(
        fly0_kp, fly1_kp, kp_names, relink_cfg,
    )

    n_pairs = 0
    n_pairs_swap = 0
    total_frames = 0
    swapped_frames = 0
    swap_segments = 0
    colocated_frames = 0
    n_pairs_colocated = 0
    per_bout: dict[str, dict] = {}

    for bk in sorted(rl0_dict.keys()):
        n_pairs += 1
        log = logs[bk]
        swap_state = np.asarray(log["swap_state"], dtype=bool)
        T = swap_state.shape[0]

        b0 = d0[bk]
        b1 = d1[bk]
        # The swap_state was discovered from orig_keypoints (raw world
        # frame) but is identity-equivariant: applying it to *every*
        # per-frame array on both bout dicts keeps everything consistent.
        # That includes the post-Procrustes ``keypoints`` array that STAC
        # IK consumes downstream.
        _swap_rows_between(b0, b1, swap_state, skip_keys=set())

        # Build the post-relink (keypoints, valid_*) views needed by
        # pair_validity. Use the post-Procrustes ``keypoints`` (in fly-body
        # units) because pair_validity's ground/floor thresholds are
        # calibrated for that scale.
        rl0_kp = np.asarray(b0.get("keypoints"))
        rl1_kp = np.asarray(b1.get("keypoints"))
        if rl0_kp.shape != rl1_kp.shape or rl0_kp.shape[0] != T:
            # Fall back to the detection-source arrays if the per-fly
            # keypoints don't line up (shouldn't happen after the trim
            # pass above, but be defensive).
            rl0_kp = rl0_dict[bk]
            rl1_kp = rl1_dict[bk]

        # Edge-NaN masks from preprocessing follow the keypoints through
        # the swap pass above (they are (T, N) per-frame arrays and therefore
        # handled by _swap_rows_between). Pass them to pair_validity so any
        # phantom frames filled by bounded linear extrapolation in
        # keypoint_filter are still marked as not-valid here.
        edge_nan0 = b0.get("edge_nan")
        edge_nan1 = b1.get("edge_nan")
        pv_out = compute_pair_validity(
            rl0_kp, rl1_kp, kp_names, cfg=pv_cfg, swap_state=swap_state,
            edge_nan_mask_fly0=edge_nan0, edge_nan_mask_fly1=edge_nan1,
        )
        for tgt in (b0, b1):
            tgt["valid_fly0"] = pv_out["valid_fly0"]
            tgt["valid_fly1"] = pv_out["valid_fly1"]
            tgt["valid_both"] = pv_out["valid_both"]
            tgt["identity_valid"] = pv_out["identity_valid"]
        # Per-fly single-fly masks for the legacy `valid_fly` consumers
        # (the bucket loop falls back to these if valid_fly0/1 are absent).
        b0["valid_fly"] = pv_out["valid_fly0"]
        b1["valid_fly"] = pv_out["valid_fly1"]
        # Surface the identity-collapse mask so downstream consumers can
        # plot it / filter on it without recomputing the distance.
        pair_colocated = pv_out.get("pair_colocated")
        if pair_colocated is not None:
            b0["pair_colocated"] = pair_colocated
            b1["pair_colocated"] = pair_colocated
        b0["swap_state"] = swap_state.copy()
        b1["swap_state"] = swap_state.copy()
        b0["n_swap_segments"] = int(log["n_swap_segments"])
        b1["n_swap_segments"] = int(log["n_swap_segments"])
        b0["fraction_swapped"] = float(log["fraction_swapped"])
        b1["fraction_swapped"] = float(log["fraction_swapped"])

        n_swap = int(swap_state.sum())
        n_coloc = int(pv_out.get("n_colocated", 0))
        total_frames += T
        swapped_frames += n_swap
        swap_segments += int(log["n_swap_segments"])
        colocated_frames += n_coloc
        if log["n_swap_segments"] > 0:
            n_pairs_swap += 1
        if n_coloc > 0:
            n_pairs_colocated += 1
        per_bout[bk] = {
            "n_frames": T,
            "n_swapped": n_swap,
            "n_swap_segments": int(log["n_swap_segments"]),
            "fraction_swapped": float(log["fraction_swapped"]),
            "n_colocated": n_coloc,
        }

    summary = {
        "n_pairs": n_pairs,
        "n_pairs_swap": n_pairs_swap,
        "total_frames": total_frames,
        "swapped_frames": swapped_frames,
        "fraction_swapped": (swapped_frames / total_frames) if total_frames else 0.0,
        "n_swap_segments": swap_segments,
        "colocated_frames": colocated_frames,
        "n_pairs_colocated": n_pairs_colocated,
        "fraction_colocated": (colocated_frames / total_frames) if total_frames else 0.0,
        "min_pair_separation_mm": float(pv_cfg.min_pair_separation_mm),
        "trimmed_bouts": trimmed,
        "per_bout": per_bout,
    }
    if verbose:
        print(f"  [joint-relink] {n_pairs} bouts, {n_pairs_swap} with swap, "
              f"{swap_segments} swap segments, "
              f"{summary['fraction_swapped']*100:.2f}% frames swapped"
              + (f", trimmed={len(trimmed)}" if trimmed else ""))
        if pv_cfg.min_pair_separation_mm > 0:
            print(f"  [joint-relink] identity-collapse guard: "
                  f"{n_pairs_colocated}/{n_pairs} bouts affected, "
                  f"{colocated_frames} frames dropped "
                  f"({summary['fraction_colocated']*100:.2f}%) "
                  f"[min_pair_separation_mm={pv_cfg.min_pair_separation_mm}]")
    return summary


def _apply_sex_cleaning(
    kp: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Apply despike + optional medfilt_despike to a keypoint array.

    Parameters
    ----------
    kp : (T, N, 3) array
    cfg : dict with 'despike_iterations' and optional 'medfilt_despike' sub-dict

    Returns
    -------
    cleaned : (T, N, 3) array
    stats : dict with counts of replaced frames
    """
    stats: dict = {}
    iters = int(cfg.get("despike_iterations", 1))
    if iters > 1:
        kp, n_fixed = despike_isolated_spikes(kp, max_iterations=iters)
        stats["despike_replaced"] = int(n_fixed)

    mfd = cfg.get("medfilt_despike", {})
    if mfd.get("enabled", False):
        kp, n_mfd = medfilt_despike(
            kp,
            kernel=int(mfd.get("kernel", 7)),
            threshold_factor=float(mfd.get("threshold_factor", 10.0)),
        )
        stats["medfilt_replaced"] = int(n_mfd)

    return kp, stats


def sex_clean_shared_bouts(
    d0: dict,
    d1: dict,
    shared: list[str],
    kp_names: list[str],
    sex_cleaning_cfg: dict,
    verbose: bool = True,
) -> dict:
    """Run song analysis + sex ID on each shared bout, then apply per-sex
    aggressive cleaning to keypoints in-place.

    Mutates d0 and d1. Returns a summary dict.
    """
    male_cfg = sex_cleaning_cfg.get("male", {})
    female_cfg = sex_cleaning_cfg.get("female", {})
    per_bout: dict[str, dict] = {}
    n_male_fly0 = 0
    n_male_fly1 = 0

    for bk in shared:
        b0 = d0[bk]
        b1 = d1[bk]

        kp0 = np.asarray(b0.get("keypoints"))
        kp1 = np.asarray(b1.get("keypoints"))
        if kp0.ndim != 3 or kp1.ndim != 3:
            continue

        # Clip to common length
        T = min(kp0.shape[0], kp1.shape[0])
        kp0 = kp0[:T]
        kp1 = kp1[:T]

        # Song analysis (world-frame keypoints only — no xpos_ego/qpos pre-IK)
        try:
            song0 = analyze_fly_song(kp0, None, None, kp_names)
            song1 = analyze_fly_song(kp1, None, None, kp_names)
            sex = identify_male_female(song0, song1, kp0, kp1, kp_names)
        except Exception as e:
            if verbose:
                print(f"  [sex-clean] {bk}: song/sex failed ({e}) — skipping")
            continue

        male_id = sex["male_id"]  # 'fly0' or 'fly1'
        bout_info: dict = {"male_id": male_id}

        if male_id == "fly0":
            n_male_fly0 += 1
            male_bout, female_bout = b0, b1
        else:
            n_male_fly1 += 1
            male_bout, female_bout = b1, b0

        # Clean female keypoints aggressively
        fem_kp = np.asarray(female_bout["keypoints"])[:T]
        fem_kp, fem_stats = _apply_sex_cleaning(fem_kp, female_cfg)
        female_bout["keypoints"] = fem_kp
        bout_info["female_stats"] = fem_stats

        # Clean male keypoints conservatively (usually no-op)
        male_kp_arr = np.asarray(male_bout["keypoints"])[:T]
        male_kp_arr, male_stats = _apply_sex_cleaning(male_kp_arr, male_cfg)
        male_bout["keypoints"] = male_kp_arr
        bout_info["male_stats"] = male_stats

        # Store sex_id in bout dicts for provenance
        b0["sex_id"] = male_id
        b1["sex_id"] = male_id
        b0["song_fraction_fly0"] = float(song0["summary"]["song_fraction"])
        b0["song_fraction_fly1"] = float(song1["summary"]["song_fraction"])
        b1["song_fraction_fly0"] = b0["song_fraction_fly0"]
        b1["song_fraction_fly1"] = b0["song_fraction_fly1"]

        per_bout[bk] = bout_info

    summary = {
        "n_bouts": len(per_bout),
        "n_male_fly0": n_male_fly0,
        "n_male_fly1": n_male_fly1,
        "per_bout": per_bout,
    }
    if verbose:
        print(f"  [sex-clean] {len(per_bout)}/{len(shared)} bouts: "
              f"male=fly0 in {n_male_fly0}, male=fly1 in {n_male_fly1}")
        # Aggregate female cleaning stats
        total_despike = sum(
            v.get("female_stats", {}).get("despike_replaced", 0)
            for v in per_bout.values()
        )
        total_medfilt = sum(
            v.get("female_stats", {}).get("medfilt_replaced", 0)
            for v in per_bout.values()
        )
        if total_despike or total_medfilt:
            print(f"  [sex-clean] female cleaning: {total_despike} despike + "
                  f"{total_medfilt} medfilt replacements across all bouts")
    return summary


def _find_folders(base_dir: Path) -> list[Path]:
    if base_dir.is_dir() and base_dir.match("Predictions_3D_*"):
        return [base_dir]
    return sorted({p for p in base_dir.rglob("Predictions_3D_*") if p.is_dir()})


def _get_thresholds(d0: dict) -> tuple[int, int]:
    """Pull min_solo_frames / min_paired_frames from preprocessed info if present."""
    pv = (d0.get("info", {}) or {}).get("pair_validity", {}) or {}
    min_solo = int(pv.get("min_solo_frames", 30))
    min_paired = int(pv.get("min_paired_frames", 30))
    return min_solo, min_paired


def _bout_count(mask) -> int:
    return int(np.asarray(mask, dtype=bool).sum())


def _copy_bout(src_bout: dict) -> dict:
    """Shallow copy a bout dict — values are arrays/lists we don't mutate."""
    return {k: v for k, v in src_bout.items()}


def _apply_mask(src_bout: dict, mask: np.ndarray) -> dict:
    """Return a new bout dict with per-frame arrays sliced by ``mask``.

    This drops every frame where ``mask`` is False, so the resulting bout
    contains only the frames where that fly's tracking was trusted. Any value
    whose first axis matches ``T = mask.size`` is treated as per-frame and
    sliced; everything else (scalars, nested dicts like ``alignment_info``,
    kp_names, skeleton_edges, …) is passed through unchanged.

    Trimming here is what keeps NaN/untracked frames out of STAC's fit
    window. Without it, ``batch_run_stac.merge_fly_preprocessed`` would
    concatenate the untrimmed bouts into ``_merged.h5`` and STAC's first
    500-frame offset optimization would see all-NaN rows and produce
    ``Final error of nan``.
    """
    mask = np.asarray(mask, dtype=bool)
    T = int(mask.size)
    out: dict = {}
    for k, v in src_bout.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == T:
            out[k] = v[mask]
        else:
            out[k] = v
    return out


def classify_and_split(
    fly0_path: Path,
    fly1_path: Path,
    out_paths: dict[str, Path],
    index_path: Path,
    force: bool,
    relink_enabled: bool = True,
    relink_cfg: RelinkConfig | None = None,
    sex_cleaning_cfg: dict | None = None,
) -> dict:
    """Read per-fly preprocessed h5s and emit bucket files. Returns manifest.

    When ``relink_enabled`` is True (the default) we run a single bout-aware
    joint relink across every bout that exists in both per-fly h5s, then
    recompute pair_validity using the *real* (fly0, fly1) pair before the
    bucket loop. This is the only place in the pipeline where both flies
    coexist in memory, so it's the right home for the cross-fly identity
    check — see plans/concurrent-leaping-liskov.md.

    When ``sex_cleaning_cfg`` is provided and enabled, we run song analysis +
    sex ID per bout after identity relink, then apply aggressive cleaning to
    the female fly and conservative cleaning to the male fly before pair
    validity and bucket splitting.
    """
    if not force and all(p.exists() for p in out_paths.values()) and index_path.exists():
        with open(index_path) as fh:
            return json.load(fh)

    d0 = ioh5.load(fly0_path, enable_jax=False)
    d1 = ioh5.load(fly1_path, enable_jax=False)

    info0 = d0.get("info", {}) or {}
    info1 = d1.get("info", {}) or {}
    min_solo, min_paired = _get_thresholds(d0)

    bout_keys0 = sorted(k for k in d0.keys() if k != "info")
    bout_keys1 = sorted(k for k in d1.keys() if k != "info")
    shared = sorted(set(bout_keys0) & set(bout_keys1))

    # Resolve kp_names once (needed by relink, sex cleaning, pair validity)
    kp_names: list[str] | None = None
    try:
        kp_names = _resolve_kp_names(d0, d1)
    except KeyError as e:
        print(f"  [kp_names] not found ({e})")

    # Joint bout-aware identity relink + real pair_validity. Mutates d0/d1
    # in place so the bucket loop below sees corrected keypoints and the
    # _both.h5 we write later contains the relinked data — STAC IK then
    # runs on cross-fly-correct keypoints.
    relink_summary: dict | None = None
    if relink_enabled and shared and kp_names is not None:
        pv_cfg = pair_validity_config_from_dict(info0.get("pair_validity"))
        cfg_obj = relink_cfg if relink_cfg is not None else RelinkConfig()
        relink_summary = joint_relink_shared_bouts(
            d0, d1, shared, kp_names, cfg_obj, pv_cfg, verbose=True,
        )

    # Per-sex aggressive cleaning: song analysis → sex ID → clean female.
    # Config source priority: CLI override > h5 info (echoed by preprocess).
    sex_summary: dict | None = None
    _sc_cfg = sex_cleaning_cfg
    if _sc_cfg is None or not _sc_cfg.get("enabled", False):
        _sc_cfg = dict(info0.get("sex_cleaning", {}) or {})
    if (_sc_cfg.get("enabled", False)
            and shared
            and kp_names is not None):
        sex_summary = sex_clean_shared_bouts(
            d0, d1, shared, kp_names, _sc_cfg, verbose=True,
        )

    fly_ids0 = list(info0.get("fly_ids", []))
    fly_ids1 = list(info1.get("fly_ids", []))
    clip_lengths0 = list(info0.get("clip_lengths", []))
    start_frames0 = list(info0.get("start_frames", []))
    end_frames0 = list(info0.get("end_frames", []))
    start_frames1 = list(info1.get("start_frames", []))
    end_frames1 = list(info1.get("end_frames", []))

    def _idx(keys, k):
        try:
            return keys.index(k)
        except ValueError:
            return None

    bucket_data: dict[str, dict] = {b: {} for b in BUCKETS}
    bucket_info: dict[str, dict] = {
        b: {"fly_ids": [], "source_flies": [], "clip_lengths": [],
            "start_frames": [], "end_frames": [], "bucket": []}
        for b in BUCKETS
    }
    for b in BUCKETS:
        for k in ("kp_names", "skeleton_edges"):
            if k in info0:
                bucket_info[b][k] = info0[k]
        # Echo pair_validity config so downstream stages can introspect
        if "pair_validity" in info0:
            bucket_info[b]["pair_validity"] = info0["pair_validity"]

    counters = {b: 0 for b in BUCKETS}
    index_entries: list[dict] = []

    for bk in shared:
        b0 = d0[bk]
        b1 = d1[bk]
        # Prefer the cross-fly masks installed by the joint relink pass; fall
        # back to per-fly single-fly masks if relink was disabled or skipped.
        if "valid_fly0" in b0 and "valid_fly1" in b0 and "valid_both" in b0:
            v0 = np.asarray(b0["valid_fly0"], dtype=bool)
            v1 = np.asarray(b0["valid_fly1"], dtype=bool)
            v_both = np.asarray(b0["valid_both"], dtype=bool)
        else:
            v0 = np.asarray(b0.get("valid_fly", []), dtype=bool)
            v1 = np.asarray(b1.get("valid_fly", []), dtype=bool)
            v_both = v0 & v1 if v0.shape == v1.shape else np.zeros(0, dtype=bool)
        if v0.shape != v1.shape or v0.size == 0:
            # Defensive: if masks are missing or shape-mismatched, skip the bout.
            index_entries.append({
                "src_bout_key": bk,
                "skipped": True,
                "reason": f"valid_fly shape mismatch or empty (v0={v0.shape}, v1={v1.shape})",
            })
            continue

        n0 = _bout_count(v0)
        n1 = _bout_count(v1)
        n_both = _bout_count(v_both)

        i0 = _idx(bout_keys0, bk)
        T = int(clip_lengths0[i0]) if i0 is not None and i0 < len(clip_lengths0) else int(v0.size)
        base_id0 = fly_ids0[i0] if i0 is not None and i0 < len(fly_ids0) else f"{bk}_fly0"
        sf0 = int(start_frames0[i0]) if i0 is not None and i0 < len(start_frames0) else -1
        ef0 = int(end_frames0[i0]) if i0 is not None and i0 < len(end_frames0) else -1
        i1 = _idx(bout_keys1, bk)
        base_id1 = fly_ids1[i1] if i1 is not None and i1 < len(fly_ids1) else f"{bk}_fly1"
        sf1 = int(start_frames1[i1]) if i1 is not None and i1 < len(start_frames1) else -1
        ef1 = int(end_frames1[i1]) if i1 is not None and i1 < len(end_frames1) else -1

        entry = {
            "src_bout_key": bk,
            "n_frames": T,
            "n_valid_fly0": n0,
            "n_valid_fly1": n1,
            "n_valid_both": n_both,
            "bucket": None,
            "out_keys": {},
        }

        if n_both >= min_paired:
            # 'both' wins — emit two entries (fly0 then fly1) so STAC sees
            # both. Trim to the filter_ok overlap (both flies well-tracked)
            # rather than the stricter v_both (which also requires ground
            # contact). This keeps the full temporal context for STAC while
            # preserving the v_both/v0/v1 masks as metadata for downstream
            # analysis that needs ground-contact filtering.
            filt0 = np.asarray(b0.get("filter_ok", v0), dtype=bool)
            filt1 = np.asarray(b1.get("filter_ok", v1), dtype=bool)
            filt_both = filt0 & filt1
            n_filt_both = int(filt_both.sum())
            # Fall back to v_both if filter_ok is absent
            trim_mask = filt_both if n_filt_both >= n_both else v_both
            n_stored = int(trim_mask.sum())

            new_key0 = f"bout_{counters['both']:03d}"
            counters["both"] += 1
            new_key1 = f"bout_{counters['both']:03d}"
            counters["both"] += 1

            # Fix inter-fly distance distortion from per-fly Procrustes
            _fix_interfly_scale(b0, b1)
            bucket_data["both"][new_key0] = _apply_mask(b0, trim_mask)
            bucket_data["both"][new_key1] = _apply_mask(b1, trim_mask)
            for new_key, base, src_fly, sf, ef in (
                (new_key0, base_id0, "fly0", sf0, ef0),
                (new_key1, base_id1, "fly1", sf1, ef1),
            ):
                bucket_info["both"]["fly_ids"].append(str(base))
                bucket_info["both"]["source_flies"].append(src_fly)
                bucket_info["both"]["clip_lengths"].append(n_stored)
                bucket_info["both"]["start_frames"].append(int(sf))
                bucket_info["both"]["end_frames"].append(int(ef))
                bucket_info["both"]["bucket"].append("both")
            entry["bucket"] = "both"
            entry["out_keys"] = {"fly0": new_key0, "fly1": new_key1}
            entry["n_stored"] = {"fly0": n_stored, "fly1": n_stored}
        else:
            if n0 >= min_solo:
                # fly0_only: trim to frames where fly0 was valid; stored len = n0
                new_key = f"bout_{counters['fly0_only']:03d}"
                counters["fly0_only"] += 1
                bucket_data["fly0_only"][new_key] = _apply_mask(b0, v0)
                bucket_info["fly0_only"]["fly_ids"].append(str(base_id0))
                bucket_info["fly0_only"]["source_flies"].append("fly0")
                bucket_info["fly0_only"]["clip_lengths"].append(n0)
                bucket_info["fly0_only"]["start_frames"].append(int(sf0))
                bucket_info["fly0_only"]["end_frames"].append(int(ef0))
                bucket_info["fly0_only"]["bucket"].append("fly0_only")
                entry["bucket"] = "fly0_only"
                entry["out_keys"] = {"fly0": new_key}
                entry["n_stored"] = {"fly0": n0}
            if n1 >= min_solo:
                # fly1_only: trim to frames where fly1 was valid; stored len = n1
                new_key = f"bout_{counters['fly1_only']:03d}"
                counters["fly1_only"] += 1
                bucket_data["fly1_only"][new_key] = _apply_mask(b1, v1)
                bucket_info["fly1_only"]["fly_ids"].append(str(base_id1))
                bucket_info["fly1_only"]["source_flies"].append("fly1")
                bucket_info["fly1_only"]["clip_lengths"].append(n1)
                bucket_info["fly1_only"]["start_frames"].append(int(sf1))
                bucket_info["fly1_only"]["end_frames"].append(int(ef1))
                bucket_info["fly1_only"]["bucket"].append("fly1_only")
                # If neither solo passed, bucket stays None.
                if entry["bucket"] is None:
                    entry["bucket"] = "fly1_only"
                    entry["out_keys"] = {"fly1": new_key}
                    entry["n_stored"] = {"fly1": n1}
                elif entry["bucket"] == "fly0_only":
                    entry["bucket"] = "fly0_only+fly1_only"
                    entry["out_keys"]["fly1"] = new_key
                    entry.setdefault("n_stored", {})["fly1"] = n1

        index_entries.append(entry)

    # Write bucket files (skip empties)
    written = {}
    for b in BUCKETS:
        if counters[b] == 0:
            continue
        bucket_data[b]["info"] = bucket_info[b]
        out = out_paths[b]
        out.parent.mkdir(parents=True, exist_ok=True)
        ioh5.save(out, bucket_data[b])
        written[b] = str(out)
        print(f"  ✓ {b}: {counters[b]} entries → {out.name}")

    manifest = {
        "fly0_h5": str(fly0_path),
        "fly1_h5": str(fly1_path),
        "min_solo_frames": min_solo,
        "min_paired_frames": min_paired,
        "n_shared_bouts": len(shared),
        "counts": counters,
        "outputs": written,
        "bouts": index_entries,
    }
    if relink_summary is not None:
        manifest["identity_relink"] = {
            k: v for k, v in relink_summary.items() if k != "per_bout"
        }
    if sex_summary is not None:
        manifest["sex_cleaning"] = {
            k: v for k, v in sex_summary.items() if k != "per_bout"
        }
    # Atomic write: json.dump to a .tmp sibling, fsync, then os.replace. This
    # keeps the manifest in lockstep with the bucket h5s — if a SLURM
    # preemption kills us mid-write we never leave a truncated index_path.
    tmp_index_path = index_path.with_suffix(index_path.suffix + ".tmp")
    with open(tmp_index_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_index_path, index_path)
    return manifest


def process_folder(folder: Path, anatomy: str, dataset: str,
                   force: bool, dry_run: bool,
                   relink_enabled: bool = True,
                   relink_cfg: RelinkConfig | None = None,
                   sex_cleaning_cfg: dict | None = None) -> dict:
    preproc = folder / "preprocessing"
    result = {"folder": str(folder), "status": "skipped", "message": "", "counts": {}}
    if not preproc.exists():
        result["message"] = "no preprocessing/ dir"
        return result

    stem = f"preprocessed_bout_{anatomy}_{dataset}"
    fly0 = preproc / f"{stem}_fly0.h5"
    fly1 = preproc / f"{stem}_fly1.h5"
    if not fly0.exists() or not fly1.exists():
        result["message"] = "no fly0/fly1 pair (single-fly dataset)"
        return result

    out_paths = {
        "fly0_only": preproc / f"{stem}_fly0_only.h5",
        "fly1_only": preproc / f"{stem}_fly1_only.h5",
        "both": preproc / f"{stem}_both.h5",
    }
    index_path = preproc / f"{stem}_split_index.json"

    print(f"\n[split] {folder.name}")
    if dry_run:
        result["status"] = "dry-run"
        result["message"] = f"would split {fly0.name} + {fly1.name}"
        return result

    try:
        manifest = classify_and_split(
            fly0, fly1, out_paths, index_path, force=force,
            relink_enabled=relink_enabled, relink_cfg=relink_cfg,
            sex_cleaning_cfg=sex_cleaning_cfg,
        )
        counts = manifest.get("counts", {})
        result["counts"] = counts
        total = sum(counts.get(b, 0) for b in BUCKETS)
        if total == 0:
            print(f"  [skip] no valid bouts captured in {folder.name}")
            result["status"] = "no_bouts"
            result["message"] = "no bouts above thresholds"
        else:
            result["status"] = "success"
            result["message"] = ", ".join(f"{b}={counts.get(b,0)}" for b in BUCKETS)
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["status"] = "error"
        result["message"] = str(e)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--anatomy", default="v1")
    ap.add_argument("--base-dir", type=Path, required=True)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    # Joint bout-aware identity relink (runs before bucket loop). Default ON.
    ap.add_argument("--no-identity-relink", dest="identity_relink",
                    action="store_false", default=True,
                    help="Disable the joint bout-aware identity relink "
                         "(restores legacy per-fly behavior).")
    ap.add_argument("--relink-swap-ratio", type=float, default=0.7)
    ap.add_argument("--relink-bl-tube-factor", type=float, default=0.25)
    # Body-length-relative step ceiling — unit-agnostic for cm / mm / etc.
    ap.add_argument("--relink-max-step-bl", type=float, default=0.5)
    ap.add_argument("--relink-max-step-abs", type=float, default=0.0)
    ap.add_argument("--relink-nan-resume-frames", type=int, default=3)
    ap.add_argument("--relink-velocity-alpha", type=float, default=0.5)
    ap.add_argument("--relink-body-length-alpha", type=float, default=0.05)
    ap.add_argument("--relink-body-length-weight", type=float, default=0.5)
    # Per-sex aggressive cleaning (song analysis → sex ID → clean female).
    ap.add_argument("--sex-cleaning", action="store_true", default=False,
                    help="Enable per-sex aggressive cleaning (female gets "
                         "multi-pass despike + medfilt, male stays conservative).")
    ap.add_argument("--sex-cleaning-config", type=Path, default=None,
                    help="YAML file with sex_cleaning config. If omitted, "
                         "uses defaults or reads from the preprocessed h5 info.")
    args = ap.parse_args()

    relink_cfg = RelinkConfig(
        swap_ratio=args.relink_swap_ratio,
        bl_tube_factor=args.relink_bl_tube_factor,
        max_step_bl=args.relink_max_step_bl,
        max_step_abs=args.relink_max_step_abs,
        nan_resume_frames=args.relink_nan_resume_frames,
        velocity_alpha=args.relink_velocity_alpha,
        body_length_alpha=args.relink_body_length_alpha,
        body_length_weight=args.relink_body_length_weight,
    )

    # Build sex_cleaning config
    sex_cleaning_cfg: dict | None = None
    if args.sex_cleaning:
        if args.sex_cleaning_config is not None and args.sex_cleaning_config.exists():
            import yaml
            with open(args.sex_cleaning_config) as fh:
                sex_cleaning_cfg = yaml.safe_load(fh).get("sex_cleaning", {})
            sex_cleaning_cfg["enabled"] = True
        else:
            # Sensible defaults matching preprocessing.yaml
            sex_cleaning_cfg = {
                "enabled": True,
                "female": {
                    "despike_iterations": 6,
                    "medfilt_despike": {"enabled": True, "kernel": 7, "threshold_factor": 10.0},
                },
                "male": {
                    "despike_iterations": 1,
                    "medfilt_despike": {"enabled": False},
                },
            }
        print(f"[sex-cleaning] enabled: female despike_iter="
              f"{sex_cleaning_cfg['female']['despike_iterations']}, "
              f"medfilt={sex_cleaning_cfg['female'].get('medfilt_despike', {}).get('enabled', False)}")

    if not args.base_dir.exists():
        print(f"Error: base dir not found: {args.base_dir}", file=sys.stderr)
        sys.exit(1)

    folders = _find_folders(args.base_dir)
    if not folders:
        print(f"No Predictions_3D_* folders under {args.base_dir}")
        sys.exit(0)

    print(f"Found {len(folders)} prediction folder(s)")
    results = [
        process_folder(
            f, args.anatomy, args.dataset, args.force, args.dry_run,
            relink_enabled=args.identity_relink, relink_cfg=relink_cfg,
            sex_cleaning_cfg=sex_cleaning_cfg,
        )
        for f in folders
    ]

    print("\n" + "=" * 60)
    print("SPLIT SUMMARY")
    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # 'no_bouts' and 'skipped' are not failures — pipeline should continue.
    if counts.get("error", 0) > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
